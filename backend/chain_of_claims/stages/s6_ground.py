"""Stage 6 — grounding: mark which evidence chunks are relevant to each claim.

Retrieval-equalized: candidate chunks come only from the user-supplied sources, so we
measure verification rather than retrieval luck. To keep LLM calls bounded, we first
SHORTLIST candidate chunks, then ask the model to judge which actually support the
claim.

Shortlisting is numeric-aware. Financial claims hinge on figures, and the true
supporting evidence is often a short table cell (e.g. "Gross profit / Q3 FY2024 = 500").
Plain Jaccard (intersection / union) penalises short cells because the union is
dominated by claim tokens the cell lacks, so the right cell falls outside the top-k and
the fact-checker later abstains for "not enough evidence". We fix this with:
  * an OVERLAP-COEFFICIENT token score (intersection / size of the smaller set), which
    does not punish short chunks, and
  * NUMBER NORMALISATION + a strong boost when a chunk shares a figure with the claim,
  * a GUARANTEE that any chunk sharing a normalised number with the claim is included,
    even beyond the top-k ranked by text.
"""

from __future__ import annotations

import re

from ..config import TRANSFORM_MODEL
from ..llm.client import get_provider
from ..models import Claim, EvidenceChunk
from .s2_prune import _tokens  # reuse the tokenizer

from pydantic import BaseModel


class _RelevantLocators(BaseModel):
    locators: list[str]


_SYSTEM = (
    "You decide which evidence snippets support a financial claim. A snippet is "
    "relevant only if it provides evidence bearing on the claim's truth (a figure, a "
    "fact, a comparison). Return the locators of relevant snippets only."
)

_PROMPT = """CLAIM: {claim}

CANDIDATE EVIDENCE (locator: text):
{candidates}

Return the locators of the snippets that are relevant to the claim.
"""

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
# tokens worth matching on: 3+ char words (drops "a", "of", "in", "was")
_STOP = {"the", "and", "for", "was", "were", "with", "from", "its", "has", "had"}


def _norm_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").rstrip("%").strip()


def _numbers(text: str) -> set[str]:
    return {_norm_num(n) for n in _NUM.findall(text) if _norm_num(n)}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= 3 and t not in _STOP}


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _score(claim_toks: set[str], claim_nums: set[str], chunk: EvidenceChunk) -> float:
    """Higher is more relevant. Numeric agreement dominates; text overlap breaks ties."""
    ctoks = _content_tokens(chunk.text)
    cnums = _numbers(chunk.text)
    text_score = _overlap_coeff(claim_toks, ctoks)
    shared_nums = len(claim_nums & cnums)
    # A shared figure is the strongest signal for a financial claim.
    num_score = 1.0 if shared_nums else 0.0
    return num_score * 2.0 + text_score


def _shortlist(claim: Claim, chunks: list[EvidenceChunk], k: int = 16) -> list[EvidenceChunk]:
    claim_toks = _content_tokens(claim.text)
    claim_nums = _numbers(claim.text)

    scored = sorted(
        chunks, key=lambda c: _score(claim_toks, claim_nums, c), reverse=True
    )
    top = [c for c in scored if _score(claim_toks, claim_nums, c) > 0][:k]

    # Guarantee: every chunk that shares a figure with the claim is a candidate,
    # even if it ranked below the top-k on text. This is what stops the fact-checker
    # from abstaining on a true figure whose cell was crowded out. Capped so a common
    # figure (e.g. a share count) cannot balloon the prompt.
    if claim_nums:
        top_ids = {c.id for c in top}
        extra = 0
        for c in scored:  # scored is ranked, so we add the most relevant matches first
            if extra >= k:
                break
            if c.id not in top_ids and (claim_nums & _numbers(c.text)):
                top.append(c)
                top_ids.add(c.id)
                extra += 1

    # Fall back to the best-by-text handful if nothing scored (e.g. pure prose claim).
    return top or scored[:k]


def run(claim: Claim, chunks: list[EvidenceChunk]) -> list[tuple[int, int, bool, str | None]]:
    """Return grounding rows: (claim_id, chunk_id, relevant, rationale)."""
    shortlist = _shortlist(claim, chunks)
    if not shortlist:
        return []
    provider = get_provider()
    candidates = "\n".join(f"{c.locator}: {c.text}" for c in shortlist)
    result = provider.structured_output(
        system=_SYSTEM,
        prompt=_PROMPT.format(claim=claim.text, candidates=candidates),
        schema=_RelevantLocators,
        model=TRANSFORM_MODEL,
    )
    relevant = set(result.locators)
    rows: list[tuple[int, int, bool, str | None]] = []
    for c in shortlist:
        rows.append((claim.id, c.id, c.locator in relevant, None))
    return rows
