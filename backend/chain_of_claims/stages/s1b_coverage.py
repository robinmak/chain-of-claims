"""Stage 1b — gold-claim coverage (recall).

Turns Stage 1's raw atomic count into a real coverage metric: given a human-annotated
GOLD set of claims the report *should* contain, measure how many were recovered by
extraction (recall) and how many extracted claims matched a gold claim (precision).

Matching must be SEMANTIC, not literal: the extractor legitimately rewords claims
("Net income reached $210M" vs "net income was $210 million"). We ask the model which
gold claims are covered by the extracted set. A deterministic numeric+lexical matcher
backs the offline path and serves as a sanity floor.
"""

from __future__ import annotations

from ..config import TRANSFORM_MODEL
from ..llm.client import get_provider
from ..models import Claim, CoverageMatch, GoldClaim
from .s6_ground import _content_tokens, _numbers, _overlap_coeff

_SYSTEM = (
    "You judge claim coverage. Given a list of GOLD claims (facts a report should "
    "state) and a list of EXTRACTED claims, decide which gold claims are covered by at "
    "least one extracted claim. Two claims match if they assert the same fact, even if "
    "worded differently or split differently. Return the indices of covered gold claims."
)

_PROMPT = """GOLD claims (0-based index):
{gold}

EXTRACTED claims:
{extracted}

Return the indices of the GOLD claims that are covered by at least one extracted claim.
"""


def _semantic_match(gold: list[GoldClaim], extracted: list[Claim]) -> set[int]:
    provider = get_provider()
    gold_block = "\n".join(f"{i}: {g.text}" for i, g in enumerate(gold))
    ext_block = "\n".join(f"- {c.text}" for c in extracted)
    result: CoverageMatch = provider.structured_output(
        system=_SYSTEM,
        prompt=_PROMPT.format(gold=gold_block, extracted=ext_block),
        schema=CoverageMatch,
        model=TRANSFORM_MODEL,
    )
    return {i for i in result.matched_gold_indices if 0 <= i < len(gold)}


def compute(gold: list[GoldClaim], extracted: list[Claim]) -> tuple[int, int, int]:
    """Return (n_gold_matched, n_gold, n_extracted_matched).

    recall    = n_gold_matched / n_gold        (gold claims recovered by extraction)
    precision = n_extracted_matched / n_extracted (extracted claims that hit some gold)

    These use different numerators: a gold claim may be covered by several extracted
    claims (atomic splitting), and an extracted claim may cover a gold claim. We count
    both sides so precision can never exceed 1.
    """
    if not gold:
        return 0, 0, 0

    matched_gold, matched_ext = self_match(gold, extracted)
    return len(matched_gold), len(gold), len(matched_ext)


def self_match(gold: list[GoldClaim], extracted: list[Claim]) -> tuple[set[int], set[int]]:
    """Compute covered gold indices AND matched extracted indices in one pass."""
    try:
        sem_gold = _semantic_match(gold, extracted)
    except Exception:  # noqa: BLE001
        sem_gold = set()

    matched_gold: set[int] = set(sem_gold)
    matched_ext: set[int] = set()

    ext = [(_content_tokens(c.text), _numbers(c.text)) for c in extracted]
    for gi, g in enumerate(gold):
        gt, gn = _content_tokens(g.text), _numbers(g.text)
        for ei, (et, en) in enumerate(ext):
            nums_ok = (not gn) or bool(gn & en)
            if nums_ok and _overlap_coeff(gt, et) >= 0.6:
                matched_gold.add(gi)
                matched_ext.add(ei)
    return matched_gold, matched_ext
