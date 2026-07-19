"""Stage 4b — causal attribution check (Part C).

Runs AFTER Stage 6 grounding, only for causal warrants whose claim has grounded
evidence. For each cause->effect pair extracted in Stage 4 (Part B), it asks whether
the grounded evidence STATES/SUPPORTS the causal link — a grounding question — versus
merely mentioning both relata (co-occurrence), describing a purpose/concessive relation
(the dominant error class in FinCausal 2025's analysis), or contradicting it.

Boundary (Corr2Cause [Jin et al. 2024]): this NEVER asks whether the causation is true
in the world — off-the-shelf LLMs infer causation-from-correlation near-random. There is
no "true"/"false" verdict; only attribution against the supplied evidence.

The result is DIAGNOSTIC: it is surfaced in the per-claim audit and does not gate the
hallucination score.
"""

from __future__ import annotations

from ..config import JUDGE_MODEL, settings
from ..llm.client import get_provider
from ..models import (
    CausalAttribution,
    CausalPairVerdict,
    EvidenceChunk,
    WarrantAudit,
)

_SYSTEM = (
    "You judge whether a causal link is STATED or SUPPORTED by the given evidence. You "
    "are NOT judging whether the causation is true in the world — only whether the "
    "evidence asserts this cause->effect link. Choose one attribution:\n"
    "- attributed: the evidence states or supports the causal link.\n"
    "- co_occurrence_only: the evidence mentions both items but not a causal link "
    "between them.\n"
    "- purpose_or_concessive: the evidence describes a purpose/goal or a concession "
    "(although, despite), not a cause.\n"
    "- contradicted: the evidence states a different or opposite link.\n"
    "Be conservative: if the link is not actually stated, do not answer 'attributed'."
)

_PROMPT = """CAUSE: {cause}
EFFECT: {effect}

EVIDENCE (grounded chunks with locators):
{evidence_block}
"""


def run(audit: WarrantAudit, grounded: list[EvidenceChunk]) -> WarrantAudit:
    """Augment a WarrantAudit with Part-C attribution for its extracted causal pairs.

    Skipped (fields stay None) when: attribution is disabled (mode != 'full'), the
    warrant is non-causal / had no extracted pairs, or the claim has no grounded
    evidence. Absence of evidence is not treated as an error.
    """
    if not settings.causal_attribution_enabled:
        return audit
    if not audit.causal_pairs:
        return audit
    if not grounded:
        # No grounding -> attribution not applicable (consistent with S == empty rule).
        return audit

    provider = get_provider()
    evidence_block = "\n".join(f"{ch.locator}: {ch.text}" for ch in grounded)

    verdicts: list[CausalPairVerdict] = []
    for pair in audit.causal_pairs:
        v: CausalPairVerdict = provider.structured_output(
            system=_SYSTEM,
            prompt=_PROMPT.format(
                cause=pair.cause,
                effect=pair.effect,
                evidence_block=evidence_block,
            ),
            schema=CausalPairVerdict,
            model=JUDGE_MODEL,  # judgement-heavy -> stronger tier
        )
        verdicts.append(v)

    n = len(verdicts)
    attributed = sum(1 for v in verdicts if v.attribution == CausalAttribution.ATTRIBUTED)
    return audit.model_copy(
        update={
            "causal_attribution": verdicts,
            "causal_attribution_score": (attributed / n) if n else None,
        }
    )
