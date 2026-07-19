"""Stage 9 — citation status.

Two distinct failure modes (DEER + attribution-alignment framing):
- out_of_context: the report cited a source, but the grounded evidence supports a
  DIFFERENT claim than the one it's attached to (mis-attribution even if the evidence
  is itself true).
- uncited_unsupported: no citation AND no supporting evidence in the sources — the
  worst case.
We derive citation status from (a) whether the report attached a citation and (b) the
Stage-6 grounding / Stage-8 verdict, so it stays consistent with the evidence layer.
"""

from __future__ import annotations

from ..models import CitationStatus, Claim, ClaimType, FactCheckResult, Verdict


def compute(claim: Claim, has_grounding: bool, verdict: Verdict) -> CitationStatus:
    if claim.type == ClaimType.FORWARD_LOOKING or not claim.checkworthy:
        return CitationStatus.NOT_APPLICABLE

    cited = bool(claim.cited_source)
    supported = verdict.factcheck_result == FactCheckResult.SUPPORTED

    if cited:
        # Citation present. If the claim is not actually supported by grounded
        # evidence, the citation points somewhere that doesn't back this claim.
        if has_grounding and supported:
            return CitationStatus.SUPPORTED
        return CitationStatus.OUT_OF_CONTEXT

    # No citation attached by the report.
    if has_grounding and supported:
        return CitationStatus.UNCITED_SUPPORTED
    return CitationStatus.UNCITED_UNSUPPORTED
