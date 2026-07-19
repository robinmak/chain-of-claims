"""Stage 10 — Hallucination score.

A claim counts as a hallucination if it is REFUTED, or has a citation failure
(out_of_context / uncited_unsupported). Reported overall and per claim type, so
computational hallucination (derived_quantity) is separable from textual — the
diagnostic FinGround emphasises.
"""

from __future__ import annotations

from collections import defaultdict

from .. import db
from ..models import Claim, CitationStatus, FactCheckResult, RunScores


_CITATION_FAILURES = {CitationStatus.OUT_OF_CONTEXT, CitationStatus.UNCITED_UNSUPPORTED}


def _is_hallucination(fc: FactCheckResult, cite: CitationStatus) -> bool:
    return fc == FactCheckResult.REFUTED or cite in _CITATION_FAILURES


def compute(
    run_id: str,
    claims: list[Claim],
    explainability: float | None,
    coverage: tuple[int, int, int] | None = None,
    verification_skipped: bool = False,
) -> RunScores:
    """coverage = (n_gold_matched, n_gold, n_extracted_matched); None if no gold set.

    When verification_skipped (no source materials), grounding/fact-check/citation did
    not run: explainability and hallucination are reported as None (not applicable)
    rather than 0, since absence of evidence is not evidence of error. Claim extraction,
    typing, argument structure, and (if a gold set is given) coverage still apply.
    """
    checkworthy = [c for c in claims if c.checkworthy]
    n_ck = len(checkworthy)

    if verification_skipped:
        scores = RunScores(
            explainability=None,
            hallucination=None,
            hallucination_by_type={},
            verification_skipped=True,
            n_claims=len(claims),
            n_checkworthy=n_ck,
            coverage_note="verification skipped — no source materials supplied",
        )
        _apply_coverage(scores, coverage, claims)
        return scores

    total_hallu = 0
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_hallu: dict[str, int] = defaultdict(int)

    for c in checkworthy:
        v = db.get_verdict(c.id) if c.id is not None else None
        fc = v.factcheck_result if v else FactCheckResult.NOT_CHECKED
        cite = v.citation_status if v else CitationStatus.UNCITED_UNSUPPORTED
        by_type_total[c.type.value] += 1
        if _is_hallucination(fc, cite):
            total_hallu += 1
            by_type_hallu[c.type.value] += 1

    hallucination = (total_hallu / n_ck) if n_ck else 0.0
    by_type = {
        t: (by_type_hallu[t] / by_type_total[t]) for t in by_type_total if by_type_total[t]
    }

    scores = RunScores(
        explainability=round(explainability, 4) if explainability is not None else None,
        hallucination=round(hallucination, 4),
        hallucination_by_type={k: round(v, 4) for k, v in by_type.items()},
        n_claims=len(claims),
        n_checkworthy=n_ck,
    )
    _apply_coverage(scores, coverage, claims)
    return scores


def _apply_coverage(scores: RunScores, coverage, claims) -> None:
    if coverage is not None and coverage[1] > 0:
        matched, n_gold, ext_matched = coverage
        # recall = gold claims recovered; precision = extracted claims that hit gold
        scores.coverage = round(matched / n_gold, 4)
        scores.coverage_precision = (
            round(ext_matched / len(claims), 4) if claims else 0.0
        )
        scores.n_gold = n_gold
        scores.n_gold_matched = matched
        scores.coverage_note = (
            f"coverage=recall {matched}/{n_gold} gold claims recovered"
        )
