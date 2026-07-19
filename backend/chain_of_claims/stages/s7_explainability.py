"""Stage 7 — Explainability score.

The headline grounding-coverage metric: of the check-worthy claims, what fraction are
grounded in at least one relevant evidence chunk. This is the axis DEER's rubric and
FinGround's grounding stage both target, so external baselines are comparable.
"""

from __future__ import annotations

from .. import db
from ..models import Claim


def compute(run_id: str, claims: list[Claim]) -> float:
    checkworthy = [c for c in claims if c.checkworthy]
    if not checkworthy:
        return 0.0
    grounded = 0
    for c in checkworthy:
        if c.id is not None and db.get_relevant_chunk_ids(c.id):
            grounded += 1
    return grounded / len(checkworthy)
