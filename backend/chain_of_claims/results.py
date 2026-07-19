"""Assemble a full run result (claims + structure + evidence + verdicts) for the API."""

from __future__ import annotations

import json

from . import db
from .models import RunScores


def build_result(run_id: str) -> dict | None:
    run = db.get_run(run_id)
    if not run:
        return None

    claims = db.get_claims(run_id)
    chunks = {ch.id: ch for ch in db.get_chunks(run_id)}

    claim_views = []
    for c in claims:
        triplet = db.get_triplet(c.id) if c.id is not None else None
        audit = db.get_warrant_audit(c.id) if c.id is not None else None
        verdict = db.get_verdict(c.id) if c.id is not None else None
        rel_ids = db.get_relevant_chunk_ids(c.id) if c.id is not None else []
        evidence = [
            {"locator": chunks[i].locator, "text": chunks[i].text, "source": chunks[i].source}
            for i in rel_ids
            if i in chunks
        ]
        claim_views.append(
            {
                "id": c.id,
                "text": c.text,
                "type": c.type.value,
                "checkworthy": c.checkworthy,
                "cited_source": c.cited_source,
                "triplet": triplet.model_dump() if triplet else None,
                "warrant_audit": audit.model_dump() if audit else None,
                "verdict": verdict.model_dump(mode="json") if verdict else None,
                "evidence": evidence,
            }
        )

    scores = None
    if run.get("scores_json"):
        scores = json.loads(run["scores_json"])

    return {
        "id": run_id,
        "status": run["status"],
        "stage": run["stage"],
        "error": run.get("error"),
        "scores": scores,
        "claims": claim_views,
        "n_chunks": len(chunks),
    }
