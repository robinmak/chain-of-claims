"""Pipeline controller: orchestrates stages A->D deterministically.

Each stage writes its output to SQLite before the next runs, so a run is inspectable
and a failure localizes to a stage. Progress is emitted through an optional callback
(the API turns it into Server-Sent Events).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Callable

from . import db
from .config import settings
from .ingest.chunker import chunk_blocks
from .ingest.documents import parse_document
from .models import GoldSet, RunScores, RunStatus, Triplet
from .stages import (
    s1_extract,
    s1b_coverage,
    s2_prune,
    s3_toulmin,
    s4_warrant_audit,
    s4b_causal_attribution,
    s6_ground,
    s7_explainability,
    s8_factcheck,
    s9_citations,
    s10_hallucination,
)

ProgressFn = Callable[[str, str], None]  # (stage_label, human_message)


def _load_gold(path: str | None) -> GoldSet | None:
    """Load an optional gold-claim set (JSON: {"claims":[{"text":...}, ...]})."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return GoldSet.model_validate(data)
    except Exception:  # noqa: BLE001 — a malformed gold file must not fail the run
        return None


def _read_report(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        blocks = parse_document(path)
        parts: list[str] = []
        for b in blocks:
            parts.extend(b.paragraphs)
        return "\n\n".join(parts)
    return p.read_text(encoding="utf-8", errors="replace")


def _context_for(report_text: str, claim_text: str, window: int = 240) -> str:
    idx = report_text.find(claim_text[:40])
    if idx < 0:
        return ""
    start = max(0, idx - window)
    end = min(len(report_text), idx + len(claim_text) + window)
    return report_text[start:end]


def run_pipeline(
    run_id: str,
    report_path: str,
    source_paths: list[str],
    progress: ProgressFn | None = None,
    gold_path: str | None = None,
) -> RunScores:
    def emit(stage: str, msg: str) -> None:
        db.set_run_stage(run_id, stage)
        if progress:
            progress(stage, msg)

    db.set_run_status(run_id, RunStatus.RUNNING, stage="starting")
    try:
        # --- Ingest sources (Stage 5 assets) ---
        emit("ingest", "Parsing source documents")
        all_chunks = []
        for sp in source_paths:
            blocks = parse_document(sp)
            all_chunks.extend(chunk_blocks(blocks))
        chunks = db.replace_chunks(run_id, all_chunks)
        emit("ingest", f"Extracted {len(chunks)} evidence chunks")

        report_text = _read_report(report_path)

        # --- Stage 1: extract claims ---
        emit("s1_extract", "Extracting atomic claims")
        claim_list = s1_extract.run(report_text)

        # --- Stage 2: prune ---
        emit("s2_prune", "Pruning duplicates and boilerplate")
        pruned = s2_prune.run(claim_list.claims)
        claims = db.replace_claims(run_id, pruned)
        emit("s2_prune", f"{len(claims)} claims after pruning")

        # --- Stage 1b: gold-claim coverage (optional) ---
        coverage = None
        gold = _load_gold(gold_path)
        if gold and gold.claims:
            emit("s1b_coverage", "Scoring claim coverage against gold set")
            coverage = s1b_coverage.compute(gold.claims, claims)
            emit("s1b_coverage", f"coverage {coverage[0]}/{coverage[1]} gold claims")

        # --- Stage 3+4: Toulmin structure + warrant audit ---
        emit("s3_toulmin", "Reconstructing Toulmin structure and auditing warrants")
        audits_by_claim = {}
        for c in claims:
            ctx = _context_for(report_text, c.text)
            triplet: Triplet = s3_toulmin.run(c, ctx)
            db.save_triplet(triplet)
            audit = s4_warrant_audit.run(c.text, triplet)
            db.save_warrant_audit(audit)
            if c.id is not None:
                audits_by_claim[c.id] = audit

        # Verification (grounding, fact-check, citation) requires source materials.
        # Without them we still extract claims and reconstruct argument structure, but
        # the evidence-dependent scores are reported as not-applicable rather than 0.
        has_sources = len(chunks) > 0

        if has_sources:
            # --- Stage 6: grounding ---
            emit("s6_ground", "Grounding claims in evidence")
            for c in claims:
                rows = s6_ground.run(c, chunks)
                if rows:
                    db.save_groundings(rows)

            # --- Stage 4b: causal attribution (Part C) — needs grounded evidence ---
            if settings.causal_attribution_enabled:
                emit("s4b_causal", "Checking causal attribution against evidence")
                chunk_by_id = {ch.id: ch for ch in chunks}
                for c in claims:
                    audit = audits_by_claim.get(c.id)
                    if audit is None or not audit.causal_pairs:
                        continue
                    relevant_ids = (
                        db.get_relevant_chunk_ids(c.id) if c.id is not None else []
                    )
                    grounded = [
                        chunk_by_id[i] for i in relevant_ids if i in chunk_by_id
                    ]
                    audit = s4b_causal_attribution.run(audit, grounded)
                    db.save_warrant_audit(audit)

            # --- Stage 7: explainability ---
            emit("s7_explainability", "Computing explainability score")
            explainability = s7_explainability.compute(run_id, claims)

            # --- Stage 8+9: fact-check + citation status ---
            emit("s8_factcheck", "Type-routed fact-checking")
            chunk_by_id = {ch.id: ch for ch in chunks}
            for c in claims:
                relevant_ids = db.get_relevant_chunk_ids(c.id) if c.id is not None else []
                grounded = [chunk_by_id[i] for i in relevant_ids if i in chunk_by_id]
                verdict = s8_factcheck.run(c, grounded)
                citation = s9_citations.compute(c, bool(grounded), verdict)
                verdict = verdict.model_copy(update={"citation_status": citation})
                db.save_verdict(verdict)
        else:
            emit("s6_ground", "No source materials — skipping evidence verification")
            explainability = None

        # --- Stage 10: hallucination score ---
        emit("s10_hallucination", "Computing scores")
        scores = s10_hallucination.compute(
            run_id, claims, explainability, coverage, verification_skipped=not has_sources
        )
        db.save_scores(run_id, scores)

        db.set_run_status(run_id, RunStatus.DONE, stage="done")
        emit("done", "Complete")
        return scores

    except Exception as e:  # noqa: BLE001
        db.set_run_status(
            run_id, RunStatus.FAILED, stage="error", error=f"{e}\n{traceback.format_exc()}"
        )
        if progress:
            progress("error", str(e))
        raise
