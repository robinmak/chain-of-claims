"""End-to-end pipeline tests (offline provider).

Run: cd backend && COC_OFFLINE=1 pytest
These lock in the behaviors the design promises: atomic typing, cell-level table
chunking, a caught computational/extracted error, forward-looking skipped, and
populated per-type scores.
"""

import os
import uuid

os.environ.setdefault("COC_OFFLINE", "1")
os.environ.setdefault("COC_DATA_DIR", "/tmp/coc_test_data")
os.environ.setdefault("COC_DB_PATH", "/tmp/coc_test_data/coc_test.db")

from pathlib import Path

from chain_of_claims import db, results
from chain_of_claims.config import settings
from chain_of_claims.ingest.chunker import chunk_blocks
from chain_of_claims.ingest.documents import parse_document
from chain_of_claims.models import ChunkKind, FactCheckResult
from chain_of_claims.pipeline import run_pipeline

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _fresh_run():
    settings.ensure_dirs()
    db.init_db()
    run_id = uuid.uuid4().hex[:12]
    report = str(SAMPLES / "report.md")
    source = str(SAMPLES / "source_filing.md")
    db.create_run(run_id, report, [source])
    scores = run_pipeline(run_id, report, [source])
    return run_id, scores


def test_table_chunking_emits_cells():
    blocks = parse_document(str(SAMPLES / "source_filing.md"))
    chunks = chunk_blocks(blocks)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.TABLE_CELL in kinds
    assert ChunkKind.TABLE_ROW in kinds
    # a specific cell should be locatable
    assert any("Net income" in c.text and c.kind == ChunkKind.TABLE_CELL for c in chunks)


def test_pipeline_scores_populated():
    _, scores = _fresh_run()
    assert scores.n_claims > 0
    assert scores.n_checkworthy > 0
    assert 0.0 <= scores.explainability <= 1.0
    assert 0.0 <= scores.hallucination <= 1.0
    # per-type breakdown must be present
    assert scores.hallucination_by_type


def test_net_income_error_is_caught():
    run_id, _ = _fresh_run()
    result = results.build_result(run_id)
    net = [c for c in result["claims"] if "Net income" in c["text"]]
    assert net, "expected a net income claim"
    assert net[0]["verdict"]["factcheck_result"] == FactCheckResult.REFUTED.value


def test_forward_looking_not_checked():
    run_id, _ = _fresh_run()
    result = results.build_result(run_id)
    fwd = [c for c in result["claims"] if c["type"] == "forward_looking"]
    assert fwd, "expected a forward-looking claim"
    assert fwd[0]["verdict"]["factcheck_result"] == FactCheckResult.NOT_CHECKED.value


def test_no_sources_skips_verification_but_extracts_claims():
    """A document with no source materials still yields claims + structure; the
    verification-dependent scores are None (not 0), and verification_skipped is set."""
    settings.ensure_dirs()
    db.init_db()
    run_id = uuid.uuid4().hex[:12]
    report = str(SAMPLES / "report.md")
    db.create_run(run_id, report, [])
    scores = run_pipeline(run_id, report, [])  # no sources

    assert scores.n_claims > 0                    # claims still extracted
    assert scores.verification_skipped is True
    assert scores.explainability is None          # not applicable, not 0
    assert scores.hallucination is None
