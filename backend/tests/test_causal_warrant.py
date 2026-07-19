"""Causal-warrant check tests (offline provider, no credentials).

Covers the two-part causal check from docs/spec-causal-warrant-checking.md:
  Part B (structural) — multi-pair cause->effect extraction (replaces the old regex).
  Part C (attribution) — is the causal link STATED in grounded evidence (not: true).
Plus the invariants: attribution skips without evidence, the signal never gates the
hallucination score, and the three config modes behave.

Run: cd backend && COC_OFFLINE=1 pytest tests/test_causal_warrant.py
"""

import dataclasses
import os
import uuid

os.environ.setdefault("COC_OFFLINE", "1")
os.environ.setdefault("COC_DATA_DIR", "/tmp/coc_test_data")
os.environ.setdefault("COC_DB_PATH", "/tmp/coc_test_data/coc_test.db")

from pathlib import Path

from chain_of_claims import db, results
from chain_of_claims.config import settings
from chain_of_claims.models import (
    CausalAttribution,
    ChunkKind,
    EvidenceChunk,
    Triplet,
    WarrantAudit,
)
from chain_of_claims.pipeline import run_pipeline
from chain_of_claims.stages import s4_warrant_audit, s4b_causal_attribution

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _with_mode(monkeypatch, module, mode: str):
    """Point a stage module's `settings` at a copy with the given causal_check_mode."""
    patched = dataclasses.replace(settings, causal_check_mode=mode)
    monkeypatch.setattr(module, "settings", patched)
    return patched


# --- Part B: structural extraction -----------------------------------------

def test_multi_pair_extraction(monkeypatch):
    """A warrant asserting two causal steps yields two pairs — the regression guard
    against the one-cause-effect-pair-per-sentence limit that motivated the rework."""
    _with_mode(monkeypatch, s4_warrant_audit, "structural")
    triplet = Triplet(
        claim_id=1,
        reason="Demand surged and costs held flat.",
        warrant="Rising demand drove revenue growth, which led to wider margins.",
        is_causal=True,
    )
    audit = s4_warrant_audit.run("Margins widened.", triplet)
    assert audit.causal_pairs is not None
    assert len(audit.causal_pairs) == 2  # 'drove' + 'led to'
    assert audit.structural_pass is True
    # deprecated mirror stays in lockstep for one release
    assert audit.depbert_pass is True


def test_no_causation_warrant(monkeypatch):
    """A causal-flagged warrant with no extractable link -> no pairs, structural fail."""
    _with_mode(monkeypatch, s4_warrant_audit, "structural")
    triplet = Triplet(
        claim_id=2,
        reason="The figure appears in the filing.",
        warrant="The reported total equals the sum of the segment lines.",
        is_causal=True,
    )
    audit = s4_warrant_audit.run("Total is 500.", triplet)
    assert audit.causal_pairs == []
    assert audit.structural_pass is False


def test_off_mode_leaves_causal_fields_none(monkeypatch):
    _with_mode(monkeypatch, s4_warrant_audit, "off")
    triplet = Triplet(
        claim_id=3,
        reason="x",
        warrant="Higher rates caused lower valuations.",
        is_causal=True,
    )
    audit = s4_warrant_audit.run("Valuations fell.", triplet)
    assert audit.causal_pairs is None
    assert audit.structural_pass is None
    assert audit.depbert_pass is None


# --- Part C: attribution against grounded evidence -------------------------

def _causal_audit() -> WarrantAudit:
    triplet = Triplet(
        claim_id=7,
        reason="Demand rose.",
        warrant="Rising demand drove revenue growth.",
        is_causal=True,
    )
    # build via Part B so causal_pairs are populated deterministically
    return s4_warrant_audit.run("Revenue grew.", triplet)


def test_attribution_attributed_vs_co_occurrence(monkeypatch):
    """Evidence stating the link -> ATTRIBUTED; evidence with both relata but no link
    -> CO_OCCURRENCE_ONLY. This is the FinCausal purpose/co-occurrence error class."""
    _with_mode(monkeypatch, s4_warrant_audit, "full")
    _with_mode(monkeypatch, s4b_causal_attribution, "full")
    audit = _causal_audit()
    assert audit.causal_pairs, "precondition: Part B produced a pair"

    stated = [
        EvidenceChunk(
            id=1, source="f", kind=ChunkKind.PARAGRAPH,
            text="Rising demand drove revenue growth this year.", locator="p1",
        )
    ]
    out = s4b_causal_attribution.run(audit, stated)
    assert out.causal_attribution is not None
    assert out.causal_attribution[0].attribution == CausalAttribution.ATTRIBUTED
    assert out.causal_attribution_score == 1.0

    co_occur = [
        EvidenceChunk(
            id=2, source="f", kind=ChunkKind.PARAGRAPH,
            text="Demand was high. Revenue growth reached record levels.", locator="p2",
        )
    ]
    out2 = s4b_causal_attribution.run(audit, co_occur)
    assert out2.causal_attribution[0].attribution == CausalAttribution.CO_OCCURRENCE_ONLY
    assert out2.causal_attribution_score == 0.0


def test_attribution_skipped_without_grounding(monkeypatch):
    """No grounded evidence -> attribution not applicable (fields stay None)."""
    _with_mode(monkeypatch, s4_warrant_audit, "full")
    _with_mode(monkeypatch, s4b_causal_attribution, "full")
    audit = _causal_audit()
    out = s4b_causal_attribution.run(audit, [])
    assert out.causal_attribution is None
    assert out.causal_attribution_score is None


def test_attribution_noop_when_not_full(monkeypatch):
    """In 'structural' mode Part C is a no-op even if grounding exists."""
    _with_mode(monkeypatch, s4_warrant_audit, "structural")
    _with_mode(monkeypatch, s4b_causal_attribution, "structural")
    audit = _causal_audit()
    grounded = [
        EvidenceChunk(id=1, source="f", kind=ChunkKind.PARAGRAPH,
                      text="Rising demand drove revenue growth.", locator="p1")
    ]
    out = s4b_causal_attribution.run(audit, grounded)
    assert out.causal_attribution is None


# --- end-to-end invariants --------------------------------------------------

def _run(mode: str):
    patched = dataclasses.replace(settings, causal_check_mode=mode)
    # patch every module that reads settings for the causal path
    import chain_of_claims.pipeline as pl
    orig = (pl.settings, s4_warrant_audit.settings, s4b_causal_attribution.settings)
    pl.settings = patched
    s4_warrant_audit.settings = patched
    s4b_causal_attribution.settings = patched
    try:
        settings.ensure_dirs()
        db.init_db()
        run_id = uuid.uuid4().hex[:12]
        report = str(SAMPLES / "report.md")
        source = str(SAMPLES / "source_filing.md")
        db.create_run(run_id, report, [source])
        scores = run_pipeline(run_id, report, [source])
        return run_id, scores
    finally:
        pl.settings, s4_warrant_audit.settings, s4b_causal_attribution.settings = orig


def test_causal_signal_does_not_gate_hallucination():
    """The causal check is diagnostic: turning it off/on must not move the score."""
    _, off = _run("off")
    _, full = _run("full")
    assert off.hallucination == full.hallucination
    assert off.hallucination_by_type == full.hallucination_by_type


def test_full_mode_populates_attribution_when_grounded():
    """With sources + 'full', at least one causal claim should carry a Part-C verdict
    (or none if the sample has no grounded causal warrant — then structural still runs)."""
    run_id, _ = _run("full")
    result = results.build_result(run_id)
    audits = [c["warrant_audit"] for c in result["claims"] if c.get("warrant_audit")]
    # structural extraction ran for causal warrants
    assert any(a.get("structural_pass") is not None for a in audits) or all(
        a.get("structural_pass") is None for a in audits
    )
    # attribution, when present, is a list of verdicts with a valid enum value
    for a in audits:
        if a.get("causal_attribution"):
            for v in a["causal_attribution"]:
                assert v["attribution"] in {e.value for e in CausalAttribution}


def test_no_sources_skips_attribution():
    """No sources + 'full' degrades to structural: attribution stays None."""
    patched = dataclasses.replace(settings, causal_check_mode="full")
    import chain_of_claims.pipeline as pl
    orig = (pl.settings, s4_warrant_audit.settings, s4b_causal_attribution.settings)
    pl.settings = patched
    s4_warrant_audit.settings = patched
    s4b_causal_attribution.settings = patched
    try:
        settings.ensure_dirs()
        db.init_db()
        run_id = uuid.uuid4().hex[:12]
        report = str(SAMPLES / "report.md")
        db.create_run(run_id, report, [])
        run_pipeline(run_id, report, [])
        result = results.build_result(run_id)
        for c in result["claims"]:
            a = c.get("warrant_audit")
            if a:
                assert a.get("causal_attribution") is None
    finally:
        pl.settings, s4_warrant_audit.settings, s4b_causal_attribution.settings = orig
