"""Live Anthropic smoke test.

Exercises the REAL Claude path end-to-end. Skips cleanly when ANTHROPIC_API_KEY is
absent, so it is safe in CI. Run explicitly with a key:

    cd backend && source .venv/bin/activate
    ANTHROPIC_API_KEY=sk-... python -m pytest tests/test_anthropic_smoke.py -v -s

Keep it small: a couple of structured-output/tool-loop calls plus one tiny pipeline
run, so the bill is a few cents. Override models via COC_TRANSFORM_MODEL /
COC_JUDGE_MODEL if the defaults are not enabled on your account.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

_HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

# Only force real mode when credentials are present, so merely *collecting* this
# module in a keyless run (e.g. CI) does not clobber the shared offline settings that
# the rest of the suite relies on.
if _HAS_KEY:
    os.environ["COC_OFFLINE"] = "0"
os.environ.setdefault("COC_DATA_DIR", "/tmp/coc_smoke_data")
os.environ.setdefault("COC_DB_PATH", "/tmp/coc_smoke_data/coc_smoke.db")
# A small panel keeps the live warrant-audit cost down.
os.environ.setdefault("COC_CQ_PANEL_SIZE", "1")

pytestmark = pytest.mark.skipif(
    not _HAS_KEY,
    reason="No Anthropic credentials set; skipping live smoke test",
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _provider():
    # Import here so the module imports even when the SDK/key are absent.
    from chain_of_claims.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def test_structured_output_roundtrip():
    """The provider returns a schema-valid, correctly-typed object from a real call."""
    from chain_of_claims.config import TRANSFORM_MODEL
    from chain_of_claims.models import ClaimList

    provider = _provider()
    report = "Acme revenue was $1,250 million in Q3 FY2024, up 25% year over year."
    out = provider.structured_output(
        system="You decompose financial text into atomic claims.",
        prompt=(
            "Decompose into atomic claims; assign a claim type and checkworthy flag.\n\n"
            f"REPORT:\n{report}"
        ),
        schema=ClaimList,
        model=TRANSFORM_MODEL,
    )
    assert isinstance(out, ClaimList)
    assert len(out.claims) >= 1
    # every claim has a valid enum type (validation already guarantees this)
    assert all(c.text for c in out.claims)


def test_causal_extraction_multi_pair():
    """Part B: the real model recovers BOTH causal links in a two-step warrant —
    the multi-pair property that the retired one-pair regex/tagger could not provide."""
    from chain_of_claims.config import TRANSFORM_MODEL
    from chain_of_claims.models import Triplet
    from chain_of_claims.stages import s4_warrant_audit

    triplet = Triplet(
        claim_id=1,
        reason="Demand surged while unit costs stayed flat.",
        warrant="Rising demand drove revenue growth, which in turn led to wider margins.",
        is_causal=True,
    )
    audit = s4_warrant_audit.run("Margins widened.", triplet)
    assert audit.causal_pairs is not None
    # two asserted links: demand->revenue growth, revenue growth->wider margins
    assert len(audit.causal_pairs) >= 2
    assert audit.structural_pass is True


def test_causal_attribution_not_asserted_for_co_occurrence():
    """Part C: when evidence merely states two facts without a causal link, the live
    model must NOT report ATTRIBUTED (the FinCausal co-occurrence error class)."""
    from chain_of_claims.models import (
        CausalAttribution,
        ChunkKind,
        EvidenceChunk,
        Triplet,
    )
    from chain_of_claims.stages import s4_warrant_audit, s4b_causal_attribution

    triplet = Triplet(
        claim_id=1,
        reason="Both figures appear in the filing.",
        warrant="The marketing spend increase drove the rise in customer count.",
        is_causal=True,
    )
    audit = s4_warrant_audit.run("Customer count rose.", triplet)
    # evidence states both facts but asserts NO causal link between them
    grounded = [
        EvidenceChunk(id=1, source="f", kind=ChunkKind.PARAGRAPH,
                      text="Marketing spend rose 12% in FY2024.", locator="p1"),
        EvidenceChunk(id=2, source="f", kind=ChunkKind.PARAGRAPH,
                      text="Customer count reached 4.1 million at year end.", locator="p2"),
    ]
    out = s4b_causal_attribution.run(audit, grounded)
    if out.causal_attribution:  # attribution only runs in 'full' mode
        assert all(
            v.attribution != CausalAttribution.ATTRIBUTED for v in out.causal_attribution
        ), "co-occurrence without a stated link must not be judged 'attributed'"


def test_tool_loop_uses_calculator():
    """The agentic loop can call the calculator tool and reach a verdict token."""
    from chain_of_claims.config import JUDGE_MODEL
    from chain_of_claims.llm.tools import CALCULATOR_TOOL, LOOKUP_TOOL, build_tool_impls
    from chain_of_claims.models import ChunkKind, EvidenceChunk

    provider = _provider()
    chunks = [
        EvidenceChunk(id=1, source="src", kind=ChunkKind.TABLE_CELL,
                      text="Income: Gross profit = 500", locator="t1 r1c1"),
        EvidenceChunk(id=2, source="src", kind=ChunkKind.TABLE_CELL,
                      text="Income: Total revenue = 1250", locator="t1 r0c1"),
    ]
    out = provider.tool_loop(
        system=(
            "You verify a computed figure. Use the calculator tool to reconstruct it. "
            "Conclude with one token: SUPPORTED, REFUTED, or NOT_ENOUGH_EVIDENCE."
        ),
        prompt=(
            "CLAIM: Gross margin was 40% (gross profit / revenue).\n\n"
            "EVIDENCE:\n[t1 r1c1] Gross profit = 500\n[t1 r0c1] Total revenue = 1250"
        ),
        tools=[CALCULATOR_TOOL, LOOKUP_TOOL],
        tool_impls=build_tool_impls(chunks),
        model=JUDGE_MODEL,
    )
    upper = out.upper()
    assert any(tok in upper for tok in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE"))


def test_full_pipeline_live():
    """One real end-to-end run on the sample; scores populate and the net-income
    error ($210M report vs $190M source) is caught as refuted by the live model."""
    from chain_of_claims import db, results
    from chain_of_claims.config import settings
    from chain_of_claims.models import FactCheckResult
    from chain_of_claims.pipeline import run_pipeline

    assert not settings.offline, "smoke test must run against the real provider"
    settings.ensure_dirs()
    db.init_db()

    run_id = uuid.uuid4().hex[:12]
    report = str(SAMPLES / "report.md")
    source = str(SAMPLES / "source_filing.md")
    db.create_run(run_id, report, [source])
    scores = run_pipeline(run_id, report, [source])

    assert scores.n_claims > 0
    assert 0.0 <= scores.explainability <= 1.0
    assert 0.0 <= scores.hallucination <= 1.0

    result = results.build_result(run_id)
    net = [c for c in result["claims"] if "net income" in c["text"].lower()]
    assert net, "expected a net-income claim to be extracted"
    # The live model should refute the wrong figure ($210M) given the source ($190M).
    assert any(
        c["verdict"]["factcheck_result"] == FactCheckResult.REFUTED.value for c in net
    ), "expected the incorrect net-income figure to be refuted"
