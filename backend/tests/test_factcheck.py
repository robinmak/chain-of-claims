"""Stage-8 verdict parsing tests (deterministic, no LLM).

Locks in the robust parse: the model may reason (mentioning tokens mid-text) before its
final 'VERDICT:' line. We must read the final verdict, not the first token seen.
"""

from chain_of_claims.models import FactCheckResult
from chain_of_claims.stages.s8_factcheck import _parse, _method_for
from chain_of_claims.models import ClaimType


def test_parse_prefers_final_verdict_marker():
    text = (
        "The claim says NOT_ENOUGH_EVIDENCE might apply, but I found both inputs.\n"
        "Computed 500/1250 = 0.4 = 40%, which matches.\n"
        "VERDICT: SUPPORTED"
    )
    result, _ = _parse(text)
    assert result == FactCheckResult.SUPPORTED


def test_parse_refuted_marker():
    text = "Reconstructed net income = 190, claim says 210.\nVERDICT: REFUTED"
    assert _parse(text)[0] == FactCheckResult.REFUTED


def test_parse_not_enough_evidence():
    text = "Revenue input is missing from the evidence.\nVERDICT: NOT_ENOUGH_EVIDENCE"
    assert _parse(text)[0] == FactCheckResult.NOT_ENOUGH_EVIDENCE


def test_parse_fallback_without_marker():
    # No explicit marker: fall back to the last bare token.
    text = "I lean SUPPORTED after checking, though initially unsure. Final: REFUTED."
    assert _parse(text)[0] == FactCheckResult.REFUTED


def test_comparative_routes_to_comparison():
    assert _method_for(ClaimType.COMPARATIVE) == "comparison"
    assert _method_for(ClaimType.DERIVED_QUANTITY) == "formula_reconstruction"
    assert _method_for(ClaimType.FORWARD_LOOKING) == "not_checked"
