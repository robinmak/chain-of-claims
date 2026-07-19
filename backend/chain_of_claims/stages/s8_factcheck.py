"""Stage 8 — type-routed fact-checking.

The FinGround insight: verify each claim with a strategy matched to its type, because
~43% of financial errors are computational and are missed by uniform text-entailment.

Routing:
- derived_quantity  -> formula reconstruction: identify the formula, look up each input
                       cell, compute with the calculator, compare to the claimed value.
- comparative       -> comparison: look up the two figures being related, compute the
                       relationship (difference/ratio), decide if the claim holds.
- temporal          -> time-consistency against grounded evidence.
- textual_assertion -> lookup + entailment.
- extracted_metric  -> figure lookup + entailment.
- forward_looking   -> NOT_CHECKED (not verifiable against historical sources).

Both computational routes get the calculator AND evidence-lookup tools. The prompts
push the model to actually fetch inputs and compute rather than abstain: if every input
needed is present in the evidence, NOT_ENOUGH_EVIDENCE is not an allowed answer.

Only grounded (relevant) chunks are passed in, keeping the check retrieval-equalized.
"""

from __future__ import annotations

import re

from ..config import JUDGE_MODEL
from ..llm.client import get_provider
from ..llm.tools import CALCULATOR_TOOL, LOOKUP_TOOL, build_tool_impls
from ..models import (
    Claim,
    ClaimType,
    EvidenceChunk,
    FactCheckResult,
    Verdict,
)

_VERDICT_RULE = (
    "End your reply with a final line of exactly one verdict token: "
    "'VERDICT: SUPPORTED', 'VERDICT: REFUTED', or 'VERDICT: NOT_ENOUGH_EVIDENCE'."
)

_ENTAIL_SYSTEM = (
    "You are a financial fact-checker. Decide whether the EVIDENCE supports or refutes "
    "the CLAIM. Use lookup_evidence to find the relevant figures or statements. "
    "Only answer NOT_ENOUGH_EVIDENCE if the evidence genuinely does not address the "
    "claim. " + _VERDICT_RULE
)

_COMPUTE_SYSTEM = (
    "You are a financial fact-checker verifying a COMPUTED figure. Work step by step:\n"
    "1. State the formula for the claimed quantity (e.g. gross margin = gross profit / "
    "revenue; growth = (new - old) / old).\n"
    "2. Use lookup_evidence to find each input value in the evidence; note its locator "
    "and unit/scale.\n"
    "3. Use the calculator tool to compute the result from those inputs.\n"
    "4. Compare your computed result to the claimed value, allowing small rounding "
    "differences (within ~1%, or +/-0.5 percentage points for a percentage).\n"
    "SUPPORTED if they match, REFUTED if they clearly differ. You may answer "
    "NOT_ENOUGH_EVIDENCE ONLY if one of the required input values is absent from the "
    "evidence. If all inputs are present you MUST compute and decide. " + _VERDICT_RULE
)

_COMPARE_SYSTEM = (
    "You are a financial fact-checker verifying a COMPARATIVE claim (e.g. X is greater "
    "than Y, X grew faster than Y, X exceeds Y). Work step by step:\n"
    "1. Identify the two (or more) quantities being compared and the relation asserted.\n"
    "2. Use lookup_evidence to find each quantity's value in the evidence.\n"
    "3. Use the calculator tool if a difference, ratio, or growth rate is needed to "
    "judge the relation.\n"
    "4. Decide whether the asserted relation holds given the figures.\n"
    "Answer NOT_ENOUGH_EVIDENCE ONLY if a value needed for the comparison is absent. "
    "If the values are present you MUST compare and decide. " + _VERDICT_RULE
)


def _method_for(t: ClaimType) -> str:
    if t == ClaimType.DERIVED_QUANTITY:
        return "formula_reconstruction"
    if t == ClaimType.COMPARATIVE:
        return "comparison"
    if t == ClaimType.TEMPORAL:
        return "temporal"
    if t == ClaimType.FORWARD_LOOKING:
        return "not_checked"
    return "lookup_entailment"


_TOKEN = re.compile(r"\bVERDICT:\s*(SUPPORTED|REFUTED|NOT[_ ]ENOUGH[_ ]EVIDENCE)\b", re.I)
_FALLBACK = re.compile(r"\b(SUPPORTED|REFUTED|NOT[_ ]ENOUGH[_ ]EVIDENCE)\b", re.I)


def _parse(text: str) -> tuple[FactCheckResult, str]:
    """Robustly extract the verdict.

    Prefer the explicit 'VERDICT:' marker; take the LAST match so the model's final
    conclusion wins over any tokens mentioned mid-reasoning. Fall back to a bare token.
    """
    matches = list(_TOKEN.finditer(text)) or list(_FALLBACK.finditer(text))
    if not matches:
        return FactCheckResult.NOT_ENOUGH_EVIDENCE, text
    tok = matches[-1].group(1).upper().replace(" ", "_")
    if tok == "SUPPORTED":
        return FactCheckResult.SUPPORTED, text
    if tok == "REFUTED":
        return FactCheckResult.REFUTED, text
    return FactCheckResult.NOT_ENOUGH_EVIDENCE, text


_SYSTEM_FOR = {
    "formula_reconstruction": _COMPUTE_SYSTEM,
    "comparison": _COMPARE_SYSTEM,
    "temporal": _ENTAIL_SYSTEM,
    "lookup_entailment": _ENTAIL_SYSTEM,
}


def run(claim: Claim, grounded: list[EvidenceChunk]) -> Verdict:
    method = _method_for(claim.type)
    if method == "not_checked" or not claim.checkworthy:
        return Verdict(
            claim_id=claim.id,
            factcheck_result=FactCheckResult.NOT_CHECKED,
            method_used="not_checked",
            detail="forward-looking or non-check-worthy",
        )

    provider = get_provider()
    evidence_text = "\n".join(f"[{c.locator}] {c.text}" for c in grounded) or "(none)"
    prompt = f"CLAIM: {claim.text}\n\nEVIDENCE:\n{evidence_text}"

    # Computational routes get the calculator; entailment routes need only lookup.
    if method in ("formula_reconstruction", "comparison"):
        tools = [CALCULATOR_TOOL, LOOKUP_TOOL]
    else:
        tools = [LOOKUP_TOOL]

    out = provider.tool_loop(
        system=_SYSTEM_FOR[method],
        prompt=prompt,
        tools=tools,
        tool_impls=build_tool_impls(grounded),
        model=JUDGE_MODEL,
        max_turns=8,  # allow lookup(s) + calculate + conclude
    )

    result, detail = _parse(out)
    return Verdict(
        claim_id=claim.id,
        factcheck_result=result,
        method_used=method,
        detail=detail[:500],
    )
