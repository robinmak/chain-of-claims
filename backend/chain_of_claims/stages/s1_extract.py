"""Stage 1 — extract atomic claims and tag each with a financial claim type.

Atomic decomposition (one verifiable assertion per claim) is the FactScore unit; the
claim-type tag follows the FinGround taxonomy and *routes* Stage-8 verification. In a
thin v1 without a gold annotated set, "coverage" is reported as the raw atomic count.
"""

from __future__ import annotations

from ..config import TRANSFORM_MODEL
from ..llm.client import get_provider
from ..models import ClaimList

_SYSTEM = (
    "You are a meticulous financial-analysis auditor. You decompose a research report "
    "into ATOMIC claims: each claim states exactly one verifiable fact. Split compound "
    "sentences. Preserve figures, units, and time references exactly as written. "
    "Do not invent claims not present in the report."
)

_PROMPT = """Decompose the following financial research report into atomic claims.

For each claim, assign:
- type: one of extracted_metric (a figure taken from a source), derived_quantity
  (a computed/aggregated figure such as a total, margin, ratio, or growth rate),
  textual_assertion (a qualitative factual statement), comparative (X vs Y, ranking,
  relative change), temporal (time-indexed or trend-over-period), forward_looking
  (a projection, forecast, or guidance).
- checkworthy: false only for pure opinion/framing/boilerplate.
- cited_source: if the report attributes the claim to a source, copy that reference; else null.

REPORT:
{report}
"""


def run(report_text: str) -> ClaimList:
    provider = get_provider()
    return provider.structured_output(
        system=_SYSTEM,
        prompt=_PROMPT.format(report=report_text),
        schema=ClaimList,
        model=TRANSFORM_MODEL,
    )
