"""Stage 3 — reconstruct the Toulmin triplet for each claim.

Following Gupta, Zuckerman & O'Connor (2024): the claim and its reason (grounds) are
taken from the report; the warrant — the implicit inferential link — is generated. We
flag the warrant as machine-generated so a Stage-4 failure can be attributed to the
report vs. our own generator. We also detect whether the warrant is causal, which
gates the optional structural check in Stage 4.
"""

from __future__ import annotations

from ..config import TRANSFORM_MODEL
from ..llm.client import get_provider
from ..models import Claim, Triplet

_SYSTEM = (
    "You analyse arguments using Toulmin's model. Given a claim and the surrounding "
    "report context, identify the REASON (the grounds the author gives, quoted from "
    "the report) and generate the implicit WARRANT: the general principle that licenses "
    "moving from the reason to the claim. State the warrant as a single clear sentence."
)

_PROMPT = """Reconstruct the Toulmin structure for this claim.

CLAIM: {claim}

REASON_CONTEXT (report text near the claim; extract the grounds verbatim if present):
{context}

Set is_causal=true only if the warrant asserts a cause->effect relationship.
warrant_generated must be true.
"""


def run(claim: Claim, context: str) -> Triplet:
    provider = get_provider()
    t = provider.structured_output(
        system=_SYSTEM,
        prompt=_PROMPT.format(claim=claim.text, context=context or "(none provided)"),
        schema=Triplet,
        model=TRANSFORM_MODEL,
    )
    return t.model_copy(update={"claim_id": claim.id, "warrant_generated": True})
