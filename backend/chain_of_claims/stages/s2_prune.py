"""Stage 2 — prune claims.

Deduplicate near-identical claims, drop "motherhood" boilerplate, and gate
non-check-worthy claims. Dedup uses a cheap lexical similarity (token Jaccard) so it
runs without an embedding model in v1; the LLM already set `checkworthy` in Stage 1,
which we respect here and additionally harden with a boilerplate filter.
"""

from __future__ import annotations

import re

from ..models import Claim

_MOTHERHOOD = (
    "important to note", "it is worth", "as always", "past performance",
    "no guarantee", "consult your", "for informational purposes",
    "we remain committed", "long-term value", "well positioned",
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(claims: list[Claim], sim_threshold: float = 0.85) -> list[Claim]:
    kept: list[Claim] = []
    kept_tokens: list[set[str]] = []
    for c in claims:
        lower = c.text.lower()
        # Boilerplate / motherhood -> keep but mark not check-worthy.
        if any(m in lower for m in _MOTHERHOOD):
            c = c.model_copy(update={"checkworthy": False})
        toks = _tokens(c.text)
        # Drop near-duplicate of an already-kept claim.
        if any(_jaccard(toks, kt) >= sim_threshold for kt in kept_tokens):
            continue
        kept.append(c)
        kept_tokens.append(toks)
    return kept
