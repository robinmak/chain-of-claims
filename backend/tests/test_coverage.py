"""Stage-1b coverage tests (deterministic lexical matcher; offline provider).

Locks in recall/precision semantics and semantic-vs-literal matching behavior.
"""

import os

os.environ.setdefault("COC_OFFLINE", "1")

from chain_of_claims.models import Claim, ClaimType, GoldClaim, GoldSet
from chain_of_claims.stages import s1b_coverage


def _c(text, t=ClaimType.EXTRACTED_METRIC):
    return Claim(text=text, type=t)


def _g(text):
    return GoldClaim(text=text)


def test_reworded_claim_counts_as_covered():
    # Extraction rewords the gold claim; lexical matcher (figures + tokens) still matches.
    gold = [_g("Net income for Q3 FY2024 was $210 million.")]
    extracted = [_c("Acme's net income reached $210 million in Q3 FY2024.")]
    matched, n, ext_matched = s1b_coverage.compute(gold, extracted)
    assert (matched, n, ext_matched) == (1, 1, 1)


def test_missing_gold_claim_lowers_recall():
    gold = [
        _g("Total revenue was $1,250 million."),
        _g("Operating income was $200 million."),  # not extracted
    ]
    extracted = [_c("Total revenue was $1,250 million in Q3.")]
    matched, n, _ext = s1b_coverage.compute(gold, extracted)
    assert n == 2 and matched == 1  # recall = 0.5


def test_different_figure_not_matched():
    # A claim about a different figure must NOT count as covering the gold claim.
    gold = [_g("Net income was $190 million.")]
    extracted = [_c("Net income was $210 million.")]
    matched, n, ext_matched = s1b_coverage.compute(gold, extracted)
    assert (matched, n, ext_matched) == (0, 1, 0)


def test_precision_never_exceeds_one():
    # Two extracted claims both cover one gold claim; precision uses extracted-side count.
    gold = [_g("Total revenue was $1,250 million.")]
    extracted = [
        _c("Total revenue was $1,250 million."),
        _c("Total revenue reached $1,250 million in the quarter."),
    ]
    matched, n, ext_matched = s1b_coverage.compute(gold, extracted)
    assert matched == 1 and n == 1
    assert ext_matched <= len(extracted)
    assert (ext_matched / len(extracted)) <= 1.0


def test_goldset_ignores_comment_key():
    gs = GoldSet.model_validate({"_comment": "note", "claims": [{"text": "x"}]})
    assert len(gs.claims) == 1
