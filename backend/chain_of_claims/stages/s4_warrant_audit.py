"""Stage 4 — warrant critical-question audit (+ optional causal structural check).

Applies the eight Critical-Questions-of-Thought (Castagna, Sassoon & Parsons 2024) to
each warrant, using a PANEL of independent verifiers and reporting agreement — because
warrant acceptability is intrinsically subjective (Gupta et al. found gold warrants
accepted only ~46% of the time, kappa~0.18). This is a DIAGNOSTIC score, not a hard
gate.

The optional causal check (`is_causal` warrants only) has two parts:

- Part B (here): STRUCTURAL extraction — recover every cause->effect pair the warrant
  asserts, via an LLM structured-output call. This replaces the old one-pair regex and
  is multi-pair by construction; it needs no training data and no domain transfer
  (see docs/spec-causal-warrant-checking.md for why the DEPBERT fine-tune plan was
  retired in favour of this). It does NOT judge whether the causation is true.
- Part C (s4b_causal_attribution, run after grounding): ATTRIBUTION — whether the
  grounded evidence actually states the causal link. Kept separate so Stage 4 stays
  evidence-independent and still runs in the no-sources path.
"""

from __future__ import annotations

from ..config import JUDGE_MODEL, TRANSFORM_MODEL, settings
from ..llm.client import get_provider
from ..models import CausalPairs, CQVerdict, Triplet, WarrantAudit

# The 8 CQoT critical questions (targeting Data, Warrant, Backing, Claim, Qualifier).
CRITICAL_QUESTIONS = [
    "Does the reasoning start with clearly defined premises?",
    "Are the premises supported by evidence or accepted facts?",
    "Does the reasoning use logical connections between premises and conclusion?",
    "Are those logical connections valid?",
    "Does the reasoning avoid fallacies or logical errors?",
    "Is the conclusion logically derived from the premises?",
    "Is the reasoning consistent with established financial knowledge or principles?",
    "Does the reasoning lead to a plausible, reasonable conclusion?",
]

_SYSTEM = (
    "You are a strict, critical reasoner auditing an argument's warrant. Answer each "
    "question with a boolean. Be conservative: answer false when in doubt."
)

_PROMPT = """Assess this argument's warrant against the eight critical questions.

CLAIM: {claim}
REASON: {reason}
WARRANT: {warrant}

Return exactly eight booleans, one per question, in order:
{questions}
"""

# Part B: multi-pair cause->effect extraction. EXTRACTION, not truth judgement.
_CAUSAL_EXTRACT_SYSTEM = (
    "You extract cause->effect relationships from a single sentence. Return every "
    "distinct causal link the sentence ASSERTS. Do NOT judge whether the causation is "
    "true; only extract what is claimed. Quote the cause and effect phrases from the "
    "sentence. If the sentence asserts no causation, return an empty list."
)

_CAUSAL_EXTRACT_PROMPT = "WARRANT: {warrant}\n"


def run(claim_text: str, triplet: Triplet) -> WarrantAudit:
    provider = get_provider()
    panel: list[list[bool]] = []
    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(CRITICAL_QUESTIONS))
    for _ in range(max(1, settings.cq_panel_size)):
        verdict: CQVerdict = provider.structured_output(
            system=_SYSTEM,
            prompt=_PROMPT.format(
                claim=claim_text,
                reason=triplet.reason,
                warrant=triplet.warrant,
                questions=questions_block,
            ),
            schema=CQVerdict,
            model=JUDGE_MODEL,
            temperature=0.6,  # some diversity across panel members
        )
        answers = (verdict.answers + [False] * 8)[:8]
        panel.append(answers)

    n = len(panel)
    per_q = [sum(1 for a in panel if a[i]) / n for i in range(8)]
    cq_score = sum(per_q) / 8
    # agreement = fraction of questions where the whole panel gave the same answer
    unanimous = sum(1 for i in range(8) if all(panel[0][i] == a[i] for a in panel))
    cq_agreement = unanimous / 8

    causal_pairs = None
    structural_pass = None
    if settings.enable_causal_check and triplet.is_causal:
        pairs = _extract_causal_pairs(triplet.warrant)
        causal_pairs = pairs
        structural_pass = len(pairs) >= 1

    return WarrantAudit(
        claim_id=triplet.claim_id,
        cq_score=cq_score,
        cq_agreement=cq_agreement,
        per_question_pass_rate=per_q,
        causal_pairs=causal_pairs,
        structural_pass=structural_pass,
        # deprecated mirror kept for one release (frontend/DB back-compat)
        depbert_pass=structural_pass,
    )


def _extract_causal_pairs(warrant: str):
    """Part B: LLM multi-pair cause->effect extraction (replaces the old regex).

    Returns a list of CausalPair. Multi-pair by construction — a warrant asserting
    two linked causal steps yields two pairs, dissolving the one-pair-per-sentence
    limit of both the retired regex and a DEPBERT-style tagger. STRUCTURAL only:
    records what causation is claimed, not whether it is true.
    """
    provider = get_provider()
    result: CausalPairs = provider.structured_output(
        system=_CAUSAL_EXTRACT_SYSTEM,
        prompt=_CAUSAL_EXTRACT_PROMPT.format(warrant=warrant),
        schema=CausalPairs,
        model=TRANSFORM_MODEL,  # mechanical extraction -> cheaper tier
    )
    return result.pairs
