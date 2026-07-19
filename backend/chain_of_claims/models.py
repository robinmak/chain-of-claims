"""Domain models for the claim-evidence verification pipeline.

These Pydantic schemas are the contract between stages and the persistence layer.
They double as the JSON Schemas handed to the LLM for structured output, so field
descriptions matter: they are read by the model.
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class ClaimType(str, enum.Enum):
    """FinGround-style financial claim taxonomy. Drives Stage-8 verifier routing."""

    EXTRACTED_METRIC = "extracted_metric"      # a figure lifted from a source table/text
    DERIVED_QUANTITY = "derived_quantity"      # a computed/aggregated figure (needs arithmetic)
    TEXTUAL_ASSERTION = "textual_assertion"    # a qualitative factual statement
    COMPARATIVE = "comparative"                # X > Y, ranking, relative change
    TEMPORAL = "temporal"                      # time-indexed / trend-over-period claim
    FORWARD_LOOKING = "forward_looking"        # projection/guidance (not verifiable vs. sources)


class CitationStatus(str, enum.Enum):
    SUPPORTED = "supported"                    # cited evidence genuinely supports the claim
    OUT_OF_CONTEXT = "out_of_context"          # cited evidence supports a *different* claim
    UNCITED_SUPPORTED = "uncited_supported"    # no citation, but evidence exists in sources
    UNCITED_UNSUPPORTED = "uncited_unsupported"  # no citation and no supporting evidence
    NOT_APPLICABLE = "not_applicable"          # e.g. forward-looking / opinion


class FactCheckResult(str, enum.Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    NOT_CHECKED = "not_checked"                # non-check-worthy or forward-looking


class ChunkKind(str, enum.Enum):
    PARAGRAPH = "paragraph"
    TABLE_CELL = "table_cell"
    TABLE_ROW = "table_row"


# --- Stage 1/2: claims ------------------------------------------------------

class Claim(BaseModel):
    id: Optional[int] = None
    text: str = Field(description="A single atomic assertion (one verifiable fact).")
    type: ClaimType = Field(description="Financial claim type; routes verification.")
    checkworthy: bool = Field(
        default=True,
        description="False for opinion/framing/boilerplate not worth verifying.",
    )
    # Optional citation the report itself attached to this claim (source ref / locator).
    cited_source: Optional[str] = Field(
        default=None, description="Source reference the report attributes this claim to."
    )
    gold_matched: Optional[bool] = None  # set only when a gold claim set is supplied


class ClaimList(BaseModel):
    """Structured-output wrapper for Stage 1/2."""

    claims: list[Claim]


# --- Stage 3: Toulmin structure ---------------------------------------------

class Triplet(BaseModel):
    claim_id: Optional[int] = None
    reason: str = Field(
        description="The report's stated grounds for the claim, quoted verbatim."
    )
    warrant: str = Field(
        description="The implicit inferential link from reason to claim (LLM-generated)."
    )
    warrant_generated: bool = Field(
        default=True,
        description="True when the warrant was produced by us, not present in the report.",
    )
    is_causal: bool = Field(
        default=False,
        description="True when the warrant asserts a cause->effect relationship.",
    )


# --- Stage 4: warrant critical-question audit -------------------------------

class CQVerdict(BaseModel):
    """One verifier's Yes/No answers to the 8 CQoT critical questions."""

    answers: list[bool] = Field(
        description="Eight booleans, one per critical question (True = passes)."
    )


class CausalPair(BaseModel):
    """One cause->effect link asserted by a warrant (Part B, structural).

    This is an EXTRACTION, not a truth judgement: it records what causation the
    warrant claims, regardless of whether that causation actually holds.
    """

    cause: str = Field(description="The cause phrase, quoted from the warrant.")
    effect: str = Field(description="The effect phrase, quoted from the warrant.")
    implicit: bool = Field(
        default=False,
        description="True if the link is implied rather than marked by a connective.",
    )


class CausalPairs(BaseModel):
    """Structured-output wrapper for Part B extraction."""

    pairs: list[CausalPair] = Field(
        default_factory=list,
        description="All cause->effect pairs asserted by the warrant; [] if none.",
    )


class CausalAttribution(str, enum.Enum):
    """Whether the evidence STATES the causal link — never whether it is TRUE.

    (Corr2Cause [Jin et al. 2024]: LLMs infer causation-from-correlation near-random,
    so substantive truth is deliberately out of scope; this is a grounding question.)
    """

    ATTRIBUTED = "attributed"                    # evidence states/supports the link
    CO_OCCURRENCE_ONLY = "co_occurrence_only"    # both relata present, link NOT stated
    PURPOSE_OR_CONCESSIVE = "purpose_or_concessive"  # a goal/despite relation, not cause
    CONTRADICTED = "contradicted"                # evidence states a different/opposite link
    NOT_APPLICABLE = "not_applicable"            # no grounding / S == empty


class CausalPairVerdict(BaseModel):
    """Structured-output wrapper for Part C, per extracted pair."""

    attribution: CausalAttribution = Field(
        description="Whether the grounded evidence STATES this cause->effect link."
    )
    rationale: Optional[str] = Field(
        default=None, description="One sentence citing the evidence locator, if any."
    )


class WarrantAudit(BaseModel):
    claim_id: Optional[int] = None
    # mean pass-rate across the verifier panel, 0..1
    cq_score: float = 0.0
    # panel agreement (fraction of questions where all verifiers agreed), 0..1
    cq_agreement: float = 0.0
    per_question_pass_rate: list[float] = Field(default_factory=list)

    # --- causal check (replaces the single depbert_pass bool) ---
    # Part B: structural extraction. None when not a causal warrant / check disabled.
    causal_pairs: Optional[list[CausalPair]] = None
    structural_pass: Optional[bool] = None       # len(causal_pairs) >= 1
    # Part C: attribution against grounded evidence. None when S == empty / no grounding.
    causal_attribution: Optional[list[CausalPairVerdict]] = None
    # Diagnostic roll-up in [0,1]: fraction of extracted pairs attributed to evidence.
    # None when Part C did not run. Low-weighted; never gates scoring.
    causal_attribution_score: Optional[float] = None

    # Deprecated: retained one release as a mirror of structural_pass so existing DB
    # rows and the frontend graph keep working. New code should read structural_pass.
    depbert_pass: Optional[bool] = None


# --- Stage 5: evidence chunks -----------------------------------------------

class EvidenceChunk(BaseModel):
    id: Optional[int] = None
    source: str = Field(description="Source document name.")
    kind: ChunkKind
    text: str
    locator: str = Field(
        description="Human-readable position, e.g. 'p3¶2' or 'TableA r4 c2'."
    )


# --- Stage 6: grounding -----------------------------------------------------

class Grounding(BaseModel):
    claim_id: int
    chunk_id: int
    relevant: bool
    rationale: Optional[str] = None


# --- Stage 8/9: verdict per claim -------------------------------------------

class Verdict(BaseModel):
    claim_id: int
    factcheck_result: FactCheckResult = FactCheckResult.NOT_CHECKED
    method_used: str = ""  # 'lookup_entailment' | 'formula_reconstruction' | 'temporal'
    citation_status: CitationStatus = CitationStatus.NOT_APPLICABLE
    detail: Optional[str] = None  # short explanation / caught computation


# --- Stage 1b: gold-claim coverage -----------------------------------------

class GoldClaim(BaseModel):
    """A claim a correct report SHOULD contain, from a human-annotated gold set."""

    text: str = Field(description="The atomic fact the report is expected to state.")
    type: Optional[ClaimType] = None  # optional expected type


class GoldSet(BaseModel):
    claims: list[GoldClaim]


class CoverageMatch(BaseModel):
    """Structured-output wrapper: which gold claims are covered by the extraction."""

    matched_gold_indices: list[int] = Field(
        description="0-based indices of gold claims covered by >=1 extracted claim."
    )


# --- Stage 7/10: scores -----------------------------------------------------

class RunScores(BaseModel):
    # Explainability: fraction of check-worthy claims grounded in >=1 relevant chunk.
    # None when no source materials were supplied (verification not possible).
    explainability: Optional[float] = 0.0
    # Hallucination: fraction of check-worthy claims that are refuted OR have a
    # citation failure. None when no source materials were supplied.
    hallucination: Optional[float] = 0.0
    hallucination_by_type: dict[str, float] = Field(default_factory=dict)
    # True when no source materials were supplied: claims + argument structure are
    # produced, but grounding/fact-check/citation stages are skipped.
    verification_skipped: bool = False
    n_claims: int = 0
    n_checkworthy: int = 0
    # Coverage: fraction of GOLD claims recovered by extraction (recall). None when no
    # gold set was supplied. Precision = fraction of extracted claims that matched gold.
    coverage: Optional[float] = None
    coverage_precision: Optional[float] = None
    n_gold: int = 0
    n_gold_matched: int = 0
    coverage_note: str = "coverage=raw_atomic_count (no gold set supplied)"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
