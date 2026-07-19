// Shared API types mirroring the backend result shape (chain_of_claims/results.py).

export interface Triplet {
  reason: string;
  warrant: string;
  warrant_generated: boolean;
  is_causal: boolean;
}

export interface CausalPair {
  cause: string;
  effect: string;
  implicit: boolean;
}

export interface CausalPairVerdict {
  attribution:
    | "attributed"
    | "co_occurrence_only"
    | "purpose_or_concessive"
    | "contradicted"
    | "not_applicable";
  rationale: string | null;
}

export interface WarrantAudit {
  cq_score: number;
  cq_agreement: number;
  per_question_pass_rate: number[];
  // Part B: structural cause->effect extraction (multi-pair). null when non-causal.
  causal_pairs: CausalPair[] | null;
  structural_pass: boolean | null;
  // Part C: attribution against grounded evidence. null when not run.
  causal_attribution: CausalPairVerdict[] | null;
  causal_attribution_score: number | null;
  // deprecated mirror of structural_pass (kept one release for back-compat)
  depbert_pass: boolean | null;
}

export interface Verdict {
  factcheck_result: string;
  method_used: string;
  citation_status: string;
  detail: string | null;
}

export interface Evidence {
  locator: string;
  text: string;
  source: string;
}

export interface ClaimView {
  id: number;
  text: string;
  type: string;
  checkworthy: boolean;
  cited_source: string | null;
  triplet: Triplet | null;
  warrant_audit: WarrantAudit | null;
  verdict: Verdict | null;
  evidence: Evidence[];
}

export interface Scores {
  explainability: number | null;
  hallucination: number | null;
  hallucination_by_type: Record<string, number>;
  verification_skipped: boolean;
  n_claims: number;
  n_checkworthy: number;
  coverage: number | null;
  coverage_precision: number | null;
  n_gold: number;
  n_gold_matched: number;
  coverage_note: string;
}

export interface RunResult {
  id: string;
  status: string;
  stage: string;
  error: string | null;
  scores: Scores | null;
  claims: ClaimView[];
  n_chunks: number;
}
