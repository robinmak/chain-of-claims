# Design Spec — Strengthening Causal-Warrant Checking (Stage 4)

**Status:** Draft · **Scope:** `backend/chain_of_claims/stages/s4_warrant_audit.py`,
`models.py`, `config.py`, offline provider · **Supersedes:** the v1 "fine-tune DEPBERT"
plan recorded in README §8.

---

## 1. Motivation and what changed

Stage 4 audits each machine-generated warrant with a panel of critical-question
verifiers (a diagnostic signal, not a gate). For warrants that assert causation it
*also* runs an optional causal check. Today that check is a single regex
(`_CAUSAL_HINT`) testing whether a causal connective ("because", "due to", "led to", …)
sits between two content phrases. It has two acknowledged weaknesses:

1. **One cause→effect pair per sentence.** The regex finds the first connective and
   splits once. A warrant asserting two linked causal steps is scored as one.
2. **Out-of-domain / no grounding.** It is a surface pattern; it neither understands
   financial phrasing nor checks the causation against evidence. It cannot tell a
   *stated* causal link from mere co-occurrence dressed up with a connective.

The v1 remediation was to replace the regex with a **DEPBERT-style tagger fine-tuned on
financial text**. Analysis of three papers retires that plan and reframes the fix.

### 1.1 Evidence against the literal DEPBERT fix (Option A)

- **UniCausal** [Tan et al., 2023] is the natural Option-A resource: a unified,
  fine-tunable causal-text-mining benchmark spanning sequence classification, span
  detection, and pair classification. But (a) **all six** of its consolidated corpora
  (AltLex, BECAUSE, CTB, ESL, PDTB, SemEval) are news / web / general text — **none
  financial** — so it does not resolve the domain-transfer constraint; and (b) its own
  span-detection baseline "can only predict one cause-effect relation per input
  sequence," i.e. a fine-tuned tagger *reproduces* limitation (1) rather than fixing it.
  A general-domain tagger is therefore doubly mismatched to our need.

- **FinCausal 2025** [Moreno-Sandoval et al.] is the financial-domain anchor and shows
  the field has already moved: after several editions of extractive cause–effect *span*
  detection, the 2025 shared task was **reframed as causal QA**, and the leaderboard is
  dominated by **fine-tuned / prompted generative LLMs** (gpt-4o-mini, LoRA-tuned
  Llama-3.1) — no span tagger competitive at the top. The genre-appropriate tool for
  financial causal extraction today is an LLM, not a BERT tagger.

**Conclusion:** do not fine-tune a tagger. Replace the regex with an **LLM
structured-output extractor** (Option B). It needs no training data, has no
domain-transfer problem, and is multi-pair by construction — dissolving both
weaknesses at once.

### 1.2 Evidence for the attribution check (Option C) and its hard boundary

- **FinCausal 2025** error analysis identifies the dominant real-world failure exactly:
  "purpose-based relationships are often confused with cause-effect," and
  concessive-relationship fragments ("although", "despite") get folded into supposedly
  causal answers. In our terms: a warrant asserts "X drove Y" when the evidence only
  supports "X and Y both occurred" (or "X was done *in order to* achieve Y"). Catching
  this — a **causal attribution / grounding** question — is the high-value move.

- **Corr2Cause** [Jin et al., 2024] fixes the boundary. Seventeen off-the-shelf LLMs
  score **near-random** on *pure* causal inference (best F1 ≈ 33%, several below random),
  and fine-tuned models collapse out-of-distribution. So the check must **never** ask
  "is this causation *true*?" — LLMs cannot answer it reliably. It may only ask "is this
  causation *stated in / entailed by* the grounded evidence?" — a grounding question the
  rest of the pipeline already answers well.

### 1.3 The three "causal warrant checking" senses (kept distinct)

| Sense | Question | In scope? | Mechanism |
|---|---|---|---|
| **Structural** | Does the warrant *contain* a well-formed cause→effect pair? | Yes (kept, upgraded) | Option B — LLM multi-pair extractor |
| **Attributive** | Is that causation *stated in the evidence*, or invented? | **Yes (new, high-value)** | Option C — grounding/entailment check |
| **Substantive** | Is the causal claim *true* of the world? | **No — out of scope** | none (Corr2Cause: LLMs ≈ random) |

---

## 2. Design overview

Replace `_causal_structural_check` with a two-part causal assessment, both operating
only on `is_causal` warrants and only when `settings.enable_causal_check` is set:

- **Part B — Structural extraction.** An LLM structured-output call extracts the set of
  cause→effect pairs asserted by the warrant. `structural_pass = len(pairs) >= 1`. This
  replaces the regex and is multi-pair.
- **Part C — Attribution check.** *Only when source materials exist and the claim has
  grounded evidence chunks*, an LLM entailment call asks, per extracted pair, whether the
  grounded evidence **states/supports the causal link** (`attributed`), merely mentions
  the two relata without the link (`co_occurrence_only`), or contradicts it
  (`purpose_or_concessive` / `unsupported`). Attribution is skipped (`None`) when
  `S = ∅` or the claim has no grounding, consistent with the pipeline's existing
  "absence of evidence ≠ error" rule.

Both parts feed a `WarrantAudit`. Neither gates hallucination scoring. The attribution
result is surfaced diagnostically and low-weighted, per the subjectivity/robustness
posture already established for the CQ panel.

### 2.1 Dependency change (grounding must precede the causal check)

Part C consumes grounded evidence, so it needs Stage 6 output. Two options:

- **(preferred) Split the causal check out of Stage 4** into a late sub-step that runs
  after Stage 6 grounding, feeding its result back onto the claim's `WarrantAudit`
  record. Stage 4 keeps only the CQ panel + Part B (structural), which need no evidence.
- (fallback) Keep it in Stage 4 but pass grounded chunks in; requires reordering so
  Stage 4 runs after Stage 6 for causal claims only.

The preferred split keeps Stage 4 evidence-independent (so it still runs in the `S = ∅`
argument-structure-only path) and confines the evidence dependency to Part C. The
controller already persists per-stage state, so writing an augmented `WarrantAudit`
after grounding is a localised change.

---

## 3. Data model changes (`models.py`)

Add causal-extraction types and extend `WarrantAudit`. Field descriptions are read by
the model (they are the JSON Schema), so they carry the prompt intent.

```python
class CausalPair(BaseModel):
    """One cause->effect link asserted by a warrant (Part B, structural)."""
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
    ATTRIBUTED = "attributed"                # evidence states/supports the causal link
    CO_OCCURRENCE_ONLY = "co_occurrence_only"  # both relata present, link NOT stated
    PURPOSE_OR_CONCESSIVE = "purpose_or_concessive"  # a goal/despite relation, not cause
    CONTRADICTED = "contradicted"            # evidence states a different/opposite link
    NOT_APPLICABLE = "not_applicable"        # no grounding / S == empty


class CausalPairVerdict(BaseModel):
    """Structured-output wrapper for Part C, per extracted pair."""
    attribution: CausalAttribution
    rationale: Optional[str] = Field(
        default=None, description="One sentence citing the evidence locator, if any."
    )
```

Extend `WarrantAudit`:

```python
class WarrantAudit(BaseModel):
    claim_id: Optional[int] = None
    cq_score: float = 0.0
    cq_agreement: float = 0.0
    per_question_pass_rate: list[float] = Field(default_factory=list)

    # --- causal check (replaces the single depbert_pass bool) ---
    # Part B: structural extraction. None when not a causal warrant / check disabled.
    causal_pairs: Optional[list[CausalPair]] = None
    structural_pass: Optional[bool] = None          # len(causal_pairs) >= 1
    # Part C: attribution against grounded evidence. None when S == empty or no grounding.
    causal_attribution: Optional[list[CausalPairVerdict]] = None
    # Diagnostic roll-up in [0,1]: fraction of extracted pairs attributed to evidence.
    # None when Part C did not run. Low-weighted; never gates scoring.
    causal_attribution_score: Optional[float] = None

    # Back-compat: retain until UI/persistence migrate off it.
    depbert_pass: Optional[bool] = None
```

`depbert_pass` is kept as a deprecated mirror of `structural_pass` for one release so
existing DB rows and the frontend graph do not break; new code reads `structural_pass`.

---

## 4. Stage logic

### 4.1 Part B — structural extraction (in Stage 4, evidence-free)

Replaces `_causal_structural_check`. Runs on `is_causal` warrants when the check is
enabled.

```
system: "You extract cause->effect relationships from a single sentence. Return every
         distinct causal link the sentence ASSERTS. Do not judge whether the causation
         is true; only extract what is claimed. If the sentence asserts no causation,
         return an empty list."
prompt:  WARRANT: {warrant}
schema:  CausalPairs
model:   TRANSFORM_MODEL   # mechanical extraction, cheaper tier
```

`structural_pass = len(result.pairs) >= 1`. This is the multi-pair replacement for the
regex; the regex is deleted.

### 4.2 Part C — attribution check (after Stage 6 grounding)

Runs per extracted pair, only when the claim has ≥1 relevant grounded chunk. Evidence
text is the concatenation of the claim's grounded chunk texts (with locators).

```
system: "You judge whether a causal link is STATED or SUPPORTED by the given evidence.
         You are NOT judging whether the causation is true in the world — only whether
         the evidence asserts this cause->effect link. Distinguish: the evidence states
         the link (attributed); the evidence mentions both items but not a causal link
         between them (co_occurrence_only); the evidence describes a purpose/goal or a
         concession, not a cause (purpose_or_concessive); the evidence states a
         different or opposite link (contradicted)."
prompt:  CAUSE: {pair.cause}
         EFFECT: {pair.effect}
         EVIDENCE (grounded chunks with locators):
         {evidence_block}
schema:  CausalPairVerdict
model:   JUDGE_MODEL       # judgement-heavy, stronger tier
```

`causal_attribution_score = mean(1 if v.attribution == ATTRIBUTED else 0)` over pairs.
When `S = ∅` or no grounding: `causal_attribution = None`, `causal_attribution_score =
None`, and each pair (if surfaced) reads `NOT_APPLICABLE`.

**Boundary guardrail (Corr2Cause).** The prompts above forbid substantive truth
judgement in both the system message and the enum design (there is no "true"/"false"
verdict). The check can only answer the grounding question the pipeline is competent at.

### 4.3 Scoring interaction (Stage 10)

Unchanged for v1: the causal signal does **not** enter the hallucination score. Rationale
— warrant acceptability is subjective (CQ panel κ ≈ 0.18) and, per Corr2Cause, causal
judgement is unreliable; the attribution result is a **diagnostic** shown in the
per-claim audit. A later revision may fold a *strongly-attributed-negative* signal
(`CONTRADICTED` on a check-worthy causal claim with good grounding) into citation-status
as an `out_of_context`-like failure, gated behind a config flag and validated first.

---

## 5. Config (`config.py`)

```python
# Causal-warrant check: "off" | "structural" | "full".
#   off        -> no causal check (causal_* fields stay None)
#   structural -> Part B only (LLM multi-pair extraction; replaces the regex)
#   full       -> Part B + Part C attribution (requires source materials)
causal_check_mode: str = os.environ.get("COC_CAUSAL_CHECK_MODE", "structural")
```

Migration: the existing boolean `enable_causal_check` maps to `mode != "off"`; keep it
as a derived property for back-compat. `full` degrades gracefully to `structural`
behaviour whenever `S = ∅` (Part C simply reports `NOT_APPLICABLE`).

---

## 6. Offline provider (`offline_provider.py`)

Add deterministic stubs so the pipeline stays runnable with `COC_OFFLINE=1`:

- `CausalPairs`: reuse the retired regex logic as the *offline heuristic only* — find a
  causal connective, split once, emit a single `CausalPair`; empty list if no connective.
  (The regex is fine as a deterministic stand-in; it is only unfit as the *production*
  check.) This keeps the offline structural path meaningful without a model.
- `CausalPairVerdict`: deterministic attribution — `ATTRIBUTED` if both the cause tokens
  and effect tokens appear in the evidence block, else `CO_OCCURRENCE_ONLY`; a
  connective-in-evidence check can promote to `ATTRIBUTED`. No world knowledge, matching
  the offline provider's "plausible, inspectable, not random" contract.

---

## 7. Tests

New `tests/test_causal_warrant.py` (offline, no credentials):

1. **Multi-pair extraction** — a warrant asserting two causal steps yields
   `len(causal_pairs) == 2` (guards against the one-pair-per-sentence regression that
   defined the old limitation).
2. **No-causation warrant** — a non-causal warrant yields `causal_pairs == []`,
   `structural_pass is False`.
3. **Attribution skipped when `S = ∅`** — `causal_attribution is None`,
   `causal_attribution_score is None`; assert the run still completes (argument-structure
   path unaffected).
4. **Co-occurrence vs. attributed** — evidence containing both relata but no link →
   `CO_OCCURRENCE_ONLY`; evidence stating the link → `ATTRIBUTED`.
5. **Scoring isolation** — the causal signal does not move the Stage-10 hallucination
   score (regression guard on the "diagnostic, not a gate" invariant).
6. **Config modes** — `off` leaves all `causal_*` fields `None`; `structural` populates
   Part B only; `full` with grounding populates Part C.

Extend the live smoke test to assert that on a warrant with a spurious causal claim
(two grounded facts, no stated link) the attribution verdict is not `ATTRIBUTED`.

---

## 8. Migration & rollout

1. Land model changes with `depbert_pass` mirrored from `structural_pass` (no DB break).
2. Ship Part B behind `COC_CAUSAL_CHECK_MODE=structural` (default) — pure win over the
   regex, no evidence dependency, no reorder.
3. Split the causal sub-step to run post-grounding; ship Part C behind
   `COC_CAUSAL_CHECK_MODE=full`, opt-in.
4. Migrate the frontend graph/detail view to read `structural_pass` +
   `causal_attribution`; then remove `depbert_pass`.
5. Update README §8 (done) and, if a paper follows, the related-work contrast:
   Option A (UniCausal) rejected on domain + one-pair grounds; Option B (LLM extraction,
   FinCausal-era) adopted; Option C (attribution, bounded away from Corr2Cause's
   substantive-truth failure) adopted.

---

## 9. Open questions

- **Warrant vs. reason as the causal source.** We currently extract from the
  machine-generated `warrant`. Should Part B also/instead read the report-quoted
  `reason`, so a failure is attributable to the document rather than our generator?
  (Consistent with the Stage-3 `warrant_generated` provenance concern.)
- **Attribution granularity.** Per-pair vs. per-warrant roll-up when a warrant has
  multiple pairs with mixed verdicts — spec picks per-pair with a mean roll-up; confirm
  the UI can render per-pair.
- **When (if ever) attribution should touch the score.** Deferred to a validated later
  revision (§4.3); needs a labelled financial sample before any gating.

---

## References

- Jin, Z., et al. (2024). *Can Large Language Models Infer Causation from Correlation?*
  ICLR. arXiv:2306.05836. — bounds Part C away from substantive causal-truth judgement.
- Moreno-Sandoval, A., et al. (2025). *The Financial Document Causality Detection Shared
  Task (FinCausal 2025).* FinNLP/FNP/LLMFinLegal. — LLM-based causal QA supersedes span
  tagging; purpose/concessive-vs-cause is the dominant error class.
- Tan, F. A., Zuo, X., & Ng, S.-K. (2023). *UniCausal: Unified Benchmark and Repository
  for Causal Text Mining.* arXiv:2208.09163. — Option-A resource; general-domain corpora
  and one-pair-per-sentence span baseline argue against the fine-tuned-tagger path.
- Kabir, M. A., et al. (2025). *DEPBERT: Extracting Cause-Effect Pairs from a Sentence
  with a Dependency-Aware Transformer Model.* arXiv:2507.09925. — the v1 baseline whose
  constraints (one pair/sentence, out-of-domain) this spec removes.
