"""Deterministic offline provider.

Lets the full pipeline run end-to-end with no API key (tests, CI, demos). It returns
schema-valid outputs using light heuristics so the results are *plausible and
inspectable*, not random. It is NOT a model: it exists to exercise wiring and to make
the app demonstrable. Enable with COC_OFFLINE=1.
"""

from __future__ import annotations

import re
from typing import Callable, Type, TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")

# Offline-only causal connective pattern. This is the retired production regex, kept
# here purely as a DETERMINISTIC stand-in so the offline structural path is meaningful
# without a model. Unlike the old code it splits on EVERY connective, so a warrant
# asserting two causal steps yields two pairs.
_CAUSAL_HINT = re.compile(
    r"\b(because|due to|led to|leads to|drove|drives|caused|causes|resulted in|"
    r"results in|owing to|as a result)\b",
    re.I,
)


class OfflineProvider(LLMProvider):
    def structured_output(
        self,
        *,
        system: str,
        prompt: str,
        schema: Type[T],
        model: str,
        temperature: float = 0.0,
    ) -> T:
        name = schema.__name__

        if name == "ClaimList":
            return self._claims(prompt, schema)
        if name == "Triplet":
            return self._triplet(prompt, schema)
        if name == "CQVerdict":
            return schema.model_validate({"answers": [True, True, True, False, True, True, True, True]})
        if name == "CausalPairs":
            return self._causal_pairs(prompt, schema)
        if name == "CausalPairVerdict":
            return self._causal_verdict(prompt, schema)
        if name == "_RelevantLocators":
            return self._relevant(prompt, schema)
        if name == "CoverageMatch":
            # Offline: let the deterministic lexical matcher own coverage. Returning
            # no semantic matches means s1b_coverage falls back to its lexical floor.
            return schema.model_validate({"matched_gold_indices": []})
        # generic
        return schema.model_validate(self._defaults(schema))

    def _causal_pairs(self, prompt: str, schema):
        # Deterministic multi-pair extraction: split the warrant on each causal
        # connective and emit a (cause, effect) pair per connective, provided both
        # sides carry content. STRUCTURAL stand-in only; no truth judgement.
        warrant = prompt.split("WARRANT:", 1)[-1].strip()
        pairs = []
        for m in _CAUSAL_HINT.finditer(warrant):
            left = warrant[: m.start()].strip(" ,.;")
            right = warrant[m.end():].strip(" ,.;")
            # For "effect BECAUSE cause"-style connectives the cause is on the right;
            # for "cause LED TO effect" it is on the left. We keep the surface order
            # (left->right) since this is only a structural stand-in.
            if len(left.split()) >= 2 and len(right.split()) >= 2:
                pairs.append({"cause": left[:200], "effect": right[:200], "implicit": False})
        return schema.model_validate({"pairs": pairs})

    def _causal_verdict(self, prompt: str, schema):
        # Deterministic attribution: 'attributed' when both the cause tokens and the
        # effect tokens appear in the evidence block AND a causal connective is present
        # in the evidence; else 'co_occurrence_only'. No world knowledge, matching the
        # provider's "plausible, inspectable, not random" contract.
        cause = prompt.split("CAUSE:", 1)[-1].split("EFFECT:", 1)[0]
        effect = prompt.split("EFFECT:", 1)[-1].split("EVIDENCE", 1)[0]
        evidence = prompt.split("locators):", 1)[-1] if "locators):" in prompt else ""
        ev_l = evidence.lower()

        def _content(s: str) -> set[str]:
            return {t for t in re.findall(r"[a-zA-Z0-9$%.]+", s.lower()) if len(t) >= 4}

        cause_hit = bool(_content(cause) & _content(evidence)) or not _content(cause)
        effect_hit = bool(_content(effect) & _content(evidence)) or not _content(effect)
        if cause_hit and effect_hit and _CAUSAL_HINT.search(ev_l):
            attribution = "attributed"
        elif cause_hit and effect_hit:
            attribution = "co_occurrence_only"
        else:
            attribution = "co_occurrence_only"
        return schema.model_validate(
            {"attribution": attribution, "rationale": "offline heuristic"}
        )

    def _relevant(self, prompt: str, schema):
        # Mark a candidate relevant if it shares a salient token (number or 4+ char
        # word) with the claim. Deterministic stand-in for a relevance judge.
        claim = prompt.split("CLAIM:", 1)[-1].split("CANDIDATE", 1)[0]
        claim_tokens = {t for t in re.findall(r"[a-zA-Z0-9$%.,]+", claim.lower()) if len(t) >= 4}
        claim_nums = {_norm_num(n) for n in _NUM.findall(claim)}
        locators = []
        block = prompt.split("locator: text):", 1)[-1]
        for line in block.splitlines():
            if ":" not in line:
                continue
            loc, _, text = line.partition(":")
            loc = loc.strip()
            if not loc:
                continue
            text_l = text.lower()
            text_nums = {_norm_num(n) for n in _NUM.findall(text)}
            shares_word = any(t in text_l for t in claim_tokens)
            shares_num = bool(claim_nums & text_nums)
            if shares_word or shares_num:
                locators.append(loc)
        return schema.model_validate({"locators": locators})

    def _claims(self, prompt: str, schema):
        # Split the report into sentences; keep those that look like assertions.
        body = prompt.split("REPORT:", 1)[-1]
        # Drop markdown heading lines so titles don't leak into claim text.
        body = "\n".join(
            ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
        )
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            has_num = bool(_NUM.search(s))
            lower = s.lower()
            if any(w in lower for w in ("grew", "rose", "fell", "increased", "decreased", "than", "compared")):
                ctype = "comparative"
            elif any(w in lower for w in ("total", "sum", "margin", "ratio", "per ", "growth of")):
                ctype = "derived_quantity"
            elif any(w in lower for w in ("will", "expect", "forecast", "guidance", "project")):
                ctype = "forward_looking"
            elif any(w in lower for w in ("in 20", "quarter", "fiscal", "year-over-year", "yoy")):
                ctype = "temporal"
            elif has_num:
                ctype = "extracted_metric"
            else:
                ctype = "textual_assertion"
            checkworthy = ctype != "forward_looking" and not lower.startswith(("we believe", "in our view"))
            claims.append({"text": s, "type": ctype, "checkworthy": checkworthy})
        if not claims:
            claims = [{"text": body.strip()[:200] or "No claims found.",
                       "type": "textual_assertion", "checkworthy": True}]
        return schema.model_validate({"claims": claims})

    def _triplet(self, prompt: str, schema):
        claim = prompt.split("CLAIM:", 1)[-1].strip()[:300]
        lower = claim.lower()
        is_causal = any(w in lower for w in ("because", "due to", "led to", "drove", "caused", "resulted in"))
        reason = ""
        if "REASON_CONTEXT:" in prompt:
            reason = prompt.split("REASON_CONTEXT:", 1)[-1].strip()[:300]
        return schema.model_validate({
            "reason": reason or "(no explicit grounds located in report)",
            "warrant": f"If the stated grounds hold, then it follows that {claim[:120]}",
            "warrant_generated": True,
            "is_causal": is_causal,
        })

    def _defaults(self, schema):
        # Build minimal valid instance from schema defaults / first enum values.
        out = {}
        for fname, field in schema.model_fields.items():
            if field.is_required():
                ann = field.annotation
                if ann is bool:
                    out[fname] = True
                elif ann is int:
                    out[fname] = 0
                elif ann is float:
                    out[fname] = 0.0
                elif ann is str:
                    out[fname] = ""
                else:
                    out[fname] = None
        return out

    def tool_loop(
        self,
        *,
        system: str,
        prompt: str,
        tools: list[dict],
        tool_impls: dict[str, Callable[[dict], str]],
        model: str,
        max_turns: int = 6,
    ) -> str:
        # Deterministic fact-check heuristic used by Stage 8 in offline mode.
        # Compares numbers in the claim against numbers in the provided evidence.
        after_claim = prompt.split("CLAIM:", 1)[-1]
        # The claim portion ends where the EVIDENCE block begins; isolate it so
        # evidence figures don't leak into the claim's number set.
        claim = after_claim.split("EVIDENCE:", 1)[0]
        evidence = after_claim.split("EVIDENCE:", 1)[1] if "EVIDENCE:" in after_claim else ""
        claim_nums = set(_norm_num(n) for n in _NUM.findall(claim))
        ev_nums = set(_norm_num(n) for n in _NUM.findall(evidence))
        if not claim_nums:
            return "no numeric assertion to contradict\nVERDICT: SUPPORTED"
        if claim_nums & ev_nums:
            return "figure matches evidence\nVERDICT: SUPPORTED"
        if ev_nums:
            return "figure not found in evidence (offline numeric check)\nVERDICT: REFUTED"
        return "no comparable figures in evidence\nVERDICT: NOT_ENOUGH_EVIDENCE"


def _norm_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").rstrip("%").strip()
