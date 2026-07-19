"""Tools for the Stage-8 fact-check agent loop.

`calculator` lets the model reconstruct derived quantities rather than eyeball them
(the FinGround insight: ~43% of financial errors are computational). `lookup_evidence`
lets it pull specific grounded chunks by keyword within the retrieval-equalized set.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

from ..models import EvidenceChunk

# Safe arithmetic evaluator (no names, no calls) for the calculator tool.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def safe_calc(expr: str) -> str:
    try:
        return str(_eval(ast.parse(expr, mode="eval").body))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression to reconstruct a derived figure.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

LOOKUP_TOOL = {
    "name": "lookup_evidence",
    "description": "Search the grounded evidence chunks for a keyword; returns matches.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


def build_tool_impls(chunks: list[EvidenceChunk]) -> dict[str, Callable[[dict], str]]:
    def _calc(inp: dict) -> str:
        return safe_calc(str(inp.get("expression", "")))

    def _lookup(inp: dict) -> str:
        q = str(inp.get("query", "")).lower()
        hits = [f"[{c.locator}] {c.text}" for c in chunks if q in c.text.lower()]
        return "\n".join(hits[:8]) if hits else "no matches"

    return {"calculator": _calc, "lookup_evidence": _lookup}
