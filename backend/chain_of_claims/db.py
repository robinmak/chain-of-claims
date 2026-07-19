"""SQLite persistence.

State is written after each stage so a run is inspectable and resumable. We use
plain sqlite3 (stdlib) with a thin row->model mapping rather than an ORM to keep
the dependency surface small and the schema legible.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings
from .models import (
    CausalPair,
    CausalPairVerdict,
    Claim,
    ClaimType,
    EvidenceChunk,
    ChunkKind,
    RunScores,
    RunStatus,
    Triplet,
    Verdict,
    WarrantAudit,
    CitationStatus,
    FactCheckResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    report_path TEXT,
    source_paths TEXT,          -- json list
    stage TEXT,                 -- current/last stage label
    error TEXT,
    scores_json TEXT
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    text TEXT NOT NULL,
    type TEXT NOT NULL,
    checkworthy INTEGER NOT NULL,
    cited_source TEXT,
    gold_matched INTEGER
);
CREATE TABLE IF NOT EXISTS triplets (
    claim_id INTEGER PRIMARY KEY,
    reason TEXT,
    warrant TEXT,
    warrant_generated INTEGER,
    is_causal INTEGER
);
CREATE TABLE IF NOT EXISTS warrant_audits (
    claim_id INTEGER PRIMARY KEY,
    cq_score REAL,
    cq_agreement REAL,
    per_question_pass_rate TEXT,   -- json list
    depbert_pass INTEGER,           -- nullable; deprecated mirror of structural_pass
    causal_pairs TEXT,              -- json list[CausalPair]; nullable
    structural_pass INTEGER,        -- nullable
    causal_attribution TEXT,        -- json list[CausalPairVerdict]; nullable
    causal_attribution_score REAL   -- nullable
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    locator TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS groundings (
    claim_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    relevant INTEGER NOT NULL,
    rationale TEXT,
    PRIMARY KEY (claim_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS verdicts (
    claim_id INTEGER PRIMARY KEY,
    factcheck_result TEXT,
    method_used TEXT,
    citation_status TEXT,
    detail TEXT
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for DBs created before a column existed.

    SQLite ADD COLUMN is cheap and idempotent-guarded here, so upgrading an existing
    run database does not require a rebuild.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(warrant_audits)")}
    for name, decl in (
        ("causal_pairs", "TEXT"),
        ("structural_pass", "INTEGER"),
        ("causal_attribution", "TEXT"),
        ("causal_attribution_score", "REAL"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE warrant_audits ADD COLUMN {name} {decl}")


# --- runs -------------------------------------------------------------------

def create_run(run_id: str, report_path: str, source_paths: list[str]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, status, created_at, report_path, source_paths, stage) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                RunStatus.PENDING.value,
                time.time(),
                report_path,
                json.dumps(source_paths),
                "created",
            ),
        )


def set_run_status(
    run_id: str,
    status: RunStatus,
    *,
    stage: str | None = None,
    error: str | None = None,
) -> None:
    with connect() as conn:
        fields = ["status = ?"]
        params: list[object] = [status.value]
        if stage is not None:
            fields.append("stage = ?")
            params.append(stage)
        if error is not None:
            fields.append("error = ?")
            params.append(error)
        params.append(run_id)
        conn.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", params)


def set_run_stage(run_id: str, stage: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET stage = ? WHERE id = ?", (stage, run_id))


def save_scores(run_id: str, scores: RunScores) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET scores_json = ? WHERE id = ?",
            (scores.model_dump_json(), run_id),
        )


def get_run(run_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


# --- claims -----------------------------------------------------------------

def replace_claims(run_id: str, claims: list[Claim]) -> list[Claim]:
    """Overwrite the claim set for a run; returns claims with assigned ids."""
    with connect() as conn:
        conn.execute("DELETE FROM claims WHERE run_id = ?", (run_id,))
        out: list[Claim] = []
        for c in claims:
            cur = conn.execute(
                "INSERT INTO claims (run_id, text, type, checkworthy, cited_source, gold_matched) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    c.text,
                    c.type.value,
                    int(c.checkworthy),
                    c.cited_source,
                    None if c.gold_matched is None else int(c.gold_matched),
                ),
            )
            out.append(c.model_copy(update={"id": cur.lastrowid}))
        return out


def get_claims(run_id: str) -> list[Claim]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM claims WHERE run_id = ?", (run_id,)).fetchall()
        return [
            Claim(
                id=r["id"],
                text=r["text"],
                type=ClaimType(r["type"]),
                checkworthy=bool(r["checkworthy"]),
                cited_source=r["cited_source"],
                gold_matched=None if r["gold_matched"] is None else bool(r["gold_matched"]),
            )
            for r in rows
        ]


# --- triplets ---------------------------------------------------------------

def save_triplet(t: Triplet) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO triplets (claim_id, reason, warrant, warrant_generated, is_causal) "
            "VALUES (?, ?, ?, ?, ?)",
            (t.claim_id, t.reason, t.warrant, int(t.warrant_generated), int(t.is_causal)),
        )


def get_triplet(claim_id: int) -> Triplet | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM triplets WHERE claim_id = ?", (claim_id,)).fetchone()
        if not r:
            return None
        return Triplet(
            claim_id=r["claim_id"],
            reason=r["reason"],
            warrant=r["warrant"],
            warrant_generated=bool(r["warrant_generated"]),
            is_causal=bool(r["is_causal"]),
        )


# --- warrant audits ---------------------------------------------------------

def save_warrant_audit(a: WarrantAudit) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO warrant_audits "
            "(claim_id, cq_score, cq_agreement, per_question_pass_rate, depbert_pass, "
            " causal_pairs, structural_pass, causal_attribution, causal_attribution_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                a.claim_id,
                a.cq_score,
                a.cq_agreement,
                json.dumps(a.per_question_pass_rate),
                None if a.depbert_pass is None else int(a.depbert_pass),
                None
                if a.causal_pairs is None
                else json.dumps([p.model_dump() for p in a.causal_pairs]),
                None if a.structural_pass is None else int(a.structural_pass),
                None
                if a.causal_attribution is None
                else json.dumps(
                    [v.model_dump(mode="json") for v in a.causal_attribution]
                ),
                a.causal_attribution_score,
            ),
        )


def get_warrant_audit(claim_id: int) -> WarrantAudit | None:
    with connect() as conn:
        r = conn.execute(
            "SELECT * FROM warrant_audits WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if not r:
            return None
        keys = r.keys()
        raw_pairs = r["causal_pairs"] if "causal_pairs" in keys else None
        raw_attr = r["causal_attribution"] if "causal_attribution" in keys else None
        structural = r["structural_pass"] if "structural_pass" in keys else None
        attr_score = (
            r["causal_attribution_score"]
            if "causal_attribution_score" in keys
            else None
        )
        return WarrantAudit(
            claim_id=r["claim_id"],
            cq_score=r["cq_score"],
            cq_agreement=r["cq_agreement"],
            per_question_pass_rate=json.loads(r["per_question_pass_rate"] or "[]"),
            depbert_pass=None if r["depbert_pass"] is None else bool(r["depbert_pass"]),
            causal_pairs=None
            if raw_pairs is None
            else [CausalPair.model_validate(p) for p in json.loads(raw_pairs)],
            structural_pass=None if structural is None else bool(structural),
            causal_attribution=None
            if raw_attr is None
            else [CausalPairVerdict.model_validate(v) for v in json.loads(raw_attr)],
            causal_attribution_score=attr_score,
        )


# --- chunks -----------------------------------------------------------------

def replace_chunks(run_id: str, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
        out: list[EvidenceChunk] = []
        for ch in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (run_id, source, kind, text, locator) VALUES (?, ?, ?, ?, ?)",
                (run_id, ch.source, ch.kind.value, ch.text, ch.locator),
            )
            out.append(ch.model_copy(update={"id": cur.lastrowid}))
        return out


def get_chunks(run_id: str) -> list[EvidenceChunk]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE run_id = ?", (run_id,)).fetchall()
        return [
            EvidenceChunk(
                id=r["id"],
                source=r["source"],
                kind=ChunkKind(r["kind"]),
                text=r["text"],
                locator=r["locator"],
            )
            for r in rows
        ]


# --- groundings -------------------------------------------------------------

def save_groundings(rows: list[tuple[int, int, bool, str | None]]) -> None:
    """rows = (claim_id, chunk_id, relevant, rationale)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO groundings (claim_id, chunk_id, relevant, rationale) "
            "VALUES (?, ?, ?, ?)",
            [(c, ch, int(rel), rat) for (c, ch, rel, rat) in rows],
        )


def get_relevant_chunk_ids(claim_id: int) -> list[int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM groundings WHERE claim_id = ? AND relevant = 1",
            (claim_id,),
        ).fetchall()
        return [r["chunk_id"] for r in rows]


# --- verdicts ---------------------------------------------------------------

def save_verdict(v: Verdict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO verdicts "
            "(claim_id, factcheck_result, method_used, citation_status, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                v.claim_id,
                v.factcheck_result.value,
                v.method_used,
                v.citation_status.value,
                v.detail,
            ),
        )


def get_verdict(claim_id: int) -> Verdict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM verdicts WHERE claim_id = ?", (claim_id,)).fetchone()
        if not r:
            return None
        return Verdict(
            claim_id=r["claim_id"],
            factcheck_result=FactCheckResult(r["factcheck_result"]),
            method_used=r["method_used"],
            citation_status=CitationStatus(r["citation_status"]),
            detail=r["detail"],
        )
