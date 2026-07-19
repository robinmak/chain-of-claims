"""FastAPI service.

Endpoints:
  POST /runs                 upload a report + source docs, start a pipeline run
  GET  /runs/{id}            full run result (claims, structure, evidence, scores)
  GET  /runs/{id}/stream     Server-Sent Events stream of stage progress
  GET  /health

The pipeline runs in a background thread; progress messages are pushed onto a per-run
queue that the SSE endpoint drains.
"""

from __future__ import annotations

import queue
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from . import db, results
from .config import settings
from .ingest.webfetch import fetch_url_text
from .pipeline import run_pipeline

app = FastAPI(title="Chain-of-Claims", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-run progress queues for SSE.
_queues: dict[str, "queue.Queue[dict]"] = {}


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()
    db.init_db()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "offline": settings.offline, "provider": settings.provider}


def _save_upload(run_dir: Path, up: UploadFile) -> str:
    dest = run_dir / (up.filename or "file")
    dest.write_bytes(up.file.read())
    return str(dest)


def _save_text(run_dir: Path, name: str, text: str) -> str:
    dest = run_dir / name
    dest.write_text(text, encoding="utf-8")
    return str(dest)


@app.post("/runs")
async def create_run(
    report: UploadFile | None = File(None),
    report_text: str | None = Form(None),
    sources: list[UploadFile] = File(default=[]),
    source_text: str | None = Form(None),
    source_urls: str | None = Form(None),
    gold: UploadFile | None = File(None),
) -> JSONResponse:
    run_id = uuid.uuid4().hex[:12]
    run_dir = settings.data_dir / "uploads" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # The document to analyze may be uploaded as a file OR pasted as text (e.g. a single
    # statement, paragraph, or section). Pasted text is written to a file so the rest of
    # the pipeline is format-agnostic.
    if report is not None and report.filename:
        report_path = _save_upload(run_dir, report)
    elif report_text and report_text.strip():
        report_path = _save_text(run_dir, "document.md", report_text)
    else:
        raise HTTPException(
            status_code=400, detail="Provide a document file or document text."
        )

    # Sources are optional: uploaded files, pasted text, and/or fetched web URLs.
    source_paths = [
        _save_upload(run_dir, s) for s in sources if s is not None and s.filename
    ]
    if source_text and source_text.strip():
        source_paths.append(_save_text(run_dir, "source_pasted.md", source_text))

    # Web sources: fetch each URL, extract text, save as an evidence file. Failures are
    # recorded as notes and skipped rather than aborting the run.
    url_notes: list[str] = []
    if source_urls and source_urls.strip():
        urls = [u.strip() for u in re.split(r"[\s,]+", source_urls) if u.strip()]
        for i, url in enumerate(urls):
            try:
                text = fetch_url_text(url)
                if text.strip():
                    header = f"Source URL: {url}\n\n"
                    source_paths.append(
                        _save_text(run_dir, f"source_url_{i}.md", header + text)
                    )
                else:
                    url_notes.append(f"{url}: no extractable text")
            except Exception as e:  # noqa: BLE001
                url_notes.append(f"{url}: fetch failed ({type(e).__name__})")

    gold_path = _save_upload(run_dir, gold) if gold is not None else None

    db.create_run(run_id, report_path, source_paths)
    q: "queue.Queue[dict]" = queue.Queue()
    _queues[run_id] = q

    def progress(stage: str, msg: str) -> None:
        q.put({"stage": stage, "message": msg})

    def worker() -> None:
        try:
            for note in url_notes:
                progress("webfetch", f"skipped {note}")
            run_pipeline(run_id, report_path, source_paths, progress=progress,
                         gold_path=gold_path)
        except Exception:  # noqa: BLE001
            pass
        finally:
            q.put({"stage": "_end", "message": "stream closed"})

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"run_id": run_id})


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    result = results.build_result(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="run not found")
    return result


@app.get("/runs/{run_id}/stream")
async def stream(run_id: str) -> EventSourceResponse:
    q = _queues.get(run_id)
    if q is None:
        raise HTTPException(status_code=404, detail="no active stream for run")

    async def event_gen():
        import asyncio

        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.2)
                continue
            if item.get("stage") == "_end":
                yield {"event": "end", "data": "done"}
                break
            yield {"event": "progress", "data": f'{item["stage"]}: {item["message"]}'}

    return EventSourceResponse(event_gen())
