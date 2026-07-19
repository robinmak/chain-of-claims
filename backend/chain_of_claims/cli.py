"""CLI: run the pipeline on local files and print scores + a claim summary.

Usage:
  COC_OFFLINE=1 python -m chain_of_claims.cli REPORT SOURCE [SOURCE ...] [--gold GOLD.json]
Example:
  COC_OFFLINE=1 python -m chain_of_claims.cli samples/report.md samples/source_filing.md \\
      --gold samples/gold_claims.json
"""

from __future__ import annotations

import sys
import uuid

from . import db, results
from .config import settings
from .pipeline import run_pipeline


def main(argv: list[str]) -> int:
    gold_path = None
    if "--gold" in argv:
        i = argv.index("--gold")
        try:
            gold_path = argv[i + 1]
        except IndexError:
            print("error: --gold requires a path")
            return 2
        argv = argv[:i] + argv[i + 2:]

    if len(argv) < 2:
        print("usage: python -m chain_of_claims.cli REPORT SOURCE [SOURCE ...] [--gold GOLD.json]")
        return 2

    settings.ensure_dirs()
    db.init_db()

    report_path, *source_paths = argv
    run_id = uuid.uuid4().hex[:12]
    db.create_run(run_id, report_path, source_paths)

    def progress(stage: str, msg: str) -> None:
        print(f"  [{stage}] {msg}")

    print(f"Run {run_id} (offline={settings.offline})")
    scores = run_pipeline(
        run_id, report_path, source_paths, progress=progress, gold_path=gold_path
    )

    print("\n=== SCORES ===")
    expl = "N/A" if scores.explainability is None else f"{scores.explainability:.2f}"
    hallu = "N/A" if scores.hallucination is None else f"{scores.hallucination:.2f}"
    print(f"Explainability: {expl}")
    print(f"Hallucination:  {hallu}")
    print(f"By type: {scores.hallucination_by_type}")
    print(f"Claims: {scores.n_claims}  check-worthy: {scores.n_checkworthy}")
    if scores.verification_skipped:
        print("Verification: skipped (no source materials supplied)")
    if scores.coverage is not None:
        print(
            f"Coverage (recall): {scores.coverage:.2f} "
            f"({scores.n_gold_matched}/{scores.n_gold} gold)  "
            f"precision: {scores.coverage_precision:.2f}"
        )
    else:
        print(f"Coverage: {scores.coverage_note}")

    result = results.build_result(run_id)
    print("\n=== CLAIMS ===")
    for c in result["claims"]:
        v = c["verdict"] or {}
        print(
            f"- [{c['type']}] {c['text'][:70]!r}\n"
            f"    fact={v.get('factcheck_result')} "
            f"cite={v.get('citation_status')} "
            f"method={v.get('method_used')} "
            f"evidence={len(c['evidence'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
