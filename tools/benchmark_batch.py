"""Run the hidden-contract benchmark over many repositories, verifying each one's contracts first.

    uv run python tools/benchmark_batch.py candidates.csv --out reports/learned-rules-batch.csv [--limit 60] [--keep]

candidates.csv needs a Repository column (owner/name). For each: partial clone (blob:none), read
the real import-linter record at HEAD (a listed count is not trusted), skip when fewer than
--min-contracts; otherwise run tools/eval_learned_rules.py (full history; sampling every 25 / 50 /
100 commits by history length) and append the eval.json numbers to the CSV. Unless --keep, the clone,
worktrees and cache are removed afterwards to save disk. Safe to re-run: done rows are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from sarms.mining.common import (
    ensure_clone,
    find_il_configs_anywhere,
    git,
    read_il_options,
    repo_name,
    resolve_ref,
)

COLS = ["repo", "status", "contracts", "contract_types", "commits", "stars", "sampled", "components", "hard_pairs",
        "hard_visible", "adopted", "adopted_correct", "precision", "recall", "all_candidates", "all_precision", "all_recall",
        "unreadable", "seconds", "note"]


def read_done(out):
    if not Path(out).exists():
        return {}
    return {r["repo"]: r for r in csv.DictReader(open(out))}


def append(out, row):
    new = not Path(out).exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", default="reports/learned-rules-batch.csv")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-contracts", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="run every stride-th candidate (for parallel workers)")
    a = ap.parse_args()
    cand = pd.read_csv(a.candidates, encoding="utf-8-sig")
    done = read_done(a.out)
    todo = [r for r in cand.to_dict("records") if r["Repository"] not in done][a.offset::a.stride][:a.limit]
    print(f"{len(todo)} repositories to run ({len(done)} already done)", flush=True)
    for r in todo:
        repo = r["Repository"]
        name = repo_name(repo)
        t0 = time.time()
        row = {"repo": repo, "stars": r.get("Stars", ""), "commits": r.get("Commits", "")}
        print(f"\n### {repo}", flush=True)
        try:
            raw = ensure_clone(repo, checkout=False)
        except Exception as e:
            append(a.out, {**row, "status": "clone_failed", "note": str(e)[:200]})
            continue
        try:
            head = git(raw, "rev-parse", resolve_ref(raw)).strip()
            fname, opts = None, None
            for cand_f, text in find_il_configs_anywhere(raw, head):
                o = read_il_options(cand_f, text)
                if o and o.contracts_options:
                    fname, opts = cand_f, o
                    break
            n = len(opts.contracts_options) if opts else 0
            row["note"] = fname or ""
            types = sorted({c.get("type", "?") for c in opts.contracts_options}) if opts else []
            n_commits = int(git(raw, "rev-list", "--first-parent", "--count", head).strip())
            row.update({"contracts": n, "contract_types": "+".join(types), "commits": n_commits})
            if n < a.min_contracts:
                append(a.out, {**row, "status": "no_contracts" if n == 0 else "too_few_contracts", "seconds": round(time.time() - t0)})
                print(f"  skipped: {n} contract(s) at HEAD", flush=True)
                continue
            every = 25 if n_commits <= 3000 else 50 if n_commits <= 12000 else 100
            cmd = [sys.executable, "tools/eval_learned_rules.py", repo, "--every", str(every)]
            if fname and "/" in fname:
                cmd += ["--config", fname]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            ev = Path("data") / name / "rms" / "learn" / "eval.json"
            if p.returncode != 0 or not ev.exists():
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:]
                append(a.out, {**row, "status": "eval_failed", "seconds": round(time.time() - t0), "note": (tail[0] if tail else "")[:200]})
                print(f"  eval failed: {tail}", flush=True)
                continue
            res = json.loads(ev.read_text())
            s, al = res["adopted"], res["all"]
            append(a.out, {**row, "status": "ok", "sampled": res["sampled_commits"], "components": res["components"],
                           "hard_pairs": s.get("hard"), "hard_visible": s.get("hard_known"), "adopted": s["learned"],
                           "adopted_correct": s["correct"], "precision": s["precision"], "recall": s["recall"],
                           "all_candidates": al["learned"], "all_precision": al["precision"], "all_recall": al["recall"],
                           "unreadable": json.loads((ev.parent / "learn.json").read_text()).get("unreadable_commits", ""),
                           "seconds": round(time.time() - t0)})
            print(f"  ok: adopted {s['learned']} precision {s['precision']} recall {s['recall']} ({round(time.time() - t0)}s)", flush=True)
        except subprocess.TimeoutExpired:
            append(a.out, {**row, "status": "timeout", "seconds": round(time.time() - t0)})
        except Exception as e:
            append(a.out, {**row, "status": "error", "note": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.time() - t0)})
            print(f"  error: {e}", flush=True)
        finally:
            if not a.keep and name not in ("kedro", "checkov", "wemake-python-styleguide", "sqlfluff", "faststream", "ramses_rf", "posthog", "qiskit-serverless"):
                for suffix in ("", ".wt", ".ilcache", ".draft", ".fi"):
                    shutil.rmtree(Path("data/raw") / f"{name}{suffix}", ignore_errors=True)


if __name__ == "__main__":
    main()
