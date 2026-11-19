"""Run the SARMS product loop (dry run, --template) over many repositories; one CSV row each.

    uv run python tools/product_batch.py data/candidates.csv --out reports/product-batch.csv [--limit 2] [--since 2023-06-01]

The CSV needs a Repository column (owner/name or Git URL). Re-runnable: rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from sarms.mining.common import ensure_clone, git, repo_name, resolve_ref

COLS = ["repo", "status", "commits", "every", "head", "written_rules", "certified_rules", "verified",
        "violations", "enforceable", "applied", "left_standing", "seconds", "note"]


def done_repos(out):
    return {r["repo"] for r in csv.DictReader(open(out, encoding="utf-8"))} if Path(out).exists() else set()


def append(out, row):
    new = not Path(out).exists()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", default="reports/product-batch.csv")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--since", help="mine only the change stream after YYYY-MM-DD (large histories)")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per repository")
    ap.add_argument("--tests", help='optional, e.g. "python -m pytest -q" (needs each repo\'s deps installed)')
    a = ap.parse_args()
    cand = pd.read_csv(a.candidates, encoding="utf-8-sig")
    done = done_repos(a.out)
    todo = [r for r in cand["Repository"].dropna().astype(str) if r not in done][: a.limit]
    print(f"{len(todo)} repositories to run ({len(done)} already done)", flush=True)
    for repo in todo:
        name, t0 = repo_name(repo), time.time()
        row = {"repo": repo}
        print(f"\n### {repo}", flush=True)
        try:
            raw = ensure_clone(repo)
            ref = resolve_ref(raw)
            git(raw, "merge", "--ff-only", "-q", ref, check=False)  # local HEAD = the tip that gets mined
            n = int(git(raw, "rev-list", "--first-parent", "--count", ref).strip())
            every = 25 if n <= 3000 else 50 if n <= 12000 else 100
            row.update(commits=n, every=every)
            url = repo if ("://" in repo or repo.startswith("git@")) else f"https://github.com/{repo}.git"
            cmd = ["uv", "run", "--no-sync", "sarms", "auto", url, "--template", "--every", str(every)]
            if a.since:
                cmd += ["--since", a.since]
            if a.tests:
                cmd += ["--tests", a.tests]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            rms = Path("data") / name / "rms"
            rms.mkdir(parents=True, exist_ok=True)
            (rms / "batch.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
            rj = rms / "run.json"
            if p.returncode != 0 or not rj.exists():
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:]
                append(a.out, {**row, "status": "run_failed", "seconds": round(time.time() - t0),
                               "note": (tail[0] if tail else "")[:200]})
                continue
            r = json.loads(rj.read_text())
            outc = r["outcomes"]
            append(a.out, {**row, "status": "ok", "head": r["head"][:12], "written_rules": r["rules"]["written"],
                           "certified_rules": r["rules"]["certified"], "verified": r["rules"]["verified"],
                           "violations": r["violations"], "enforceable": r["enforceable"],
                           "applied": sum(o["outcome"] == "applied" for o in outc),
                           "left_standing": sum(o["outcome"] != "applied" for o in outc),
                           "seconds": round(time.time() - t0)})
        except subprocess.TimeoutExpired:
            append(a.out, {**row, "status": "timeout", "seconds": round(time.time() - t0)})
        except Exception as e:
            append(a.out, {**row, "status": "error", "note": f"{type(e).__name__}: {e}"[:200],
                           "seconds": round(time.time() - t0)})
        print(f"  done in {round(time.time() - t0)}s", flush=True)


if __name__ == "__main__":
    main()
