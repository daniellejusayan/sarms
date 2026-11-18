"""Hidden-contract benchmark: can the rule learner rediscover a team's written rules from history alone?

    uv run python tools/eval_learned_rules.py kedro-org/kedro                 # full history, up to the saved replay's last commit
    uv run python tools/eval_learned_rules.py kedro-org/kedro --pre-contract  # only history from BEFORE the first contract was written
    uv run python tools/eval_learned_rules.py kedro-org/kedro --every 50 --min-evidence 2

The learner (sarms.rules.evidence) never reads the import-linter record; this script does, after
the fact, to grade it. Hard rules = the team's contracts compiled to forbidden pairs (forbidden,
independence, and layers expanded to lower -> higher pairs) plus allowed pairs (layers higher -> lower).

A learned pair (A, B) is scored against the hard pairs:
  exact      A == F and B == T
  narrower   A under F and B under T (a correct, more specific rule)
  broader    F under A and T under B (over-general: covers a real rule but may forbid more)
  contradicted  it forbids a pair a layers contract explicitly allows   -> counted wrong
  unknown    nothing in the record says either way                      -> counted wrong for precision
Precision = (exact + narrower + broader) / learned; recall = hard pairs matched by any learned pair /
hard pairs whose ends are both known components (the learner cannot find rules about packages it
cannot see). Both are reported for all candidates and for the adopted set (>= --min-evidence kinds).
Writes data/REPO/rms/learn[-pre]/eval.json and appends a row to reports/learned-rules.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from sarms.mining.common import git, resolve_ref
from sarms.rules.compile import compile_repo
from sarms.rules.evidence import learn

ap = argparse.ArgumentParser()
ap.add_argument("owner_repo")
ap.add_argument("--every", type=int, default=25)
ap.add_argument("--pre-contract", action="store_true")
ap.add_argument("--min-evidence", type=int, default=2, help="(informational; adoption = stability + one more kind)")
ap.add_argument("--src", nargs="*")
ap.add_argument("--config")
ap.add_argument("--subdir", help="monorepo: folder holding the package; default = the folder of --config")
a = ap.parse_args()
subdir = a.subdir or (str(Path(a.config).parent) if a.config and "/" in a.config else None)
name = a.owner_repo.split("/")[-1]
raw, data = Path("data/raw") / name, Path("data") / name
if not raw.exists():
    sys.exit(f"no clone at {raw}")
record = data / "record" / "contracts.csv"
head = git(raw, "rev-parse", resolve_ref(raw)).strip()

# ---- where the learner's history ends
until = head
if (data / "sample.csv").exists():
    until = pd.read_csv(data / "sample.csv").sort_values("idx").iloc[-1].sha
tag = "learn"
if a.pre_contract:
    if not record.exists():
        sys.exit("--pre-contract needs data/REPO/record/contracts.csv")
    intro = min(pd.Timestamp(r["introduced_at"]) for r in csv.DictReader(open(record)) if r["introduced_at"])
    before = git(raw, "rev-list", "-n1", "--first-parent", f"--before={intro.isoformat()}", resolve_ref(raw)).strip()
    if not before:
        sys.exit(f"no commit before the first contract ({intro:%Y-%m-%d}); nothing to learn from")
    until, tag = before, "learn-pre"
    n_pre = int(git(raw, "rev-list", "--first-parent", "--count", before).strip())
    print(f"pre-contract setting: first contract {intro:%Y-%m-%d}; learning from the {n_pre} commits before it (ends {before[:12]})")
out = data / "rms" / tag

# ---- learn (blind)
meta, ev, cand = learn(raw, out, a.every, until, None, a.src, subdir, log=lambda s: print("   ", s, flush=True))
comps = set(meta["components"])

# ---- grade (the record is opened only here)
_, rules, _ = compile_repo(raw, sha=head, config=a.config, out_dir=out / "hard-rules-for-grading", repo_label=name)
hard = [(r["from"], r["to"], r["source"]["id"], r["source"].get("sentence", "")) for r in rules
        if r["kind"] == "forbidden" and r.get("evaluable", True) and "*" not in r["from"] + r["to"]]
allowed = {(r["from"], r["to"]) for r in rules if r["kind"] == "allowed"}


def under(x, y):
    return x == y or x.startswith(y + ".")


def grade(A, B):
    for F, T, cid, _ in hard:
        if A == F and B == T:
            return "exact", cid
    for F, T, cid, _ in hard:
        if under(A, F) and under(B, T):
            return "narrower", cid
    for F, T, cid, _ in hard:
        if under(F, A) and under(T, B):
            return "broader", cid
    for F, T in allowed:
        if (under(A, F) and under(B, T)) or (under(F, A) and under(T, B)):
            return "contradicted", None
    return "unknown", None


def known(F, T):
    return any(under(F, c) or under(c, F) for c in comps) and any(under(T, c) or under(c, T) for c in comps)


rows = []
for r in cand.to_dict("records"):
    g, cid = grade(r["from"], r["to"])
    rows.append({"from": r["from"], "to": r["to"], "rule_kind": r["rule_kind"], "evidence": r["evidence"],
                 "n_evidence": r["n_evidence"], "adopt": bool(r["adopt"]), "ba_share": r["ba_share"], "grade": g, "matches": cid or ""})
graded = pd.DataFrame(rows, columns=["from", "to", "rule_kind", "evidence", "n_evidence", "adopt", "ba_share", "grade", "matches"])
graded.to_csv(out / "candidates_graded.csv", index=False)

hard_known = [(F, T, cid) for F, T, cid, _ in hard if known(F, T)]


def summarize(df):
    if df.empty:
        return {"learned": 0, "correct": 0, "precision": None, "recall": None, "hard_known": len(hard_known), "hard": len(hard)}
    correct = df.grade.isin(["exact", "narrower", "broader"]).sum()
    found = 0
    for F, T, _ in hard_known:
        if any((under(A, F) and under(B, T)) or (under(F, A) and under(T, B)) for A, B in zip(df["from"], df["to"], strict=True)):
            found += 1
    return {"learned": int(len(df)), "correct": int(correct), "precision": round(correct / len(df), 2),
            "recall": round(found / len(hard_known), 2) if hard_known else None, "found": found,
            "hard_known": len(hard_known), "hard": len(hard),
            "grades": df.grade.value_counts().to_dict(),
            "by_evidence": {k: {"n": int(len(v)), "precision": round(v.grade.isin(["exact", "narrower", "broader"]).mean(), 2)}
                            for k, v in df.groupby("evidence")}}


res = {"repo": a.owner_repo, "setting": "pre-contract" if a.pre_contract else "full-history", "until": until,
       "sampled_commits": meta["sampled"], "window": f"{meta['first_ts'][:10]} .. {meta['last_ts'][:10]}",
       "components": len(comps), "all": summarize(graded),
       "adopted": summarize(graded[graded.adopt] if len(graded) else graded), "min_evidence": a.min_evidence}
(out / "eval.json").write_text(json.dumps(res, indent=1))

print(f"\n{a.owner_repo} — {res['setting']} — {meta['sampled']} commits sampled ({res['window']}), {len(comps)} components")
print(f"hard rules: {len(hard)} forbidden pairs, {len(hard_known)} between components the learner can see")
for key in ("all", "adopted"):
    s = res[key]
    print(f"{key:8s}: {s['learned']} learned, {s['correct']} correct -> precision {s['precision']}, recall {s['recall']}  grades {s.get('grades')}")
print("by evidence (adopted):", json.dumps(res["adopted"].get("by_evidence", {})))
missed = [(F, T) for F, T, _ in hard_known if not any((under(A, F) and under(B, T)) or (under(F, A) and under(T, B))
                                                        for A, B in zip(graded["from"], graded["to"], strict=True))]
print("hard pairs not found:", missed[:12], "..." if len(missed) > 12 else "")
wrong = graded[graded.adopt & ~graded.grade.isin(["exact", "narrower", "broader"])]
print("adopted but not in the record:", [(r["from"], r["to"], r["grade"]) for _, r in wrong.head(12).iterrows()])

summary = Path("reports") / "learned-rules.md"
line = (f"| {a.owner_repo} | {res['setting']} | {meta['sampled']} | {len(comps)} | {len(hard)} / {len(hard_known)} | "
        f"{res['adopted']['learned']} | {res['adopted']['precision']} | {res['adopted']['recall']} | "
        f"{res['all']['learned']} | {res['all']['precision']} | {res['all']['recall']} |")
headhdr = ["# Hidden-contract benchmark: learned rules vs the team's written rules", "",
           "The learner (sarms.rules.evidence) sees only the import history; the contracts are compiled afterwards to grade it. "
           "Adopted = candidates with stability (coexisted >= 2 years, both changed) plus at least one more evidence kind (asymmetry, repair, lexical); names veto forbidding a higher layer from importing a lower one. "
           "Precision counts exact, narrower and broader matches as correct; recall is over hard pairs whose both ends the learner could see. "
           "`pre-contract` learns only from commits before the first contract existed (no CI enforcement shaping the history).", "",
           "| Repository | setting | commits sampled | components | hard pairs (all / visible) | adopted | precision | recall | all candidates | precision | recall |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
old = summary.read_text().splitlines() if summary.exists() else []
body = [ln for ln in old if ln.startswith("| ") and not ln.startswith("| Repository") and not ln.startswith("| ---")]
body = [ln for ln in body if not ln.startswith(f"| {a.owner_repo} | {res['setting']} |")] + [line]
summary.parent.mkdir(exist_ok=True)
summary.write_text("\n".join(headhdr + body) + "\n")
print(f"-> {out / 'eval.json'} ; {summary}")
