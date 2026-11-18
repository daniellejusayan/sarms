"""Does behaviour in git (hotspots, change coupling, authorship) help the rule learner find better rules?

    uv run python tools/behavioural_evidence.py --all            # every repo with a clone and graded candidates
    uv run python tools/behavioural_evidence.py kedro sqlfluff

For each learner candidate "A must not import B" (adopted or not) this adds behavioural features computed
from the same first-parent history, bulk commits (> 5 components) excluded from coupling:
  rev_a, rev_b          revisions of each component in the learner's window
  co, degree            commits touching both; degree = co / min(rev_a, rev_b)
  author_jaccard        overlap of the author sets (Conway: disjoint authors suggest a real boundary)
  churn_ratio           rev_b / rev_a  (behavioural stability: the forbidden target should be the more volatile side)
Then three questions, graded against the hidden contracts (opened only here, after the fact):
  Q1  do written rules differ from unwritten ones on these features?            (discrimination)
  Q2  does a behavioural gate added to the strict filter raise the written share? (precision proxy)
  Q3  can behaviour ADOPT candidates the structural learner left out, correctly?  (recall)
Writes data/REPO/rms/learn/behavioural.csv per repo and reports/behavioural-evidence.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from sarms.rules.evidence import comp_of
from sarms.rules.viability import score

MAX_COMPS = 5


def module_of(path: str, roots):
    if not path.endswith(".py") or "/tests/" in f"/{path}" or path.startswith("tests/"):
        return None
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for i, p in enumerate(parts):
        if p in roots:
            return ".".join(parts[i:])
    return None


def behaviour(repo_dir: Path, learn_dir: Path):
    meta = json.loads((learn_dir / "learn.json").read_text())
    roots = set(meta["roots"])
    hc = pd.read_parquet(learn_dir / "history_components.parquet")
    comps = sorted(hc[hc.idx == hc.idx.max()].comp.tolist(), key=len, reverse=True)
    args = ["git", "-C", str(repo_dir), "log", "--first-parent", f"--since={meta['first_ts']}", f"--until={meta['last_ts']}",
            "--format=@%H%x09%an", "--name-only"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("@"):
            cur = {"author": line[1:].split("\t")[1], "comps": set()}
            commits.append(cur)
        elif line.strip() and cur is not None:
            m = module_of(line.strip(), roots)
            c = comp_of(m, comps) if m else None
            if c:
                cur["comps"].add(c)
    rev, authors, co = Counter(), defaultdict(set), Counter()
    for c in commits:
        if not c["comps"]:
            continue
        for x in c["comps"]:
            rev[x] += 1
            authors[x].add(c["author"])
        if len(c["comps"]) > MAX_COMPS:
            continue
        for x, y in itertools.combinations(sorted(c["comps"]), 2):
            co[(x, y)] += 1
    return rev, authors, co, sum(1 for c in commits if c["comps"])


def features(repo: str):
    learn = Path("data") / repo / "rms" / "learn"
    raw = Path("data/raw") / repo
    g = pd.read_csv(learn / "candidates_graded.csv")
    meta = json.loads((learn / "learn.json").read_text())
    top = subprocess.run(["git", "-C", str(raw), "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    has = subprocess.run(["git", "-C", str(raw), "cat-file", "-e", f"{meta['last']}^{{commit}}"], capture_output=True).returncode == 0
    if Path(top).resolve() != raw.resolve() or not has:
        raise RuntimeError("no clone with the learner's history (git would walk up to the parent repository)")
    # tiers and the strict gates are defined over the loosely adopted set, as in `sarms adopt --strict`
    v = score(learn, "adopted")
    if v.empty:
        return None
    v = v.set_index(["from", "to"])
    rev, authors, co, n_commits = behaviour(raw, learn)
    rows = []
    for r in g.to_dict("records"):
        a, b = r["from"], r["to"]
        key = (a, b) if (a, b) in v.index else None
        ra, rb = rev[a], rev[b]
        c = co[tuple(sorted((a, b)))]
        au_a, au_b = authors[a], authors[b]
        jac = len(au_a & au_b) / len(au_a | au_b) if (au_a | au_b) else None
        rows.append({"repo": repo, "from": a, "to": b, "adopt": bool(r["adopt"]), "evidence": r["evidence"], "grade": r["grade"],
                     "written": r["grade"] in ("exact", "narrower", "broader"),
                     "strict": bool(v.loc[key, "strict"]) if key else False, "tier": v.loc[key, "tier"] if key else "",
                     "ba_share": r["ba_share"], "rev_a": ra, "rev_b": rb, "co": c, "degree": round(c / min(ra, rb), 3) if min(ra, rb) else None,
                     "author_jaccard": round(jac, 3) if jac is not None else None, "churn_ratio": round(rb / ra, 2) if ra else None,
                     "n_commits": n_commits})
    df = pd.DataFrame(rows)
    df.to_csv(learn / "behavioural.csv", index=False)
    return df


def pct(x):
    return f"{100 * x:.0f}%" if pd.notna(x) else "–"


def section(allr, title):
    ad = allr[allr.adopt].copy()
    st = ad[ad.strict]
    sg = ad[ad.tier == "strong"]
    w, u = ad[ad.written], ad[~ad.written]
    L = ["", f"## {title}", "", f"{len(ad)} adopted rules, {int(ad.written.sum())} written ({pct(ad.written.mean())}); strict {len(st)} ({pct(st.written.mean()) if len(st) else '–'} written); strong tier {len(sg)} ({pct(sg.written.mean()) if len(sg) else '–'}).", "",
         "### Q1. Written vs unwritten adopted rules", "", "| feature | written | unwritten |", "| --- | --- | --- |"]
    for name, fn in [("never co-changed (co = 0)", lambda d: (d.co == 0).mean()),
                     ("low coupling (degree <= 0.10)", lambda d: (d.degree.fillna(0) <= 0.10).mean()),
                     ("disjoint authors (jaccard <= 0.10)", lambda d: (d.author_jaccard.fillna(0) <= 0.10).mean()),
                     ("source hotter: revised >= 1.5x the target (churn_ratio <= 0.67)", lambda d: (d.churn_ratio.fillna(9) <= 0.67).mean()),
                     ("target hotter (churn_ratio >= 1.5)", lambda d: (d.churn_ratio.fillna(0) >= 1.5).mean())]:
        L.append(f"| {name} | {fn(w):.2f} | {fn(u):.2f} |" if len(w) and len(u) else f"| {name} | – | – |")
    L += ["", "### Q2. Behavioural gates on top of the structural ones", "", "| gate | rules kept | written share |", "| --- | --- | --- |"]
    for base_name, base in (("strict", st), ("strong tier", sg)):
        L.append(f"| {base_name} alone | {len(base)} | {pct(base.written.mean()) if len(base) else '–'} |")
        for name, mask in [("+ never co-changed", base.co == 0), ("+ disjoint authors", base.author_jaccard.fillna(0) <= 0.10),
                           ("+ source hotter", base.churn_ratio.fillna(9) <= 0.67), ("+ target hotter", base.churn_ratio.fillna(0) >= 1.5),
                           ("+ never co-changed + source hotter", (base.co == 0) & (base.churn_ratio.fillna(9) <= 0.67)),
                           ("+ disjoint authors + source hotter", (base.author_jaccard.fillna(0) <= 0.10) & (base.churn_ratio.fillna(9) <= 0.67))]:
            k = base[mask]
            L.append(f"| {base_name} {name} | {len(k)} | {pct(k.written.mean()) if len(k) else '–'} |")
    na = allr[~allr.adopt & (allr.rev_a >= 5) & (allr.rev_b >= 5)]
    L += ["", "### Q3. Candidates the structural learner did NOT adopt (both sides revised >= 5 times)", "",
          f"{len(na)} such candidates. Written share of what a behavioural rule would add:", "",
          "| behavioural rule | would adopt | written share |", "| --- | --- | --- |"]
    for name, mask in [("never co-changed + disjoint authors", (na.co == 0) & (na.author_jaccard.fillna(0) <= 0.10)),
                       ("never co-changed + disjoint authors + direction >= 80%", (na.co == 0) & (na.author_jaccard.fillna(0) <= 0.10) & (na.ba_share >= 0.8)),
                       ("never co-changed + direction >= 80%", (na.co == 0) & (na.ba_share >= 0.8)),
                       ("never co-changed + direction >= 80% + source hotter", (na.co == 0) & (na.ba_share >= 0.8) & (na.churn_ratio.fillna(9) <= 0.67)),
                       ("never co-changed + disjoint authors + direction >= 80% + source hotter", (na.co == 0) & (na.author_jaccard.fillna(0) <= 0.10) & (na.ba_share >= 0.8) & (na.churn_ratio.fillna(9) <= 0.67))]:
        k = na[mask]
        L.append(f"| {name} | {len(k)} | {pct(k.written.mean()) if len(k) else '–'} |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repos", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip", nargs="*", default=["kedro-org__kedro", "ICHNedx"], help="forks of other subjects")
    ap.add_argument("--holdout", nargs="*", default=["nilearn", "mlrun", "ggshield", "cloudai"], help="repositories not looked at while choosing the gates")
    a = ap.parse_args()
    repos = a.repos
    if a.all:
        repos = [d.parts[1] for d in sorted(Path("data").glob("*/rms/learn"))
                 if (d / "candidates_graded.csv").exists() and ((Path("data/raw") / d.parts[1] / ".git").exists() or (Path("data/raw") / d.parts[1] / "HEAD").exists())
                 and d.parts[1] not in a.skip]
    frames = []
    for r in repos:
        try:
            df = features(r)
        except Exception as e:  # noqa: BLE001
            print(f"{r}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        if df is None or df.empty:
            continue
        ad = df[df.adopt]
        print(f"{r}: {len(df)} candidates, {len(ad)} adopted ({int(ad.written.sum())} written), {df.n_commits.iloc[0]} commits", flush=True)
        frames.append(df)
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv("reports/behavioural-evidence.csv", index=False)
    L = ["# Behavioural evidence for learned rules (hotspots, change coupling, authorship)", "",
         f"{len(frames)} repositories with a clone and graded candidates; {len(allr)} candidates, {int(allr.adopt.sum())} adopted by the structural learner. "
         "Features from the same first-parent history (bulk commits > 5 components excluded from coupling). Contracts opened only to grade. "
         "The gates below were chosen while looking at the TRAINING repositories; the HELD-OUT block is the honest number.", ""]
    train = allr[~allr.repo.isin(a.holdout)]
    hold = allr[allr.repo.isin(a.holdout)]
    L += section(train, "Training repositories: " + ", ".join(sorted(train.repo.unique())))
    if len(hold):
        L += section(hold, "HELD-OUT repositories: " + ", ".join(sorted(hold.repo.unique())))
    L += ["", "## Per repository (adopted rules)", "",
          "| repo | adopted | written | strict | written | strong + never co-changed + source hotter | written | strict + never co-changed + source hotter | written |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ad = allr[allr.adopt]
    for repo, grp in ad.groupby("repo"):
        s_ = grp[grp.strict]
        sg = grp[(grp.tier == "strong") & (grp.co == 0) & (grp.churn_ratio.fillna(9) <= 0.67)]
        ss = s_[(s_.co == 0) & (s_.churn_ratio.fillna(9) <= 0.67)]
        tag = " (held out)" if repo in a.holdout else ""
        L.append(f"| {repo}{tag} | {len(grp)} | {pct(grp.written.mean())} | {len(s_)} | {pct(s_.written.mean()) if len(s_) else '–'} | "
                 f"{len(sg)} | {pct(sg.written.mean()) if len(sg) else '–'} | {len(ss)} | {pct(ss.written.mean()) if len(ss) else '–'} |")
    Path("reports/behavioural-evidence.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
