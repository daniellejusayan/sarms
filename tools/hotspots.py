"""Behavioural signals in the CodeScene style, at the component level: hotspots, change coupling, erosion pressure.

    uv run python tools/hotspots.py data/raw/kedro --learn data/kedro/rms/learn --rules data/kedro/rms/rules data/kedro/rms/rules-definitive \
        --replay data/kedro/rms/differ --months 24 --out data/kedro/rms/hotspots

Three signals, all from git history plus the component graph the learner already built (no config read):
  hotspot          revisions of a component in the window x its size (lines of Python at HEAD): where work concentrates
  change coupling  two components modified in the same commit; degree = co-commits / min(revisions), kept when
                   >= --min-co co-commits and both have >= --min-rev revisions. Bulk commits (more than --max-comps
                   components, e.g. a release-wide rename) count as revisions but not as coupling, as CodeScene
                   also excludes them, otherwise every pair looks coupled. Each coupled pair is classified:
                     import     an import edge exists between them today (expected coupling)
                     hidden     no import either way: a logical dependency the import graph cannot see
                     forbidden  a rule (hard or definitive) forbids one direction: erosion pressure, developers keep
                                touching both sides together and the next shortcut is a rule violation
                   trend = degree in the second half of the window minus the first half (rising / stable / falling)
  erosion heat     per component: violation episodes that started in the window (from the replay), forbidden-coupled
                   partners, hidden-coupled partners, hotspot rank -> one 0..1 score for the heat map
Writes components.csv, coupling.csv, heatmap.json (matrix of coupling degree with the class per cell) and a summary.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from sarms.mining.common import git
from sarms.rules.evidence import comp_of


def load_components(learn_dir: Path):
    hc = pd.read_parquet(learn_dir / "history_components.parquet")
    he = pd.read_parquet(learn_dir / "history_edges.parquet")
    last = hc.idx.max()
    comps = sorted(hc[hc.idx == last].comp.tolist(), key=len, reverse=True)
    edges = {(r.comp_from, r.comp_to) for r in he[he.idx == last].itertuples(index=False)}
    return comps, edges


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


def commits_with_components(repo_dir: Path, comps, roots, months: int):
    out = subprocess.run(["git", "-C", str(repo_dir), "log", "--first-parent", f"--since={months} months ago",
                          "--format=@%H%x09%at%x09%an", "--name-only"], capture_output=True, text=True, check=True).stdout
    commits = []
    cur = None
    for line in out.splitlines():
        if line.startswith("@"):
            sha, ts, author = line[1:].split("\t")
            cur = {"sha": sha, "ts": int(ts), "author": author, "comps": set()}
            commits.append(cur)
        elif line.strip() and cur is not None:
            m = module_of(line.strip(), roots)
            c = comp_of(m, comps) if m else None
            if c:
                cur["comps"].add(c)
    return [c for c in commits if c["comps"]]


def loc_at_head(repo_dir: Path, comps, roots):
    files = git(repo_dir, "ls-files", "*.py").splitlines()
    loc = Counter()
    for f in files:
        m = module_of(f, roots)
        c = comp_of(m, comps) if m else None
        if c:
            try:
                loc[c] += sum(1 for _ in (repo_dir / f).open(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return loc


def forbidden_pairs(rule_dirs):
    pairs = {}
    for d in rule_dirs:
        for f in Path(d).glob("*.yaml"):
            for r in yaml.safe_load(f.read_text()) or []:
                if isinstance(r, dict) and r.get("kind") == "forbidden" and "*" not in r["from"] + r["to"]:
                    pairs[(r["from"], r["to"])] = r["id"]
    return pairs


def under(x, y):
    return x == y or x.startswith(y + ".")


def forbids(pairs, a, b):
    for (F, T), rid in pairs.items():
        if under(a, F) and under(b, T):
            return rid
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo_dir")
    ap.add_argument("--learn", required=True)
    ap.add_argument("--rules", nargs="*", default=[])
    ap.add_argument("--replay", help="replay dir with edges.parquet, for violation episodes per component")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--min-rev", type=int, default=5)
    ap.add_argument("--min-co", type=int, default=3)
    ap.add_argument("--max-comps", type=int, default=5, help="commits touching more components than this are bulk: no coupling")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, out = Path(a.repo_dir).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.learn) / "learn.json").read_text())
    roots = set(meta["roots"])
    comps, edges = load_components(Path(a.learn))
    commits = commits_with_components(repo, comps, roots, a.months)
    if not commits:
        raise SystemExit("no commits touching the components in the window")
    mid = (commits[0]["ts"] + commits[-1]["ts"]) / 2  # log is newest first
    rev, rev_half, authors, co, co_half = Counter(), defaultdict(Counter), defaultdict(set), Counter(), defaultdict(Counter)
    bulk = 0
    for c in commits:
        half = "late" if c["ts"] >= mid else "early"
        for x in c["comps"]:
            rev[x] += 1
            rev_half[half][x] += 1
            authors[x].add(c["author"])
        if len(c["comps"]) > a.max_comps:
            bulk += 1
            continue
        for x, y in itertools.combinations(sorted(c["comps"]), 2):
            co[(x, y)] += 1
            co_half[half][(x, y)] += 1
    loc = loc_at_head(repo, comps, roots)
    rules = forbidden_pairs(a.rules)

    # ---- episodes per component from the replay (violations that started in the window)
    episodes = Counter()
    if a.replay and (Path(a.replay) / "edges.parquet").exists():
        e = pd.read_parquet(Path(a.replay) / "edges.parquet")
        e = e[e.label != "CONFIRMED"]
        since = pd.Timestamp.now("UTC") - pd.DateOffset(months=a.months)
        first = e.groupby(["rule_id", "edge_from", "edge_to"]).ts.min().reset_index()
        for r in first[first.ts >= since].itertuples(index=False):
            c = comp_of(r.edge_from, comps)
            if c:
                episodes[c] += 1

    # ---- coupling table
    rows = []
    for (x, y), n in co.items():
        if n < a.min_co or min(rev[x], rev[y]) < a.min_rev:
            continue
        degree = n / min(rev[x], rev[y])
        imp = (x, y) in edges or (y, x) in edges
        rid = forbids(rules, x, y) or forbids(rules, y, x)
        direction = f"{x} -> {y}" if forbids(rules, x, y) else (f"{y} -> {x}" if forbids(rules, y, x) else "")
        klass = "forbidden" if rid else ("import" if imp else "hidden")

        def deg(half, x=x, y=y):
            m = min(rev_half[half][x], rev_half[half][y])
            return co_half[half][(x, y)] / m if m else 0.0
        trend_v = deg("late") - deg("early")
        trend = "rising" if trend_v > 0.1 else "falling" if trend_v < -0.1 else "stable"
        rows.append({"a": x, "b": y, "co_commits": n, "revisions_a": rev[x], "revisions_b": rev[y], "degree": round(degree, 2),
                     "class": klass, "rule_id": rid or "", "forbidden_direction": direction, "has_import": imp,
                     "degree_early": round(deg("early"), 2), "degree_late": round(deg("late"), 2), "trend": trend})
    coupling = pd.DataFrame(rows).sort_values(["class", "degree"], ascending=[True, False]) if rows else pd.DataFrame()
    coupling.to_csv(out / "coupling.csv", index=False)

    # ---- component table and erosion heat
    comp_rows = []
    for c in sorted(set(rev) | set(loc)):
        hot = rev[c] * loc[c]
        mine = coupling[(coupling.a == c) | (coupling.b == c)] if len(coupling) else pd.DataFrame(columns=["class"])
        forb = int((mine["class"] == "forbidden").sum())
        hid = int((mine["class"] == "hidden").sum())
        comp_rows.append({"component": c, "revisions": rev[c], "authors": len(authors[c]), "loc": loc[c], "hotspot": hot,
                          "episodes_started": episodes[c], "forbidden_coupled_partners": forb, "hidden_coupled_partners": hid})
    comp = pd.DataFrame(comp_rows)
    if len(comp):
        comp["hotspot_rank"] = comp.hotspot.rank(pct=True)
        mx = lambda s: s / s.max() if s.max() else s  # noqa: E731
        comp["erosion_heat"] = (0.4 * mx(comp.episodes_started) + 0.3 * mx(comp.forbidden_coupled_partners)
                                + 0.15 * mx(comp.hidden_coupled_partners) + 0.15 * comp.hotspot_rank).round(2)
        comp = comp.sort_values("erosion_heat", ascending=False)
    comp.to_csv(out / "components.csv", index=False)

    # ---- heat map: coupling degree matrix over the components that have any coupling, class per cell
    active = sorted({*coupling.a, *coupling.b}) if len(coupling) else []
    heat = {"repo": repo.name, "window_months": a.months, "commits": len(commits), "bulk_excluded": bulk, "components": active,
            "cells": [{"a": r["a"], "b": r["b"], "degree": r["degree"], "class": r["class"], "trend": r["trend"], "co": int(r["co_commits"]),
                       "rule_id": r["rule_id"], "direction": r["forbidden_direction"]} for r in coupling.to_dict("records")],
            "component_rows": comp.to_dict("records")}
    (out / "heatmap.json").write_text(json.dumps(heat, indent=1, default=str))

    print(f"{repo.name}: {len(commits)} commits in the last {a.months} months touched {len(rev)} components ({bulk} bulk commits excluded from coupling); "
          f"{len(coupling)} coupled pairs ({coupling['class'].value_counts().to_dict() if len(coupling) else {}})")
    print("\nTop hotspots (revisions x lines):")
    for r in comp.sort_values("hotspot", ascending=False).head(8).itertuples(index=False):
        print(f"  {r.component:40s} rev {r.revisions:4d}  loc {r.loc:6d}  authors {r.authors:3d}  episodes {r.episodes_started}")
    print("\nErosion pressure: coupling across a forbidden boundary")
    for r in coupling[coupling["class"] == "forbidden"].head(12).itertuples(index=False):
        print(f"  {r.a} ~ {r.b}: {r.co_commits} co-commits, degree {r.degree}, {r.trend}; rule {r.rule_id} ({r.forbidden_direction})")
    print("\nHidden coupling (co-change, no import either way):")
    for r in coupling[coupling["class"] == "hidden"].head(12).itertuples(index=False):
        print(f"  {r.a} ~ {r.b}: {r.co_commits} co-commits, degree {r.degree}, {r.trend}")
    print("\nErosion heat (0..1):")
    for r in comp.head(10).itertuples(index=False):
        print(f"  {r.component:40s} {r.erosion_heat:.2f}  episodes {r.episodes_started} forbidden-coupled {r.forbidden_coupled_partners} hidden {r.hidden_coupled_partners} hot-rank {r.hotspot_rank:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
