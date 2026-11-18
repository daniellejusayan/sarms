"""The rule learner, deterministic half: what the import history says about a repository's
components, and which rules that evidence supports. No model here, and NO access to the
repository's import-linter record: this module never reads .importlinter / setup.cfg /
pyproject.toml, so the hard rules stay hidden from it (tools/eval_learned_rules.py relies on that).

    uv run python -m sarms.rules.evidence data/raw/kedro --out data/kedro/rms/learn [--every 25] [--until SHA]

Steps
  1. roots       top-level importable packages found on disk (dirs with __init__.py at the repo
                 root or under src/), tests/docs/examples excluded
  2. history     every k-th first-parent commit: the module import graph (the lint worker with
                 no contracts), rolled up to components = depth-2 packages, split to depth 3
                 where a depth-2 package is itself a tree of >= 3 sub-packages (kedro.framework.*)
  3. evidence    for every ordered pair (A, B) of components alive at the end:
                   commits_both   sampled commits where both existed
                   ab / ba        commits with a direct module import A->B / B->A
                   present_now    A->B exists at the last commit
                   repaired       A->B existed, then vanished and stayed gone
                   span_days      how long both have coexisted
                   changed        both components changed size over the window
  4. candidates  forbidden A->B where A->B is absent now, with evidence kinds (adopt = stability plus
                 at least one more kind; names veto a candidate that would forbid a higher layer
                 importing a lower one unless the structure insists):
                   asymmetry  B->A present in >= 50% of the commits where both existed
                   repair     A->B was removed and stayed removed
                   stability  coexisted >= 2 years and both kept changing (the absence is not luck)
                   lexical    names say A is a lower layer than B (cli > service > core > io > config/utils)
                 independence A|B (both directions absent, siblings, both stable) as two forbidden pairs
Outputs <out>/{history_edges,history_components,evidence,candidates}.parquet/csv. Adoption
(thresholds, judge) is sarms.synthesize.learn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from sarms.mining.common import first_parent, git, resolve_ref, worktree_path
from sarms.mining.tree import checkout, run_worker

SKIP_DIRS = {"tests", "test", "docs", "doc", "examples", "example", "scripts", "benchmarks", "features", "templates",
             "build", "dist", "site-packages", "node_modules", ".venv", "venv"}

# a small layer vocabulary: lower number = higher layer (closer to the user / entry point)
LAYER_WORDS = [
    ("cli", "commands", "command", "console", "entrypoint", "entrypoints", "main", "app", "apps", "ui", "web", "api", "server", "serve", "rest", "http"),
    ("session", "framework", "application", "services", "service", "orchestration", "workflow", "manager", "managers"),
    ("runner", "runners", "engine", "pipeline", "pipelines", "core", "domain", "logic", "rules", "linter", "checker"),
    ("io", "dataset", "datasets", "storage", "db", "database", "persistence", "repository", "repositories", "adapters", "adapter", "backend", "backends", "client", "clients"),
    ("config", "configuration", "settings", "utils", "util", "utilities", "common", "helpers", "helper", "types", "typing", "constants", "exceptions", "errors", "models", "schema", "schemas", "protocol", "protocols"),
]


def layer_rank(component: str):
    """Rank of the most specific name part that is in the vocabulary (the last part first), so
    `app.config` is a config layer, not an app layer."""
    for part in reversed(component.split(".")[1:] or component.split(".")):
        for i, words in enumerate(LAYER_WORDS):
            if part.lower() in words:
                return i
    return None


def detect_roots(tree: Path):
    """[(root package name, base dir)] without reading any config."""
    out = []
    for base in (tree, tree / "src"):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "__init__.py").exists() and d.name not in SKIP_DIRS and not d.name.startswith((".", "_")):
                out.append((d.name, base))
    seen, roots = set(), []
    for name, base in out:
        if name not in seen:
            seen.add(name)
            roots.append((name, base))
    return roots


def components_of(modules, roots, min_sub=3):
    """Depth-2 packages; split to depth 3 where the depth-2 package holds >= min_sub sub-packages."""
    mods = set(modules)
    depth2 = {m for m in mods if m.count(".") == 1 and m.split(".")[0] in roots}
    comps = set()
    for d2 in depth2:
        depth3 = {m for m in mods if m.startswith(d2 + ".") and m.count(".") == 2}
        subpkgs = {m for m in depth3 if any(x.startswith(m + ".") for x in mods)}
        if len(subpkgs) >= min_sub:
            comps.update(depth3)   # every depth-3 package or module is its own component (framework.project, framework.startup)
            comps.add(d2)          # the package's own __init__ stays as d2
        else:
            comps.add(d2)
    return sorted(comps)


def comp_of(module, comps_sorted_longest_first):
    for c in comps_sorted_longest_first:
        if module == c or module.startswith(c + "."):
            return c
    return None


def scan_history(repo_dir, every=25, until=None, since=None, src=None, roots=None, subdir=None, log=print):
    repo_dir = Path(repo_dir)
    ref = resolve_ref(repo_dir)
    chain = first_parent(repo_dir, ref, since=since)
    if until:
        full = git(repo_dir, "rev-parse", until).strip()
        pos = [i for i, c in enumerate(chain) if c[1] == full]
        if not pos:
            sys.exit(f"--until {until[:12]} is not on the first-parent chain")
        chain = chain[:pos[0] + 1]
    sample = [c for n, c in enumerate(chain) if n % every == 0]
    if sample[-1] != chain[-1]:
        sample.append(chain[-1])
    tree = checkout(repo_dir, sample[-1][1])
    base = tree / subdir if subdir else tree
    found = detect_roots(base) if roots is None else []
    if roots is None:
        roots = [r for r, _ in found]
    if not roots:
        sys.exit("no importable top-level package found" + (f" under {subdir}" if subdir else ""))
    if src is None and found:
        src = sorted({str(b.relative_to(tree)) for _, b in found})   # where the roots live, relative to the repo
    bundle = {"session": {"root_packages": roots}, "contracts_strict": [], "contracts_asis": []}
    cache = worktree_path(repo_dir, "ilcache").resolve()
    edges_rows, comp_rows, t0, comps, errors = [], [], time.time(), None, []
    for n, (idx, sha, ts, _m) in enumerate(sample, 1):
        tree = checkout(repo_dir, sha)
        res = run_worker(tree, bundle, cache, src, want_edges=True, want_details=False)
        if "graph_error" in res:
            errors.append((idx, sha, res["graph_error"]))
            if len(errors) <= 3:
                log(f"  graph error at {sha[:12]} ({ts[:10]}): {res['graph_error'][:160]}")
            continue
        edges = [tuple(e) for e in res["edges"]]
        modules = set(res.get("module_list") or [m for e in edges for m in e])
        edges_rows.append((idx, sha, ts, edges))
        comp_rows.append((idx, sha, ts, sorted(modules)))
        if n % 20 == 0 or n == len(sample):
            el = time.time() - t0
            log(f"  {n}/{len(sample)} commits ({el:.0f}s)")
    if errors:
        log(f"  {len(errors)} of {len(sample)} commits could not be read (first at {errors[0][1][:12]}); evidence uses the rest")
    if not comp_rows:
        sys.exit("no commit could be read")
    last_modules = comp_rows[-1][3]
    comps = components_of(last_modules, roots)
    longest = sorted(comps, key=len, reverse=True)
    he, hc = [], []
    for (idx, sha, ts, edges), (_, _, _, modules) in zip(edges_rows, comp_rows, strict=True):
        counts = {}
        for a, b in edges:
            ca, cb = comp_of(a, longest), comp_of(b, longest)
            if ca and cb and ca != cb:
                counts[(ca, cb)] = counts.get((ca, cb), 0) + 1
        he.extend({"idx": idx, "sha": sha, "ts": ts, "comp_from": a, "comp_to": b, "n": c} for (a, b), c in counts.items())
        sizes = {}
        for m in modules:
            c = comp_of(m, longest)
            if c:
                sizes[c] = sizes.get(c, 0) + 1
        hc.extend({"idx": idx, "sha": sha, "ts": ts, "comp": c, "n_modules": k} for c, k in sizes.items())
    he, hc = pd.DataFrame(he), pd.DataFrame(hc)
    for df in (he, hc):
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return he, hc, comps, roots, sample, errors


def evidence(he, hc, comps):
    """One row per ordered pair of components alive at the last commit."""
    last_idx = hc.idx.max()
    alive = set(hc[hc.idx == last_idx].comp)
    present = he.groupby(["comp_from", "comp_to"]).idx.apply(set).to_dict()
    exists = hc.groupby("comp").idx.apply(set).to_dict()
    first_ts = hc.groupby("comp").ts.min().to_dict()
    sizes = hc.groupby("comp").n_modules.agg(["min", "max"]).to_dict("index")
    ts_of = dict(zip(hc.idx, hc.ts, strict=True))
    rows = []
    for a in sorted(alive):
        for b in sorted(alive):
            if a == b:
                continue
            both = exists[a] & exists[b]
            if not both:
                continue
            ab, ba = present.get((a, b), set()) & both, present.get((b, a), set()) & both
            span_days = (ts_of[max(both)] - max(first_ts[a], first_ts[b])).total_seconds() / 86400
            repaired = bool(ab) and last_idx not in ab and max(ab) < max(both) and (max(both) - max(ab)) >= (sorted(both)[1] - sorted(both)[0] if len(both) > 1 else 1)
            rows.append({"comp_from": a, "comp_to": b, "commits_both": len(both), "ab": len(ab), "ba": len(ba),
                         "present_now": last_idx in ab, "ever": bool(ab), "repaired": repaired,
                         "ba_share": len(ba) / len(both), "span_days": round(span_days, 1),
                         "changed": sizes[a]["max"] > sizes[a]["min"] and sizes[b]["max"] > sizes[b]["min"]})
    return pd.DataFrame(rows)


def candidates(ev, min_commits=8, min_span_days=365, asym_share=0.5, stable_days=730):
    """Forbidden-pair candidates with their evidence kinds. Only pairs ABSENT now (so adopting
    one never touches existing code)."""
    out = []
    for r in ev.itertuples(index=False):
        if r.present_now or r.commits_both < min_commits or r.span_days < min_span_days:
            continue
        kinds = []
        if r.ba_share >= asym_share:
            kinds.append("asymmetry")
        if r.repaired:
            kinds.append("repair")
        if r.span_days >= stable_days and r.changed:
            kinds.append("stability")
        ra, rb = layer_rank(r.comp_from), layer_rank(r.comp_to)
        if ra is not None and rb is not None and ra > rb:
            kinds.append("lexical")
        # the names can also veto: if the vocabulary says A is a HIGHER layer than B, forbidding
        # A -> B contradicts them, and only strong structural asymmetry may override that
        if ra is not None and rb is not None and ra < rb and r.ba_share < asym_share:
            continue
        if not kinds:
            continue
        rule_kind = "layer" if "asymmetry" in kinds else "forbidden"
        out.append({"from": r.comp_from, "to": r.comp_to, "rule_kind": rule_kind, "evidence": "+".join(kinds),
                    "n_evidence": len(kinds), "adopt": len(kinds) >= 2 and "stability" in kinds,
                    "commits_both": r.commits_both, "ba_share": round(r.ba_share, 2),
                    "span_days": r.span_days, "repaired": r.repaired})
    cand = pd.DataFrame(out)
    # independence: siblings under the same parent, absent both ways, both stable
    if len(ev):
        parent = lambda c: c.rsplit(".", 1)[0]  # noqa: E731
        pairs = {(r.comp_from, r.comp_to): r for r in ev.itertuples(index=False)}
        for (a, b), r in pairs.items():
            if a < b and parent(a) == parent(b) and not r.ever and (b, a) in pairs and not pairs[(b, a)].ever \
                    and r.commits_both >= min_commits and r.span_days >= stable_days and r.changed:
                for x, y in ((a, b), (b, a)):
                    cand = pd.concat([cand, pd.DataFrame([{"from": x, "to": y, "rule_kind": "independence",
                                                           "evidence": "mutual-absence+stability", "n_evidence": 2, "adopt": True,
                                                           "commits_both": r.commits_both, "ba_share": 0.0,
                                                           "span_days": r.span_days, "repaired": False}])], ignore_index=True)
    if len(cand):
        cand = cand.drop_duplicates(["from", "to"]).sort_values(["n_evidence", "ba_share"], ascending=False).reset_index(drop=True)
    return cand


def learn(repo_dir, out_dir, every=25, until=None, since=None, src=None, subdir=None, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    he, hc, comps, roots, sample, errors = scan_history(repo_dir, every, until, since, src, subdir=subdir, log=log)
    ev = evidence(he, hc, comps)
    cand = candidates(ev)
    he.to_parquet(out / "history_edges.parquet", index=False)
    hc.to_parquet(out / "history_components.parquet", index=False)
    ev.to_csv(out / "evidence.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    meta = {"repo": str(repo_dir), "roots": roots, "components": comps, "every": every, "sampled": len(sample),
            "first": sample[0][1], "last": sample[-1][1], "first_ts": sample[0][2], "last_ts": sample[-1][2],
            "pairs": int(len(ev)), "candidates": int(len(cand)), "unreadable_commits": len(errors),
            "last_read": comps and (hc.sha.iloc[-1] if len(hc) else None)}
    (out / "learn.json").write_text(json.dumps(meta, indent=1))
    log(f"{len(comps)} components, {len(ev)} pairs with evidence, {len(cand)} candidate rules -> {out}")
    return meta, ev, cand


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--until")
    ap.add_argument("--since")
    ap.add_argument("--src", nargs="*")
    ap.add_argument("--subdir", help="monorepo: the folder that holds the package (api/, libs/x/)")
    a = ap.parse_args()
    learn(a.repo_dir, a.out, a.every, a.until, a.since, a.src, a.subdir, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
