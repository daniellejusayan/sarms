"""Track C, part 1: rules and hotspots from the change stream alone (no import graph).

Directional co-change confidence (per component, or per file):
    C(x -> y) = |commits changing x and y| / |commits changing x|
A pair with C(x -> y) > THRESHOLD and enough support is a *directional coupling*: when x changes, y usually
changes too. With the layer vocabulary (sarms.rules.evidence.LAYER_WORDS; level 0 = config/utils at the bottom,
level 4 = cli/ui/app at the top) a coupling whose low-level side is driven by a high-level side,
    C(x -> y) > THRESHOLD and level(x) < level(y),
is reported as a layer-constraint violation candidate: low-level logic that moves whenever high-level
infrastructure moves. The constraint written for it is "x must not depend on y" (same card format as every
other rule), and it is then checked against the import graph like any other rule.

Hotspots (per file): H(f) = commits(f) x churn(f). Critical = Pareto top 5% by H or a statistical outlier
(z >= 2 on log H); quiet debt = high commits with low churn, or low commits with high churn (medians split).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sarms.rules.evidence import layer_rank

THRESHOLD = 0.35
MIN_SUPPORT = 5
PARETO = 0.05
Z = 2.0


def level(component: str):
    """0 = bottom (config, utils, models), 4 = top (cli, ui, app). None when the name says nothing."""
    r = layer_rank(component)
    return None if r is None else 4 - r


def confidence(df: pd.DataFrame, unit: str = "component", min_support: int = MIN_SUPPORT) -> pd.DataFrame:
    """C(x -> y) for every ordered pair seen together in at least one commit. unit = 'component' | 'path'."""
    sets = df.dropna(subset=[unit]).groupby("idx")[unit].apply(lambda s: frozenset(s)).tolist()
    single, pair = Counter(), Counter()
    for s in sets:
        for x in s:
            single[x] += 1
        for x in s:
            for y in s:
                if x != y:
                    pair[(x, y)] += 1
    rows = [{"x": x, "y": y, "commits_x": single[x], "commits_xy": n, "C": n / single[x]} for (x, y), n in pair.items() if single[x] >= min_support]
    out = pd.DataFrame(rows, columns=["x", "y", "commits_x", "commits_xy", "C"])
    if len(out):
        out["level_x"] = out.x.map(level)
        out["level_y"] = out.y.map(level)
    return out.sort_values("C", ascending=False)


def couplings(conf: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    return conf[conf.C > threshold].copy()


def layer_violations(conf: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """C(x -> y) > threshold and level(x) < level(y): the low side is driven by the high side."""
    c = couplings(conf, threshold)
    v = c[c.level_x.notna() & c.level_y.notna() & (c.level_x < c.level_y)].copy()
    v["constraint"] = v.apply(lambda r: f"{r.x} must not depend on {r.y}", axis=1)
    return v


def hotspots(df: pd.DataFrame, pareto: float = PARETO, z: float = Z) -> pd.DataFrame:
    """Per file: commits, churn (added + deleted), H = commits x churn, Pareto/outlier flags and the quadrant."""
    g = df.groupby("path").agg(commits=("idx", "nunique"), added=("added", "sum"), deleted=("deleted", "sum"), component=("component", "first")
                               if "component" in df.columns else ("path", "first"))
    g["churn"] = g.added + g.deleted
    g["H"] = g.commits * g.churn
    g = g[g.H > 0].copy()
    if g.empty:
        return g
    cut = g.H.quantile(1 - pareto)
    logH = np.log1p(g.H)
    g["z"] = (logH - logH.mean()) / (logH.std() or 1.0)
    g["pareto_top"] = g.H >= cut
    g["outlier"] = g.z >= z
    mc, mch = g.commits.median(), g.churn.median()
    g["critical"] = g.pareto_top | g.outlier
    hi_c, hi_ch = g.commits > mc, g.churn > mch
    g["quadrant"] = np.select([g.critical, hi_c & ~hi_ch, hi_ch & ~hi_c],
                              ["critical", "quiet debt (many small edits)", "quiet debt (few large edits)"], default="normal")
    return g.sort_values("H", ascending=False)


def stability(df: pd.DataFrame, windows: int = 4) -> dict:
    """How stable is the hotspot set over time: Jaccard of the critical set between consecutive equal-size windows."""
    idxs = sorted(df.idx.unique())
    if len(idxs) < windows * 10:
        return {"windows": 0}
    bounds = np.array_split(np.array(idxs), windows)
    sets = []
    for b in bounds:
        h = hotspots(df[df.idx.isin(set(b))])
        sets.append(set(h.index[h.critical]) if len(h) else set())
    jac = [len(a & b) / len(a | b) if (a | b) else None for a, b in zip(sets[:-1], sets[1:], strict=True)]
    return {"windows": windows, "critical_per_window": [len(s) for s in sets], "jaccard_consecutive": jac}


def summarize(conf, viol, hot):
    by = defaultdict(int)
    for q in hot.quadrant:
        by[q] += 1
    return {"pairs_with_support": int(len(conf)), "couplings_over_threshold": int((conf.C > THRESHOLD).sum()),
            "layer_violation_candidates": int(len(viol)), "files": int(len(hot)), "critical": int(hot.critical.sum()), "quadrants": dict(by)}
