"""Viability of a learned rule, judged on its own merits: never against the team's contracts.

Each candidate "A must not import B" is scored from the import history and the current component graph
that the learner (sarms.rules.evidence) already wrote to its output directory. Two levels:

Loose criteria (tiers strong / plausible / weak, used for analysis and for the review sheets):
  clean_chain    A reaches B neither directly nor through any chain of components today (reported only)
  grounded       B has depended on A in >= 50% of the sampled commits where both existed
  sdp            Stable-Dependencies Principle: B is at least as volatile as A (instability I = out/(in+out))
  never_crossed  A -> B appeared in no sampled commit, or once and was removed for good (reported only)
  not_vacuous    both components have >= 2 modules and share a neighbour
  consistent     no rule B -> A in the same set (unless independence siblings); on no cycle of layer rules
  names_agree    the layer vocabulary does not place A above B

Strict gates (every one must hold; this is the adoption rule for DEFINITIVE rules, no person involved):
  direction_80          B -> A in >= 80% of the sampled commits where both existed
  sdp_margin            B at least 0.10 more unstable than A
  clean_chain           no path A -> B today, direct or through other components
  never_crossed_at_all  A -> B in no sampled commit, not even one later repaired
  substantial           both sides >= 3 modules and >= 2 shared neighbours
  three_years           coexisted >= 1095 days
  consistent, names_agree   as above
  selective             A forbids at most 40% of the components it could reach (a component forbidden from
                        nearly everything is a flat-project artefact, not a decision)

Why strict is safe to adopt without a person: a strict rule is absent from the code now, so adopting it
rewrites nothing; it can only act on a future push that adds exactly that import, and then on one
import line at a time, in a commit that reverts in one step (gate G7). The worst an oversight costs is
one re-routed import, and a developer revert demotes the rule (see sarms.synthesize.learn).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from sarms.rules.evidence import layer_rank

STRICT_GATES = ["direction_80", "sdp_margin", "clean_chain", "never_crossed_at_all", "substantial", "three_years",
                "consistent", "names_agree", "selective"]
LOOSE_CRITERIA = ["clean_chain", "grounded", "sdp", "never_crossed", "not_vacuous", "consistent", "names_agree"]


def head_component_graph(learn_dir: Path):
    he = pd.read_parquet(learn_dir / "history_edges.parquet")
    hc = pd.read_parquet(learn_dir / "history_components.parquet")
    last = hc.idx.max()
    g = nx.DiGraph()
    g.add_nodes_from(hc[hc.idx == last].comp)
    for r in he[he.idx == last].itertuples(index=False):
        g.add_edge(r.comp_from, r.comp_to, n=int(r.n))
    sizes = dict(zip(hc[hc.idx == last].comp, hc[hc.idx == last].n_modules, strict=True))
    return g, sizes


def instability(g, c):
    out_, in_ = g.out_degree(c), g.in_degree(c)
    return out_ / (in_ + out_) if (in_ + out_) else 0.5


def score(learn_dir: Path, which: str = "adopted") -> pd.DataFrame:
    """Score the learner's candidates in learn_dir. which = "adopted" (the loose-adopted set) or "all"."""
    learn_dir = Path(learn_dir)
    try:
        cand = pd.read_csv(learn_dir / "candidates.csv")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "adopt" not in cand.columns or not (learn_dir / "history_edges.parquet").exists():
        return pd.DataFrame()
    ev = pd.read_csv(learn_dir / "evidence.csv").set_index(["comp_from", "comp_to"])
    g, sizes = head_component_graph(learn_dir)
    rows_in = cand[cand.adopt].copy() if which == "adopted" else cand.copy()
    if rows_in.empty:
        return pd.DataFrame()
    recs = rows_in.to_dict("records")
    pairs = {(r["from"], r["to"]) for r in recs}
    indep = {(r["from"], r["to"]) for r in recs if r["rule_kind"] == "independence"}
    # layering cycles: an edge A -> B in L means "A is below B" (A must not import B, and B depends on A)
    L = nx.DiGraph()
    for r in recs:
        if r["rule_kind"] == "layer":
            L.add_edge(r["from"], r["to"])
    on_cycle = set()
    for comp in nx.strongly_connected_components(L):
        if len(comp) > 1:
            on_cycle.update(comp)
    rows = []
    for r in recs:
        a, b = r["from"], r["to"]
        e = ev.loc[(a, b)] if (a, b) in ev.index else None
        reach = nx.has_path(g, a, b) if a in g and b in g else False
        sound = not reach
        grounded = bool(e is not None and e.ba_share >= 0.5)
        ia, ib = instability(g, a), instability(g, b)
        sdp = ib >= ia - 0.05
        never = bool(e is not None and (e.ab == 0 or e.repaired))
        na, nb = sizes.get(a, 0), sizes.get(b, 0)
        nbrs_a = set(g.predecessors(a)) | set(g.successors(a)) if a in g else set()
        nbrs_b = set(g.predecessors(b)) | set(g.successors(b)) if b in g else set()
        shared = len((nbrs_a & nbrs_b) - {a, b})
        not_vac = na >= 2 and nb >= 2 and shared >= 1
        reverse = (b, a) in pairs and not ((a, b) in indep and (b, a) in indep)
        consistent = not reverse and not (a in on_cycle and b in on_cycle and L.has_edge(a, b))
        ra, rb = layer_rank(a), layer_rank(b)
        names_ok = not (ra is not None and rb is not None and ra < rb)
        crits = {"clean_chain": sound, "grounded": grounded, "sdp": sdp, "never_crossed": never, "not_vacuous": not_vac,
                 "consistent": consistent, "names_agree": names_ok}
        if grounded and sdp and consistent and not_vac and names_ok:
            tier = "strong"
        elif consistent and names_ok and sum([grounded, sdp, not_vac, sound]) >= 2:
            tier = "plausible"
        else:
            tier = "weak"
        span = float(e.span_days) if e is not None else 0.0
        strict_crits = {"direction_80": bool(e is not None and e.ba_share >= 0.8), "sdp_margin": ib - ia >= 0.10,
                        "clean_chain": sound, "never_crossed_at_all": bool(e is not None and e.ab == 0),
                        "substantial": na >= 3 and nb >= 3 and shared >= 2, "three_years": span >= 1095,
                        "consistent": consistent, "names_agree": names_ok}
        rows.append({"from": a, "to": b, "rule_kind": r["rule_kind"], "evidence": r["evidence"], "adopt": bool(r["adopt"]), **crits,
                     "score": sum(crits.values()), "tier": tier,
                     "ba_share": round(float(e.ba_share), 2) if e is not None else None,
                     "commits_both": int(e.commits_both) if e is not None else 0,
                     "instability_from": round(ia, 2), "instability_to": round(ib, 2), "modules_from": na, "modules_to": nb,
                     "shared_neighbours": shared, "span_days": round(span), **{f"s_{k}": v for k, v in strict_crits.items()},
                     "strict_score": sum(strict_crits.values()), "strict": all(strict_crits.values())})
    df = pd.DataFrame(rows)
    # selectivity, per source component: how many of the components A could reach does it forbid?
    n_comps = max(len(g.nodes) - 1, 1)
    per_src = df.groupby("from").size()
    df["s_selective"] = df["from"].map(lambda c: per_src[c] / n_comps <= 0.40)
    df["strict"] = df["strict"] & df["s_selective"]
    df["strict_score"] = df["strict_score"] + df["s_selective"].astype(int)
    df["failed_gates"] = df.apply(lambda r: "+".join(k for k in STRICT_GATES if not r[f"s_{k}"]), axis=1)
    return df
