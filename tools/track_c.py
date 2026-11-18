"""Track C end to end, as it would run in production: learn on the first 70% of a repository's life, then
replay the last 30% commit window by commit window with belief-function enforcement and feedback.

    uv run python tools/track_c.py kedro [sqlfluff ...] [--days 90] [--split 0.7]   # -> data/REPO/rms/track-c/, reports/track-c.md

Per repository:
  1. git log stream, noise removed (sarms.mining.gitlog), mapped to the learner's components
  2. co-change confidence C(x->y), couplings > 0.35, layer-violation candidates, file hotspots (sarms.rules.cochange)
  3. hypergraph encoder trained on the past windows; learned architecture matrix; coupling rules at 0.98; drift events;
     forward accuracy against a plain count baseline (sarms.rules.hgnn)
  4. candidate constraints = co-change layer candidates + hgnn layer candidates + structurally certified rules (A1-A6 on the past)
  5. belief functions combine every source per candidate -> enforce / monitor / drop (sarms.rules.belief)
  6. production replay over the future windows: crossings, repairs, feedback updates, coupling anomalies, hotspot persistence
  7. afterwards only: grade the final enforced set against the team's written rules (written share, contradicted)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from sarms.mining.gitlog import stream, to_frame
from sarms.rules import belief as B
from sarms.rules import certify as C
from sarms.rules import cochange as CC
from sarms.rules import hgnn as G
from sarms.rules.evidence import comp_of


def under(x, y):
    return x == y or x.startswith(y + ".")


def written_rules(learn_dir: Path):
    hard, allowed = [], set()
    for f in (learn_dir / "hard-rules-for-grading").glob("*.yaml"):
        if f.name == "acknowledged.yaml":
            continue
        for r in yaml.safe_load(f.read_text()) or []:
            if not isinstance(r, dict):
                continue
            if r["kind"] == "forbidden" and r.get("evaluable", True) and "*" not in r["from"] + r["to"]:
                hard.append((r["from"], r["to"]))
            elif r["kind"] == "allowed":
                allowed.add((r["from"], r["to"]))
    return hard, allowed


def grade(a, b, hard, allowed):
    for F, T in hard:
        if (under(a, F) and under(b, T)) or (under(F, a) and under(T, b)):
            return "written"
    for F, T in allowed:
        if (under(a, F) and under(b, T)) or (under(F, a) and under(T, b)):
            return "contradicted"
    return "unknown"


def run(repo: str, days: int, split: float, epochs: int, log=print, since=None):
    raw, learn = Path("data/raw") / repo, Path("data") / repo / "rms" / "learn"
    out = Path("data") / repo / "rms" / "track-c"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((learn / "learn.json").read_text())
    comps = sorted(meta["components"], key=len, reverse=True)
    log(f"## {repo}: streaming git log")
    df = to_frame(stream(raw, max_files=50, since=since), roots=meta["roots"], component_of=lambda m: comp_of(m, comps))
    df = df.dropna(subset=["component"]) if "component" in df.columns else df
    if df.empty or df.idx.nunique() < 60:
        return {"repo": repo, "status": "too little history"}
    t_cut = df.ts.min() + (df.ts.max() - df.ts.min()) * split
    past, future = df[df.ts < t_cut], df[df.ts >= t_cut]
    nodes = sorted(df.component.unique())

    # ---- part 1: co-change and hotspots on the past
    conf = CC.confidence(past)
    CC.couplings(conf).to_csv(out / "couplings.csv", index=False)
    viol = CC.layer_violations(conf)
    hot = CC.hotspots(past)
    hot_future = CC.hotspots(future)
    crit_past = set(hot.index[hot.critical]) if len(hot) else set()
    crit_future = set(hot_future.index[hot_future.critical]) if len(hot_future) else set()
    hotspot_persistence = len(crit_past & crit_future) / len(crit_past) if crit_past else None
    conf.to_csv(out / "cochange_confidence.csv", index=False)
    hot.to_csv(out / "hotspots.csv")

    # ---- part 2: hypergraph encoder
    try:
        fit = G.train(df, nodes, days=days, epochs=epochs, train_until=t_cut, log=log)
        coupling_rules, hgnn_layer = G.rules_from_matrix(fit)
        forward = G.evaluate_forward(fit)
        forward_loose = G.evaluate_forward(fit, confidence=0.8)
        G.save(fit, out, coupling_rules, hgnn_layer, forward)
        drift_future = [d for d in fit["drift_events"] if pd.Timestamp(d["start"], tz="UTC") >= t_cut]
    except Exception as e:  # noqa: BLE001
        log(f"   hgnn skipped: {type(e).__name__}: {e}")
        fit, coupling_rules, hgnn_layer, forward, forward_loose, drift_future = None, [], [], {"error": str(e)[:100]}, {}, []

    # ---- part 4: candidates and their evidence
    history = C.load_history(learn)
    idxs, ts, hcomps, sizes, hedges = history
    past_idxs = [i for i in idxs if ts[i] < t_cut]
    future_idxs = [i for i in idxs if ts[i] >= t_cut]
    if len(past_idxs) < 5 or not future_idxs:
        return {"repo": repo, "status": "no sampled history on one side of the split"}
    past_hist = (past_idxs, ts, {i: hcomps[i] for i in past_idxs}, {k: v for k, v in sizes.items() if k[0] in set(past_idxs)}, {i: hedges[i] for i in past_idxs})
    last_past = past_idxs[-1]
    cands = {}
    for r in viol.to_dict("records"):
        cands.setdefault((r["x"], r["y"]), set()).add("cochange_layer")
    for r in hgnn_layer:
        cands.setdefault((r["x"], r["y"]), set()).add("hgnn_predicted")
    # structural: certify every pair alive at the cut (what adopt --certified would do on the past)
    alive = sorted(hcomps[last_past])
    for a in alive:
        for b in alive:
            if a != b:
                _, _, verdict = C.evaluate(past_hist, a, b)
                if all(verdict.values()):
                    cands.setdefault((a, b), set()).add("certificate_all_axioms")
                elif (a, b) in cands and sum(verdict.values()) >= 3:
                    cands[(a, b)].add("certificate_partial")
    for (a, b), src in cands.items():
        if (a, b) in hedges[last_past]:
            src.add("imports_now")
    hard, allowed = written_rules(learn)
    assessed = {}
    for (a, b), src in cands.items():
        assessed[(a, b)] = B.assess(sorted(src))
    decisions = Counter(s["decision"] for s in assessed.values())

    # ---- part 6: production replay over the future, window by window (sampled commits as the clock)
    timeline, outcomes = [], Counter()
    enforced = {p for p, s in assessed.items() if s["decision"] == "enforce"}
    state = dict(assessed)
    for i in future_idxs:
        events = []
        for p in list(enforced):
            a, b = p
            if a not in hcomps[i] or b not in hcomps[i]:
                continue
            crossed = p in hedges[i]
            if crossed:
                kept = p in hedges[idxs[-1]]
                src = "outcome_crossed_kept" if kept else "outcome_crossed_repaired"
                outcomes[src] += 1
            else:
                src = "outcome_never_crossed"
            new = B.feedback(state[p], src)
            if new["decision"] != state[p]["decision"]:
                events.append({"rule": f"{a} must not depend on {b}", "from": state[p]["decision"], "to": new["decision"], "because": src, "bel": new["bel"]})
                if new["decision"] != "enforce":
                    enforced.discard(p)
            state[p] = new
        if events:
            timeline.append({"commit_idx": int(i), "date": str(ts[i].date()), "events": events})
    # coupling anomalies: future commits that changed x without y for a 0.98 coupling rule
    anomalies = 0
    for r in coupling_rules:
        for _, g in future.groupby("idx"):
            s = set(g.component)
            if r["x"] in s and r["y"] not in s:
                anomalies += 1
    # ---- part 7: grade, afterwards only
    final_enforced = {p for p, s in state.items() if s["decision"] == "enforce"}
    sets = {"cochange_layer_candidates": {(r["x"], r["y"]) for r in viol.to_dict("records")},
            "hgnn_layer_candidates": {(r["x"], r["y"]) for r in hgnn_layer},
            "certified_on_past": {p for p, s in cands.items() if "certificate_all_axioms" in s},
            "ds_enforced_initial": {p for p, s in assessed.items() if s["decision"] == "enforce"},
            "ds_enforced_final": final_enforced}
    grades = {k: Counter(grade(a, b, hard, allowed) for a, b in v) for k, v in sets.items()}
    # did the future cross them?
    crossings = {}
    for k, v in sets.items():
        c_kept = c_rep = 0
        for a, b in v:
            fut = [(a, b) in hedges[i] for i in future_idxs if a in hcomps[i] and b in hcomps[i]]
            if any(fut):
                if (a, b) in hedges[idxs[-1]]:
                    c_kept += 1
                else:
                    c_rep += 1
        crossings[k] = {"n": len(v), "crossed_kept": c_kept, "crossed_repaired": c_rep}
    res = {"repo": repo, "status": "ok", "commits_after_noise_filter": int(df.idx.nunique()), "files": int(df.path.nunique()), "components": len(nodes),
           "split_date": str(t_cut.date()), "past_commits": int(past.idx.nunique()), "future_commits": int(future.idx.nunique()),
           "cochange": CC.summarize(conf, viol, hot), "hotspot_persistence": hotspot_persistence,
           "hgnn": {"windows": len(fit["windows"]) if fit else 0, "train_windows": fit["n_train_windows"] if fit else 0,
                    "coupling_rules_098": len(coupling_rules), "layer_candidates": len(hgnn_layer), "drift_events_future": len(drift_future),
                    "forward_098": forward, "forward_080": forward_loose},
           "candidates": len(cands), "decisions_initial": dict(decisions), "decisions_final": dict(Counter(s["decision"] for s in state.values())),
           "feedback_outcomes": dict(outcomes), "decision_changes": sum(len(t["events"]) for t in timeline), "coupling_anomalies_future": anomalies,
           "grades": {k: dict(v) for k, v in grades.items()}, "future_crossings": crossings,
           "enforced_rules": sorted(f"{a} -> {b}" for a, b in final_enforced)}
    (out / "track-c.json").write_text(json.dumps({**res, "assessed": {f"{a} -> {b}": s for (a, b), s in state.items()}, "timeline": timeline}, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--redo", action="store_true", help="recompute repositories that already have a track-c.json")
    ap.add_argument("--since", help="git --since for the stream (a partial clone fetches file contents lazily; e.g. '4 years ago')")
    a = ap.parse_args()
    results = []
    for r in a.repos:
        prev = Path("data") / r / "rms" / "track-c" / "track-c.json"
        if prev.exists() and not a.redo:
            d = json.loads(prev.read_text())
            results.append({k: v for k, v in d.items() if k not in ("assessed", "timeline")})
            print(f"{r}: reusing {prev}", flush=True)
            continue
        try:
            results.append(run(r, a.days, a.split, a.epochs, since=a.since))
        except Exception as e:  # noqa: BLE001
            results.append({"repo": r, "status": f"error {type(e).__name__}: {str(e)[:120]}"})
        print(json.dumps(results[-1], indent=1, default=str)[:1500], flush=True)
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "track-c.json").write_text(json.dumps(results, indent=1, default=str))
    ok = [r for r in results if r.get("status") == "ok"]
    L = ["# Track C in production replay (learn on the first 70% of each repository, enforce on the last 30%)", "",
         "| repo | commits (filtered) | split | co-change couplings > .35 | layer candidates | critical hotspots | hotspot persistence | hgnn 0.98 rules | hgnn vs count MAE | drift events (future) | DS enforced initial → final | future crossings of enforced (kept / repaired) | written / contradicted among final |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in ok:
        f = r["hgnn"]["forward_098"]
        mae = f"{f['hgnn']['mae_all_pairs']} / {f['count']['mae_all_pairs']}" if "hgnn" in f else "–"
        cr = r["future_crossings"]["ds_enforced_final"]
        g = r["grades"]["ds_enforced_final"]
        L.append(f"| {r['repo']} | {r['commits_after_noise_filter']} | {r['split_date']} | {r['cochange']['couplings_over_threshold']} | {r['cochange']['layer_violation_candidates']} | "
                 f"{r['cochange']['critical']} | {r['hotspot_persistence'] if r['hotspot_persistence'] is None else round(r['hotspot_persistence'], 2)} | {r['hgnn']['coupling_rules_098']} | {mae} | {r['hgnn']['drift_events_future']} | "
                 f"{r['decisions_initial'].get('enforce', 0)} → {r['decisions_final'].get('enforce', 0)} | {cr['crossed_kept']} / {cr['crossed_repaired']} of {cr['n']} | {g.get('written', 0)} / {g.get('contradicted', 0)} of {sum(g.values())} |")
    tot = Counter()
    for r in ok:
        for k in ("cochange_layer_candidates", "hgnn_layer_candidates", "certified_on_past", "ds_enforced_final"):
            g, c = r["grades"][k], r["future_crossings"][k]
            tot.update({f"{k}_n": c["n"], f"{k}_written": g.get("written", 0), f"{k}_contradicted": g.get("contradicted", 0), f"{k}_kept": c["crossed_kept"], f"{k}_rep": c["crossed_repaired"]})
    L += ["", "Pooled, by source of the candidate rule:", "", "| source | rules | written | contradicted | crossed & kept later (wrong) | crossed & repaired later |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("cochange_layer_candidates", "hgnn_layer_candidates", "certified_on_past", "ds_enforced_final"):
        n = tot[f"{k}_n"] or 1
        L.append(f"| {k} | {tot[f'{k}_n']} | {tot[f'{k}_written']} ({100 * tot[f'{k}_written'] / n:.0f}%) | {tot[f'{k}_contradicted']} | {tot[f'{k}_kept']} | {tot[f'{k}_rep']} |")
    L += ["", "Skipped: " + ", ".join(f"{r['repo']} ({r['status']})" for r in results if r.get("status") != "ok")]
    (Path("reports") / "track-c.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
