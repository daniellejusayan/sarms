"""The DST harness: run every scenario with many seeds through the real code, check the invariants after every
step, label each run by risk, and write the risk report.

    uv run sarms dst --seeds 50 --level 1                 # ledger + certificate engine + enforcement decisions, in memory
    uv run sarms dst --seeds 10 --level 2                 # a real git repository per seed, the whole loop incl. gate and kernel

Invariants (docs/spec amendments A-2026-11-07-*):
  I1 push always lands            the developer's commit is never rewritten (level 2: HEAD of main == the developer's sha)
  I2 no applied fix adds a violation   (level 2: every accepted decision record has G1.new_violations == [])
  I3 nothing left standing silently    every crossing of an enforced rule has a ledger action (apply / escalate / retire)
  I4 ledger replayable            building the ledger twice from the same history gives the same state hash
  I5 every fix reverts in one commit   (level 2: kernel's own re-check; a revert restores the tree)
  I6 adoption touches nothing     every certified pair is absent from the graph at adoption
  I7 rule set consistent          no reverse pairs, 'below' acyclic
  I8 re-certification idempotent  same history -> same rules and ids
  I9 certificates sound           every certified rule satisfies A1-A3 in every ledger commit it covers

Risk labels per run:
  CRITICAL  any invariant violated (seed and step recorded; the same seed replays it)
  HIGH      near miss: an enforced rule's A1 margin within 0.05 of DELTA at a crossing; a rule retired and re-adopted
            inside the scenario (flapping); a bulk commit that retired > 30% of the rules; the same rule demoted twice
  MEDIUM    rule churn (new + retired) above 20% of the set during the scenario; any retire
  LOW       nominal
"""
from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sarms.ledger.hypergraph import Ledger
from sarms.rules import certify as C
from sarms.sim.generator import N_BASE, all_scenarios, generate, to_frames

NEAR = 0.05


def _ledger_from(he, hc, st, where):
    led = Ledger(where)
    if len(st):
        led.ingest_stream(st)
    led.ingest_graph(he, hc)
    return led


def _comp_edges_at(he, idx):
    g = he[he.idx == idx]
    return {(a, b) for a, b in zip(g.comp_from, g.comp_to, strict=True)}


def run_level1(seed: int, name: str, scenario: list[str], workdir: Path, hysteresis: bool = True):
    r = generate(seed, scenario)
    he, hc, st = to_frames(r)
    past_idx = N_BASE - 1
    past = (he[he.idx <= past_idx], hc[hc.idx <= past_idx], st[st.idx <= past_idx] if len(st) else st)
    issues, flags, transitions = [], Counter(), Counter()
    d1, d2 = workdir / "l1", workdir / "l2"
    led = _ledger_from(*past, d1)
    led2 = _ledger_from(*past, d2)
    # I4: replay determinism
    if led.state_hash() != led2.state_hash():
        issues.append(("I4", "ledger state differs between two builds from the same history"))
    led2.close()
    shutil.rmtree(d2, ignore_errors=True)
    hist = led.history()
    rules, certs, changes = C.adopt_from_history(hist, "sim")
    rules_b, _, _ = C.adopt_from_history(hist, "sim")
    certs_live = dict(certs)
    # I8 idempotence
    if [(x["id"], x["from"], x["to"]) for x in rules] != [(x["id"], x["from"], x["to"]) for x in rules_b]:
        issues.append(("I8", "re-certification on the same history gave a different set"))
    # I6, I7, I9
    last_edges = hist[4][hist[0][-1]]
    pairs = [(x["from"], x["to"]) for x in rules]
    for a, b in pairs:
        if (a, b) in last_edges:
            issues.append(("I6", f"{a} -> {b} adopted while present"))
    ok, witness = C.consistency(pairs)
    if not ok:
        issues.append(("I7", json.dumps(witness)))
    for a, b in pairs:
        rows, _, verdict = C.evaluate(hist, a, b)
        if not (verdict["A1_direction"] and verdict["A2_history"] and verdict["A3_reachability"]):
            issues.append(("I9", f"{a} -> {b} certified but axioms fail on recomputation"))
    transitions["adopt"] += len(rules)
    # ---- the future: scenario commits, one by one, with re-certification and enforcement decisions
    enforced = {p: x["id"] for p, x in zip(pairs, rules, strict=True)}
    ever_retired, demoted = set(), Counter()
    previous = {(x["from"], x["to"]): x for x in rules}
    min_margin, retired_by_bulk = 9.9, 0
    n0 = len(rules)
    churn_new = churn_ret = 0
    for i in range(past_idx + 1, int(hc.idx.max()) + 1):
        ev = r.history[i]["event"]
        led.ingest_graph(he[he.idx == i], hc[hc.idx == i])
        hist = led.history()
        edges_now = hist[4][i]
        # crossings of enforced rules at this commit
        for (a, b), rid in list(enforced.items()):
            if (a, b) in edges_now:
                ce, ca = C.degrees(edges_now)
                iA, iB = C.instability_plus(edges_now, ce, ca, a, b)
                margin = iB - iA
                min_margin = min(min_margin, margin) if margin >= C.DELTA else min_margin
                if margin >= (C.KEEP_DELTA if hysteresis else C.DELTA):
                    transitions["enforce"] += 1
                    led.record_action(f"{i:040x}", "apply", rid, {"sim": "scripted fix", "event": ev, "from": a, "to": b})
                    if ev == "revert_fix":
                        demoted[rid] += 1
                        transitions["demote"] += 1
                        led.record_action(f"{i:040x}", "demote", rid, {})
                else:
                    transitions["retired_at_violation"] += 1
                    led.record_action(f"{i:040x}", "retire", rid, {"reason": f"A1 margin {margin:.2f} < {C.DELTA}", "from": a, "to": b})
                    enforced.pop((a, b), None)
                    ever_retired.add((a, b))
        # I3: every crossing of an enforced rule above has an action (by construction here; check the ledger agrees)
        # re-certify on the grown ledger
        rules, certs, changes = C.adopt_from_history(hist, "sim", previous=previous, excused=led.excused(), hysteresis=hysteresis)
        certs_live = dict(certs)
        new_pairs = {(x["from"], x["to"]) for x in rules}
        retired_now = set(previous) - new_pairs
        churn_new += len(new_pairs - set(previous))
        churn_ret += len(retired_now)
        if previous and len(retired_now) / len(previous) > 0.30:
            retired_by_bulk += 1
            flags[f"cascade:{ev}"] += 1
        for p in retired_now:
            ever_retired.add(p)
            enforced.pop(p, None)
            transitions["retire"] += 1
        for p in new_pairs - set(previous):
            if p in ever_retired:
                flags["flapping"] += 1
            enforced[p] = next(x["id"] for x in rules if (x["from"], x["to"]) == p)
            transitions["adopt"] += 1
        previous = {(x["from"], x["to"]): x for x in rules}
        # I9 on the live set, under each certificate's own recorded margin and excused crossings
        for x in rules:
            k = certs_live[x["id"]]["constants"]
            exc = {(j, x["from"], x["to"]) for j in k.get("excused", [])}
            rows, _, verdict = C.evaluate(hist, x["from"], x["to"], exc, k.get("delta", C.DELTA))
            if not (verdict["A1_direction"] and verdict["A2_history"] and verdict["A3_reachability"]):
                issues.append(("I9", f"{x['from']} -> {x['to']} certified at step {i} but axioms fail"))
                break
    acts = led.actions()
    crossings_with_action = int((acts.kind.isin(["apply", "retire"])).sum()) if len(acts) else 0
    if transitions["enforce"] + transitions["retired_at_violation"] != crossings_with_action:
        issues.append(("I3", "a crossing of an enforced rule left no ledger action"))
    led.close()
    # ---- label
    label = "LOW"
    reasons = []
    if issues:
        label = "CRITICAL"
        reasons = [f"{k}: {v}" for k, v in issues[:3]]
    else:
        if min_margin < C.DELTA + NEAR:
            reasons.append(f"near miss: A1 margin {min_margin:.2f} at a crossing")
        if flags["flapping"]:
            reasons.append(f"flapping: {flags['flapping']} rule(s) retired then re-adopted")
        if retired_by_bulk:
            reasons.append("cascade: one commit retired > 30% of the rules (" + ", ".join(k.split(":")[1] for k in flags if k.startswith("cascade:")) + ")")
        if any(v >= 2 for v in demoted.values()):
            reasons.append("the same rule demoted twice")
        if reasons:
            label = "HIGH"
        elif n0 and (churn_new + churn_ret) / n0 > 0.20 or transitions["retire"]:
            label = "MEDIUM"
            reasons.append(f"rule churn {churn_new} new / {churn_ret} retired of {n0}")
    return {"level": 1, "scenario": name, "seed": seed, "hysteresis": hysteresis, "label": label, "reasons": reasons, "rules_at_start": n0,
            "transitions": dict(transitions), "issues": issues, "events": sorted(set(scenario))}


def run_level2(seed: int, name: str, scenario: list[str], workdir: Path, model_behaviour: str = "correct"):
    """A real git repository per seed: the generated life is materialised as Python packages and commits, then the
    scenario commits are pushed one by one and `product.run` (Differ -> draft with a scripted reply -> gate -> kernel)
    runs after each. Checks I1, I2, I3, I5 against git and the decision records."""
    import subprocess

    from sarms import product
    from sarms.sim.materialize import commit_step, make_repo, scripted_reply

    r = generate(seed, scenario, n_base=30)
    repo = make_repo(r, workdir / "repo", n_base=30)
    out = workdir / "rms"
    issues, transitions = [], Counter()
    replies = workdir / "replies"
    replies.mkdir(exist_ok=True)
    for i in range(30, len(r.history)):
        rec = r.history[i]
        dev_sha = commit_step(repo, r, rec, i)  # the developer's push
        for f in replies.glob("*.txt"):
            f.unlink()
        (replies / "1.txt").write_text(scripted_reply(repo, rec, model_behaviour))
        try:
            rep = product.run(repo, out, base_url=f"fake:{replies}", tests="python -m pytest -q -p no:cacheprovider", every=1, onto="main", log=lambda s: None)
        except Exception as e:  # noqa: BLE001
            issues.append(("LOOP", f"{type(e).__name__}: {str(e)[:120]}"))
            break
        # I1: the developer's commit is still on main, unchanged (the system only adds follow-up commits after it)
        anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", dev_sha, "HEAD"], capture_output=True)
        tree_same = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{dev_sha}^{{commit}}"], capture_output=True).returncode == 0
        if anc.returncode != 0 or not tree_same:
            issues.append(("I1", "the developer's commit is no longer on the branch"))
        transitions["pushes"] += 1
        for o in rep["outcomes"]:
            transitions[o["outcome"]] += 1
            if o["outcome"] == "applied":
                rec_ = json.loads(Path(o["decision"]).read_text())
                g1 = next(x for x in rec_["gate"] if x["line"] == "G1")
                if g1["evidence"]["new_violations"]:
                    issues.append(("I2", "an applied fix added a violation"))
                g7 = next(x for x in rec_["gate"] if x["line"] == "G7")
                if not g7["pass"]:
                    issues.append(("I5", "applied without a clean revert dry run"))
        if rep["enforceable"] and not rep["outcomes"]:
            issues.append(("I3", "an enforceable violation produced no outcome"))
    label = "CRITICAL" if issues else ("MEDIUM" if transitions.get("left_standing") else "LOW")
    return {"level": 2, "scenario": name, "seed": seed, "model": model_behaviour, "label": label, "reasons": [f"{k}: {v}" for k, v in issues[:3]],
            "transitions": dict(transitions), "issues": issues, "events": sorted(set(scenario))}


def main(seeds=50, level=1, out="reports/dst-risk.md", scenario=None, log=print):
    scen = all_scenarios()
    if scenario == "named":
        from sarms.sim.generator import NAMED  # noqa: PLC0415
        scen = dict(NAMED)
    elif scenario:
        scen = {k: v for k, v in scen.items() if k == scenario or k.startswith(scenario)}
    runs = []
    base = Path(tempfile.mkdtemp(prefix="sarms-dst-"))
    try:
        for name, seq in scen.items():
            for k in range(seeds):
                seed = 2569 + k
                wd = base / f"{name.replace(':', '_').replace('+', '_')}-{seed}"
                wd.mkdir(parents=True, exist_ok=True)
                if level == 1:
                    runs.append(run_level1(seed, name, seq, wd))
                else:
                    for mb in ("correct", "fake_module", "adds_violation", "noop"):
                        runs.append(run_level2(seed, name, seq, wd / mb, mb))
                shutil.rmtree(wd, ignore_errors=True)
            lab = Counter(x["label"] for x in runs if x["scenario"] == name)
            log(f"{name:32s} {dict(lab)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    report(runs, out, level, seeds)
    return runs


def report(runs, out, level, seeds):
    by = defaultdict(Counter)
    for x in runs:
        by[x["scenario"]][x["label"]] += 1
    cov_events, cov_trans = Counter(), Counter()
    for x in runs:
        cov_events.update(x["events"])
        cov_trans.update(x["transitions"].keys())
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    L = [f"# DST risk report (level {level}, {seeds} seeds per scenario, {len(runs)} runs)", "",
         "| scenario | CRITICAL | HIGH | MEDIUM | LOW | worst |", "| --- | --- | --- | --- | --- | --- |"]
    for name, c in sorted(by.items(), key=lambda kv: (-kv[1]["CRITICAL"], -kv[1]["HIGH"], kv[0])):
        worst = next(l_ for l_ in order if c[l_])
        L.append(f"| {name} | {c['CRITICAL']} | {c['HIGH']} | {c['MEDIUM']} | {c['LOW']} | {worst} |")
    tot = Counter(x["label"] for x in runs)
    L += ["", f"Totals: {dict(tot)}", "", "## CRITICAL and HIGH runs (seed replays the run)", ""]
    for x in runs:
        if x["label"] in ("CRITICAL", "HIGH"):
            L.append(f"- {x['label']} {x['scenario']} seed {x['seed']}" + (f" model {x['model']}" if "model" in x else "") + ": " + "; ".join(x["reasons"]))
    L += ["", "## Coverage", "", f"events exercised: {sorted(cov_events)}", f"loop transitions exercised: {sorted(cov_trans)}",
          "", "Labels: CRITICAL = an invariant I1-I9 was violated; HIGH = near miss (A1 margin within 0.05 of the threshold at a crossing, "
          "flapping, a bulk commit retiring > 30% of rules, double demotion); MEDIUM = rule churn or a retire; LOW = nominal."]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n")
    Path(out).with_suffix(".json").write_text(json.dumps(runs, indent=1, default=str))
    print("\n".join(L[:40]))
