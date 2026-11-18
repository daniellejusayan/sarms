"""Proof-carrying rules, simulation-style: seeded random histories -> learner -> certified rules -> independent
verification. Level-1 deterministic simulation (in-memory histories, no git, fixed seed per run)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sarms.rules import certify as C
from sarms.rules import evidence as E
from sarms.rules.viability import score

SEEDS = range(2569, 2569 + 40)


def random_history(seed, n_commits=40, n_layers=4, per_layer=3):
    """A layered system that mostly obeys 'higher imports lower', with noise: random extra edges, one rule crossing
    in some commits, components appearing late, sizes drifting. Returns (history_edges, history_components)."""
    rng = np.random.default_rng(seed)
    comps = [f"app.l{layer}.c{k}" for layer in range(n_layers) for k in range(per_layer)]
    layer_of = {c: int(c.split(".")[1][1]) for c in comps}
    born = {c: (0 if rng.random() < 0.7 else int(rng.integers(1, n_commits // 2))) for c in comps}
    base_edges = set()
    for a in comps:
        for b in comps:
            if layer_of[a] > layer_of[b] and rng.random() < 0.5:   # higher layer imports lower layer
                base_edges.add((a, b))
    crossing = (comps[0], comps[-1]) if rng.random() < 0.5 else None  # a lower->higher import in a few commits
    he_rows, hc_rows = [], []
    t0 = pd.Timestamp("2018-01-01", tz="UTC")
    for i in range(n_commits):
        ts = t0 + pd.Timedelta(days=40 * i)
        alive = [c for c in comps if born[c] <= i]
        for c in alive:
            hc_rows.append({"idx": i, "sha": f"{i:040x}", "ts": ts, "comp": c, "n_modules": 3 + int(rng.integers(0, 4)) + (i // 10)})
        edges = {(a, b) for a, b in base_edges if a in alive and b in alive}
        if rng.random() < 0.3:  # a same-layer or noise edge that may come and go
            a, b = rng.choice(alive, 2, replace=False)
            edges.add((str(a), str(b)))
        if crossing and crossing[0] in alive and crossing[1] in alive and 10 <= i <= 12:
            edges.add(crossing)
        for a, b in edges:
            he_rows.append({"idx": i, "sha": f"{i:040x}", "ts": ts, "comp_from": a, "comp_to": b, "n": 1})
    return pd.DataFrame(he_rows), pd.DataFrame(hc_rows), crossing


def write_learn_dir(tmp, he, hc):
    he.to_parquet(tmp / "history_edges.parquet", index=False)
    hc.to_parquet(tmp / "history_components.parquet", index=False)
    comps = sorted(hc.comp.unique())
    ev = E.evidence(he, hc, comps)
    cand = E.candidates(ev)
    ev.to_csv(tmp / "evidence.csv", index=False)
    cand.to_csv(tmp / "candidates.csv", index=False)
    return cand


def certified_rules(tmp):
    """The generator under test: loose adoption -> strict gates -> axioms A1-A6 -> set consistency."""
    v = score(tmp, "adopted")
    if v.empty:
        return [], {}
    history = C.load_history(tmp)
    rules, certs = [], {}
    for n, r in enumerate(v[v.strict].to_dict("records"), 1):
        rows, summary, verdict = C.evaluate(history, r["from"], r["to"])
        if all(verdict.values()):
            rid = f"R-sim-certified-{n}"
            rules.append({"id": rid, "kind": "forbidden", "from": r["from"], "to": r["to"], "source": {"type": "recovered", "id": "learned:certified"},
                          "compiled_at": "0" * 12, "severity": "medium", "evaluable": True})
            certs[rid] = {"from": r["from"], "to": r["to"], "constants": {"DELTA": C.DELTA, "K": C.K, "T_DAYS": C.T_DAYS, "M": C.M, "S": C.S},
                          "axioms": verdict, "summary": summary, "per_commit": rows}
    ok, witness = C.consistency([(r["from"], r["to"]) for r in rules])
    assert ok, witness
    return rules, certs


def write_rules(tmp, rules, certs):
    import yaml
    rd = tmp / "rules"
    rd.mkdir(exist_ok=True)
    (rd / "soft.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    ok, witness = C.consistency([(r["from"], r["to"]) for r in rules])
    (rd / "certificates.json").write_text(json.dumps({"spec": "sarms rule correctness v1 (A1-A6)", "A5_consistency": {"holds": ok, **witness},
                                                      "certified": sorted(f"{r['from']} -> {r['to']}" for r in rules), "rules": certs}))
    return rd


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_emits_only_rules_the_independent_checker_accepts(tmp_path, seed):
    he, hc, crossing = random_history(seed)
    write_learn_dir(tmp_path, he, hc)
    rules, certs = certified_rules(tmp_path)
    rd = write_rules(tmp_path, rules, certs)
    ok, failures = C.verify(tmp_path, rd, log=lambda s: None)
    assert ok, failures
    pairs = {(r["from"], r["to"]) for r in rules}
    # I9: every certified rule satisfies A1 and A3 in EVERY commit, A2 (never crossed), A6 (absent now)
    history = C.load_history(tmp_path)
    for a, b in pairs:
        rows, _, verdict = C.evaluate(history, a, b)
        assert all(verdict.values())
        assert all(r["margin"] >= C.DELTA and not r["imp"] and not r["path"] for r in rows)
    # the planted crossing can never be certified (A2), nor can any pair present now (A6)
    if crossing:
        assert crossing not in pairs
    last = history[0][-1]
    assert not (pairs & history[4][last])
    # never both directions; the 'below' relation is acyclic (A5)
    assert not any((b, a) in pairs for a, b in pairs)


def test_tampered_certificate_is_rejected(tmp_path):
    he, hc, _ = random_history(2571)   # a seed known to certify rules (25 of the 40 seeds do)
    write_learn_dir(tmp_path, he, hc)
    rules, certs = certified_rules(tmp_path)
    assert rules
    rd = write_rules(tmp_path, rules, certs)
    doc = json.loads((rd / "certificates.json").read_text())
    rid = next(iter(doc["rules"]))
    doc["rules"][rid]["per_commit"][0]["margin"] = 0.99          # forged evidence
    (rd / "certificates.json").write_text(json.dumps(doc))
    ok, failures = C.verify(tmp_path, rd, log=lambda s: None)
    assert not ok and any(rid == f[0] for f in failures)
    # a rule smuggled in without satisfying the axioms is rejected too
    doc = json.loads((rd / "certificates.json").read_text())
    a, b = doc["rules"][rid]["from"], doc["rules"][rid]["to"]
    doc["rules"][rid]["per_commit"][0]["margin"] = C.evaluate(C.load_history(tmp_path), a, b)[0][0]["margin"]
    doc["rules"]["R-sim-forged"] = {"from": b, "to": a, "constants": doc["rules"][rid]["constants"],
                                    "axioms": dict.fromkeys(C.AXIOMS[:4] + ("A6_safety",), True), "summary": {}, "per_commit": []}
    doc["certified"].append(f"{b} -> {a}")
    (rd / "certificates.json").write_text(json.dumps(doc))
    ok, failures = C.verify(tmp_path, rd, log=lambda s: None)
    assert not ok


def test_determinism_same_history_same_rules(tmp_path):
    he, hc, _ = random_history(2600)
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    write_learn_dir(d1, he, hc)
    write_learn_dir(d2, he, hc)
    r1, _ = certified_rules(d1)
    r2, _ = certified_rules(d2)
    assert [(r["from"], r["to"]) for r in r1] == [(r["from"], r["to"]) for r in r2]
    Path(d1 / "rules").mkdir(exist_ok=True)


def test_certified_rule_is_enforced_only_while_a1_still_holds():
    """At violation time the Differ recomputes A1 on the current graph: if the stability margin is gone the
    push is recorded (enforce False, retired_reason) instead of fixed."""
    from sarms.differ.diff import label

    rule = {"id": "R-app-certified-1", "kind": "forbidden", "from": "app.core", "to": "app.web",
            "source": {"type": "recovered", "id": "learned:certified", "sentence": "A1-A6"},
            "compiled_at": "abcdef123456", "severity": "medium", "evaluable": True}
    comps = ["app.core", "app.web", "app.api", "app.cli", "app.db"]
    bundle = {"session": {"root_packages": ["app"], "components": comps}, "contracts_strict": [], "contracts_asis": []}
    # sound: everyone depends on core, core depends on nothing else; web depends on several -> core stable, web unstable
    edges_sound = [["app.core.x", "app.web.v"],                       # the violation
                   ["app.web.v", "app.api.a"], ["app.web.v", "app.db.d"], ["app.web.v", "app.cli.c"],
                   ["app.api.a", "app.core.x"], ["app.cli.c", "app.core.x"], ["app.db.d", "app.core.x"]]
    out = label(bundle, [rule], {"results": [], "results_asis": [], "edges": edges_sound, "details": {}}, "a" * 40, "2026-01-01T00:00:00+00:00")
    v = [r for r in out["rows"] if r["label"] != "CONFIRMED"]
    assert len(v) == 1 and v[0]["enforce"] is True and v[0]["soundness_margin"] >= C.DELTA
    # no longer sound: core now imports everything and nothing imports web except core -> core is the unstable one
    edges_unsound = [["app.core.x", "app.web.v"], ["app.core.x", "app.api.a"], ["app.core.x", "app.db.d"], ["app.core.x", "app.cli.c"],
                     ["app.api.a", "app.web.v"], ["app.db.d", "app.web.v"], ["app.cli.c", "app.web.v"]]
    out = label(bundle, [rule], {"results": [], "results_asis": [], "edges": edges_unsound, "details": {}}, "a" * 40, "2026-01-01T00:00:00+00:00")
    v = [r for r in out["rows"] if r["label"] != "CONFIRMED"]
    assert len(v) == 1 and v[0]["enforce"] is False and "A1 no longer holds" in v[0]["retired_reason"]
    # the draft picker never picks a retired rule
    import pytest as _pt

    from sarms.synthesize.draft import pick
    with _pt.raises(SystemExit):
        pick({"rows": v})
