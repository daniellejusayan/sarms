"""Proof-carrying rules: a six-axiom specification of when "A must not import B" is correct, a
certificate per rule, and an independent checker.

    uv run sarms certify      --learn data/kedro/rms/learn --rules data/kedro/rms/rules-definitive
    uv run sarms verify-rules --learn data/kedro/rms/learn --rules data/kedro/rms/rules-definitive

The specification (constants are declared, not tuned per repository). Over the sampled history
c_1..c_n with component graphs G_c, for a rule r = (A, B):

  A1 direction     in every commit where A and B coexist, I+(B) - I+(A) >= DELTA, where I+ is the
                   instability I = Ce/(Ca+Ce) recomputed as if the edge A->B were present
                   (Stable-Dependencies Principle, counterfactual form)
  A2 history       A -> B appears in no sampled commit; A and B coexist in >= K commits spanning >= T days
  A3 reachability  in no sampled commit is there any path A ->* B through other components
  A4 substance     at the last commit both have >= M modules and share >= S neighbours
  A5 consistency   over the whole set: never both (A,B) and (B,A); the relation "A below B" is acyclic
  A6 safety        A -> B is absent at the last commit (adopting rewrites nothing); enforcement is one
                   import line per fix, revertible in one commit (gate G7, checked by the kernel)

Correct(r) = A1 & A2 & A3 & A4 & A6;  Correct(R) = all r Correct(r) & A5(R).

`certify` evaluates the axioms from the learner's history tables and writes certificates.json: one
record per rule with the per-commit values, so the proof can be re-checked. `verify` is the
independent checker: it recomputes every axiom from the raw tables with its own code (it does not
import sarms.rules.viability), compares with the certificate, and fails on any rule that does not hold.
A rule is "never wrong" exactly in this sense: it is a consequence of the declared axioms and the
observed history, and a second program can re-derive the proof.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

DELTA = 0.10     # A1 stability margin for adopting a rule
KEEP_DELTA = 0.05  # hysteresis: an already-certified rule is kept while its margin stays above this (never below 0)
K = 8            # A2 minimum coexisting sampled commits
T_DAYS = 1095    # A2 minimum coexistence span
M = 3            # A4 minimum modules per side
S = 2            # A4 minimum shared neighbours
AXIOMS = ("A1_direction", "A2_history", "A3_reachability", "A4_substance", "A5_consistency", "A6_safety")


def load_history(source):
    """History tuple from a learner directory (parquet), a ledger directory (ledger.duckdb) or a Ledger object."""
    if hasattr(source, "history"):
        return source.history()
    source = Path(source)
    if (source / "ledger.duckdb").exists():
        from sarms.ledger.hypergraph import Ledger  # noqa: PLC0415  (rules may read the ledger; never the reverse)
        return Ledger(source).history()
    learn_dir = source
    he = pd.read_parquet(learn_dir / "history_edges.parquet")
    hc = pd.read_parquet(learn_dir / "history_components.parquet")
    idxs = sorted(hc.idx.unique())
    ts = dict(zip(hc.idx, pd.to_datetime(hc.ts, utc=True), strict=False))
    comps = {i: set(g.comp) for i, g in hc.groupby("idx")}
    sizes = {(r.idx, r.comp): int(r.n_modules) for r in hc.itertuples(index=False)}
    edges = {i: {(a, b) for a, b in zip(g.comp_from, g.comp_to, strict=True) if a != b} for i, g in he.groupby("idx")}
    for i in idxs:
        edges.setdefault(i, set())
    return idxs, ts, comps, sizes, edges


def reach(edges, src, dst):
    """Is dst reachable from src over the component edges? Plain BFS (no networkx: the checker is independent)."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
    seen, stack = {src}, [src]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y == dst:
                return True
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def degrees(edges):
    ce, ca = defaultdict(int), defaultdict(int)
    for a, b in edges:
        ce[a] += 1
        ca[b] += 1
    return ce, ca


def instability_plus(edges, ce, ca, A, B):
    """I+(A), I+(B): instability with the edge A->B counted as present."""
    add = 0 if (A, B) in edges else 1
    ceA, caA, ceB, caB = ce[A] + add, ca[A], ce[B], ca[B] + add
    iA = ceA / (caA + ceA) if (caA + ceA) else 1.0
    iB = ceB / (caB + ceB) if (caB + ceB) else 1.0
    return iA, iB


def evaluate(history, A, B, excused=frozenset(), delta=DELTA):
    """Per-commit values and the axiom verdicts for one rule. history = load_history(...).
    excused: {(idx, A, B)} crossings the loop itself handled (an `apply` or `retire` action is on the ledger); those
    commits are left out of the rule's own history, because the crossing never stood. A rule being adopted for the
    first time has no excuses: A2 is 'never crossed at all'."""
    idxs, ts, comps, sizes, edges = history
    rows = []
    for i in idxs:
        if A not in comps[i] or B not in comps[i] or (i, A, B) in excused:
            continue
        E = edges[i]
        ce, ca = degrees(E)
        iA, iB = instability_plus(E, ce, ca, A, B)
        rows.append({"idx": int(i), "ts": ts[i].isoformat(), "imp": (A, B) in E, "path": reach(E, A, B),
                     "iA_plus": round(iA, 4), "iB_plus": round(iB, 4), "margin": round(iB - iA, 4)})
    last = idxs[-1]
    E_last = edges[last]
    nbr = defaultdict(set)
    for a, b in E_last:
        nbr[a].add(b)
        nbr[b].add(a)
    shared = len((nbr[A] & nbr[B]) - {A, B})
    span = ((ts[rows[-1]["idx"]] - ts[rows[0]["idx"]]).total_seconds() / 86400) if rows else 0.0
    verdict = {
        "A1_direction": bool(rows) and all(r["margin"] >= delta for r in rows),
        "A2_history": bool(rows) and not any(r["imp"] for r in rows) and len(rows) >= K and span >= T_DAYS,
        "A3_reachability": bool(rows) and not any(r["path"] for r in rows),
        "A4_substance": sizes.get((last, A), 0) >= M and sizes.get((last, B), 0) >= M and shared >= S,
        "A6_safety": (A, B) not in E_last,
    }
    summary = {"coexist_commits": len(rows), "span_days": round(span, 1), "min_margin": min((r["margin"] for r in rows), default=None),
               "ever_imp": any(r["imp"] for r in rows), "ever_path": any(r["path"] for r in rows),
               "modules_A": sizes.get((last, A), 0), "modules_B": sizes.get((last, B), 0), "shared_neighbours": shared}
    return rows, summary, verdict


def consistency(pairs):
    """A5: no reverse pairs; 'A below B' acyclic. Returns (ok, witness topological order or the offending cycle)."""
    s = set(pairs)
    reverse = sorted(f"{a} -> {b}" for a, b in s if (b, a) in s)
    if reverse:
        return False, {"reverse_pairs": reverse}
    nodes = {x for p in s for x in p}
    indeg = {n: 0 for n in nodes}
    out = defaultdict(set)
    for a, b in s:          # a below b
        out[a].add(b)
        indeg[b] += 1
    order, ready = [], sorted(n for n in nodes if indeg[n] == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(out[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(order) != len(nodes):
        return False, {"cycle_among": sorted(n for n in nodes if indeg[n] > 0)}
    return True, {"topological_order": order}


def load_rules(rules_dir: Path):
    rules = []
    for f in sorted(Path(rules_dir).glob("*.yaml")):
        if f.name == "acknowledged.yaml":
            continue
        for r in yaml.safe_load(f.read_text()) or []:
            if isinstance(r, dict) and r.get("kind") == "forbidden":
                rules.append(r)
    return rules


def certify(learn_dir, rules_dir, out=None, log=print):
    learn_dir, rules_dir = Path(learn_dir), Path(rules_dir)
    history = load_history(learn_dir)
    rules = load_rules(rules_dir)
    certs, ok_pairs = {}, []
    for r in rules:
        rows, summary, verdict = evaluate(history, r["from"], r["to"])
        certs[r["id"]] = {"from": r["from"], "to": r["to"], "constants": {"DELTA": DELTA, "K": K, "T_DAYS": T_DAYS, "M": M, "S": S},
                          "axioms": verdict, "summary": summary, "per_commit": rows}
        if all(verdict.values()):
            ok_pairs.append((r["from"], r["to"]))
    a5_ok, witness = consistency(ok_pairs)
    doc = {"spec": "sarms rule correctness v1 (A1-A6)", "learn_dir": str(learn_dir), "rules_dir": str(rules_dir),
           "A5_consistency": {"holds": a5_ok, **witness}, "certified": sorted(f"{a} -> {b}" for a, b in ok_pairs) if a5_ok else [],
           "rules": certs}
    out = Path(out) if out else rules_dir / "certificates.json"
    out.write_text(json.dumps(doc, indent=1))
    n_ok = len(ok_pairs) if a5_ok else 0
    log(f"{n_ok} of {len(rules)} rules certified under A1-A6 (A5 set consistency: {a5_ok}) -> {out}")
    return doc


def verify(learn_dir, rules_dir, log=print):
    """Independent check: recompute every axiom from the raw tables and compare with the certificate."""
    learn_dir, rules_dir = Path(learn_dir), Path(rules_dir)
    cert_path = rules_dir / "certificates.json"
    if not cert_path.exists():
        raise SystemExit(f"no certificates.json in {rules_dir}; run `sarms certify` first")
    doc = json.loads(cert_path.read_text())
    history = load_history(learn_dir)
    failures, ok_pairs = [], []
    ledger_excused = None
    if (learn_dir / "ledger.duckdb").exists():
        from sarms.ledger.hypergraph import Ledger  # noqa: PLC0415
        ledger_excused = Ledger(learn_dir).excused()
    for rid, c in doc["rules"].items():
        k = c["constants"]
        if {x: k.get(x) for x in ("DELTA", "K", "T_DAYS", "M", "S")} != {"DELTA": DELTA, "K": K, "T_DAYS": T_DAYS, "M": M, "S": S}:
            failures.append((rid, "constants differ from the checker's specification"))
            continue
        delta = k.get("delta", DELTA)
        if delta not in (DELTA, KEEP_DELTA):
            failures.append((rid, f"unknown margin {delta}"))
            continue
        exc = {(i, c["from"], c["to"]) for i in k.get("excused", [])}
        if exc and ledger_excused is not None and not exc <= ledger_excused:
            failures.append((rid, "excused crossings are not on the ledger"))
            continue
        rows, summary, verdict = evaluate(history, c["from"], c["to"], exc, delta)
        if verdict != c["axioms"]:
            failures.append((rid, f"axiom verdicts differ: certificate {c['axioms']} recomputed {verdict}"))
            continue
        if [(r["idx"], r["imp"], r["path"], r["margin"]) for r in rows] != [(r["idx"], r["imp"], r["path"], r["margin"]) for r in c["per_commit"]]:
            failures.append((rid, "per-commit evidence differs from the recomputation"))
            continue
        if all(verdict.values()):
            ok_pairs.append((c["from"], c["to"]))
        else:
            failing = [k for k, v in verdict.items() if not v]
            failures.append((rid, f"axioms not satisfied: {failing}"))
    a5_ok, witness = consistency(ok_pairs)
    if not a5_ok:
        failures.append(("SET", f"A5 consistency fails: {witness}"))
    claimed = set(doc.get("certified", []))
    recomputed = {f"{a} -> {b}" for a, b in ok_pairs} if a5_ok else set()
    if claimed != recomputed:
        failures.append(("SET", f"certified set differs: claimed {sorted(claimed - recomputed)} extra, missing {sorted(recomputed - claimed)}"))
    for rid, why in failures:
        log(f"FAIL {rid}: {why}")
    log(f"{'VERIFIED' if not failures else 'NOT VERIFIED'}: {len(recomputed)} rules hold under A1-A6; {len(failures)} failure(s)")
    return not failures, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["certify", "verify"])
    ap.add_argument("--learn", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode == "certify":
        certify(a.learn, a.rules, a.out)
    else:
        ok, _ = verify(a.learn, a.rules)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()


# ---- the certificate engine as the rule generator (reads a ledger or any history tuple) ------------------------

def under(x, y):
    return x == y or x.startswith(y + ".")


def adopt_from_history(history, label, written_allowed=(), written_forbidden=(), previous=None, compiled_at="0" * 12, excused=frozenset(),
                       hysteresis=True):
    """Every ordered pair of components alive at the last commit is a candidate; a rule is emitted only when A1-A6
    hold, a written rule does not already allow or forbid the pair (precedence), and the set is consistent (A5).
    Ids are stable per pair across runs (previous = {(from, to): rule dict}). Returns (rules, certificates, changes)."""
    idxs, ts, comps, sizes, edges = history
    last = idxs[-1]
    alive = sorted(comps[last])
    previous = previous or {}
    used = {int(r["id"].rsplit("-", 1)[1]) for r in previous.values() if r["id"].rsplit("-", 1)[1].isdigit()}
    n = max(used, default=0)
    kept, certs, rejected = [], {}, []
    for a in alive:
        for b in alive:
            if a == b:
                continue
            allow = [f"{F} -> {T}" for F, T in written_allowed if (under(a, F) and under(b, T)) or (under(F, a) and under(T, b))]
            if allow:
                continue
            dup = [f"{F} -> {T}" for F, T in written_forbidden if under(a, F) and under(b, T)]
            if dup:
                continue
            existing = (a, b) in previous
            used_delta = KEEP_DELTA if (existing and hysteresis) else DELTA
            used_excused = frozenset(x for x in excused if x[1] == a and x[2] == b) if existing else frozenset()
            rows, summary, verdict = evaluate(history, a, b, used_excused, used_delta)
            if not all(verdict.values()):
                if sum(verdict.values()) >= 4:
                    rejected.append({"from": a, "to": b, "failed": [k for k, v in verdict.items() if not v]})
                continue
            kept.append((a, b, rows, summary, verdict, used_delta, sorted(i for i, _, _ in used_excused)))
    ok, witness = consistency([(a, b) for a, b, *_ in kept])
    if not ok:
        bad = set(witness.get("cycle_among", [])) | {p.split(" -> ")[0] for p in witness.get("reverse_pairs", [])} | {p.split(" -> ")[1] for p in witness.get("reverse_pairs", [])}
        kept = [k for k in kept if k[0] not in bad and k[1] not in bad]
        ok, witness = consistency([(a, b) for a, b, *_ in kept])
    rules = []
    for a, b, rows, summary, verdict, used_delta, used_excused in kept:
        prev = previous.get((a, b))
        if prev:
            rid = prev["id"]
        else:
            n += 1
            rid = f"R-{label}-certified-{n}"
        rules.append({"id": rid, "kind": "forbidden", "from": a, "to": b,
                      "source": {"type": "recovered", "id": "learned:certified",
                                 "sentence": f"A1-A6 over {summary['coexist_commits']} sampled commits, {summary['span_days']} days; min margin {summary['min_margin']}; "
                                             f"{summary['modules_A']}x{summary['modules_B']} modules, {summary['shared_neighbours']} shared neighbours"},
                      "compiled_at": compiled_at[:12], "severity": "medium", "evaluable": True})
        certs[rid] = {"from": a, "to": b, "constants": {"DELTA": DELTA, "K": K, "T_DAYS": T_DAYS, "M": M, "S": S, "delta": used_delta, "excused": used_excused},
                      "axioms": verdict, "summary": summary, "per_commit": rows}
    now = {(r["from"], r["to"]) for r in rules}
    changes = {"new": sorted(f"{a} -> {b}" for a, b in now - set(previous)), "kept": sorted(f"{a} -> {b}" for a, b in now & set(previous)),
               "retired": sorted(f"{a} -> {b}" for a, b in set(previous) - now), "near_misses": rejected[:200], "A5": {"holds": ok, **witness}}
    return rules, certs, changes
