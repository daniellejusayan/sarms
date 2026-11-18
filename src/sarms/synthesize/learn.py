"""Autonomous rule adoption: candidates from the history (sarms.rules.evidence) -> soft rules in
.rms/rules-soft/, in the same schema the Differ reads (source.type = recovered).

    uv run python -m sarms.synthesize.learn data/raw/kedro --learn data/kedro/rms/learn --out data/kedro/rms/rules-soft
    uv run python -m sarms.synthesize.learn data/raw/kedro --learn ... --out ... --judge            # ask the model too

Adoption, with no human step:
  1. the candidate is absent from the code now (by construction in the learner): adopting it never
     rewrites existing code, it only guards the future
  2. stability plus at least one more independent evidence kind (asymmetry, repair, lexical, or the
     judge), and the names do not say the opposite
  3. with --judge: one model call per surviving candidate, given the two components' names, their
     docstrings and the structural evidence; the model answers DECISION or ACCIDENT with one sentence.
     ACCIDENT with the word "certain" vetoes; DECISION adds the evidence kind "judge". The model never
     adds a rule the structure did not propose.
Soft rules carry severity "low": the kernel applies only small reversible fixes under them and records
the rest as left_standing; two reverts under a rule demote it (sarms.kernel, Weeks 4-5).

--strict: DEFINITIVE rules. Only candidates that pass all nine strict gates of sarms.rules.viability are
adopted, with severity "medium" and source id "learned:strict": the Differ treats them like the team's own
rules (a push that crosses one gets a draft fix and the full gate). They are safe to make definitive
without a person because (a) each is absent from the code now, so adopting it rewrites nothing, (b) it
acts only on a future push that adds exactly that import, one import line per fix, in a one-commit
revertible change (G7), and (c) a developer revert of such a fix demotes the rule to "low" (recorded in
the adoption log as `demotions`). Re-run as the repository grows: ids stay stable per pair; rules that
no longer pass are dropped and logged as `no_longer_strict`, new ones as `new`.

    uv run sarms adopt data/raw/kedro --learn data/kedro/rms/learn --out data/kedro/rms/rules-definitive --strict
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from sarms.mining.common import git
from sarms.rules.compile import validate
from sarms.synthesize import llm

JUDGE_SYSTEM = ("You are a software architect reviewing dependency rules inferred from a repository's history. "
                "Answer in the exact format requested, nothing else.")


def docstring(tree: Path, component: str, roots_paths):
    rel = component.replace(".", "/")
    for base in roots_paths:
        for cand in (Path(base) / rel / "__init__.py", Path(base) / f"{rel}.py"):
            if cand.exists():
                m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', cand.read_text(encoding="utf-8", errors="replace"), re.S)
                return (m.group(1).strip().split("\n\n")[0][:400] if m else "")
    return ""


def judge_prompt(c, doc_from, doc_to):
    return (f"Repository components: '{c['from']}' and '{c['to']}'.\n"
            f"Docstring of {c['from']}: {doc_from or '(none)'}\n"
            f"Docstring of {c['to']}: {doc_to or '(none)'}\n\n"
            f"Evidence from the import history: {c['from']} has never imported {c['to']} "
            f"(or did and the import was removed); {c['to']} imports {c['from']} in {int(c['ba_share'] * 100)}% of sampled commits; "
            f"they have coexisted for {int(c['span_days'])} days; evidence kinds: {c['evidence']}.\n\n"
            f"Question: is '{c['from']} must not import {c['to']}' most likely a deliberate architectural decision of this project, "
            f"or an accident of what happened to be needed so far?\n"
            "Reply exactly:\nVERDICT: DECISION or ACCIDENT\nCONFIDENCE: likely or certain\nWHY: <one sentence>")


def previous_rules(out: Path):
    """The last adopted set in out_dir, keyed by pair, so ids stay stable across re-runs."""
    f = out / "soft.yaml"
    if not f.exists():
        return {}
    return {(r["from"], r["to"]): r for r in (yaml.safe_load(f.read_text()) or [])}


def adopt(repo_dir, learn_dir, out_dir, judge=False, model=None, base_url=None, log=print, strict=False, demotions=(), certified=False):
    repo_dir, learn_dir, out = Path(repo_dir).resolve(), Path(learn_dir), Path(out_dir)
    meta = json.loads((learn_dir / "learn.json").read_text())
    cand = pd.read_csv(learn_dir / "candidates.csv")
    sha = git(repo_dir, "rev-parse", meta["last"]).strip()
    label = repo_dir.name
    out.mkdir(parents=True, exist_ok=True)
    before = previous_rules(out)
    demoted = {tuple(d.split(" -> ")) if isinstance(d, str) else tuple(d) for d in demotions}
    gates = {}
    if strict:
        from sarms.rules.viability import STRICT_GATES, score  # noqa: PLC0415
        v = score(learn_dir, "adopted")
        gates = {(r["from"], r["to"]): r for r in v.to_dict("records")} if len(v) else {}
    rules, log_rows = [], []
    used = {int(r["id"].rsplit("-", 1)[1]) for r in before.values() if r["id"].rsplit("-", 1)[1].isdigit()}
    n = max(used, default=0)
    tag = "definitive" if strict else "soft"
    for c in cand.to_dict("records"):
        kinds = c["evidence"].split("+")
        pair = (c["from"], c["to"])
        decision = {"from": c["from"], "to": c["to"], "evidence": c["evidence"], "adopt": bool(c["adopt"])}
        if strict:
            g = gates.get(pair)
            ok = bool(g is not None and g["strict"])
            decision.update({"strict": ok, "failed_gates": g["failed_gates"] if g is not None else "not adopted loosely",
                             "adopt": decision["adopt"] and ok})
            if not ok:
                log_rows.append(decision)
                continue
            decision["gates"] = {k: bool(g[f"s_{k}"]) for k in STRICT_GATES}
        if judge and c["adopt"]:
            from sarms.mining.tree import pythonpath  # noqa: PLC0415
            roots_paths, _ = pythonpath(repo_dir, meta["roots"])
            text = judge_prompt(c, docstring(repo_dir, c["from"], roots_paths), docstring(repo_dir, c["to"], roots_paths))
            try:
                reply, rec = llm.complete(text, JUDGE_SYSTEM, model=model, base_url=base_url)
            except llm.LLMError as e:
                reply, rec = f"VERDICT: ERROR\nWHY: {e}", {}
            verdict = (re.search(r"VERDICT:\s*(\w+)", reply) or [None, "?"])[1].upper()
            conf = (re.search(r"CONFIDENCE:\s*(\w+)", reply) or [None, "?"])[1].lower()
            why = (re.search(r"WHY:\s*(.+)", reply) or [None, ""])[1].strip()
            decision.update({"judge": verdict, "judge_confidence": conf, "judge_why": why, "judge_model": rec.get("model")})
            if verdict == "DECISION":
                kinds.append("judge")
            elif verdict == "ACCIDENT" and conf == "certain":
                decision["adopt"] = False
                decision["vetoed_by"] = "judge"
        if pair in demoted:
            decision.update({"demoted": True, "severity": "low"})
        log_rows.append(decision)
        if not decision["adopt"]:
            continue
        prev = before.get(pair)
        if prev:
            rid = prev["id"]
        else:
            n += 1
            rid = f"R-{label}-{tag}-{n}"
        severity = "low" if (not strict or pair in demoted) else "medium"
        sentence = (f"evidence: {'+'.join(kinds)}; {c['to']} imports {c['from']} in {int(c['ba_share'] * 100)}% of "
                    f"{c['commits_both']} sampled commits; coexisted {int(c['span_days'])} days")
        if strict:
            g = gates[pair]
            sentence += (f"; strict: never crossed, no path today, stability margin {g['instability_to'] - g['instability_from']:.2f}, "
                         f"{g['modules_from']}x{g['modules_to']} modules, {g['shared_neighbours']} shared neighbours")
        rules.append({"id": rid, "kind": "forbidden", "from": c["from"], "to": c["to"],
                      "source": {"type": "recovered", "id": "learned:strict" if strict else f"learned:{c['rule_kind']}", "sentence": sentence},
                      "compiled_at": sha[:12], "severity": severity, "evaluable": True})
    uncertified = []
    if certified:
        # proof-carrying rules: keep only rules that satisfy the six-axiom specification, and write the certificates
        from sarms.rules import certify as C  # noqa: PLC0415
        history = C.load_history(learn_dir)
        # precedence: the team's written rules beat derived ones. A pair a written rule ALLOWS is never adopted;
        # a pair a written rule already FORBIDS is redundant and left to that rule. (The learner never saw these;
        # the enforcement side may.)
        written_allowed, written_forbidden = [], []
        try:
            from sarms.rules.compile import compile_repo  # noqa: PLC0415
            _, wr, _ = compile_repo(repo_dir, sha=sha, out_dir=out / "written-for-precedence", repo_label=label)
            written_allowed = [(w["from"], w["to"]) for w in wr if w["kind"] == "allowed"]
            written_forbidden = [(w["from"], w["to"]) for w in wr if w["kind"] == "forbidden" and w.get("evaluable", True)]
        except SystemExit:
            pass  # no written rules in this repository: nothing to defer to

        def under(x, y):
            return x == y or x.startswith(y + ".")

        kept, certs = [], {}
        for r in rules:
            a, b = r["from"], r["to"]
            allow = [f"{F} -> {T}" for F, T in written_allowed if (under(a, F) and under(b, T)) or (under(F, a) and under(T, b))]
            if allow:
                uncertified.append({"id": r["id"], "from": a, "to": b, "failed": ["written rule allows it: " + allow[0]]})
                continue
            dup = [f"{F} -> {T}" for F, T in written_forbidden if under(a, F) and under(b, T)]
            if dup:
                uncertified.append({"id": r["id"], "from": a, "to": b, "failed": ["already enforced by a written rule: " + dup[0]]})
                continue
            rows_, summary, verdict = C.evaluate(history, r["from"], r["to"])
            certs[r["id"]] = {"from": r["from"], "to": r["to"], "constants": {"DELTA": C.DELTA, "K": C.K, "T_DAYS": C.T_DAYS, "M": C.M, "S": C.S},
                              "axioms": verdict, "summary": summary, "per_commit": rows_}
            if all(verdict.values()):
                kept.append(r)
            else:
                uncertified.append({"id": r["id"], "from": r["from"], "to": r["to"], "failed": [k for k, v in verdict.items() if not v]})
        a5_ok, witness = C.consistency([(r["from"], r["to"]) for r in kept])
        if not a5_ok:  # drop the rules involved in the inconsistency, then re-check
            bad = set(witness.get("cycle_among", [])) | {p.split(" -> ")[0] for p in witness.get("reverse_pairs", [])} | {p.split(" -> ")[1] for p in witness.get("reverse_pairs", [])}
            kept = [r for r in kept if r["from"] not in bad and r["to"] not in bad]
            a5_ok, witness = C.consistency([(r["from"], r["to"]) for r in kept])
        rules = kept
        for r in rules:
            r["source"]["id"] = "learned:certified"
        (out / "certificates.json").write_text(json.dumps({"spec": "sarms rule correctness v1 (A1-A6)", "learn_dir": str(learn_dir), "rules_dir": str(out),
                                                           "A5_consistency": {"holds": a5_ok, **witness},
                                                           "certified": sorted(f"{r['from']} -> {r['to']}" for r in rules),
                                                           "rules": {r["id"]: certs[r["id"]] for r in rules}}, indent=1))
    errors = validate(rules)
    (out / "soft.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    now_pairs = {(r["from"], r["to"]) for r in rules}
    changes = {"new": sorted(f"{a} -> {b}" for a, b in now_pairs - set(before)),
               "kept": sorted(f"{a} -> {b}" for a, b in now_pairs & set(before)),
               "no_longer_strict" if strict else "dropped": sorted(f"{a} -> {b}" for a, b in set(before) - now_pairs),
               "demotions": sorted(f"{a} -> {b}" for a, b in demoted), "uncertified": uncertified}
    # a bundle without contracts, so the Differ knows the root packages and checks these rules itself
    (out / "contracts.json").write_text(json.dumps({"repo": label, "compiled_at": sha, "config_file": None,
                                                    "session": {"root_packages": meta["roots"], "components": meta.get("components", [])},
                                                    "contracts_strict": [], "contracts_asis": []}, indent=1))
    policy = {"status": "definitive" if strict else "soft",
              "worst_case_of_a_wrong_rule": "one future import line re-routed per push, in a one-commit revertible change (G7); "
                                           "nothing existing is rewritten because every adopted pair is absent from the code at adoption",
              "demotion": "a developer revert of a fix made under the rule sets its severity to low (pass --demote 'A -> B' on the next run)",
              "re-evaluation": "re-run adopt after each replay window; ids are stable per pair"}
    (out / "adoption-log.json").write_text(json.dumps({"repo": label, "at": sha, "judge": judge, "strict": strict, "policy": policy,
                                                       "changes": changes, "decisions": log_rows}, indent=1))
    kind = "certified" if certified else "definitive" if strict else "soft"
    log(f"{len(rules)} {kind} rules adopted of {len(cand)} candidates "
        f"({sum(1 for d in log_rows if d.get('vetoed_by'))} vetoed by the judge; "
        f"{len(changes['new'])} new, {len(changes['kept'])} kept, {len(changes.get('no_longer_strict', changes.get('dropped', [])))} dropped) -> {out / 'soft.yaml'}"
        + (f"; SCHEMA ERRORS {errors}" if errors else ""))
    return rules, log_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo_dir")
    ap.add_argument("--learn", required=True, help="the learner's output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--strict", action="store_true", help="definitive rules: all nine strict gates, severity medium")
    ap.add_argument("--demote", action="append", default=[], metavar="'A -> B'", help="pair whose fix a developer reverted: keep it at severity low")
    ap.add_argument("--certified", action="store_true", help="keep only rules that satisfy the six-axiom specification; write certificates.json (implies --strict)")
    a = ap.parse_args()
    adopt(a.repo_dir, a.learn, a.out, a.judge, a.model, a.base_url, log=lambda s: print(s, flush=True), strict=a.strict or a.certified,
          demotions=a.demote, certified=a.certified)


if __name__ == "__main__":
    main()
