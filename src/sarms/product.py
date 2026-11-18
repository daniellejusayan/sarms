"""sarms as a product: one command runs the whole workflow on a repository.

    uv run sarms mine    REPO [--out DIR] [--every 25]          mining -> the spatio-temporal hypergraph ledger
    uv run sarms certify-ledger REPO [--out DIR]                 the certificate engine reads the ledger -> verified rules
    uv run sarms run     REPO [--out DIR] [--model ...] [--tests "..."] [--onto main] [--dry-run]
                                                                 mine (if needed) -> certify -> Differ on HEAD -> draft -> gate -> kernel -> ledger
    uv run sarms status  [--out DIR]                             one screen from the ledger
    uv run sarms dst     [--seeds 200] [--level 1|2] [--out reports/dst-risk.md]   deterministic simulation: every scenario, risk labels

Workflow (the product): mining -> STHG ledger -> architecture certificates -> loop (synthesizer, gate, kernel) -> DST verification.
Humans appear nowhere in it; they read the ledger afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import time
from pathlib import Path

import yaml

from sarms.differ.diff import diff_commit
from sarms.kernel.apply import apply_decision, human_fixed
from sarms.ledger.hypergraph import Ledger
from sarms.mining.common import git, resolve_repo_input, sarms_data_paths
from sarms.mining.gitlog import stream, to_frame
from sarms.rules import certify as C
from sarms.rules.compile import compile_repo, load_rules
from sarms.rules.evidence import comp_of, scan_history
from sarms.synthesize.draft import draft_one
from sarms.verify.gate import run_gate


def _terminal_table(title, rows, log=print):
    log("")
    log(f"=== {title} ===")
    width = max(len(str(label)) for label, _ in rows)
    for label, value in rows:
        log(f"{str(label):<{width}} : {value}")


def rms_dir(repo, out=None):
    return Path(out) if out else Path("data") / Path(repo).name / "rms"


def mine(repo, out=None, every=25, since=None, until=None, log=print):
    """Mining -> ledger. Import graphs at every `every`-th commit, the change stream for every commit."""
    repo = resolve_repo_input(repo)
    out = rms_dir(repo, out)
    sthg = out / "sthg"
    t0 = time.time()
    log(f"[mine] {repo.name}: import graphs every {every}th commit")
    he, hc, comps, roots, sample, errors = scan_history(repo, every=every, until=until, log=lambda s: log("   " + s))
    longest = sorted(comps, key=len, reverse=True)
    log("[mine] change stream")
    df = to_frame(stream(repo, since=since, max_files=None), roots=roots, component_of=lambda m: comp_of(m, longest))
    led = Ledger(sthg)
    n_h = led.ingest_stream(df, source_of=lambda a: "sensolens-sarms" if a == "sensolens-sarms" else "human") if len(df) else 0
    n_g = led.ingest_graph(he, hc)
    meta = {"repo": str(repo), "roots": roots, "components": comps, "every": every, "sampled": len(sample), "unreadable": len(errors),
            "last": sample[-1][1] if sample else None, "seconds": round(time.time() - t0, 1)}
    (sthg / "meta.json").write_text(json.dumps(meta, indent=1))
    led.record_action(meta["last"] or "", "mine", None, {"hyperedges": n_h, "snapshots": n_g})
    led.export_parquet()
    summary = led.summary()
    _terminal_table("MINING COMPLETE", [
        ("Repository", repo.name),
        ("Ledger", str(sthg)),
        ("Commits", summary["hyperedges"]),
        ("Files", summary["files"]),
        ("Components", summary["components"]),
        ("Graph snapshots", summary["graph_snapshots"]),
        ("Graph edges now", summary["graph_edges_now"]),
        ("Elapsed", f"{meta['seconds']}s"),
    ], log)
    return led, meta


def certify_ledger(repo, out=None, log=print):
    """The certificate engine reads the ledger and writes verified rules + certificates; written rules take precedence."""
    repo = resolve_repo_input(repo)
    out = rms_dir(repo, out)
    sthg = out / "sthg"
    if not (sthg / "ledger.duckdb").exists():
        raise SystemExit(f"no ledger at {sthg}; run `sarms mine` first")
    led = Ledger(sthg)
    meta = json.loads((sthg / "meta.json").read_text())
    history = led.history()
    written_allowed, written_forbidden = [], []
    try:
        _, wr, _ = compile_repo(repo, sha=meta["last"], out_dir=out / "rules", repo_label=repo.name)
        written_allowed = [(w["from"], w["to"]) for w in wr if w["kind"] == "allowed"]
        written_forbidden = [(w["from"], w["to"]) for w in wr if w["kind"] == "forbidden" and w.get("evaluable", True)]
    except SystemExit:
        log("[certify] no written rules in this repository; the certified rules stand alone")
    prev_file = out / "rules-certified" / "soft.yaml"
    previous = {(r["from"], r["to"]): r for r in (yaml.safe_load(prev_file.read_text()) or [])} if prev_file.exists() else {}
    rules, certs, changes = C.adopt_from_history(history, repo.name, written_allowed, written_forbidden, previous, compiled_at=meta["last"] or "0" * 12,
                                                 excused=led.excused())
    rc = out / "rules-certified"
    rc.mkdir(parents=True, exist_ok=True)
    (rc / "soft.yaml").write_text(yaml.safe_dump(rules, sort_keys=False))
    doc = {"spec": "sarms rule correctness v1 (A1-A6)", "ledger": str(sthg), "certified": sorted(f"{r['from']} -> {r['to']}" for r in rules),
           "A5_consistency": changes["A5"], "rules": certs}
    (rc / "certificates.json").write_text(json.dumps(doc, indent=1, default=str))
    (rc / "contracts.json").write_text(json.dumps({"repo": repo.name, "compiled_at": meta["last"], "config_file": None,
                                                   "session": {"root_packages": meta["roots"], "components": meta["components"]},
                                                   "contracts_strict": [], "contracts_asis": []}, indent=1))
    (rc / "adoption-log.json").write_text(json.dumps({"changes": {k: v for k, v in changes.items() if k != "near_misses"}, "near_misses": changes["near_misses"]}, indent=1, default=str))
    checker_hash = hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()[:16]
    led.record_certificates(doc, meta["last"] or "", checker_hash)
    for kind in ("new", "kept", "retired"):
        for pair in changes[kind]:
            led.record_action(meta["last"] or "", {"new": "adopt", "kept": "keep", "retired": "retire"}[kind], pair, {})
    ok, failures = C.verify(sthg, rc, log=lambda s: None)
    led.record_action(meta["last"] or "", "verify", None, {"verified": ok, "failures": len(failures)})
    _terminal_table("RULE CERTIFICATION", [
        ("Verified rules", len(rules)),
        ("New", len(changes["new"])),
        ("Kept", len(changes["kept"])),
        ("Retired", len(changes["retired"])),
        ("Independent check", "VERIFIED" if ok else "NOT VERIFIED"),
        ("Output", str(rc)),
    ], log)
    # merged rule folder the Differ reads: the team's written rules + the certified ones
    merged = out / "rules-active"
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    src_rules = out / "rules"
    if (src_rules / "contracts.json").exists():
        for f in src_rules.glob("*"):
            shutil.copy(f, merged / f.name)
        bundle = json.loads((merged / "contracts.json").read_text())
        bundle["session"]["components"] = meta["components"]          # the Differ re-checks A1 on certified rules with these
        (merged / "contracts.json").write_text(json.dumps(bundle, indent=1))
    else:
        shutil.copy(rc / "contracts.json", merged / "contracts.json")
    shutil.copy(rc / "soft.yaml", merged / "certified.yaml")
    return rules, changes, ok


def run(repo, out=None, model=None, base_url=None, tests=None, onto=None, dry_run=False, every=25,
        template=False, log=print, push=False, since=None, until=None):
    """The loop on HEAD: mine if needed -> certify -> check -> for every enforceable direct violation: draft -> gate -> kernel."""
    repo = resolve_repo_input(repo)
    out = rms_dir(repo, out)
    sthg = out / "sthg"
    if not (sthg / "ledger.duckdb").exists():
        mine(repo, out, every=every, since=since, until=until, log=log)
    led = Ledger(sthg)
    rules, changes, verified = certify_ledger(repo, out, log=log)
    rules_dir = out / "rules-active"
    head = git(repo, "rev-parse", "HEAD").strip()
    # The only human signal the loop reads: a human fixed a change the system made (reverted its commit, or re-edited
    # the lines it wrote). That, and nothing else, demotes the rule: severity low, violations recorded, no more fixes.
    fixed = human_fixed(repo)
    acts = led.actions()
    seen = {k: set(zip(acts[acts.kind == k].sha, acts[acts.kind == k].rule_id, strict=True)) if len(acts) else set() for k in ("revert_seen", "demote")}
    for x in fixed:
        payload = {"fix": x["fix"], "by": x["by"], "how": x["how"], "from": x["edge_from"], "to": x["edge_to"]}
        if x["how"] == "revert" and (x["by"], x["rule_id"]) not in seen["revert_seen"]:
            led.record_action(x["by"], "revert_seen", x["rule_id"], {**payload, "reverted": x["fix"]})
        if (x["by"], x["rule_id"]) not in seen["demote"]:
            led.record_action(x["by"], "demote", x["rule_id"], payload)

    def undone_for(r):
        return [x for x in fixed if x["rule_id"] == r["rule_id"] or (x["edge_from"], x["edge_to"]) == (r["edge_from"], r["edge_to"])]

    demoted_ids = {x["rule_id"] for x in fixed}
    for f in sorted(rules_dir.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text()) or []
        if f.name != "acknowledged.yaml" and any(isinstance(x, dict) and x.get("id") in demoted_ids and x.get("severity") != "low" for x in doc):
            for x in doc:
                if isinstance(x, dict) and x.get("id") in demoted_ids:
                    x["severity"] = "low"
            f.write_text(yaml.safe_dump(doc, sort_keys=False))
    res = diff_commit(repo, head, rules_dir, out, write_files=True)
    led.record_labels([{**r, "sha": head, "ts": res["ts"]} for r in res["rows"] if r["label"] != "CONFIRMED"])
    viol = [r for r in res["rows"] if r["label"] in ("ADDITIONAL", "INVERTED", "MISSING")]
    demoted = [r for r in viol if undone_for(r)]
    enforce = [r for r in viol if r.get("enforce", True) and r.get("direct") and not undone_for(r)]
    retired = [r for r in viol if not r.get("enforce", True)]
    _terminal_table("CHECK COMPLETE", [
        ("Commit", head[:12]),
        ("Violations", len(viol)),
        ("Enforceable", len(enforce)),
        ("Retired rules", len(retired)),
        ("Demoted rules", len(demoted)),
        ("Indirect violations", len([r for r in viol if not r.get("direct")])),
    ], log)
    outcomes = []
    for r in demoted:
        x = undone_for(r)[-1]
        why = f"rule demoted: a human fixed the system's change ({x['how']} {x['by'][:12]} of {x['fix'][:12]})"
        log(f"[demoted] {r['edge_from']}->{r['edge_to']} ({r['rule_id']}): recorded, not fixed")
        outcomes.append({"edge": f"{r['edge_from']}->{r['edge_to']}", "rule": r["rule_id"], "outcome": "left_standing", "why": why})
    for r in enforce:
        edge = f"{r['edge_from']}->{r['edge_to']}"
        log(f"[draft] {edge} ({r['rule_id']})")
        # second guard: whatever a human undid anywhere is never drafted again, even under another rule
        result, d = draft_one(repo, head, rules_dir, out, edge=edge, tests=tests, model=model, base_url=base_url, template=template,
                              log=lambda s: log("   " + s), reverted=fixed)
        led.record_action(head, "draft", r["rule_id"], {"verdict": result["verdict"], "tries": result["tries"], "dir": str(d)})
        if result["verdict"] != "draft_passes":
            led.record_action(head, "escalate", r["rule_id"], {"reason": "no draft passed the static checks", "dir": str(d)})
            outcomes.append({"edge": edge, "rule": r["rule_id"], "outcome": "left_standing", "why": "no passing draft"})
            continue
        record, path = run_gate(repo, head, rules_dir, d, out, tests=tests, log=lambda s: log("   " + s))
        led.record_action(head, "gate", r["rule_id"], {"verdict": record["verdict"], "stopped_at": record.get("stopped_at"), "decision": str(path)})
        if record["verdict"] != "accept" or dry_run:
            outcomes.append({"edge": edge, "rule": r["rule_id"], "outcome": "left_standing" if record["verdict"] != "accept" else "dry_run",
                             "why": record.get("stopped_at") or "dry run", "decision": str(path)})
            if record["verdict"] != "accept":
                led.record_action(head, "escalate", r["rule_id"], {"stopped_at": record.get("stopped_at"), "decision": str(path)})
            continue
        fix = apply_decision(repo, path, d, ledger=led, onto=onto, push=push, log=lambda s: log("   " + s))
        outcomes.append({"edge": edge, "rule": r["rule_id"], "outcome": "applied", "commit": fix, "decision": str(path)})
    report = {"repo": repo.name, "head": head, "rules": {"written": len([x for x in load_rules(rules_dir) if x["source"]["type"] != "recovered"]),
                                                          "certified": len(rules), "changes": {k: len(v) for k, v in changes.items() if k in ("new", "kept", "retired")},
                                                          "verified": verified},
              "violations": len(viol), "enforceable": len(enforce), "demoted": sorted(demoted_ids), "outcomes": outcomes, "ledger": led.summary()}
    (out / "run.json").write_text(json.dumps(report, indent=1, default=str))
    led.export_parquet()
    _terminal_table("RUN COMPLETE", [
        ("Applied fixes", len([o for o in outcomes if o["outcome"] == "applied"])),
        ("Recorded/escalated", len([o for o in outcomes if o["outcome"] != "applied"])),
        ("Report", str(out / "run.json")),
    ], log)
    return report


def status(out, log=print):
    out = Path(out)
    led = Ledger(out / "sthg")
    s = led.summary()
    acts = led.actions()
    rc = out / "rules-certified" / "soft.yaml"
    n_rules = len(yaml.safe_load(rc.read_text()) or []) if rc.exists() else 0
    lines = [f"ledger: {s['hyperedges']} commits as hyperedges ({s['bulk']} bulk), {s['files']} files, {s['components']} components, "
             f"{s['graph_snapshots']} import-graph snapshots, {s['graph_edges_now']} component edges now",
             f"rules: {n_rules} certified (verified rules), {s['certificates']} certificates on file",
             f"loop: {s['labels']} labels, {s['actions']} actions"]
    if len(acts):
        tail = acts.tail(8)
        lines.append("last actions:")
        for r in tail.itertuples(index=False):
            rid = r.rule_id if isinstance(r.rule_id, str) else ""
            lines.append(f"  {str(r.recorded_at)[:19]}  {str(r.kind):12s} {rid:40s} {str(r.sha)[:12]}")
    log("\n".join(lines))
    return lines


def clean(repo, confirm=False, log=print):
    """Remove SARMS-generated data for one repository, never the remote repository."""
    if not confirm:
        raise SystemExit("nothing removed; rerun with `clean <git-url> --yes`")

    def remove_tree(path):
        def unlock_and_retry(func, target, _error):
            Path(target).chmod(stat.S_IWRITE | stat.S_IREAD)
            func(target)
        shutil.rmtree(path, onerror=unlock_and_retry)

    removed = []
    for path in sarms_data_paths(repo):
        if path.is_dir():
            remove_tree(path)
            removed.append(str(path))
        elif path.exists():
            path.unlink()
            removed.append(str(path))
    log("removed SARMS data:")
    for path in removed:
        log(f"  {path}")
    if not removed:
        log("  nothing to remove")
    return removed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sarms", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mine")
    p.add_argument("repo", help="Git URL or local repository path")
    p.add_argument("--out")
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--since")
    p.add_argument("--until")
    p = sub.add_parser("certify-ledger")
    p.add_argument("repo", help="Git URL or local repository path")
    p.add_argument("--out")
    p = sub.add_parser("run")
    p.add_argument("repo", help="Git URL or local repository path")
    p.add_argument("--tests", help="test command required for a fix to pass the safety gate")
    p.add_argument("--apply", dest="dry_run", action="store_false",
                   help="apply accepted fixes on the sarms/fixes branch")
    p.set_defaults(dry_run=True)
    p.add_argument("--push", action="store_true",
                   help="push the sarms/fixes branch (requires --apply)")
    p.add_argument("--model", help="advanced: model name for complex draft fixes")
    p.add_argument("--base-url", help="advanced: model API base URL")
    p.add_argument("--out", help="advanced: output directory")
    p.add_argument("--onto", help="advanced: target branch instead of sarms/fixes")
    p.add_argument("--template", action="store_true", help="advanced: use the simple rule-based fixer")
    p.add_argument("--every", type=int, default=25, help="advanced: sample every N history commits")
    p.add_argument("--since", help="limit first-time mining to commits after YYYY-MM-DD")
    p.add_argument("--until", help="limit first-time mining through a commit or date")
    p = sub.add_parser("clean", help="remove SARMS data for one repository")
    p.add_argument("repo", help="Git URL or local repository path")
    p.add_argument("--yes", action="store_true", help="confirm removal of the local clone and SARMS results")
    p = sub.add_parser("status")
    p.add_argument("--out", required=True)
    p = sub.add_parser("dst")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--level", type=int, default=1, choices=[1, 2])
    p.add_argument("--out", default="reports/dst-risk.md")
    p.add_argument("--scenario")
    a = ap.parse_args(argv)
    if a.cmd == "mine":
        mine(a.repo, a.out, a.every, a.since, a.until)
    elif a.cmd == "certify-ledger":
        certify_ledger(a.repo, a.out)
    elif a.cmd == "run":
        if a.push and a.dry_run:
            ap.error("run --push requires run --apply")
        run(a.repo, a.out, a.model, a.base_url, a.tests, a.onto, a.dry_run, a.every, a.template,
            push=a.push, since=a.since, until=a.until)
    elif a.cmd == "clean":
        clean(a.repo, a.yes)
    elif a.cmd == "status":
        status(a.out)
    elif a.cmd == "dst":
        from sarms.sim.harness import main as dst_main  # noqa: PLC0415
        dst_main(seeds=a.seeds, level=a.level, out=a.out, scenario=a.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
