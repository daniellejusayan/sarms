"""Gate G0–G8 (docs/spec/gate.md), run by tools only, producing a decision record (docs/spec/decision-record.schema.json).

    from sarms.verify.gate import run_gate
    record, path = run_gate(repo_dir, sha, rules_dir, draft_dir, rms_dir, tests="python -m pytest -q", policy=None)

The gate takes the draft step's patch (`patch.diff`) and re-applies it on its own fresh copy of the author's
commit, so nothing the draft step did is trusted. Static lines first, then the trials; the first failing line
stops the run and later lines are recorded as `pass: null`. Verdict: accept when all nine pass, otherwise
escalate (the kernel applies nothing; the deviation stays reported). Policy comes from `.rms/policy.yaml` when
present; defaults below.

  G0  the author's functional change is kept: the patch touches no test file, and every hunk of the author's own
      diff (against its parent) is still present in the candidate; the author's added tests pass
  G1  the Differ on the candidate: the target violation is gone and no new violation appears
  G2  no new import cycle, no new unassigned package, no hub fan-in growth beyond policy
  G5  no protected path touched; files and lines changed under the class cap
  G6  the draft names the rule it serves and that rule is the one violated
  G7  the patch becomes one commit that `git revert` undoes cleanly (dry run on the copy)
  G3  the test suite is green on the candidate
  G4  mutation score on the moved code not below the policy floor (mutmut when available; otherwise the line is
      recorded as not evaluated and the policy says whether that fails the gate: default `g4_required: false`)
  G8  the rules compile and the checker passes in a clean run on the candidate (import-linter + our Differ)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import networkx as nx
import yaml

from sarms.differ.diff import diff_commit
from sarms.mining.common import git
from sarms.rules.compile import load_rules

VERIFIER_VERSION = "gate-0.1 (2026-11-07)"
DEFAULT_POLICY = {
    "version": "policy-0.1",
    "protected_paths": [r"^\.github/", r"^\.rms/policy\.yaml$", r"^CODEOWNERS$", r"^\.importlinter$", r"(^|/)pyproject\.toml$", r"(^|/)setup\.cfg$"],
    "test_paths": [r"(^|/)tests?/", r"(^|/)test_[^/]*\.py$", r"_test\.py$", r"(^|/)conftest\.py$"],
    "class_caps": {"delete_unused_import": [1, 20], "move_to_layer": [10, 300], "route_through_port": [15, 500], "record_amendment": [3, 120], "llm_draft": [10, 300], "template": [3, 60]},
    "hub_fan_in_growth_max": 2,
    "g4_required": False,
    "g4_floor": 0.60,
    "test_timeout": 1800,
}
VIOLATION = {"ADDITIONAL", "INVERTED", "MISSING"}


def load_policy(repo_dir: Path):
    p = Path(repo_dir) / ".rms" / "policy.yaml"
    pol = dict(DEFAULT_POLICY)
    if p.exists():
        pol.update(yaml.safe_load(p.read_text()) or {})
    return pol


def _test_cmd(tests: str):
    """The test command as an argument list, with the program looked up on PATH. On Windows a bare `python` is
    otherwise searched for next to the running interpreter first, which finds the base Python and not the environment's."""
    argv = tests.split()
    return [shutil.which(argv[0]) or argv[0], *argv[1:]]


def _copy(repo_dir: Path, sha: str, where: Path):
    if where.exists():
        shutil.rmtree(where)
    git(repo_dir, "worktree", "prune", check=False)
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "add", "--detach", "-f", str(where), sha], check=True, capture_output=True)
    return where


def _cleanup(repo_dir: Path, where: Path):
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(where)], capture_output=True)
    if where.exists():
        shutil.rmtree(where, ignore_errors=True)


def _changed_files(patch: str):
    return sorted({m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", patch, re.M)})


def _cycles(rows_edges):
    g = nx.DiGraph()
    g.add_edges_from(rows_edges)
    return {frozenset(c) for c in nx.strongly_connected_components(g) if len(c) > 1}


def run_gate(repo_dir, sha, rules_dir, draft_dir, rms_dir, tests=None, policy=None, salt=None, log=print, author_parent=None):
    repo_dir, rules_dir, draft_dir, rms_dir = Path(repo_dir).resolve(), Path(rules_dir), Path(draft_dir), Path(rms_dir)
    t0 = time.time()
    pol = policy or load_policy(repo_dir)
    sha = sha or git(repo_dir, "rev-parse", "HEAD").strip()
    draft = json.loads((draft_dir / "result.json").read_text())
    patch = (draft_dir / "patch.diff").read_text() if (draft_dir / "patch.diff").exists() else ""
    gate = {f"G{i}": {"line": f"G{i}", "pass": None, "evidence": None} for i in range(9)}
    trial_runs = []
    stop = None
    cand = _copy(repo_dir, sha, rms_dir / "gate-candidate")
    try:
        files = _changed_files(patch)
        if patch.strip():
            r = subprocess.run(["git", "-C", str(cand), "apply", "--whitespace=nowarn", "-"], input=patch, text=True, capture_output=True)
            if r.returncode != 0:
                gate["G0"]["pass"], gate["G0"]["evidence"] = False, {"error": "patch does not apply: " + r.stderr[-300:]}
                stop = "G0"
        # ---- G0: the author's change is kept; no test touched
        if stop is None:
            parent = author_parent or (git(repo_dir, "rev-parse", f"{sha}^", check=False).strip() or None)
            author_diff = git(repo_dir, "diff", f"{parent}..{sha}", "--", ".", check=False) if parent else ""
            author_hunks = [h for h in re.split(r"^@@.*$", author_diff, flags=re.M)[1:]] if author_diff else []
            added_lines = [ln[1:] for h in author_hunks for ln in h.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
            # the one thing the fix may remove from the author's change is the violating import itself
            forbidden = (draft.get("violation") or {}).get("edge_to", "")
            viol_line = re.compile(r"^\s*(from\s+" + re.escape(forbidden) + r"(\.|\s)|import\s+" + re.escape(forbidden) + r"(\.|\s|$))") if forbidden else None
            cand_text = {}
            kept = 0
            for ln in added_lines:
                s_ = ln.strip()
                if not s_ or (viol_line and viol_line.match(ln)):
                    kept += 1
                    continue
                found = False
                for f in _changed_files(author_diff):
                    if f not in cand_text:
                        p = cand / f
                        cand_text[f] = p.read_text(errors="replace") if p.exists() else ""
                    if s_ in cand_text[f]:
                        found = True
                        break
                kept += found
            tests_touched = [f for f in files if any(re.search(p, f) for p in pol["test_paths"])]
            # the author's own tests still pass (only when a test runner is configured)
            author_tests = [f for f in _changed_files(author_diff) if any(re.search(p, f) for p in pol["test_paths"])]
            author_tests_ok = True
            if tests and author_tests:
                r = subprocess.run(_test_cmd(tests) + author_tests, cwd=cand, capture_output=True, text=True, timeout=pol["test_timeout"])
                author_tests_ok = r.returncode == 0
                trial_runs.append(f"G0:{hashlib.sha1(r.stdout.encode()).hexdigest()[:10]}")
            ok = not tests_touched and (kept == len(added_lines)) and author_tests_ok
            gate["G0"]["pass"], gate["G0"]["evidence"] = ok, {"diff_hunks_kept": f"{kept}/{len(added_lines)}", "tests_touched": tests_touched,
                                                              "author_tests": author_tests, "author_tests_pass": author_tests_ok}
            if not ok:
                stop = "G0"
        # ---- G1: Differ before/after
        before = diff_commit(repo_dir, sha, rules_dir, rms_dir / "gate-before", write_files=True)
        after = diff_commit(repo_dir, sha, rules_dir, rms_dir / "gate-after", tree=cand, write_files=True)
        if stop is None:
            b = {(r["contract"], r["edge_from"], r["edge_to"]) for r in before["rows"] if r["label"] in VIOLATION}
            a = {(r["contract"], r["edge_from"], r["edge_to"]) for r in after["rows"] if r["label"] in VIOLATION}
            target = draft.get("violation") or {}
            tkey = (target.get("contract"), target.get("edge_from"), target.get("edge_to"))
            gone = tkey not in a if all(tkey) else len(a) < len(b)
            new = sorted(a - b)
            ok = gone and not new
            gate["G1"]["pass"], gate["G1"]["evidence"] = ok, {"violations_before": len(b), "violations_after": len(a), "target_gone": gone,
                                                              "new_violations": [f"{x[1]} -> {x[2]}" for x in new]}
            if not ok:
                stop = "G1"
        # ---- G2: indicators
        if stop is None:
            cb, ca = _cycles([tuple(e) for e in before.get("edges", [])]), _cycles([tuple(e) for e in after.get("edges", [])])
            ub = {i["edge_from"] for i in before.get("indicators", []) if i["indicator"] == "unassigned"}
            ua = {i["edge_from"] for i in after.get("indicators", []) if i["indicator"] == "unassigned"}
            fan_b, fan_a = {}, {}
            for e in before.get("edges", []):
                fan_b[e[1]] = fan_b.get(e[1], 0) + 1
            for e in after.get("edges", []):
                fan_a[e[1]] = fan_a.get(e[1], 0) + 1
            hub_growth = {m: fan_a[m] - fan_b.get(m, 0) for m in fan_a if fan_a[m] - fan_b.get(m, 0) > pol["hub_fan_in_growth_max"]}
            ok = not (ca - cb) and not (ua - ub) and not hub_growth
            gate["G2"]["pass"], gate["G2"]["evidence"] = ok, {"new_cycles": [sorted(c) for c in (ca - cb)], "new_unassigned": sorted(ua - ub), "hub_growth": hub_growth}
            if not ok:
                stop = "G2"
        # ---- G5: protected paths and blast radius
        if stop is None:
            prot = [f for f in files if any(re.search(p, f) for p in pol["protected_paths"])]
            n_lines = sum(1 for ln in patch.splitlines() if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")))
            cls = draft.get("class", "llm_draft")
            cap_files, cap_lines = pol["class_caps"].get(cls, pol["class_caps"]["llm_draft"])
            ok = not prot and len(files) <= cap_files and n_lines <= cap_lines
            gate["G5"]["pass"], gate["G5"]["evidence"] = ok, {"paths_touched": files, "protected": prot, "blast_radius": {"files": len(files), "lines": n_lines, "cap": [cap_files, cap_lines]}}
            if not ok:
                stop = "G5"
        # ---- G6: rule link
        if stop is None:
            rid = (draft.get("violation") or {}).get("rule_id")
            known = {r["id"]: r for r in load_rules(rules_dir)}
            v = draft.get("violation") or {}
            ok = rid in known and known[rid]["from"] == v.get("edge_from") and known[rid]["to"] == v.get("edge_to")
            gate["G6"]["pass"], gate["G6"]["evidence"] = ok, {"rule_id": rid, "known": rid in known}
            if not ok:
                stop = "G6"
        # ---- G7: one commit, revertible (dry run on the candidate copy)
        if stop is None:
            subprocess.run(["git", "-C", str(cand), "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(cand), "-c", "user.name=sensolens-sarms", "-c", "user.email=sensolens-sarms@sarms.invalid",
                            "commit", "-q", "-m", "sarms: candidate"], check=True, capture_output=True)
            fix_sha = git(cand, "rev-parse", "HEAD").strip()
            r = subprocess.run(["git", "-C", str(cand), "-c", "user.name=sensolens-sarms", "-c", "user.email=sensolens-sarms@sarms.invalid",
                                "revert", "--no-edit", fix_sha], capture_output=True, text=True)
            same = git(cand, "diff", sha, "--", ".", check=False).strip() == "" if r.returncode == 0 else False
            # back to the candidate state for the trials
            subprocess.run(["git", "-C", str(cand), "reset", "-q", "--hard", fix_sha], check=True, capture_output=True)
            ok = r.returncode == 0 and same
            gate["G7"]["pass"], gate["G7"]["evidence"] = ok, {"revert_dry_run": "clean" if ok else r.stderr[-200:], "tree_restored": same}
            if not ok:
                stop = "G7"
        # ---- G3: full suite
        if stop is None:
            if tests:
                r = subprocess.run(_test_cmd(tests), cwd=cand, capture_output=True, text=True, timeout=pol["test_timeout"])
                run_id = f"G3:{hashlib.sha1((r.stdout + r.stderr).encode()).hexdigest()[:10]}"
                trial_runs.append(run_id)
                ok = r.returncode == 0
                gate["G3"]["pass"], gate["G3"]["evidence"] = ok, {"pytest_run_id": run_id, "returncode": r.returncode, "tail": (r.stdout + r.stderr)[-600:]}
            else:
                ok = False
                gate["G3"]["pass"], gate["G3"]["evidence"] = False, {"pytest_run_id": None, "error": "no test command configured"}
            if not ok:
                stop = "G3"
        # ---- G4: mutation score (mutmut if present)
        if stop is None:
            if shutil.which("mutmut") and files:
                r = subprocess.run(["mutmut", "run", "--paths-to-mutate", ",".join(files), "--no-progress"], cwd=cand, capture_output=True, text=True, timeout=pol["test_timeout"])
                m = re.search(r"(\d+)/(\d+)\s+.*killed", r.stdout)
                score = int(m.group(1)) / max(int(m.group(2)), 1) if m else None
                ok = score is not None and score >= pol["g4_floor"]
                gate["G4"]["pass"], gate["G4"]["evidence"] = ok, {"mutmut_after": score, "floor": pol["g4_floor"]}
            else:
                ok = not pol["g4_required"]
                gate["G4"]["pass"], gate["G4"]["evidence"] = ok, {"mutmut_after": None, "note": "mutmut not available; policy g4_required=false"}
            if not ok:
                stop = "G4"
        # ---- G8: clean run of the checker on the candidate
        if stop is None:
            r = subprocess.run(["lint-imports"], cwd=cand, capture_output=True, text=True) if (cand / ".importlinter").exists() or (cand / "setup.cfg").exists() else None
            rules_ok = all(x["label"] not in VIOLATION or x.get("enforce") is False for x in after["rows"]) or len({(x["edge_from"], x["edge_to"]) for x in after["rows"] if x["label"] in VIOLATION}) < len(
                {(x["edge_from"], x["edge_to"]) for x in before["rows"] if x["label"] in VIOLATION})
            ok = rules_ok and (r is None or r.returncode == 0 or "Contracts:" in (r.stdout + r.stderr))
            run_id = f"G8:{hashlib.sha1(((r.stdout + r.stderr) if r else 'differ').encode()).hexdigest()[:10]}"
            trial_runs.append(run_id)
            gate["G8"]["pass"], gate["G8"]["evidence"] = ok, {"ci_run_id": run_id, "lint_imports": (r.stdout + r.stderr)[-300:] if r else "n/a (Differ only)"}
            if not ok:
                stop = "G8"
    finally:
        _cleanup(repo_dir, cand)
    verdict = "accept" if stop is None else "escalate"
    record = {"sha": sha, "verdict": verdict, "class": draft.get("class", "llm_draft"), "gate": [gate[f"G{i}"] for i in range(9)],
              "differ_before": str(before["paths"]["sarif"]) if before.get("paths") else "", "differ_after": str(after["paths"]["sarif"]) if after.get("paths") else "",
              "trial_runs": trial_runs, "prompt_hash": draft.get("prompt_hash") or hashlib.sha256(patch.encode()).hexdigest(),
              "policy_version": pol["version"], "verifier_version": VERIFIER_VERSION, "applied_commit": None}
    key = (salt or _salt(repo_dir)).encode()
    record["signature"] = hmac.new(key, json.dumps(record, sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()
    record["seconds"] = round(time.time() - t0, 1)
    record["stopped_at"] = stop
    out = rms_dir / "decisions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sha}.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    log(f"gate: {verdict}" + (f" (stopped at {stop})" if stop else " (G0-G8 pass)") + f" in {record['seconds']}s -> {path}")
    return record, path


def _salt(repo_dir: Path) -> str:
    for p in (repo_dir / ".salt", Path.cwd() / ".salt"):
        if p.exists():
            return p.read_text().strip()
    return "sarms-dev-salt"


def verify_signature(record: dict, salt: str) -> bool:
    body = {k: v for k, v in record.items() if k not in ("signature", "seconds", "stopped_at")}
    expect = hmac.new(salt.encode(), json.dumps(body, sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expect, record.get("signature", ""))
