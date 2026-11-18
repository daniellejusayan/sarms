"""The kernel: applies an accepted decision record as the system's own commit, and nothing else.

    from sarms.kernel.apply import apply_decision, revert_own_commit
    sha_fix = apply_decision(repo_dir, record_path, draft_dir, ledger=None, branch="sarms/fixes", push=False)

Rules the kernel never relaxes (docs/spec/gate.md, point 7):
  - it executes only a decision record whose verdict is `accept` and whose signature verifies;
  - the commit is authored by sensolens-sarms, on `branch` (default a side branch; `--onto main` is a choice the
    operator makes), with the decision record id in the message, and it is exactly the gate's patch;
  - one commit, so one `git revert` undoes it (G7 re-checked here before the commit lands);
  - every step is written to the ledger as an action (apply / escalate / revert_seen).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from sarms.mining.common import git, patch_fingerprint
from sarms.verify.gate import _salt, verify_signature

BOT = ["-c", "user.name=sensolens-sarms", "-c", "user.email=sensolens-sarms@sarms.invalid"]


def apply_decision(repo_dir, record_path, draft_dir, ledger=None, branch="sarms/fixes", onto=None, push=False, log=print):
    repo_dir, record_path, draft_dir = Path(repo_dir).resolve(), Path(record_path), Path(draft_dir)
    record = json.loads(record_path.read_text())
    if record["verdict"] != "accept":
        if ledger:
            ledger.record_action(record["sha"], "escalate", (json.loads((draft_dir / "result.json").read_text()).get("violation") or {}).get("rule_id"),
                                 {"decision": str(record_path), "stopped_at": record.get("stopped_at")})
        log(f"kernel: nothing applied (verdict {record['verdict']}, stopped at {record.get('stopped_at')})")
        return None
    if not verify_signature(record, _salt(repo_dir)):
        raise PermissionError("decision record signature does not verify; the kernel refuses")
    patch = (draft_dir / "patch.diff").read_text()
    base = record["sha"]
    target = onto or branch
    # work on a detached worktree so the developer's checkout is never touched
    wt = repo_dir.parent / f"{repo_dir.name}.kernel"
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt)], capture_output=True)
    git(repo_dir, "worktree", "prune", check=False)
    subprocess.run(["git", "-C", str(repo_dir), "worktree", "add", "--detach", "-f", str(wt), base], check=True, capture_output=True)
    try:
        r = subprocess.run(["git", "-C", str(wt), "apply", "--whitespace=nowarn", "-"], input=patch, text=True, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("patch does not apply on the base commit: " + r.stderr[-300:])
        subprocess.run(["git", "-C", str(wt), "add", "-A"], check=True, capture_output=True)
        v = json.loads((draft_dir / "result.json").read_text()).get("violation") or {}
        msg = (f"sarms: {v.get('edge_from')} must not import {v.get('edge_to')} ({v.get('rule_id')})\n\n"
               f"Decision record: {record_path.name}\nGate: G0-G8 pass, verifier {record['verifier_version']}, policy {record['policy_version']}\n"
               f"Revert with: git revert <this commit>\n")
        subprocess.run(["git", "-C", str(wt), *BOT, "commit", "-q", "-m", msg], check=True, capture_output=True)
        fix = git(wt, "rev-parse", "HEAD").strip()
        # G7 once more, on the real commit
        chk = subprocess.run(["git", "-C", str(wt), *BOT, "revert", "--no-edit", fix], capture_output=True, text=True)
        if chk.returncode != 0 or git(wt, "diff", base, "--", ".", check=False).strip():
            raise RuntimeError("the applied commit does not revert cleanly; refusing to land it")
        subprocess.run(["git", "-C", str(wt), "reset", "-q", "--hard", fix], check=True, capture_output=True)
        # land it on the target branch (fast-forward only; otherwise the branch is created at the fix)
        current = git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD", check=False).strip()
        if onto and onto == current:
            # the system's follow-up commit lands on the branch the push went to; the developer's commit stays its parent
            if git(repo_dir, "status", "--short", check=False).strip():
                raise RuntimeError("the checkout has local changes; refusing to move the branch under them")
            subprocess.run(["git", "-C", str(repo_dir), "update-ref", f"refs/heads/{onto}", fix], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo_dir), "reset", "-q", "--hard", fix], check=True, capture_output=True)
        elif onto:
            subprocess.run(["git", "-C", str(repo_dir), "branch", "-f", onto, fix], check=True, capture_output=True)
        else:
            subprocess.run(["git", "-C", str(repo_dir), "branch", "-f", branch, fix], check=True, capture_output=True)
        if push:
            subprocess.run(["git", "-C", str(repo_dir), "push", "-q", "origin", f"{fix}:refs/heads/{target}"], check=True, capture_output=True)
    finally:
        subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(wt)], capture_output=True)
    record["applied_commit"] = fix
    record_path.write_text(json.dumps(record, indent=1, default=str))
    if ledger:
        ledger.record_action(base, "apply", v.get("rule_id"), {"commit": fix, "branch": target, "decision": str(record_path),
                                                              "from": v.get("edge_from"), "to": v.get("edge_to")})
    log(f"kernel: applied as {fix[:12]} on {target} (author sensolens-sarms)")
    return fix


def revert_own_commit(repo_dir, fix_sha, ledger=None, log=print):
    """Undo a system commit with one revert (what a developer would also do by hand)."""
    repo_dir = Path(repo_dir).resolve()
    r = subprocess.run(["git", "-C", str(repo_dir), *BOT, "revert", "--no-edit", fix_sha], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    new = git(repo_dir, "rev-parse", "HEAD").strip()
    if ledger:
        ledger.record_action(fix_sha, "revert_seen", None, {"revert_commit": new})
    log(f"kernel: reverted {fix_sha[:12]} with {new[:12]}")
    return new


FIX_SUBJECT = re.compile(r"^sarms: (\S+) must not import (\S+) \((.+)\)$")


BOT_NAME = "sensolens-sarms"


def _author(repo_dir, sha):
    return git(repo_dir, "log", "-1", "--format=%an", sha, check=False).strip()


def _lines_from(repo_dir, rev, path, fix):
    """Stripped text of the lines of `path` at `rev` that git blame attributes to the commit `fix`."""
    out = git(repo_dir, "blame", "--line-porcelain", rev, "--", path, check=False)
    owned, cur = set(), None
    for ln in out.splitlines():
        if re.match(r"^[0-9a-f]{40} \d+ \d+", ln):
            cur = ln[:40]
        elif ln.startswith("\t") and cur == fix and ln[1:].strip():
            owned.add(ln[1:].strip())
    return owned


def human_fixed(repo_dir):
    """System fixes on the branch that a human then fixed, read from git alone (no ledger needed). This is the one
    event that demotes a rule. A human fixed the agent's change when a commit not authored by the system either
      - reverts the system's commit (`git revert`), or
      - re-edits it: removes or rewrites lines the fix added (blamed to the fix just before that commit); for a fix
        that only deleted lines, puts one of the deleted lines back in the same file.
    Gate rejections, escalations and the system's own reverts are not in this list.
    -> [{fix, by, how: 'revert' | 're-edit', rule_id, edge_from, edge_to, patch, fingerprint}], oldest fix first."""
    repo_dir = Path(repo_dir)
    reverts = {fix: rev for rev, fix in developer_reverts(repo_dir) if _author(repo_dir, rev) != BOT_NAME}
    log_ = git(repo_dir, "log", "--first-parent", "--reverse", "--format=%H%x09%an%x09%s", "HEAD", check=False)
    out = []
    for line in log_.splitlines():
        parts = line.split("\t", 2)
        m = FIX_SUBJECT.match(parts[2]) if len(parts) == 3 and parts[1] == BOT_NAME else None
        if not m:
            continue
        fix = parts[0]
        patch = git(repo_dir, "diff", f"{fix}^", fix, "--", ".", check=False)
        fp = patch_fingerprint(patch)
        added = {(p, t) for p, s, t in fp if s == "+"}
        removed = {(p, t) for p, s, t in fp if s == "-"}
        files = sorted({p for p, _, _ in fp})
        by, how = (reverts[fix], "revert") if fix in reverts else (None, None)
        if by is None and files:
            later = git(repo_dir, "log", "--first-parent", "--reverse", "--format=%H%x09%an", f"{fix}..HEAD", "--", *files, check=False)
            for row in later.splitlines():
                c, _, an = row.partition("\t")
                if an == BOT_NAME:
                    continue
                cfp = patch_fingerprint(git(repo_dir, "diff", f"{c}^", c, "--", *files, check=False))
                if added:
                    gone = {(p, t) for p, s, t in cfp if s == "-" and (p, t) in added}
                    hit = any(t in _lines_from(repo_dir, f"{c}^", p, fix) for p, t in gone)
                else:
                    hit = any(s == "+" and (p, t) in removed for p, s, t in cfp)
                if hit:
                    by, how = c, "re-edit"
                    break
        if by:
            out.append({"fix": fix, "by": by, "how": how, "edge_from": m.group(1), "edge_to": m.group(2), "rule_id": m.group(3),
                        "patch": patch, "fingerprint": fp})
    return out


def reverted_fixes(repo_dir):
    """The system fixes a human undid with `git revert` (a subset of human_fixed)."""
    return [x for x in human_fixed(repo_dir) if x["how"] == "revert"]


def developer_reverts(repo_dir, since_sha=None):
    """Reverts of system commits seen in history (the only human signal the loop reads): list of (revert_sha, reverted_fix_sha)."""
    repo_dir = Path(repo_dir)
    rng = f"{since_sha}..HEAD" if since_sha else "HEAD"
    out = git(repo_dir, "log", "--first-parent", "--format=%H%x09%s%x09%b", rng, check=False)
    found = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and "This reverts commit" in parts[2]:
            m = re.search(r"This reverts commit ([0-9a-f]{40})", parts[2])
            if m:
                author = git(repo_dir, "log", "-1", "--format=%an", m.group(1), check=False).strip()
                if author == "sensolens-sarms":
                    found.append((parts[0], m.group(1)))
    return found
