"""Checked-out trees and the import-linter worker: shared by the replay, the Differ and the rule
learner (all above this layer), so none of them has to import another."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sarms.mining.common import git, worktree_path

WORKER = Path(__file__).with_name("_lint_worker.py")


def uses_missing(c, missing):
    """True when a contract names a module whose root package is not in this tree."""
    mods = [m for k in ("source_modules", "forbidden_modules", "modules", "layers", "containers")
            for m in (c.get(k) if isinstance(c.get(k), list) else [c.get(k)] if c.get(k) else [])]
    return any(str(m).split(".")[0].strip("() ") in missing for m in mods)


def pythonpath(wt: Path, roots, override=None):
    """(paths, missing_roots): where each root package lives in the worktree. Searches the root,
    src/, python/, lib/, and one level under packages/, libs/, apps/ and */src (monorepos such as
    PostHog's packages/owners-yaml/owners_yaml). A root that is nowhere is reported, not fatal."""
    if override:
        paths = [str(wt / p) for p in override]
        missing = [r.split(".")[0] for r in roots
                   if not any((wt / p / r.split(".")[0]).is_dir() or (wt / p / f"{r.split('.')[0]}.py").exists() for p in override)]
        return paths, missing
    bases = [wt / "src", wt, wt / "python", wt / "lib"]
    for group in ("packages", "libs", "apps"):
        g = wt / group
        if g.is_dir():
            bases += [d for d in sorted(g.iterdir()) if d.is_dir()] + [d / "src" for d in sorted(g.iterdir()) if (d / "src").is_dir()]
    bases += [d / "src" for d in sorted(wt.iterdir()) if d.is_dir() and (d / "src").is_dir()]
    paths, missing = [], []
    for r in roots:
        top = r.split(".")[0]
        for base in bases:
            if (base / top).is_dir() or (base / f"{top}.py").exists():
                paths.append(str(base))
                break
        else:
            missing.append(top)
    return list(dict.fromkeys(paths)), missing



def run_worker(tree, bundle, cache_dir=None, src=None, timeout=900, want_edges=True, want_details=True):
    """Check one checked-out tree. Returns the worker's JSON, or {"graph_error": ...}."""
    tree = Path(tree).resolve()
    session = bundle["session"]
    pp, missing = pythonpath(tree, session["root_packages"], src)
    strict = [c for c in bundle["contracts_strict"] if not uses_missing(c, missing)]
    asis = [c for c in bundle["contracts_asis"] if not uses_missing(c, missing)]
    skipped = [c["name"] for c in bundle["contracts_strict"] if uses_missing(c, missing)]
    if not pp or (not strict and bundle["contracts_strict"]):
        return {"graph_error": f"missing root(s): {' '.join(missing)}", "package_missing": skipped}
    session_here = {**session, "root_packages": [r for r in session["root_packages"] if r.split(".")[0] not in missing]}
    if not session_here["root_packages"]:
        return {"graph_error": f"missing root(s): {' '.join(missing)}", "package_missing": skipped}
    spec = {"pythonpath": pp, "session_options": session_here, "contracts": strict, "contracts_asis": asis,
            "cache_dir": str(cache_dir) if cache_dir else None, "want_edges": want_edges, "want_details": want_details}
    try:
        p = subprocess.run([sys.executable, str(WORKER)], input=json.dumps(spec), capture_output=True,
                           text=True, timeout=timeout, cwd=tree)
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"graph_error": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        out = {"graph_error": "timeout"}
    out["package_missing"] = skipped
    return out



def checkout(repo_dir, sha, worktree=None):
    """A tree at `sha`: the repository itself when its HEAD is that commit and it is a normal
    checkout, else a detached worktree <repo>.wt (reused between calls)."""
    repo_dir = Path(repo_dir)
    head = git(repo_dir, "rev-parse", "HEAD", check=False).strip()
    bare = git(repo_dir, "rev-parse", "--is-bare-repository").strip() == "true"
    populated = any(p.name != ".git" for p in repo_dir.iterdir())  # a --no-checkout clone has no files
    if head == sha and not bare and not worktree and populated:
        return repo_dir
    wt = Path(worktree) if worktree else worktree_path(repo_dir, "wt")
    valid = wt.exists() and git(wt, "rev-parse", "--is-inside-work-tree", check=False).strip() == "true"
    if not valid:
        if wt.is_dir():
            shutil.rmtree(wt)
        elif wt.exists():
            wt.unlink()
        git(repo_dir, "worktree", "prune", check=False)
        git(repo_dir, "worktree", "add", "--detach", "-f", str(wt.resolve()), sha)
    else:
        git(wt, "checkout", "-q", "--force", sha)
        git(wt, "clean", "-fdq")
    return wt
