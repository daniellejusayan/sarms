"""Git log stream with the everyday noise removed: one row per (commit, source file) with churn.

    from sarms.mining.gitlog import stream, to_frame
    df = to_frame(stream("data/raw/kedro"))          # idx, sha, ts, author, path, added, deleted, component

Noise, removed by regular expressions before anything downstream sees it: documentation, assets, third-party
and vendored code, lock files, generated code, CI and packaging metadata, and (by default) tests. What is left
is the structural change stream: source files of the repository's own packages. Binary files (numstat "-")
count as one line of churn each way so they are not invisible.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


def detect_roots(tree: Path):
    """Top-level packages (folders with __init__.py directly under the repo or under src/). Mining layer: no config read."""
    out = []
    for base in (tree, tree / "src"):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith((".", "_")) and d.name not in ("tests", "test", "docs"):
                    out.append(d.name)
    return out

NOISE = [re.compile(p, re.I) for p in (
    r"(^|/)(docs?|documentation|doc_src|site|wiki|examples?|samples?|notebooks?|tutorials?)/",
    r"\.(md|rst|txt|adoc|ipynb|pdf|csv|json|ya?ml|toml|ini|cfg|xml|html|css|scss|svg|png|jpe?g|gif|ico|webp|woff2?|ttf|otf|mp4|mov|zip|gz|whl)$",
    r"(^|/)(LICENSE|NOTICE|CHANGELOG|RELEASE|AUTHORS|CONTRIBUTORS|CODEOWNERS|MANIFEST\.in|Makefile|Dockerfile)",
    r"(^|/)(vendor|vendored|third_party|thirdparty|node_modules|external|_vendor|bower_components)/",
    r"\.min\.(js|css)$|_pb2(_grpc)?\.py$|\.pb\.go$|(^|/)migrations?/\d",
    r"(^|/)(poetry\.lock|package-lock\.json|yarn\.lock|Pipfile\.lock|uv\.lock|requirements[^/]*\.txt)$",
    r"(^|/)(\.github|\.gitlab|\.circleci|\.azure|ci|\.ci|tools?/ci)/",
    r"(^|/)(setup\.py|setup\.cfg|conftest\.py|__main__\.py)$",
)]
TESTS = re.compile(r"(^|/)(tests?|testing|test_utils|fixtures)/|(^|/)test_[^/]*\.py$|_test\.py$", re.I)
SOURCE = re.compile(r"\.(py|pyi)$", re.I)


def is_noise(path: str, keep_tests: bool = False) -> bool:
    if not SOURCE.search(path):
        return True
    if not keep_tests and TESTS.search(path):
        return True
    return any(p.search(path) for p in NOISE)


def stream(repo_dir, since=None, until=None, first_parent=True, keep_tests=False, max_files=None):
    """Yield commits {sha, ts, author, files:[(path, added, deleted)]} with noise removed; commits left with no
    source file are dropped. max_files: commits touching more source files than this are dropped as bulk."""
    args = ["git", "-C", str(repo_dir), "log", "--numstat", "--format=@%H%x09%at%x09%an", "--no-renames"]
    if first_parent:
        args.append("--first-parent")
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    out = subprocess.run(args, capture_output=True, text=True, check=True, errors="replace").stdout
    cur = None
    for line in out.splitlines():
        if line.startswith("@"):
            if cur and cur["files"] and (max_files is None or len(cur["files"]) <= max_files):
                yield cur
            sha, ts, author = line[1:].split("\t")
            cur = {"sha": sha, "ts": int(ts), "author": author, "files": []}
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            a, d, path = parts
            if is_noise(path, keep_tests):
                continue
            cur["files"].append((path, 1 if a == "-" else int(a), 1 if d == "-" else int(d)))
    if cur and cur["files"] and (max_files is None or len(cur["files"]) <= max_files):
        yield cur


def module_of(path: str, roots):
    parts = path[:-3].split("/") if path.endswith(".py") else path[:-4].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for i, p in enumerate(parts):
        if p in roots:
            return ".".join(parts[i:])
    return None


def to_frame(commits, repo_dir=None, roots=None, component_of=None):
    """Rows (idx, sha, ts, author, path, added, deleted, module[, component]). Commits are numbered oldest first.
    component_of: optional callable module -> component (the rules layer supplies the learner's components)."""
    commits = list(commits)
    commits.reverse()  # git log is newest first
    if roots is None and repo_dir is not None:
        roots = detect_roots(Path(repo_dir))
    roots = set(roots or [])
    rows = []
    for i, c in enumerate(commits):
        for path, a, d in c["files"]:
            rows.append({"idx": i, "sha": c["sha"], "ts": pd.Timestamp(c["ts"], unit="s", tz="UTC"), "author": c["author"],
                         "path": path, "added": a, "deleted": d, "module": module_of(path, roots) if roots else None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if component_of is not None:
        df["component"] = df.module.map(lambda m: component_of(m) if isinstance(m, str) else None)
    return df
