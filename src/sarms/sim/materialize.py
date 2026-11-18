"""Level 2 of the simulation: turn a generated repository life into a real git repository with Python packages,
a written layers rule file, and a test suite, so the whole loop (Differ, draft, gate, kernel) runs for real."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sarms.sim.generator import LAYERS

DEV = ["-c", "user.name=Dev Sim", "-c", "user.email=dev@sim.invalid"]


def _git(repo, *a, author=DEV):
    return subprocess.run(["git", "-C", str(repo), *author, *a], check=True, capture_output=True, text=True).stdout


def write_tree(repo: Path, rec: dict):
    """Write every module of the record as a file whose imports are the record's edges (plus a trivial function)."""
    src = repo / "app"
    if src.exists():
        for p in src.rglob("*.py"):
            p.unlink()
    imports = {}
    for a, b in rec["edges"]:
        imports.setdefault(a, set()).add(b)
    comps = {m.rsplit(".", 1)[0] for m in rec["modules"]}
    (src / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (src / "__init__.py").write_text("")
    for c in comps:
        d = repo / Path(*c.split("."))
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")
    for m in rec["modules"]:
        p = repo / (m.replace(".", "/") + ".py")
        lines = [f"import {t}" for t in sorted(imports.get(m, ()))]
        lines += ["", "", f"def f_{m.rsplit('.', 1)[1]}():", f"    return {len(lines)}", ""]
        p.write_text("\n".join(lines))
    t = repo / "tests"
    t.mkdir(exist_ok=True)
    (t / "test_smoke.py").write_text("import importlib\n\nMODULES = " + repr(sorted(rec["modules"])) + "\n\n\ndef test_import_all():\n    for m in MODULES:\n        importlib.import_module(m)\n")


def rule_file(repo: Path):
    layers = "\n".join("    " + " | ".join(f"app.{ln}{k}" for k in range(2)) for ln in reversed(LAYERS))
    (repo / ".importlinter").write_text(f"[importlinter]\nroot_package = app\n\n[importlinter:contract:layers]\nname = layers\ntype = layers\nlayers =\n{layers}\n")


def make_repo(r, path: Path, n_base: int = 30, every: int = 5):
    """Materialise the baseline: every `every`-th baseline record as a commit (dated), then HEAD at the last baseline record."""
    repo = Path(path)
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    rule_file(repo)
    picks = list(range(0, n_base, every)) + [n_base - 1]
    for i in sorted(set(picks)):
        commit_step(repo, r, r.history[i], i)
    return repo


def commit_step(repo: Path, r, rec: dict, i: int):
    write_tree(repo, rec)
    _git(repo, "add", "-A")
    env_date = f"@{1451606400 + rec['day'] * 86400} +0000"
    subprocess.run(["git", "-C", str(repo), *DEV, "commit", "-q", "--allow-empty", "-m", f"{rec['event']} #{i}", f"--date={env_date}"],
                   check=True, capture_output=True, env={"GIT_COMMITTER_DATE": env_date, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    return _git(repo, "rev-parse", "HEAD").strip()


def scripted_reply(repo: Path, rec: dict, behaviour: str) -> str:
    """The model's reply for the most recent crossing in `rec`, by behaviour: correct (delete the import), fake_module
    (import a module that does not exist), adds_violation (import another higher-layer module instead), noop."""
    cross = None
    for a, b in rec["edges"]:
        la, lb = a.split(".")[1].rstrip("0123456789_bv"), b.split(".")[1].rstrip("0123456789_bv")
        if la in LAYERS and lb in LAYERS and LAYERS.index(la) < LAYERS.index(lb):
            cross = (a, b)
    if not cross:
        return "PLAN: nothing to do\nREASON: no crossing\n"
    a, b = cross
    path = a.replace(".", "/") + ".py"
    find = f"import {b}"
    if behaviour == "correct":
        repl = ""
    elif behaviour == "fake_module":
        repl = f"import {b.rsplit('.', 1)[0]}.gateway"
    elif behaviour == "adds_violation":
        other = sorted(m for m in rec["modules"] if m.split(".")[1].startswith("cli") and m != b)
        repl = f"import {other[0]}" if other else ""
    else:
        return "PLAN: leave as is\nREASON: noop\n"
    return f"PLAN: strategy 1\nFILE: {path}\n<<<<<<< FIND\n{find}\n=======\n{repl}\n>>>>>>> REPLACE\nREASON: {behaviour}\n"
