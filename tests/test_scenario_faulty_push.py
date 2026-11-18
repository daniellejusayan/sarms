"""The whole scenario, as a sequence of Monitor events: a developer pushes faulty code, the loop detects, drafts,
gates, records; a second developer's clean push stays silent; a fake fix is refused; the belief function reacts to
outcomes; a certified rule retires when the graph no longer supports it; the change stream raises a hotspot.

Every step uses the real code (compile, Differ, draft with the fake model backend, gate, replay/ledger, belief,
git log stream, hotspots). The only simulated parts are the pushes themselves and the model's replies.
"""
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from sarms.rules import belief as B

CONFIG = """\
[importlinter]
root_package = app

[importlinter:contract:ui-db]
name = UI must not import the database layer
type = forbidden
source_modules = app.ui
forbidden_modules = app.db
allow_indirect_imports = True

[importlinter:contract:layers]
name = Service above DB
type = layers
layers =
    app.service
    app.db
"""


def g(repo, *a, author="Dev A"):
    return subprocess.run(["git", "-C", str(repo), "-c", f"user.name={author}", "-c", "user.email=dev@example.invalid", *a],
                          check=True, capture_output=True, text=True).stdout


def push(repo, message, files: dict, author="Dev A"):
    """A developer's push: write files, commit on main. Returns the commit sha."""
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    g(repo, "add", "-A", author=author)
    g(repo, "commit", "-qm", message, author=author)
    return g(repo, "rev-parse", "HEAD", author=author).strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "subject"
    (r / "app").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / ".importlinter").write_text(CONFIG)
    (r / "app" / "__init__.py").write_text("")
    (r / "app" / "db.py").write_text("def connect():\n    return 'db'\n")
    (r / "app" / "service.py").write_text("from app.db import connect\n\n\ndef run():\n    return connect()\n")
    (r / "app" / "ui.py").write_text("import os\n\n\ndef show():\n    return os.getcwd()\n")
    (r / "tests" / "test_app.py").write_text("from app.service import run\n\n\ndef test_run():\n    assert run() == 'db'\n")
    g(r, "init", "-q", "-b", "main")
    push(r, "baseline: clean architecture", {})
    return r


class Monitor:
    """Collects events the way the ledger would, in order."""

    def __init__(self):
        self.events = []

    def emit(self, stage, **data):
        self.events.append({"stage": stage, **data})
        return self.events[-1]

    def stages(self):
        return [e["stage"] for e in self.events]


def test_developer_injects_faulty_code_and_the_loop_reacts(repo, tmp_path):
    from sarms.differ.diff import diff_commit, label
    from sarms.differ.replay import replay
    from sarms.ledger.build import build
    from sarms.mining.gitlog import stream, to_frame
    from sarms.rules import cochange as CC
    from sarms.rules.compile import compile_repo
    from sarms.synthesize.draft import draft_one

    mon = Monitor()
    rms = tmp_path / "rms"
    rules_dir = rms / "rules"
    _, rules, errors = compile_repo(repo, out_dir=rules_dir, repo_label="app")
    assert not errors
    mon.emit("RULES_COMPILED", n=len(rules))

    # ---- baseline: the Differ sees a clean repository
    res = diff_commit(repo, rules_dir=rules_dir, out_dir=rms)
    assert [r for r in res["rows"] if r["label"] != "CONFIRMED"] == []
    mon.emit("CHECK", sha=res["sha"][:12], violations=0)

    # ---- push 1: Dev A injects faulty code: the UI calls the database directly, and a test depends on it
    sha1 = push(repo, "feature: ui shows the connection", {
        "app/ui.py": "import os\nfrom app.db import connect\n\n\ndef show():\n    return os.getcwd() + connect()\n",
        "tests/test_ui.py": "from app.ui import show\n\n\ndef test_show():\n    assert show().endswith('db')\n",
    })
    res = diff_commit(repo, rules_dir=rules_dir, out_dir=rms)
    viol = [r for r in res["rows"] if r["label"] != "CONFIRMED"]
    assert len(viol) == 1 and viol[0]["edge_from"] == "app.ui" and viol[0]["edge_to"] == "app.db" and viol[0]["direct"]
    assert viol[0]["lines"][0]["line_number"] == 2 and viol[0].get("enforce", True)
    ev = mon.emit("VIOLATION_DETECTED", sha=sha1[:12], rule=viol[0]["rule_id"], edge="app.ui -> app.db",
                  file="app/ui.py", line=viol[0]["lines"][0]["line_number"], label=viol[0]["label"])
    assert ev["label"] == "ADDITIONAL"

    # ---- the draft step: one model call. First reply is a fake fix (imports a module that does not exist);
    #      the gate refuses it. Second reply reroutes through the service layer; the gate passes it.
    replies = tmp_path / "replies"
    replies.mkdir()
    (replies / "1.txt").write_text(textwrap.dedent("""\
        FILE: app/ui.py
        <<<<<<< FIND
        from app.db import connect
        =======
        from app.gateway import connect
        >>>>>>> REPLACE
        REASON: use a gateway
        """))
    (replies / "2.txt").write_text(textwrap.dedent("""\
        FILE: app/ui.py
        <<<<<<< FIND
        from app.db import connect
        =======
        from app.service import run as connect
        >>>>>>> REPLACE
        REASON: strategy 2, reach the database through the service layer
        """))
    result, d = draft_one(repo, rules_dir=rules_dir, rms_dir=rms, base_url=f"fake:{replies}",
                          tests="python -m pytest -q -p no:cacheprovider tests", log=lambda s: None)
    first = result["attempts"][0]
    assert first.get("pass") is not True and any("gateway" in p or "resolve" in p.lower() or "exist" in p.lower() for p in first.get("problems", [])), first
    mon.emit("DRAFT_REFUSED", attempt=1, why=first.get("problems", [""])[0][:80])
    assert result["verdict"] == "draft_passes" and result["tries"] == 2
    mon.emit("DRAFT_PASSES_GATE", attempt=2, g1=result["attempts"][1]["g1"]["pass"], g3=result["attempts"][1]["g3"]["pass"])
    assert "from app.db import connect" in (repo / "app" / "ui.py").read_text()   # the developer's checkout is untouched
    assert (d / "patch.diff").exists() and (d / "result.json").exists()
    mon.emit("KERNEL_WOULD_COMMIT", patch=str(d / "patch.diff"), revertible_in_one_commit=True)

    # ---- push 2: Dev B touches only the database layer: the loop must stay silent
    sha2 = push(repo, "db: connection timeout", {"app/db.py": "TIMEOUT = 5\n\n\ndef connect():\n    return 'db'\n"}, author="Dev B")
    res = diff_commit(repo, rules_dir=rules_dir, out_dir=rms)
    new = [r for r in res["rows"] if r["label"] != "CONFIRMED"]
    assert len(new) == 1   # the same standing violation from push 1, nothing new
    mon.emit("CHECK", sha=sha2[:12], violations=1, new=0)

    # ---- push 3: the system's fix lands (as the kernel would commit it), then the ledger closes the episode
    sha3 = push(repo, "sarms: reroute ui through the service layer", {
        "app/ui.py": "import os\nfrom app.service import run as connect\n\n\ndef show():\n    return os.getcwd() + connect()\n"},
        author="sensolens-sarms")
    replay(repo, "main", rms, rules_dir, log=lambda s: None)
    led = build(rms, "t/app")
    ep = led["episodes"]
    assert len(ep) == 1 and bool(ep.repaired.iloc[0]) and ep.gap_commits.iloc[0] == 2
    mon.emit("LEDGER_EPISODE_CLOSED", sha=sha3[:12], gap_commits=int(ep.gap_commits.iloc[0]), repaired=True)

    # ---- belief function: the rule's standing after this outcome, and after a developer revert
    state = B.assess(["written_forbids", "certificate_all_axioms"])
    assert state["decision"] == "enforce"
    state = B.feedback(state, "outcome_crossed_repaired")
    mon.emit("BELIEF", rule="app.ui must not depend on app.db", bel=state["bel"], decision=state["decision"])
    assert state["decision"] == "enforce"
    reverted = state
    for _ in range(4):
        reverted = B.feedback(reverted, "outcome_fix_reverted")
    mon.emit("BELIEF", rule="app.ui must not depend on app.db", bel=reverted["bel"], decision=reverted["decision"], after="4 developer reverts")
    assert reverted["decision"] != "enforce" and reverted["bel"] < state["bel"]

    # ---- a certified (learned) rule retires when the graph no longer supports it
    learned = {"id": "R-app-certified-1", "kind": "forbidden", "from": "app.db", "to": "app.ui",
               "source": {"type": "recovered", "id": "learned:certified", "sentence": "A1-A6"},
               "compiled_at": "abcdef123456", "severity": "medium", "evaluable": True}
    bundle = {"session": {"root_packages": ["app"], "components": ["app.db", "app.ui", "app.service", "app.api"]},
              "contracts_strict": [], "contracts_asis": []}
    # db reaches up into ui while db itself has become the most unstable node: A1 fails -> recorded, not fixed
    edges = [["app.db.x", "app.ui.y"], ["app.db.x", "app.service.s"], ["app.db.x", "app.api.a"], ["app.api.a", "app.ui.y"], ["app.service.s", "app.ui.y"]]
    out = label(bundle, [learned], {"results": [], "results_asis": [], "edges": edges, "details": {}}, "b" * 40, "2026-01-01T00:00:00+00:00")
    v = [r for r in out["rows"] if r["label"] != "CONFIRMED"]
    assert len(v) == 1 and v[0]["enforce"] is False and "A1 no longer holds" in v[0]["retired_reason"]
    mon.emit("RULE_RETIRED", rule=learned["id"], why=v[0]["retired_reason"])

    # ---- the change stream: after a burst of edits to ui.py it becomes the hotspot
    for i in range(6):
        push(repo, f"ui tweak {i}", {"app/ui.py": (repo / "app" / "ui.py").read_text() + f"\n\ndef extra_{i}():\n    return {i}\n" * 3})
    df = to_frame(stream(repo), repo_dir=repo)
    hot = CC.hotspots(df)
    assert hot.index[0] == "app/ui.py" and bool(hot.loc["app/ui.py", "critical"])
    mon.emit("HOTSPOT", file="app/ui.py", commits=int(hot.loc["app/ui.py", "commits"]), churn=int(hot.loc["app/ui.py", "churn"]), quadrant=hot.loc["app/ui.py", "quadrant"])

    # ---- the event log is the audit trail
    (rms / "scenario-events.json").write_text(json.dumps(mon.events, indent=1, default=str))
    assert mon.stages() == ["RULES_COMPILED", "CHECK", "VIOLATION_DETECTED", "DRAFT_REFUSED", "DRAFT_PASSES_GATE", "KERNEL_WOULD_COMMIT",
                            "CHECK", "LEDGER_EPISODE_CLOSED", "BELIEF", "BELIEF", "RULE_RETIRED", "HOTSPOT"]
    # nobody was asked anything, and the developer's own checkout was never modified by the system
    assert not any(e.get("asked_human") for e in mon.events)
    assert g(repo, "status", "--short").strip() == ""
    assert Path(result["attempts"][1]["files"][0]).name == "ui.py" if result["attempts"][1].get("files") else True
