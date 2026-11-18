"""Gate G0-G8 and the kernel on the tiny repository: an accepted record lands as a sensolens-sarms commit that reverts
in one step; a draft that touches a test fails G0; a tampered record is refused."""
import json
import textwrap

import pytest
from test_scenario_faulty_push import CONFIG, g, push


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
    push(r, "baseline", {})
    return r


def test_gate_accepts_a_real_fix_and_the_kernel_lands_it(repo, tmp_path):
    from sarms.kernel.apply import apply_decision, developer_reverts, revert_own_commit
    from sarms.ledger.hypergraph import Ledger
    from sarms.rules.compile import compile_repo
    from sarms.synthesize.draft import draft_one
    from sarms.verify.gate import run_gate, verify_signature

    rms = tmp_path / "rms"
    rules_dir = rms / "rules"
    compile_repo(repo, out_dir=rules_dir, repo_label="app")
    sha = push(repo, "feature: ui reads db", {
        "app/ui.py": "import os\nfrom app.db import connect\n\n\ndef show():\n    return os.getcwd() + connect()\n",
        "tests/test_ui.py": "from app.ui import show\n\n\ndef test_show():\n    assert show().endswith('db')\n"})
    replies = tmp_path / "replies"
    replies.mkdir()
    (replies / "1.txt").write_text(textwrap.dedent("""\
        FILE: app/ui.py
        <<<<<<< FIND
        from app.db import connect
        =======
        from app.service import run as connect
        >>>>>>> REPLACE
        REASON: reach the database through the service layer
        """))
    result, d = draft_one(repo, rules_dir=rules_dir, rms_dir=rms, base_url=f"fake:{replies}",
                          tests="python -m pytest -q -p no:cacheprovider tests", log=lambda s: None)
    assert result["verdict"] == "draft_passes"
    record, path = run_gate(repo, sha, rules_dir, d, rms, tests="python -m pytest -q -p no:cacheprovider", log=lambda s: None)
    assert record["verdict"] == "accept", [(x["line"], x["pass"], x["evidence"]) for x in record["gate"]]
    assert [x["pass"] for x in record["gate"]] == [True] * 9
    from sarms.verify.gate import _salt
    assert verify_signature(record, _salt(repo))
    led = Ledger(rms / "sthg")
    fix = apply_decision(repo, path, d, ledger=led, log=lambda s: None)
    assert fix and g(repo, "log", "-1", "--format=%an", fix).strip() == "sensolens-sarms"
    assert g(repo, "rev-parse", "sarms/fixes").strip() == fix
    assert "Decision record" in g(repo, "log", "-1", "--format=%b", fix)
    assert json.loads(path.read_text())["applied_commit"] == fix
    assert g(repo, "status", "--short").strip() == ""                      # the developer's checkout untouched
    # the developer's view: main still has the violation; the fix is one revert away on its branch
    g(repo, "checkout", "-q", "sarms/fixes")
    new = revert_own_commit(repo, fix, ledger=led, log=lambda s: None)
    assert g(repo, "diff", sha, "--", ".").strip() == ""
    assert developer_reverts(repo) == [(new, fix)]
    acts = led.actions()
    assert list(acts.kind) == ["apply", "revert_seen"]


def test_gate_stops_at_g0_when_the_draft_edits_a_test_and_refuses_tampering(repo, tmp_path):
    from sarms.kernel.apply import apply_decision
    from sarms.rules.compile import compile_repo
    from sarms.verify.gate import run_gate

    rms = tmp_path / "rms"
    rules_dir = rms / "rules"
    compile_repo(repo, out_dir=rules_dir, repo_label="app")
    sha = push(repo, "feature", {"app/ui.py": "from app.db import connect\n\n\ndef show():\n    return connect()\n"})
    d = rms / "drafts" / "fake"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"sha": sha, "class": "llm_draft", "prompt_hash": "x",
                                               "violation": {"rule_id": "R-app-UI-must-not-import-the-database-layer-1", "contract": "c", "edge_from": "app.ui", "edge_to": "app.db"}}))
    (d / "patch.diff").write_text(textwrap.dedent("""\
        diff --git a/tests/test_app.py b/tests/test_app.py
        --- a/tests/test_app.py
        +++ b/tests/test_app.py
        @@ -1,5 +1,5 @@
         from app.service import run
         
         
         def test_run():
        -    assert run() == 'db'
        +    assert True
        """))
    record, path = run_gate(repo, sha, rules_dir, d, rms, tests="python -m pytest -q -p no:cacheprovider", log=lambda s: None)
    assert record["verdict"] == "escalate" and record["stopped_at"] == "G0"
    assert [x["pass"] for x in record["gate"]] == [False] + [None] * 8
    assert apply_decision(repo, path, d, log=lambda s: None) is None
    rec = json.loads(path.read_text())
    rec["verdict"] = "accept"
    path.write_text(json.dumps(rec))
    with pytest.raises(PermissionError):
        apply_decision(repo, path, d, log=lambda s: None)
