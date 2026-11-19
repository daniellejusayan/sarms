"""The product end to end on the tiny repository: mine -> ledger -> certificates -> run (check, draft, gate, kernel) -> status."""
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
    for i in range(6):   # some history so the ledger has more than one snapshot
        push(r, f"work {i}", {"app/service.py": f"from app.db import connect\n\n\ndef run():\n    return connect()  # {i}\n"})
    return r


def test_status_requires_finished_mine(tmp_path):
    from sarms import product

    out = tmp_path / "rms"
    sthg = out / "sthg"
    sthg.mkdir(parents=True)
    (sthg / "ledger.duckdb").touch()

    with pytest.raises(SystemExit, match="run `sarms mine` first"):
        product.status(out, log=lambda s: None)


def test_mine_certify_run_status(repo, tmp_path):
    from sarms import product
    from sarms.ledger.hypergraph import Ledger

    out = tmp_path / "rms"
    led, meta = product.mine(repo, out, every=1, log=lambda s: None)
    s = led.summary()
    assert s["hyperedges"] == 7 and s["graph_snapshots"] == 7 and s["components"] >= 1
    rules, changes, verified = product.certify_ledger(repo, out, log=lambda s: None)
    assert verified and isinstance(rules, list)          # a two-week synthetic history certifies nothing (A2 needs 3 years): still VERIFIED
    # a developer pushes faulty code; run drafts (scripted model), gates and lands the fix on the side branch
    push(repo, "feature: ui reads db", {"app/ui.py": "import os\nfrom app.db import connect\n\n\ndef show():\n    return os.getcwd() + connect()\n"})
    replies = tmp_path / "replies"
    replies.mkdir()
    (replies / "1.txt").write_text(textwrap.dedent("""\
        FILE: app/ui.py
        <<<<<<< FIND
        from app.db import connect
        =======
        from app.service import run as connect
        >>>>>>> REPLACE
        REASON: through the service layer
        """))
    report = product.run(repo, out, base_url=f"fake:{replies}", tests="python -m pytest -q -p no:cacheprovider", log=lambda s: None)
    assert report["violations"] == 1 and report["enforceable"] == 1
    assert report["outcomes"][0]["outcome"] == "applied"
    fix = report["outcomes"][0]["commit"]
    assert g(repo, "log", "-1", "--format=%an", fix).strip() == "sensolens-sarms"
    assert g(repo, "status", "--short").strip() == ""
    acts = Ledger(out / "sthg").actions()
    assert ["draft", "gate", "apply"] == [k for k in acts.kind if k in ("draft", "gate", "apply")]
    lines = product.status(out, log=lambda s: None)
    assert any("certified" in ln for ln in lines) and any("apply" in ln for ln in lines)
    assert json.loads((out / "run.json").read_text())["ledger"]["actions"] >= 4


def test_git_url_resolves_to_named_checkout(tmp_path):
    from sarms.mining.common import repo_name, resolve_repo_input

    assert repo_name("https://github.com/kedro-org/kedro.git") == "kedro"
    assert repo_name("git@github.com:kedro-org/kedro.git") == "kedro"
    assert resolve_repo_input(str(tmp_path)) == tmp_path.resolve()
