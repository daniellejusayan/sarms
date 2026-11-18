"""A developer reverts a system fix: the revert is read from git, the model is told, and the same change is never drafted again."""
import pytest
from test_scenario_faulty_push import CONFIG, g, push

RULE = "R-subject-UI-must-not-import-the-database-layer-1"
BOT = "sensolens-sarms"
BAD_UI = "import os\nfrom app.db import connect\n\n\ndef show():\n    return os.getcwd() + connect()\n"
FIX_A = "import os\nfrom app.service import run as connect\n\n\ndef show():\n    return os.getcwd() + connect()\n"
REPLY_A = "FILE: app/ui.py\n<<<<<<< FIND\nfrom app.db import connect\n=======\nfrom app.service import run as connect\n>>>>>>> REPLACE\nREASON: through the service layer\n"
REPLY_B = "FILE: app/ui.py\n<<<<<<< FIND\nfrom app.db import connect\n=======\nfrom app import service\n\nconnect = service.run\n>>>>>>> REPLACE\nREASON: another route\n"


def base_repo(tmp_path):
    """main: baseline, then the developer's violation. Returns (repo, rules dir, rms dir)."""
    from sarms.rules.compile import compile_repo

    r = tmp_path / "subject"
    (r / "app").mkdir(parents=True)
    (r / ".importlinter").write_text(CONFIG)
    (r / "app" / "__init__.py").write_text("")
    (r / "app" / "db.py").write_text("def connect():\n    return 'db'\n")
    (r / "app" / "service.py").write_text("from app.db import connect\n\n\ndef run():\n    return connect()\n")
    (r / "app" / "ui.py").write_text("import os\n\n\ndef show():\n    return os.getcwd()\n")
    g(r, "init", "-q", "-b", "main")
    push(r, "baseline", {})
    rules = tmp_path / "rms" / "rules"
    compile_repo(r, out_dir=rules, repo_label="subject")
    push(r, "feature: ui reads db", {"app/ui.py": BAD_UI})
    return r, rules, tmp_path / "rms"


def system_fix(repo):
    return push(repo, f"sarms: app.ui must not import app.db ({RULE})", {"app/ui.py": FIX_A}, author=BOT)


@pytest.fixture
def reverted_repo(tmp_path):
    """main: baseline, the developer's violation, the system's fix A, the developer's revert of it."""
    r, rules, rms = base_repo(tmp_path)
    fix = system_fix(r)
    g(r, "revert", "--no-edit", fix)
    return r, rules, rms, fix


def replies(tmp_path, *texts):
    d = tmp_path / "replies"
    d.mkdir()
    for n, t in enumerate(texts, 1):
        (d / f"{n}.txt").write_text(t)
    return d


def test_reverted_fix_is_read_from_git(reverted_repo):
    from sarms.kernel.apply import reverted_fixes
    from sarms.mining.common import patch_fingerprint

    repo, _, _, fix = reverted_repo
    (x,) = reverted_fixes(repo)
    assert (x["fix"], x["rule_id"], x["edge_from"], x["edge_to"]) == (fix, RULE, "app.ui", "app.db")
    assert x["fingerprint"] == patch_fingerprint(x["patch"]) == frozenset({
        ("app/ui.py", "-", "from app.db import connect"), ("app/ui.py", "+", "from app.service import run as connect")})


def test_the_same_fix_is_refused_and_another_route_is_taken(reverted_repo, tmp_path):
    from sarms.kernel.apply import reverted_fixes
    from sarms.mining.common import patch_fingerprint
    from sarms.synthesize.draft import draft_one

    repo, rules, rms, fix = reverted_repo
    undone = reverted_fixes(repo)
    result, d = draft_one(repo, rules_dir=rules, rms_dir=rms, base_url=f"fake:{replies(tmp_path, REPLY_A, REPLY_B)}",
                          reverted=undone, log=lambda s: None)
    assert "A DEVELOPER REVERTED AN EARLIER FIX" in (d / "prompt1.txt").read_text()
    assert result["attempts"][0]["reverted_fix"] == fix and "pass" not in result["attempts"][0]
    assert result["verdict"] == "draft_passes" and result["tries"] == 2
    assert patch_fingerprint((d / "patch.diff").read_text()) != undone[0]["fingerprint"]


def test_a_model_that_only_repeats_the_reverted_fix_gets_nothing_applied(reverted_repo, tmp_path):
    from sarms.kernel.apply import reverted_fixes
    from sarms.synthesize.draft import draft_one

    repo, rules, rms, fix = reverted_repo
    result, d = draft_one(repo, rules_dir=rules, rms_dir=rms, base_url=f"fake:{replies(tmp_path, REPLY_A)}",
                          reverted=reverted_fixes(repo), log=lambda s: None)
    assert result["verdict"] == "escalate" and result["tries"] == 3
    assert all(a["reverted_fix"] == fix for a in result["attempts"]) and not (d / "patch.diff").exists()


# ---- demotion: only when a human fixed a change the system made ---------------------------------------------

def test_a_human_re_edit_of_the_systems_lines_counts_as_human_fixed(tmp_path):
    from sarms.kernel.apply import human_fixed

    repo, _, _ = base_repo(tmp_path)
    fix = system_fix(repo)
    by = push(repo, "dev: do it my way", {"app/ui.py": BAD_UI.replace("from app.db import connect", "from app import service\n\nconnect = service.run")})
    (x,) = human_fixed(repo)
    assert (x["fix"], x["by"], x["how"], x["rule_id"]) == (fix, by, "re-edit", RULE)


def test_nothing_else_demotes(tmp_path):
    from sarms.kernel.apply import human_fixed, revert_own_commit

    repo, _, _ = base_repo(tmp_path)
    assert human_fixed(repo) == []                                              # a violation alone
    fix = system_fix(repo)
    assert human_fixed(repo) == []                                              # a fix nobody touched
    push(repo, "dev: unrelated work in the same file", {"app/ui.py": FIX_A + "\n\ndef other():\n    return 1\n"})
    push(repo, "dev: unrelated work elsewhere", {"app/db.py": "def connect():\n    return 'db'  # tuned\n"})
    assert human_fixed(repo) == []                                              # human edits that leave the fix alone
    revert_own_commit(repo, fix, log=lambda s: None)
    assert human_fixed(repo) == []                                              # the system's own revert


def test_a_demoted_rule_is_recorded_and_no_longer_fixed(reverted_repo, tmp_path):
    import yaml

    from sarms import product
    from sarms.ledger.hypergraph import Ledger

    repo, _, _, fix = reverted_repo
    out = tmp_path / "product-rms"
    model = f"fake:{replies(tmp_path, REPLY_B)}"                                # a route that would pass, if it were asked
    for _ in range(2):                                                          # the second run must not record it twice
        report = product.run(repo, out, base_url=model, every=1, log=lambda s: None)
        assert report["violations"] == 1 and report["enforceable"] == 0 and report["demoted"] == [RULE]
        (o,) = report["outcomes"]
        assert o["outcome"] == "left_standing" and o["why"].startswith("rule demoted: a human fixed the system's change (revert")
    acts = Ledger(out / "sthg").actions()
    assert list(acts[acts.kind == "demote"].rule_id) == [RULE] and len(acts[acts.kind == "revert_seen"]) == 1
    assert not set(acts.kind) & {"draft", "gate", "apply"}
    rules = [x for f in (out / "rules-active").glob("*.yaml") if f.name != "acknowledged.yaml" for x in yaml.safe_load(f.read_text()) or []]
    assert [x["severity"] for x in rules if x["id"] == RULE] == ["low"]
    assert "from app.db import connect" in g(repo, "show", "HEAD:app/ui.py")     # the human's version stands
