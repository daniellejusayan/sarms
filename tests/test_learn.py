"""The rule learner: evidence -> candidates -> adoption, on crafted inputs; and the blindness guarantee."""
import re
from pathlib import Path

import pandas as pd

from sarms.rules import evidence as E


def test_learner_never_reads_the_import_linter_record():
    src = (Path(__file__).resolve().parents[1] / "src" / "sarms" / "rules" / "evidence.py").read_text()
    for forbidden in ("find_il_config", "read_il_options", "importlinter", ".importlinter", "setup.cfg", "pyproject"):
        assert not re.search(re.escape(forbidden) + r"(?![^\n]*hidden)", src.split('"""', 2)[2]), forbidden


def test_run_worker_reports_missing_root_instead_of_empty_grimp_call(tmp_path):
    from sarms.mining.tree import run_worker

    bundle = {
        "session": {"root_packages": ["missing_package"]},
        "contracts_strict": [{"name": "sample", "type": "forbidden",
                              "source_modules": ["missing_package"], "forbidden_modules": ["other"]}],
        "contracts_asis": [],
    }
    result = run_worker(tmp_path, bundle)
    assert result["graph_error"].startswith("missing root(s):")


def test_components_split_a_framework_like_package():
    mods = ["app", "app.cli", "app.cli.main", "app.fw", "app.fw.a", "app.fw.a.x", "app.fw.b", "app.fw.b.y",
            "app.fw.c", "app.fw.c.z", "app.fw.flat", "app.io", "app.io.core"]
    comps = E.components_of(mods, ["app"], min_sub=3)
    assert comps == ["app.cli", "app.fw", "app.fw.a", "app.fw.b", "app.fw.c", "app.fw.flat", "app.io"]
    longest = sorted(comps, key=len, reverse=True)
    assert E.comp_of("app.fw.a.x", longest) == "app.fw.a" and E.comp_of("app.fw.flat", longest) == "app.fw.flat"


def test_layer_rank_and_lexical_veto():
    assert E.layer_rank("app.cli") == 0 and E.layer_rank("app.config") == 4 and E.layer_rank("app.zzz") is None
    ev = pd.DataFrame([
        # config -> cli absent, cli -> config in 90%: asymmetry; names agree (config below cli)
        {"comp_from": "app.config", "comp_to": "app.cli", "commits_both": 50, "ab": 0, "ba": 45, "present_now": False,
         "ever": False, "repaired": False, "ba_share": 0.9, "span_days": 900.0, "changed": True},
        # cli -> config absent but names say cli is ABOVE config and no asymmetry: vetoed
        {"comp_from": "app.cli", "comp_to": "app.config", "commits_both": 50, "ab": 0, "ba": 0, "present_now": False,
         "ever": True, "repaired": True, "ba_share": 0.0, "span_days": 900.0, "changed": True},
        # present now: never a candidate (adoption must not touch existing code)
        {"comp_from": "app.io", "comp_to": "app.cli", "commits_both": 50, "ab": 50, "ba": 0, "present_now": True,
         "ever": True, "repaired": False, "ba_share": 0.0, "span_days": 900.0, "changed": True},
        # stability only (names unknown): candidate, not adopted
        {"comp_from": "app.zzz", "comp_to": "app.yyy", "commits_both": 50, "ab": 0, "ba": 0, "present_now": False,
         "ever": False, "repaired": False, "ba_share": 0.0, "span_days": 900.0, "changed": True},
    ])
    cand = E.candidates(ev)
    got = {(r["from"], r["to"]): (r["evidence"], bool(r["adopt"])) for r in cand.to_dict("records")}
    assert got[("app.config", "app.cli")] == ("asymmetry+stability+lexical", True)
    assert ("app.cli", "app.config") not in got
    assert ("app.io", "app.cli") not in got
    assert got[("app.zzz", "app.yyy")] == ("stability", False)


def test_soft_rules_validate_and_the_differ_checks_them(tmp_path):
    from sarms.differ.diff import label
    from sarms.rules.compile import validate

    rule = {"id": "R-app-soft-1", "kind": "forbidden", "from": "app.config", "to": "app.cli",
            "source": {"type": "recovered", "id": "learned:layer", "sentence": "evidence: asymmetry+stability"},
            "compiled_at": "abcdef123456", "severity": "low", "evaluable": True}
    assert validate([rule]) == []
    bundle = {"session": {"root_packages": ["app"]}, "contracts_strict": [], "contracts_asis": []}
    res = {"results": [], "results_asis": [], "edges": [["app.config.loader", "app.cli.main"], ["app.cli.main", "app.io.core"]], "details": {}}
    out = label(bundle, [rule], res, "a" * 40, "2026-01-01T00:00:00+00:00")
    viol = [r for r in out["rows"] if r["label"] != "CONFIRMED"]
    assert len(viol) == 1 and viol[0]["rule_id"] == "R-app-soft-1" and viol[0]["direct"] and viol[0]["label"] == "INVERTED"
    # an indirect chain does not count for a recovered rule
    res2 = {"results": [], "results_asis": [], "edges": [["app.config.loader", "app.io.core"], ["app.io.core", "app.cli.main"]], "details": {}}
    assert [r for r in label(bundle, [rule], res2, "a" * 40, "2026-01-01T00:00:00+00:00")["rows"] if r["label"] != "CONFIRMED"] == []
