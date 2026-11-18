"""Viability gates on crafted learner output, and the template drafter on plain `import a.b.c` lines."""
from pathlib import Path

import pandas as pd

from sarms.rules import viability as V
from sarms.synthesize.draft import template_draft


def write_learn_dir(d: Path):
    comps = ["app.config", "app.cli", "app.io", "app.web", "app.zzz", "app.db"]
    sizes = {"app.config": 4, "app.cli": 5, "app.io": 3, "app.web": 1, "app.zzz": 3, "app.db": 3}
    hc = pd.DataFrame([{"idx": i, "comp": c, "n_modules": sizes[c]} for i in (0, 1) for c in comps])
    # today: config is the stable bottom; cli the top; io, db, zzz in between
    edges = [("app.cli", "app.config"), ("app.cli", "app.io"), ("app.cli", "app.db"), ("app.io", "app.config"), ("app.db", "app.config"),
             ("app.web", "app.cli"), ("app.zzz", "app.config"), ("app.zzz", "app.cli")]
    he = pd.DataFrame([{"idx": i, "comp_from": a, "comp_to": b, "n": 2} for i in (0, 1) for a, b in edges])
    hc.to_parquet(d / "history_components.parquet", index=False)
    he.to_parquet(d / "history_edges.parquet", index=False)
    ev = pd.DataFrame([
        # passes every strict gate: never crossed, 100% the other way, 4 years, shared neighbours (io, zzz, web via cli)
        {"comp_from": "app.config", "comp_to": "app.cli", "commits_both": 60, "ab": 0, "ba": 60, "present_now": False,
         "ever": False, "repaired": False, "ba_share": 1.0, "span_days": 1500.0, "changed": True},
        # crossed once and repaired: loose never_crossed holds, strict never_crossed_at_all fails
        {"comp_from": "app.config", "comp_to": "app.io", "commits_both": 60, "ab": 1, "ba": 55, "present_now": False,
         "ever": True, "repaired": True, "ba_share": 0.92, "span_days": 1500.0, "changed": True},
        # too young for strict (2.5 years), otherwise fine
        {"comp_from": "app.io", "comp_to": "app.cli", "commits_both": 40, "ab": 0, "ba": 40, "present_now": False,
         "ever": False, "repaired": False, "ba_share": 1.0, "span_days": 900.0, "changed": True},
    ])
    ev.to_csv(d / "evidence.csv", index=False)
    cand = pd.DataFrame([
        {"from": "app.config", "to": "app.cli", "rule_kind": "layer", "evidence": "asymmetry+stability+lexical", "n_evidence": 3, "adopt": True,
         "ba_share": 1.0, "commits_both": 60, "span_days": 1500.0},
        {"from": "app.config", "to": "app.io", "rule_kind": "layer", "evidence": "asymmetry+repair+stability", "n_evidence": 3, "adopt": True,
         "ba_share": 0.92, "commits_both": 60, "span_days": 1500.0},
        {"from": "app.io", "to": "app.cli", "rule_kind": "layer", "evidence": "asymmetry+stability", "n_evidence": 2, "adopt": True,
         "ba_share": 1.0, "commits_both": 40, "span_days": 900.0},
    ])
    cand.to_csv(d / "candidates.csv", index=False)


def test_strict_gates_single_out_the_definitive_rule(tmp_path):
    write_learn_dir(tmp_path)
    df = V.score(tmp_path, "adopted").set_index(["from", "to"])
    assert set(V.STRICT_GATES) == {c[2:] for c in df.columns if c.startswith("s_")}
    a = df.loc[("app.config", "app.cli")]
    assert a.tier == "strong" and bool(a.strict) and a.failed_gates == ""
    b = df.loc[("app.config", "app.io")]
    assert bool(b.never_crossed) and not bool(b.s_never_crossed_at_all) and not bool(b.strict)
    assert "never_crossed_at_all" in b.failed_gates
    c = df.loc[("app.io", "app.cli")]
    assert not bool(c.strict) and "three_years" in c.failed_gates
    # strict rules are a subset of the loose-strong tier here
    assert set(df[df.strict].index) <= set(df[df.tier == "strong"].index)


def test_template_deletes_a_plain_dotted_import_whose_path_is_unused(tmp_path):
    pkg = tmp_path / "app" / "pipeline"
    pkg.mkdir(parents=True)
    (tmp_path / "app" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    # `import app.runner.thread` binds `app`, which the file uses elsewhere; the dotted path itself is unused
    (pkg / "node.py").write_text("import app.runner.thread\nfrom app.config import load\n\n\ndef run():\n    return load()\n")
    violation = {"lines": [{"edge": "app.pipeline.node->app.runner.thread", "line_number": 1, "line_contents": "import app.runner.thread"}]}
    blocks, reason = template_draft(tmp_path, violation, [str(tmp_path)])
    assert blocks == [("app/pipeline/node.py", "import app.runner.thread", "")], reason
    # when the dotted path is used, the template refuses rather than breaking the file
    (pkg / "node.py").write_text("import app.runner.thread\n\n\ndef run():\n    return app.runner.thread.go()\n")
    blocks, reason = template_draft(tmp_path, violation, [str(tmp_path)])
    assert blocks is None and "still used" in reason
