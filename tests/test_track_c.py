"""Track C pieces on crafted change streams: noise filter, co-change confidence, layer candidates, hotspots,
belief functions, and the hypergraph encoder on a tiny stream (skipped without torch)."""
import numpy as np
import pandas as pd
import pytest

from sarms.mining.gitlog import is_noise
from sarms.rules import belief as B
from sarms.rules import cochange as CC


def stream_frame():
    """12 commits: ui and app always change together; config changes alone; core changes with ui in 40% of its commits."""
    rows = []
    t0 = pd.Timestamp("2020-01-01", tz="UTC")

    def commit(i, files):
        for path, comp, a, d in files:
            rows.append({"idx": i, "sha": f"{i:040x}", "ts": t0 + pd.Timedelta(days=30 * i), "author": "a", "path": path,
                         "added": a, "deleted": d, "module": path[:-3].replace("/", "."), "component": comp})
    for i in range(12):
        files = [("app/ui/view.py", "app.ui", 10, 2), ("app/app/service.py", "app.app", 5, 1)]
        if i % 5 == 0:
            files.append(("app/core/logic.py", "app.core", 3, 3))
        if i % 3 == 0:
            files.append(("app/config/settings.py", "app.config", 1, 1))
        commit(i, files)
    for i in range(12, 17):  # core alone, so C(core -> ui) = 3/8 = 0.375 > 0.35
        commit(i, [("app/core/logic.py", "app.core", 2, 2), ("app/core/big.py", "app.core", 400, 300)])
    return pd.DataFrame(rows)


def test_noise_filter_keeps_source_and_drops_the_rest():
    assert not is_noise("kedro/io/core.py")
    for p in ("docs/index.md", "README.rst", "assets/logo.png", "vendor/lib/x.py", "poetry.lock", "tests/test_io.py", "kedro/io/data.json", ".github/workflows/ci.yml"):
        assert is_noise(p), p
    assert not is_noise("tests/test_io.py", keep_tests=True)


def test_confidence_is_directional_and_layer_candidates_point_up():
    df = stream_frame()
    conf = CC.confidence(df, min_support=3).set_index(["x", "y"])
    assert conf.loc[("app.ui", "app.app"), "C"] == 1.0 and conf.loc[("app.app", "app.ui"), "C"] == 1.0
    assert abs(conf.loc[("app.core", "app.ui"), "C"] - 3 / 8) < 1e-9
    assert conf.loc[("app.ui", "app.core"), "C"] == 3 / 12
    v = CC.layer_violations(conf.reset_index())
    # core (level 2) driven by ui (level 4): a candidate; ui -> app is same-direction noise at the top, not a violation
    assert ("app.core", "app.ui") in set(zip(v.x, v.y, strict=True))
    assert ("app.ui", "app.app") not in set(zip(v.x, v.y, strict=True))
    assert any(c == "app.core must not depend on app.ui" for c in v.constraint)


def test_hotspots_pareto_outlier_and_quadrants():
    df = stream_frame()
    h = CC.hotspots(df)
    assert h.index[0] == "app/core/big.py" and h.loc["app/core/big.py", "critical"]   # 5 commits x 3,500 lines
    assert h.loc["app/config/settings.py", "quadrant"] in ("normal", "quiet debt (many small edits)")
    assert set(h.quadrant) <= {"critical", "normal", "quiet debt (many small edits)", "quiet debt (few large edits)"}


def test_belief_functions_combine_conflict_and_feedback():
    m, k = B.combine(B.mass("certificate_all_axioms"), B.mass("cochange_layer"))
    assert m["correct"] > 0.85 and k == 0
    strong = B.assess(["certificate_all_axioms", "forward_held"])
    assert strong["decision"] == "enforce"
    weak = B.assess(["cochange_layer"])
    assert weak["decision"] == "monitor"
    conflicted = B.assess(["certificate_all_axioms", "imports_now"])
    assert conflicted["conflict"] > 0.6 and conflicted["decision"] != "enforce"
    s = strong
    for _ in range(3):
        s = B.feedback(s, "outcome_crossed_kept")
    assert s["decision"] in ("monitor", "drop") and s["bel"] < strong["bel"]
    s2 = B.feedback(strong, "outcome_never_crossed")
    assert s2["decision"] == "enforce" and s2["bel"] >= 0.8


def test_hgnn_trains_and_predicts_on_a_tiny_stream():
    pytest.importorskip("torch")
    from sarms.rules import hgnn as G
    df = stream_frame()
    nodes = sorted(df.component.unique())
    fit = G.train(df, nodes, days=120, epochs=50, log=lambda s: None)
    assert fit["matrix"].shape == (len(nodes), len(nodes)) and np.isfinite(fit["matrix"]).all()
    ui, app, cfg = nodes.index("app.ui"), nodes.index("app.app"), nodes.index("app.config")
    # ui and app always change together, config only every third commit: averaged over the windows the encoder saw,
    # the predicted C(ui -> app) must exceed C(ui -> config)
    P = np.mean([p[ui] for p in fit["preds"]], axis=0)
    assert P[app] > P[cfg]
    coupling, layer = G.rules_from_matrix(fit, confidence=0.5, min_support=1)
    assert isinstance(coupling, list) and isinstance(layer, list)
    fw = G.evaluate_forward(fit, confidence=0.5)
    assert "hgnn" in fw or fw["test_windows"] == 0
