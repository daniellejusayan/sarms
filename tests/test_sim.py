"""The simulator: generator determinism, ledger replay, level-1 invariants on a few scenarios, one level-2 run."""
import tempfile
from pathlib import Path

import pytest

from sarms.ledger.hypergraph import Ledger
from sarms.sim.generator import NAMED, all_scenarios, generate, to_frames
from sarms.sim.harness import run_level1


def test_generator_is_deterministic_and_grammar_is_enumerated():
    a, b = generate(2569, NAMED["crossing_then_repair"]), generate(2569, NAMED["crossing_then_repair"])
    assert [r["event"] for r in a.history] == [r["event"] for r in b.history]
    assert [sorted(r["edges"]) for r in a.history] == [sorted(r["edges"]) for r in b.history]
    scen = all_scenarios()
    assert set(NAMED) <= set(scen) and len(scen) > 150          # named + every ordered pair of two events


def test_ledger_replay_is_deterministic(tmp_path):
    he, hc, st = to_frames(generate(2570, NAMED["baseline"]))
    hashes = []
    for k in range(2):
        led = Ledger(tmp_path / f"l{k}")
        led.ingest_stream(st)
        led.ingest_graph(he, hc)
        hashes.append(led.state_hash())
        s = led.summary()
        assert s["hyperedges"] == len(st.sha.unique()) and s["graph_snapshots"] == hc.idx.nunique()
        led.close()
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("name", ["baseline", "crossing_then_repair", "crossing_kept", "bulk_migration", "package_split", "revert_storm", "cycle"])
def test_level1_scenarios_never_break_an_invariant(name):
    res = run_level1(2569, name, NAMED[name], Path(tempfile.mkdtemp()))
    assert res["label"] != "CRITICAL", res["issues"]
    assert res["rules_at_start"] > 0


def test_level2_gate_accepts_correct_and_refuses_bad_replies():
    from sarms.sim.harness import run_level2
    ok = run_level2(2569, "crossing_then_repair", NAMED["crossing_then_repair"], Path(tempfile.mkdtemp()), "correct")
    bad = run_level2(2569, "crossing_then_repair", NAMED["crossing_then_repair"], Path(tempfile.mkdtemp()), "adds_violation")
    # the simulated developer's next push rewrites the tree and puts the removed import back: a human fixed the system's
    # change, so the rule is demoted and the crossing is recorded, not fixed a second time (no revert fight)
    assert not ok["issues"] and ok["transitions"].get("applied", 0) == 1 and ok["transitions"].get("left_standing", 0) == 1
    assert not bad["issues"] and bad["transitions"].get("applied", 0) == 0 and bad["transitions"].get("left_standing", 0) >= 1
