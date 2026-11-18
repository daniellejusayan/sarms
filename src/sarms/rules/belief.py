"""Track C, part 3: Dempster-Shafer belief functions for enforcing, monitoring or dropping a candidate rule, with
an adaptive feedback loop.

Frame of discernment Theta = {correct, wrong}. Every source of evidence about a candidate constraint
"x must not depend on y" gives a mass function m over {correct}, {wrong}, Theta (the last is honest ignorance):

  structural certificate  A1-A6 verdicts from sarms.rules.certify      all hold -> m(correct)=0.85
  import graph now        x imports y today                             -> m(wrong)=0.80 (the team evidently wants it)
  co-change layer signal  C(x -> y) > 0.35 with level(x) < level(y)     -> m(correct)=0.35 (weak, behavioural)
  hgnn prediction         predicted C(x -> y) >= 0.98                   -> m(correct)=0.40
  forward validation      certified on the past, never crossed later    -> m(correct)=0.50; crossed and kept -> m(wrong)=0.70
  written rule            the team allows x -> y                        -> m(wrong)=0.95; the team forbids it -> m(correct)=0.95

Sources are combined with Dempster's rule (conflict renormalised; the conflict mass K is reported, high K means
the sources disagree and the decision should be 'monitor'). Decision:
  enforce  if Bel(correct) >= ENFORCE and K < CONFLICT_MAX
  drop     if Pl(correct)  <  DROP
  monitor  otherwise
Feedback: after each window, the outcome for an enforced rule (never crossed / crossed then repaired / crossed and
kept / fix reverted) is a new source, combined after discounting the previous belief by ALPHA (old evidence fades),
so the belief tracks the repository as it grows. Everything is a plain dict; no learning here, only arithmetic.
"""
from __future__ import annotations

ENFORCE = 0.80
DROP = 0.50
CONFLICT_MAX = 0.60
ALPHA = 0.90

SOURCES = {
    "certificate_all_axioms": ("correct", 0.85),
    "certificate_partial": ("correct", 0.30),
    "imports_now": ("wrong", 0.80),
    "cochange_layer": ("correct", 0.35),
    "hgnn_predicted": ("correct", 0.40),
    "forward_held": ("correct", 0.50),
    "forward_crossed_repaired": ("correct", 0.40),
    "forward_crossed_kept": ("wrong", 0.70),
    "written_allows": ("wrong", 0.95),
    "written_forbids": ("correct", 0.95),
    "outcome_never_crossed": ("correct", 0.30),
    "outcome_crossed_repaired": ("correct", 0.40),
    "outcome_crossed_kept": ("wrong", 0.70),
    "outcome_fix_reverted": ("wrong", 0.80),
}


def mass(source: str, strength: float | None = None) -> dict:
    """A simple support function: all mass on one focal element, the rest on Theta."""
    focal, s = SOURCES[source]
    s = strength if strength is not None else s
    return {focal: s, "theta": 1.0 - s}


def vacuous() -> dict:
    return {"theta": 1.0}


def discount(m: dict, alpha: float = ALPHA) -> dict:
    """Shafer discounting: keep alpha of the evidence, move the rest to ignorance."""
    out = {k: v * alpha for k, v in m.items() if k != "theta"}
    out["theta"] = 1.0 - sum(out.values())
    return out


def combine(m1: dict, m2: dict) -> tuple[dict, float]:
    """Dempster's rule on the frame {correct, wrong}. Returns (combined mass, conflict K)."""
    a = {"correct": m1.get("correct", 0.0), "wrong": m1.get("wrong", 0.0), "theta": m1.get("theta", 0.0)}
    b = {"correct": m2.get("correct", 0.0), "wrong": m2.get("wrong", 0.0), "theta": m2.get("theta", 0.0)}
    K = a["correct"] * b["wrong"] + a["wrong"] * b["correct"]
    if K >= 1.0 - 1e-12:
        return {"correct": 0.5, "wrong": 0.5, "theta": 0.0}, 1.0
    z = 1.0 - K
    c = (a["correct"] * b["correct"] + a["correct"] * b["theta"] + a["theta"] * b["correct"]) / z
    w = (a["wrong"] * b["wrong"] + a["wrong"] * b["theta"] + a["theta"] * b["wrong"]) / z
    t = (a["theta"] * b["theta"]) / z
    return {"correct": c, "wrong": w, "theta": t}, K


def combine_all(masses: list[dict]) -> tuple[dict, float]:
    m, kmax = vacuous(), 0.0
    for x in masses:
        m, k = combine(m, x)
        kmax = max(kmax, k)
    return m, kmax


def belief(m: dict) -> dict:
    """Bel and Pl of 'correct'. Bel = mass fully committed; Pl = everything not committed against."""
    bel = m.get("correct", 0.0)
    pl = m.get("correct", 0.0) + m.get("theta", 0.0)
    return {"bel": round(bel, 4), "pl": round(pl, 4), "uncertainty": round(pl - bel, 4)}


def decide(m: dict, conflict: float) -> str:
    b = belief(m)
    if b["bel"] >= ENFORCE and conflict < CONFLICT_MAX:
        return "enforce"
    if b["pl"] < DROP:
        return "drop"
    return "monitor"


def assess(sources: list[str]) -> dict:
    """One shot: sources (names from SOURCES) -> combined mass, belief, conflict, decision."""
    m, k = combine_all([mass(s) for s in sources])
    return {"sources": sources, "mass": {x: round(v, 4) for x, v in m.items()}, **belief(m), "conflict": round(k, 4), "decision": decide(m, k)}


def feedback(state: dict, outcome_source: str, alpha: float = ALPHA) -> dict:
    """Adaptive loop: discount the standing belief, combine with the new outcome, decide again."""
    m_old = discount({k: v for k, v in state["mass"].items()}, alpha)
    m, k = combine(m_old, mass(outcome_source))
    return {"sources": state["sources"] + [outcome_source], "mass": {x: round(v, 4) for x, v in m.items()}, **belief(m),
            "conflict": round(max(k, state.get("conflict", 0.0) * alpha), 4), "decision": decide(m, k)}
