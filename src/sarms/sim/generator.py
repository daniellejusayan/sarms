"""The scenario grammar and the seeded repository generator.

A simulated repository is a layered system of components (packages) with modules; its life is a sequence of
commits, each produced by one event from the grammar below. The generator yields, per commit, the module-level
import edges, the set of modules, and the files touched with churn, so the same history can be fed to the
ledger (level 1) or written to a real git repository (level 2).

EVENTS (the grammar; every scenario is a sequence of these):
  feature        edits 1-3 modules inside one layer, may add an import that respects the layering
  cross          a lower-layer module imports a higher-layer one (a rule crossing)
  repair         removes the most recent crossing
  keep           nothing (the crossing, if any, stays)
  bulk           touches > BULK modules across many components (a migration / reformat)
  split          a component splits into two (its modules move to a new sibling package)
  rename         a component is renamed
  birth          a new component appears with modules and layer-respecting imports
  death          a component is removed with all its edges
  cycle          two components import each other
  burst          many small edits to one module (a hotspot)
  quiet          time passes with no commit (a gap of QUIET days)
  revert_fix     the developer reverts the system's last fix (level 2: a real `git revert`; level 1: the edge returns)

SCENARIOS (named sequences over a seeded baseline of N_BASE feature commits spanning years):
  baseline, crossing_then_repair, crossing_kept, repeated_crossings, bulk_migration, package_split, rename,
  cycle, birth_death, revert_storm, quiet_then_burst, and pairwise: every ordered pair of two distinct events
  appended to the baseline. `all_scenarios()` lists them; the harness runs each with many seeds.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAYERS = ["config", "core", "service", "api", "cli"]      # bottom -> top; higher may import lower
BULK = 60
QUIET = 400
N_BASE = 120
EVENTS = ["feature", "cross", "repair", "keep", "bulk", "split", "rename", "birth", "death", "cycle", "burst", "quiet", "revert_fix"]
NAMED = {
    "baseline": [],
    "crossing_then_repair": ["cross", "feature", "repair"],
    "crossing_kept": ["cross", "keep", "feature", "feature"],
    "repeated_crossings": ["cross", "repair", "cross", "repair", "cross"],
    "bulk_migration": ["bulk", "feature", "bulk"],
    "package_split": ["split", "feature", "feature"],
    "rename": ["rename", "feature"],
    "cycle": ["cycle", "feature"],
    "birth_death": ["birth", "feature", "death"],
    "revert_storm": ["cross", "revert_fix", "cross", "revert_fix", "cross", "revert_fix"],
    "quiet_then_burst": ["quiet", "burst", "burst"],
}


def all_scenarios():
    out = dict(NAMED)
    for a, b in itertools.permutations(EVENTS, 2):
        out[f"pair:{a}+{b}"] = [a, b, "feature"]
    return out


@dataclass
class Repo:
    """Mutable simulated repository state."""
    rng: np.random.Generator
    root: str = "app"
    comps: dict = field(default_factory=dict)          # comp -> {"layer": int, "modules": [names]}
    edges: set = field(default_factory=set)            # (module_from, module_to)
    day: int = 0
    last_cross: tuple | None = None
    last_fix_edge: tuple | None = None
    history: list = field(default_factory=list)

    def modules(self):
        return [f"{c}.{m}" for c, d in self.comps.items() for m in d["modules"]]

    def comp_of(self, module):
        return module.rsplit(".", 1)[0]

    def layer(self, comp):
        return self.comps[comp]["layer"]


def new_repo(seed: int, n_per_layer: int = 2, modules_per_comp: int = 4) -> Repo:
    rng = np.random.default_rng(seed)
    r = Repo(rng)
    for li, lname in enumerate(LAYERS):
        for k in range(n_per_layer):
            c = f"{r.root}.{lname}{k}"
            r.comps[c] = {"layer": li, "modules": [f"m{j}" for j in range(modules_per_comp)]}
    # layer-respecting imports: higher imports lower, dense enough to make lower layers stable
    comps = list(r.comps)
    for a in comps:
        for b in comps:
            if r.layer(a) > r.layer(b) and rng.random() < 0.6:
                for _ in range(int(rng.integers(1, 3))):
                    r.edges.add((f"{a}.{rng.choice(r.comps[a]['modules'])}", f"{b}.{rng.choice(r.comps[b]['modules'])}"))
    return r


def _touch(r: Repo, modules, churn=(3, 40)):
    return [(m, int(r.rng.integers(*churn)), int(r.rng.integers(0, churn[1] // 2))) for m in modules]


def step(r: Repo, event: str):
    """Apply one event; return the commit record {event, day, touched:[(module, added, deleted)], edges:set, modules:list}."""
    rng = r.rng
    comps = list(r.comps)
    touched = []
    r.day += int(rng.integers(1, 12))
    if event == "feature":
        c = str(rng.choice(comps))
        mods = [f"{c}.{m}" for m in rng.choice(r.comps[c]["modules"], size=min(len(r.comps[c]["modules"]), int(rng.integers(1, 4))), replace=False)]
        touched = _touch(r, mods)
        if rng.random() < 0.4:
            lower = [b for b in comps if r.layer(b) < r.layer(c)]
            if lower:
                b = str(rng.choice(lower))
                r.edges.add((mods[0], f"{b}.{rng.choice(r.comps[b]['modules'])}"))
    elif event == "cross":
        lows = [c for c in comps if r.layer(c) < len(LAYERS) - 1]
        a = str(rng.choice(lows))
        highs = [c for c in comps if r.layer(c) > r.layer(a)]
        b = str(rng.choice(highs))
        e = (f"{a}.{rng.choice(r.comps[a]['modules'])}", f"{b}.{rng.choice(r.comps[b]['modules'])}")
        r.edges.add(e)
        r.last_cross = e
        touched = _touch(r, [e[0]])
    elif event == "repair":
        if r.last_cross and r.last_cross in r.edges:
            r.edges.discard(r.last_cross)
            touched = _touch(r, [r.last_cross[0]], (1, 5))
            r.last_fix_edge = r.last_cross
            r.last_cross = None
        else:
            return step(r, "feature")
    elif event == "keep":
        touched = []
    elif event == "bulk":
        mods = r.modules()
        pick = [str(m) for m in rng.choice(mods, size=min(len(mods), BULK + 10), replace=False)]
        touched = _touch(r, pick, (1, 6))
    elif event == "split":
        big = max(comps, key=lambda c: len(r.comps[c]["modules"]))
        if len(r.comps[big]["modules"]) >= 2:
            half = r.comps[big]["modules"][len(r.comps[big]["modules"]) // 2:]
            r.comps[big]["modules"] = r.comps[big]["modules"][: len(r.comps[big]["modules"]) // 2]
            new = big + "_b"
            r.comps[new] = {"layer": r.layer(big), "modules": half}
            moved = {f"{big}.{m}": f"{new}.{m}" for m in half}
            r.edges = {(moved.get(x, x), moved.get(y, y)) for x, y in r.edges}
            touched = _touch(r, list(moved.values()), (1, 3))
    elif event == "rename":
        c = str(rng.choice(comps))
        new = c + "_v2"
        r.comps[new] = r.comps.pop(c)
        r.edges = {(x.replace(c + ".", new + "."), y.replace(c + ".", new + ".")) for x, y in r.edges}
        touched = _touch(r, [f"{new}.{m}" for m in r.comps[new]["modules"]], (1, 3))
    elif event == "birth":
        li = int(rng.integers(0, len(LAYERS)))
        c = f"{r.root}.{LAYERS[li]}{len(comps)}"
        r.comps[c] = {"layer": li, "modules": [f"m{j}" for j in range(3)]}
        for b in comps:
            if r.layer(b) < li and rng.random() < 0.5:
                r.edges.add((f"{c}.m0", f"{b}.{rng.choice(r.comps[b]['modules'])}"))
        touched = _touch(r, [f"{c}.m{j}" for j in range(3)])
    elif event == "death":
        if len(comps) > 4:
            c = str(rng.choice(comps))
            r.edges = {(x, y) for x, y in r.edges if not x.startswith(c + ".") and not y.startswith(c + ".")}
            touched = _touch(r, [f"{c}.{m}" for m in r.comps[c]["modules"]], (0, 2))
            del r.comps[c]
    elif event == "cycle":
        a, b = (str(x) for x in rng.choice(comps, size=2, replace=False))
        r.edges.add((f"{a}.m0", f"{b}.m0"))
        r.edges.add((f"{b}.m0", f"{a}.m0"))
        touched = _touch(r, [f"{a}.m0", f"{b}.m0"])
    elif event == "burst":
        c = str(rng.choice(comps))
        m = f"{c}.{r.comps[c]['modules'][0]}"
        touched = _touch(r, [m], (20, 80))
    elif event == "quiet":
        r.day += QUIET
        touched = []
    elif event == "revert_fix":
        if r.last_fix_edge:
            r.edges.add(r.last_fix_edge)
            r.last_cross = r.last_fix_edge
            touched = _touch(r, [r.last_fix_edge[0]], (1, 5))
        else:
            return step(r, "cross")
    rec = {"event": event, "day": r.day, "touched": touched, "edges": set(r.edges), "modules": r.modules()}
    r.history.append(rec)
    return rec


def generate(seed: int, scenario: list[str], n_base: int = N_BASE, days_per_base: int = 12):
    """A repository life: n_base feature commits spanning years (so A2's three years can be met), then the scenario."""
    r = new_repo(seed)
    events = ["feature"] * n_base + list(scenario)
    for i, ev in enumerate(events):
        step(r, ev)
        if i < n_base:
            r.day += days_per_base
    return r


def to_frames(r: Repo, root="app"):
    """History tables in the ledger's shape: (history_edges, history_components) at every commit, and a change stream."""
    t0 = pd.Timestamp("2016-01-01", tz="UTC")
    he, hc, st = [], [], []
    for i, rec in enumerate(r.history):
        ts = t0 + pd.Timedelta(days=rec["day"])
        sha = f"{i:040x}"
        comps = {}
        for m in rec["modules"]:
            c = m.rsplit(".", 1)[0]
            comps[c] = comps.get(c, 0) + 1
        for c, n in comps.items():
            hc.append({"idx": i, "sha": sha, "ts": ts, "comp": c, "n_modules": n})
        counts = {}
        for a, b in rec["edges"]:
            ca, cb = a.rsplit(".", 1)[0], b.rsplit(".", 1)[0]
            if ca != cb:
                counts[(ca, cb)] = counts.get((ca, cb), 0) + 1
        for (ca, cb), n in counts.items():
            he.append({"idx": i, "sha": sha, "ts": ts, "comp_from": ca, "comp_to": cb, "n": n})
        for m, add, dele in rec["touched"]:
            st.append({"idx": i, "sha": sha, "ts": ts, "author": "dev", "path": m.replace(".", "/") + ".py", "added": add, "deleted": dele,
                       "module": m, "component": m.rsplit(".", 1)[0]})
    return pd.DataFrame(he), pd.DataFrame(hc), pd.DataFrame(st)
