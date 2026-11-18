# Track C: rules from the change stream (co-change, hotspots, a hypergraph memory, belief functions)

Built 5 Nov 2026 to test, end to end and against the same benchmark as everything else, the proposal that the
most important architectural rules can be extracted from the git log alone.

## Pipeline

```
git log stream  ->  noise filter  ->  hypergraph builder  ->  temporal graph encoder  ->  learned architecture matrix
                                                                                          |
      co-change confidence C(x->y), layer candidates, hotspots  ----------------------->  belief functions  ->  enforce / monitor / drop
      structural certificate A1-A6 (sarms.rules.certify) ------------------------------>        |
                                                                                           feedback loop (outcomes per window)
```

| step | module | what it does |
| --- | --- | --- |
| 1 | `sarms.mining.gitlog` | `git log --numstat`, first-parent; regexes drop docs, assets, vendored and third-party code, lock files, generated code, CI, packaging, and tests; bulk commits (> 50 source files) dropped |
| 2 | `sarms.rules.cochange` | `C(x->y) = commits(x and y) / commits(x)`; couplings at C > 0.35 with support >= 5; layer-violation candidates where `C(x->y) > 0.35 and level(x) < level(y)` (level from the layer vocabulary: 0 config/utils … 4 cli/ui/app); hotspots `H(f) = commits x churn`, critical = Pareto top 5% or z >= 2 on log H; quiet debt = many small edits or few large edits |
| 3 | `sarms.rules.hgnn` | per 90-day window: hyperedges = commits, nodes = components, features = log activity + level; two hypergraph convolutions (Feng et al. 2019) then a GRU over windows; decoder predicts next window's C(x->y); EMA baselines of error and state give drift events; rules at predicted C >= 0.98 |
| 4 | `sarms.rules.belief` | Dempster–Shafer over {correct, wrong}: each source gives a mass; Dempster's rule combines; Bel >= 0.80 and low conflict -> enforce; Pl < 0.50 -> drop; else monitor; outcomes per window are new evidence after discounting (alpha 0.9) |
| 5 | `tools/track_c.py` | learn on the first 70% of the repository's life, replay the last 30% window by window, then grade against the team's written rules |

## Commands (PowerShell)

```powershell
uv sync --group hgnn                                   # torch, CPU build is enough
uv run python tools\track_c.py kedro sqlfluff ggshield  # -> data\<repo>\rms\track-c\, reports\track-c.md
uv run pytest -q tests\test_track_c.py
```

## What it is for, after the measurements (see reports/track-c.md and the project note)
Co-change, hotspots and the encoder are **evidence and warnings**: they feed the belief function and the Monitor, and
they never reach "enforce" on their own (their masses are set below the threshold on purpose, because out of
sample they do not hold). Enforcement still rests on the structural certificate. The belief function is where the
sources meet, with conflict measured rather than hidden, and the feedback loop is how a rule loses its standing
when the repository stops obeying it.
