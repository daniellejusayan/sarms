# Track C in production replay (learn on the first 70% of each repository, enforce on the last 30%)

| repo | commits (filtered) | split | co-change couplings > .35 | layer candidates | critical hotspots | hotspot persistence | hgnn 0.98 rules | hgnn vs count MAE | drift events (future) | DS enforced initial → final | future crossings of enforced (kept / repaired) | written / contradicted among final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kedro | 636 | 2024-07-29 | 19 | 10 | 4 | 0.25 | 0 | 0.1396 / 0.0972 | 1 | 7 → 7 | 0 / 0 of 7 | 6 / 0 of 7 |
| sqlfluff | 3626 | 2024-05-31 | 36 | 13 | 20 | 0.35 | 0 | 0.0934 / 0.08 | 0 | 0 → 0 | 0 / 0 of 0 | 0 / 0 of 0 |
| ggshield | 584 | 2024-11-28 | 74 | 5 | 11 | 0.0 | 0 | 0.0976 / 0.1206 | 1 | 0 → 0 | 0 / 0 of 0 | 0 / 0 of 0 |
| nilearn | 1233 | 2025-08-19 | 141 | 0 | 9 | 0.44 | 0 | 0.1463 / 0.1259 | 1 | 0 → 0 | 0 / 0 of 0 | 0 / 0 of 0 |
| cloudai | 623 | 2026-02-01 | 90 | 1 | 10 | 0.1 | 0 | 0.0854 / 0.0903 | 0 | 0 → 0 | 0 / 0 of 0 | 0 / 0 of 0 |
| wemake-python-styleguide | 211 | 2025-08-18 | 58 | 1 | 7 | 0.43 | 2 | 0.2962 / 0.183 | 0 | 6 → 6 | 0 / 0 of 6 | 5 / 1 of 6 |
| faststream | 484 | 2025-11-09 | 304 | 2 | 16 | 0.38 | 0 | 0.0703 / 0.0815 | 0 | 0 → 0 | 0 / 0 of 0 | 0 / 0 of 0 |
| mlrun | 2696 | 2025-08-19 | 112 | 4 | 19 | 0.26 | 0 | 0.0605 / 0.0489 | 0 | 22 → 22 | 0 / 0 of 22 | 0 / 0 of 22 |
| checkov | 3089 | 2025-08-08 | 248 | 7 | 62 | 0.08 | 0 | 0.0382 / 0.0334 | 0 | 22 → 22 | 0 / 0 of 22 | 16 / 0 of 22 |

Pooled, by source of the candidate rule:

| source | rules | written | contradicted | crossed & kept later (wrong) | crossed & repaired later |
| --- | --- | --- | --- | --- | --- |
| cochange_layer_candidates | 43 | 16 (37%) | 0 | 6 | 0 |
| hgnn_layer_candidates | 0 | 0 (0%) | 0 | 0 | 0 |
| certified_on_past | 57 | 27 (47%) | 1 | 0 | 0 |
| ds_enforced_final | 57 | 27 (47%) | 1 | 0 | 0 |

Skipped: 

## Reading (5 Nov)

- The pipeline runs end to end on nine repositories (the sandbox clones are partial, so the streams were limited to the
  last four years for the six larger ones; kedro, sqlfluff and ggshield used their full history).
- **Co-change layer candidates** (C > 0.35, low level driven by high level): 43 rules, 37% written, and **6 were later
  imported and kept by the team**: 14% of them would have been wrong rules. Useful as a warning, not as a rule.
- **Hypergraph encoder**: trains in seconds, produces **0 rules at 98% confidence** on eight of nine repositories (2 on
  wemake), 0 layer candidates, and forecasts next-window co-change **worse than plain counting** on six of nine
  (MAE column). Its drift events line up with known upheavals (kedro mid-2024 and the 1.0 migration), so it earns a
  place as a drift detector in the Monitor, not as a rule generator.
- **Hotspots**: present everywhere (Pareto/outlier), but the critical set persists into the future at 0–44%
  (median about 26%): hotspots move. A warning with a short shelf life.
- **Belief functions**: 57 rules enforced, all carried by the structural certificate; co-change and encoder evidence
  only ever added support. In the future windows **none of the 57 was crossed**, so no feedback event fired and no
  decision changed; 27 of 57 (47%) coincide with written rules, 1 contradicts one (wemake, the precedence clause in
  `adopt --certified` removes it in production).
- So, measured the way production would see it: the behavioural signals do not generate rules that are never wrong;
  the certificate does, and the belief function is the right place to let the behavioural signals speak without
  letting them decide.
