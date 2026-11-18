# Six-axiom specification over the benchmark

29 repositories. Sets: loosely adopted -> strict (nine gates) -> certified (A1-A6 with declared constants DELTA=0.1, K=8, T=1095 days, M=3, S=2). The written rules are opened only afterwards.

| set | rules | written | contradicted |
| --- | --- | --- | --- |
| adopted | 2720 | 732 (27%) | 140 (5.1%) |
| strict | 268 | 144 (54%) | 2 (0.7%) |
| certified | 224 | 115 (51%) | 2 (0.9%) |

training repositories (25):

| set | rules | written | contradicted |
| --- | --- | --- | --- |
| adopted | 2423 | 644 (27%) | 120 (5.0%) |
| strict | 221 | 121 (55%) | 2 (0.9%) |
| certified | 200 | 100 (50%) | 2 (1.0%) |

held-out repositories (4):

| set | rules | written | contradicted |
| --- | --- | --- | --- |
| adopted | 297 | 88 (30%) | 20 (6.7%) |
| strict | 47 | 23 (49%) | 0 (0.0%) |
| certified | 24 | 15 (62%) | 0 (0.0%) |

Which axiom fails (counts over all adopted rules / over strict rules):

- A1_direction: 1669 / 77
- A2_history: 1214 / 0
- A3_reachability: 1942 / 40
- A4_substance: 1041 / 0
- A6_safety: 0 / 0

Repositories where A5 (set consistency) fails among the certified rules: 0

| repo | adopted | strict | certified | certified written | certified contradicted |
| --- | --- | --- | --- | --- | --- |
| openedx-platform | 355 | 41 | 44 | 0 | 0 |
| blueking-paas | 415 | 42 | 43 | 22 | 2 |
| sqlfluff | 140 | 25 | 35 | 26 | 0 |
| SecObserve | 235 | 28 | 31 | 16 | 0 |
| bk-user | 60 | 21 | 16 | 16 | 0 |
| ggshield (held out) | 80 | 24 | 15 | 15 | 0 |
| kedro | 32 | 5 | 9 | 9 | 0 |
| mlrun (held out) | 128 | 23 | 9 | 0 | 0 |
| middleware | 462 | 9 | 8 | 0 | 0 |
| kopf | 6 | 5 | 5 | 5 | 0 |
| wemake-python-styleguide | 35 | 11 | 5 | 4 | 0 |
| gpkit-core | 7 | 4 | 2 | 0 | 0 |
| abis-mapping | 3 | 2 | 1 | 1 | 0 |
| checkov | 308 | 26 | 1 | 1 | 0 |
| evalscope | 71 | 0 | 0 | 0 | 0 |
| depmap-portal | 35 | 0 | 0 | 0 | 0 |
| codeflare-sdk | 7 | 0 | 0 | 0 | 0 |
| cloudai (held out) | 22 | 0 | 0 | 0 | 0 |
| MSS | 8 | 0 | 0 | 0 | 0 |
| SolarWindPy | 2 | 0 | 0 | 0 | 0 |
