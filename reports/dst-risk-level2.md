# DST risk report (level 2, 3 seeds per scenario, 132 runs)

| scenario | CRITICAL | HIGH | MEDIUM | LOW | worst |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 12 | LOW |
| birth_death | 0 | 0 | 0 | 12 | LOW |
| bulk_migration | 0 | 0 | 0 | 12 | LOW |
| crossing_kept | 0 | 0 | 9 | 3 | MEDIUM |
| crossing_then_repair | 0 | 0 | 9 | 3 | MEDIUM |
| cycle | 0 | 0 | 10 | 2 | MEDIUM |
| package_split | 0 | 0 | 0 | 12 | LOW |
| quiet_then_burst | 0 | 0 | 0 | 12 | LOW |
| rename | 0 | 0 | 0 | 12 | LOW |
| repeated_crossings | 0 | 0 | 9 | 3 | MEDIUM |
| revert_storm | 0 | 0 | 12 | 0 | MEDIUM |

Totals: {'LOW': 83, 'MEDIUM': 49}

## CRITICAL and HIGH runs (seed replays the run)


## Coverage

events exercised: ['birth', 'bulk', 'burst', 'cross', 'cycle', 'death', 'feature', 'keep', 'quiet', 'rename', 'repair', 'revert_fix', 'split']
loop transitions exercised: ['applied', 'left_standing', 'pushes']

Labels: CRITICAL = an invariant I1-I9 was violated; HIGH = near miss (A1 margin within 0.05 of the threshold at a crossing, flapping, a bulk commit retiring > 30% of rules, double demotion); MEDIUM = rule churn or a retire; LOW = nominal.
