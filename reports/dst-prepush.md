# DST risk report (level 1, 1 seeds per scenario, 7 runs)

| scenario | CRITICAL | HIGH | MEDIUM | LOW | worst |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 1 | LOW |
| bulk_migration | 0 | 0 | 0 | 1 | LOW |
| crossing_kept | 0 | 0 | 0 | 1 | LOW |
| crossing_then_repair | 0 | 0 | 0 | 1 | LOW |
| cycle | 0 | 0 | 1 | 0 | MEDIUM |
| package_split | 0 | 0 | 1 | 0 | MEDIUM |
| revert_storm | 0 | 0 | 1 | 0 | MEDIUM |

Totals: {'LOW': 4, 'MEDIUM': 3}

## CRITICAL and HIGH runs (seed replays the run)


## Coverage

events exercised: ['bulk', 'cross', 'cycle', 'feature', 'keep', 'repair', 'revert_fix', 'split']
loop transitions exercised: ['adopt', 'enforce', 'retire']

Labels: CRITICAL = an invariant I1-I9 was violated; HIGH = near miss (A1 margin within 0.05 of the threshold at a crossing, flapping, a bulk commit retiring > 30% of rules, double demotion); MEDIUM = rule churn or a retire; LOW = nominal.
