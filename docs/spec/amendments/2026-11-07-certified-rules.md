# Amendment A-2026-11-07-2 — certified rules as the autonomous adoption rule (additive)

Applies to `docs/spec/rules.schema.yaml` (no field added) and `docs/spec/gate.md` (one paragraph added). Targets unchanged.

1. A rule with `source.type: recovered` and `source.id: learned:certified` is one the system adopted on its own. It is
   emitted only by the certificate engine (`src/sarms/rules/certify.py`, `adopt_from_history`) when the six axioms hold:
   A1 direction (counterfactual instability margin >= DELTA 0.10 in every coexisting sampled commit), A2 history (never
   crossed; >= 8 coexisting commits over >= 1095 days), A3 reachability (no path in any commit), A4 substance (>= 3
   modules each, >= 2 shared neighbours), A5 set consistency (no reverse pairs; "below" acyclic), A6 safety (absent now).
   Constants are declared in the module and recorded in every certificate.
2. Precedence: a pair a written rule allows is never adopted; a pair a written rule forbids is left to that rule.
3. Maintenance: an existing certified rule is kept with margin >= KEEP_DELTA 0.05 (hysteresis), and crossings the loop
   itself handled (an `apply` or `retire` action on the ledger) are excused from its history. Both are recorded in the
   certificate (`constants.delta`, `constants.excused`) and re-checked by `sarms verify-rules`.
4. Enforcement: at a violation the Differ re-evaluates A1 on the live graph; a rule whose margin is gone is retired on the
   spot (`enforce: false`, `retired_reason`) and the push is recorded, never fixed. A developer revert of a system fix
   demotes the rule. No person is asked at any point; the post-hoc audit remains.
5. Certified rules carry severity `medium`; the gate and kernel treat them exactly like written rules.
