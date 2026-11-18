# Amendment A-2026-11-07-1 — the spatio-temporal hypergraph ledger (additive)

Applies to `docs/spec/ledger.md` and `docs/spec/ledger.sql`. The frozen `ledger` table and its views stay as they are;
this amendment adds tables that hold the system's memory of a repository, from which the certificate engine and the
simulator read. No target moves.

| table | one row per | purpose |
| --- | --- | --- |
| `nodes` | component or file | the vertices of the hypergraph, with first/last seen |
| `hyperedges` | commit | one hyperedge per commit: time, pseudonymised author, size, bulk flag, source (human / sensolens-sarms / test author) |
| `incidence` | commit x node | which nodes a commit touched and how much (added, deleted) |
| `components` | sampled commit x component | module count at that commit |
| `graph_state` | sampled commit x component edge | the import graph as it was |
| `graph_delta` | sampled commit x component edge | +1 / -1 between consecutive samples |
| `labels` | commit x rule x edge | the Differ's labels, with `enforce` and `retired_reason` for certified rules |
| `actions` | event | draft, gate, apply, escalate, revert_seen, adopt, keep, retire, demote, mine, verify |
| `certificates` | rule x certification run | axioms, summary, per-commit evidence, checker code hash |

Implementation: `src/sarms/ledger/hypergraph.py` (DuckDB, parquet mirror). The frozen `ledger` table continues to be
built by `sarms ledger`; `sarms status` reads both.
