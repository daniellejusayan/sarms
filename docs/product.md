# sarms as a product — the whole workflow

```
mining  ->  STHG ledger  ->  architecture certificates  ->  the loop (synthesizer, gate, kernel)
sarms mine   sarms status    sarms certify-ledger          sarms run
```

| stage | command | what happens | what it leaves |
| --- | --- | --- | --- |
| Mining | `uv run sarms mine REPO --out data\REPO\rms` | first-parent history; import graph at every 25th commit (grimp); the change stream with churn, noise removed | `sthg/ledger.duckdb` + parquet, `sthg/meta.json` |
| Ledger | (filled by every stage) | commits as hyperedges over components and files; graph state and deltas; labels; actions; certificates | the system's memory; `sarms status --out …` shows it |
| Certificates | `uv run sarms certify-ledger REPO --out …` | every pair of live components is tested against A1–A6; written rules take precedence; certificates written and independently re-checked | `rules-certified/soft.yaml`, `certificates.json`, `rules-active/` (written + certified) |
| Loop | `uv run sarms run REPO --out … --tests "python -m pytest -q" [--model …] [--onto main]` | Differ on HEAD -> for each enforceable direct violation: one model call (draft) -> gate G0–G8 -> kernel commit as `sensolens-sarms`; everything else recorded | `drafts/`, `decisions/<sha>.json`, the commit, `run.json`, ledger actions |

Model: any OpenAI-compatible endpoint (`--base-url`, `--model`, `SARMS_LLM_API_KEY`), Ollama locally, or `fake:<dir>` with scripted replies. One call per fix, three tries, no agent. Tools do all detection and verification.

What a person sees: nothing during the loop. Afterwards: `sarms status`, the decision records, the ledger, and `docs/architecture/log.md` when the kernel is pointed at a real branch.

Three dated amendments describe the additions to the frozen spec: `docs/spec/amendments/2026-11-07-*.md`.
