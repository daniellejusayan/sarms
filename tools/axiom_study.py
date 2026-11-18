"""Is the six-axiom specification (sarms.rules.certify) enough? Run it over every benchmark repository.

    uv run python tools/axiom_study.py            # -> reports/axiom-study.md

For each repository with a learner run: take the loosely adopted rules, the strict (nine-gate) rules, and the
rules that satisfy A1-A6 (certified). Count them, see which axiom removes the most, and, afterwards, open the
teams' written rules to measure agreement (written share) and disagreement (contradicted = the team allows
A -> B). The axioms never see the written rules; this is a test of the axioms, not of the rules.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from sarms.rules import certify as C
from sarms.rules.viability import score

SKIP = {"kedro-org__kedro", "ICHNedx", "edx-platform"}
HELD_OUT = {"nilearn", "mlrun", "ggshield", "cloudai"}


def main():
    rows, fails_all, fails_strict = [], Counter(), Counter()
    pooled = {"adopted": Counter(), "strict": Counter(), "certified": Counter()}
    for d in sorted(Path("data").glob("*/rms/learn")):
        name = d.parts[1]
        if name in SKIP or not (d / "history_edges.parquet").exists() or not (d / "candidates_graded.csv").exists():
            continue
        v = score(d, "adopted")
        if v.empty:
            continue
        graded = pd.read_csv(d / "candidates_graded.csv")
        grade = {(r["from"], r["to"]): r["grade"] for r in graded.to_dict("records")}
        history = C.load_history(d)
        cert = {}
        for r in v.to_dict("records"):
            _, _, verdict = C.evaluate(history, r["from"], r["to"])
            cert[(r["from"], r["to"])] = verdict
            for k, ok in verdict.items():
                if not ok:
                    fails_all[k] += 1
                    if r["strict"]:
                        fails_strict[k] += 1
        ok_pairs = [p for p, vd in cert.items() if all(vd.values())]
        a5, _ = C.consistency(ok_pairs)
        certified = set(ok_pairs) if a5 else set()
        sets = {"adopted": {(r["from"], r["to"]) for r in v.to_dict("records")},
                "strict": {(r["from"], r["to"]) for r in v.to_dict("records") if r["strict"]},
                "certified": certified}
        rec = {"repo": name, "held_out": name in HELD_OUT, "A5_holds": a5}
        for k, pairs in sets.items():
            g = Counter(grade.get(p, "unknown") for p in pairs)
            rec[f"{k}_n"] = len(pairs)
            rec[f"{k}_written"] = g["exact"] + g["narrower"] + g["broader"]
            rec[f"{k}_contradicted"] = g["contradicted"]
            pooled[k].update({"n": len(pairs), "written": rec[f"{k}_written"], "contradicted": g["contradicted"]})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv("reports/axiom-study.csv", index=False)

    def line(label, c):
        n = c["n"] or 1
        return f"| {label} | {c['n']} | {c['written']} ({100 * c['written'] / n:.0f}%) | {c['contradicted']} ({100 * c['contradicted'] / n:.1f}%) |"

    L = ["# Six-axiom specification over the benchmark", "",
         f"{len(df)} repositories. Sets: loosely adopted -> strict (nine gates) -> certified (A1-A6 with declared constants "
         f"DELTA={C.DELTA}, K={C.K}, T={C.T_DAYS} days, M={C.M}, S={C.S}). The written rules are opened only afterwards.", "",
         "| set | rules | written | contradicted |", "| --- | --- | --- | --- |",
         line("adopted", pooled["adopted"]), line("strict", pooled["strict"]), line("certified", pooled["certified"])]
    for tag, sub in (("training", df[~df.held_out]), ("held-out", df[df.held_out])):
        c = {k: Counter({"n": int(sub[f"{k}_n"].sum()), "written": int(sub[f"{k}_written"].sum()), "contradicted": int(sub[f"{k}_contradicted"].sum())})
             for k in ("adopted", "strict", "certified")}
        L += ["", f"{tag} repositories ({len(sub)}):", "", "| set | rules | written | contradicted |", "| --- | --- | --- | --- |",
              line("adopted", c["adopted"]), line("strict", c["strict"]), line("certified", c["certified"])]
    L += ["", "Which axiom fails (counts over all adopted rules / over strict rules):", ""]
    for k in ("A1_direction", "A2_history", "A3_reachability", "A4_substance", "A6_safety"):
        L.append(f"- {k}: {fails_all[k]} / {fails_strict[k]}")
    L += ["", f"Repositories where A5 (set consistency) fails among the certified rules: {int((~df.A5_holds).sum())}", "",
          "| repo | adopted | strict | certified | certified written | certified contradicted |", "| --- | --- | --- | --- | --- | --- |"]
    for r in df.sort_values("certified_n", ascending=False).head(20).to_dict("records"):
        L.append(f"| {r['repo']}{' (held out)' if r['held_out'] else ''} | {r['adopted_n']} | {r['strict_n']} | {r['certified_n']} | {r['certified_written']} | {r['certified_contradicted']} |")
    Path("reports/axiom-study.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
