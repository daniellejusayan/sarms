"""Are the learned rules viable AS RULES, judged on their own merits (not against the team's contracts)?

    uv run python tools/rule_viability.py kedro                 # one repository (data/kedro/rms/learn)
    uv run python tools/rule_viability.py --all                 # every data/*/rms/learn, summary table
    uv run python tools/rule_viability.py --all --review 40     # also a review sheet of strong-but-unwritten rules

Each adopted rule "A must not import B" is scored on criteria that use only the import history and the
current graph, never the written contracts:

  clean_chain      A reaches B neither directly nor through any chain of components today (informative:
                   cyclic codebases make most pairs reachable, including under rules people wrote)
  grounded         B has depended on A in >= 50% of the sampled commits where both existed: the rule
                   states a real layering direction, not an empty pair
  sdp              Stable-Dependencies Principle (Martin): B is at least as volatile as A
                   (instability I = out / (in + out) at the component level), so A depending on B would
                   be a dependency on something less stable
  never_crossed    A -> B appeared in no sampled commit, or appeared once and was removed for good
  not_vacuous      both components have >= 2 modules and share at least one neighbour (a common importer
                   or import), so they live in the same part of the system and could have met
  consistent       the adopted set has no rule B -> A as well (unless both sides are independent siblings)
                   and this rule sits on no cycle of layering rules
  names_agree      the layer vocabulary does not place A above B

Tiers: strong = grounded & sdp & consistent & not_vacuous & names_agree;
       plausible = consistent & names_agree & at least 2 of (grounded, sdp, not_vacuous, clean_chain);
       weak = the rest. clean_chain and never_crossed are reported but not required (see the code comment).
The contracts are read only AFTERWARDS, to cross-tab tier x (in contract / unknown / contradicted): rules
humans wrote should mostly be strong or plausible (the criteria are then credible), and strong rules that
no contract mentions are the candidate oversights, written to a review sheet for people to rate.
The criteria themselves live in sarms.rules.viability (shared with `sarms adopt --strict`).
Writes data/REPO/rms/learn/viability.csv and reports/rule-viability.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sarms.rules.viability import score


def score_repo(learn_dir: Path):
    """Criteria + strict gates from sarms.rules.viability; then, and only then, the contract grade for the cross-check."""
    df = score(learn_dir, "adopted")
    if df.empty:
        return df
    graded = pd.read_csv(learn_dir / "candidates_graded.csv") if (learn_dir / "candidates_graded.csv").exists() else None
    grade = []
    for r in df.to_dict("records"):
        g = ""
        if graded is not None:
            m = graded[(graded["from"] == r["from"]) & (graded["to"] == r["to"])]
            g = m.grade.iloc[0] if len(m) else ""
        grade.append(g)
    df["grade"] = grade
    df["in_contract"] = df.grade.isin(["exact", "narrower", "broader"])
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--review", type=int, default=0, help="write a review sheet with this many strong, unwritten rules")
    a = ap.parse_args()
    dirs = sorted(Path("data").glob("*/rms/learn")) if a.all else [Path("data") / a.repo / "rms" / "learn"]
    frames, summary = [], []
    for d in dirs:
        if not (d / "candidates.csv").exists() or not (d / "history_edges.parquet").exists():
            continue
        df = score_repo(d)
        name = d.parts[1]
        if df.empty:
            summary.append({"repo": name, "adopted": 0})
            continue
        df.insert(0, "repo", name)
        df.to_csv(d / "viability.csv", index=False)
        frames.append(df)
        t = df.tier.value_counts()
        inc = df[df.in_contract]
        summary.append({"repo": name, "adopted": len(df), "strong": int(t.get("strong", 0)), "plausible": int(t.get("plausible", 0)),
                        "weak": int(t.get("weak", 0)), "strong_share": round(t.get("strong", 0) / len(df), 2),
                        "in_contract": int(len(inc)), "in_contract_strong_or_plausible": round(float((inc.tier != "weak").mean()), 2) if len(inc) else None,
                        "unknown_strong": int(((~df.in_contract) & (df.tier == "strong")).sum())})
    s = pd.DataFrame(summary)
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lines = ["# Viability of learned rules, judged without the contracts", "",
             "Criteria (see tools/rule_viability.py): grounded, sdp, consistent, not_vacuous, names_agree (gates); clean_chain, never_crossed (reported). "
             "strong = grounded + sdp + consistent + not vacuous + names agree; plausible = consistent + names agree + 2 of (grounded, sdp, not vacuous, clean chain); weak = rest. "
             "The contracts are used only to cross-check: rules people wrote should land in strong/plausible, and strong rules no contract "
             "mentions are candidate oversights for people to rate.", "",
             "| Repository | adopted | strong | plausible | weak | strong share | in a contract | of those strong/plausible | strong but unwritten |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in s.sort_values("adopted", ascending=False).to_dict("records"):
        lines.append(f"| {r['repo']} | {r['adopted']} | {r.get('strong', '')} | {r.get('plausible', '')} | {r.get('weak', '')} | {r.get('strong_share', '')} | "
                     f"{r.get('in_contract', '')} | {r.get('in_contract_strong_or_plausible', '')} | {r.get('unknown_strong', '')} |")
    if len(allr):
        ct = pd.crosstab(allr.tier, allr.in_contract.map({True: "in a contract", False: "not written"}))
        lines += ["", "## All repositories pooled: tier x written-down", "", "| tier | in a contract | not written | precision of tier vs contracts |", "| --- | --- | --- | --- |"]
        for tier in ("strong", "plausible", "weak"):
            if tier in ct.index:
                ic, nw = int(ct.loc[tier].get("in a contract", 0)), int(ct.loc[tier].get("not written", 0))
                lines.append(f"| {tier} | {ic} | {nw} | {ic / (ic + nw):.2f} |")
        crit_cols = ["clean_chain", "grounded", "sdp", "never_crossed", "not_vacuous", "consistent", "names_agree"]
        lines += ["", "## Which criterion separates written rules from the rest", "", "| criterion | share true among rules in a contract | among unwritten rules |", "| --- | --- | --- |"]
        for c in crit_cols:
            lines.append(f"| {c} | {allr[allr.in_contract][c].mean():.2f} | {allr[~allr.in_contract][c].mean():.2f} |")
        st = allr[allr.strict]
        lines += ["", "## Strict filter: rules we would adopt with no person involved", "",
                  "All nine tightened gates must hold: B -> A in >= 80% of commits, stability margin >= 0.10, no indirect path today, "
                  "never crossed at all, both sides >= 3 modules sharing >= 2 neighbours, coexisted >= 3 years, consistent, names not against, "
                  "and the source component forbids at most 40% of what it could reach.", "",
                  "| | strict rules | of which in a contract | share written |", "| --- | --- | --- | --- |",
                  f"| all adopted | {len(st)} | {int(st.in_contract.sum())} | {st.in_contract.mean():.2f} |" if len(st) else "| all adopted | 0 | 0 | – |",
                  f"| for comparison: loose strong tier | {int((allr.tier == 'strong').sum())} | {int(((allr.tier == 'strong') & allr.in_contract).sum())} | {allr[allr.tier == 'strong'].in_contract.mean():.2f} |"]
        lines += ["", "Strict rules per repository (written / unwritten):", ""]
        for repo, grp in st.groupby("repo"):
            lines.append(f"- {repo}: {int(grp.in_contract.sum())} written, {int((~grp.in_contract).sum())} unwritten")
        strict_unwritten = st[~st.in_contract].sort_values(["repo", "ba_share"], ascending=[True, False])
        Path("data/samples").mkdir(parents=True, exist_ok=True)
        strict_unwritten.to_csv("data/samples/viable_rules_unwritten.csv", index=False)
        lines += ["", f"**{len(strict_unwritten)} strict rules that no contract mentions** -> data/samples/viable_rules_unwritten.csv (the viable set).",
                  "", "Which strict gate removes the most loose-strong rules (share failing it among unwritten strong rules):", ""]
        su = allr[(allr.tier == "strong") & (~allr.in_contract)]
        for k in ["s_direction_80", "s_sdp_margin", "s_clean_chain", "s_never_crossed_at_all", "s_substantial", "s_three_years", "s_consistent", "s_names_agree", "s_selective"]:
            lines.append(f"- {k[2:]}: {1 - su[k].mean():.2f}")
        n_strong_unwritten = int(((~allr.in_contract) & (allr.tier == "strong")).sum())
        lines += ["", f"Strong rules that no contract mentions, over all repositories: **{n_strong_unwritten}**. These are the candidate oversights; "
                      "a sample is in data/samples/rule_review.csv when --review was given."]
        if a.review:
            cand = allr[(~allr.in_contract) & (allr.tier == "strong")].sample(frac=1, random_state=2569)
            # spread over repositories: round-robin
            picked, seen = [], {}
            for r in cand.to_dict("records"):
                seen.setdefault(r["repo"], 0)
                r["_k"] = seen[r["repo"]]
                seen[r["repo"]] += 1
                picked.append(r)
            picked = sorted(picked, key=lambda r: (r["_k"], r["repo"]))[: a.review]
            sheet = pd.DataFrame(picked).drop(columns=["_k"])
            sheet["rater"] = ""
            sheet["viable_rule (y/n/unsure)"] = ""
            sheet["why"] = ""
            Path("data/samples").mkdir(parents=True, exist_ok=True)
            sheet.to_csv("data/samples/rule_review.csv", index=False)
            lines.append(f"Review sheet written: data/samples/rule_review.csv ({len(sheet)} rules).")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/rule-viability.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
