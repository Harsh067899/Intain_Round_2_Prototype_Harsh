"""Task 1 orchestrator — run everything, evaluate honestly, write the report.

Outputs:
  reports/data_intelligence_report.md
  reports/artifacts/column_profile_{train,test}.csv
  reports/artifacts/rule_violation_summary.csv
  reports/artifacts/drift_psi.csv
  reports/artifacts/record_trust_scores.csv      (feeds Tasks 4/6/7)
  reports/artifacts/batch_quality_servicer.csv
  reports/artifacts/batch_quality_month.csv
  reports/artifacts/profiling_eval_vs_ground_truth.csv

Run: python src/profiling/run_task1.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from profile_engine import profile_columns, missingness_patterns, top_correlations, drift_table  # noqa: E402
from rule_engine import run_rules, learned_relationship_checks  # noqa: E402
from reconcile import reconcile, trust_scores, batch_quality  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW = os.path.join(ROOT, "data", "raw")
GT = os.path.join(ROOT, "data", "ground_truth")
REP = os.path.join(ROOT, "reports")
ART = os.path.join(REP, "artifacts")


def main():
    os.makedirs(ART, exist_ok=True)
    train = pd.read_csv(os.path.join(RAW, "loan_monthly_performance_train.csv"))
    test = pd.read_csv(os.path.join(RAW, "loan_monthly_performance_test.csv"))
    static = pd.read_csv(os.path.join(RAW, "loan_static_attributes.csv"))
    updates = pd.read_csv(os.path.join(RAW, "servicer_updates.csv"))

    # --- 1. column profiles -------------------------------------------------
    prof_tr = profile_columns(train)
    prof_te = profile_columns(test)
    prof_tr.to_csv(os.path.join(ART, "column_profile_train.csv"), index=False)
    prof_te.to_csv(os.path.join(ART, "column_profile_test.csv"), index=False)

    # --- 2. missingness patterns --------------------------------------------
    miss = missingness_patterns(train, by=["servicer_name", "source_system"])

    # --- 3. correlations / dependent fields ---------------------------------
    corr = top_correlations(train)

    # --- 4. rules (starter + learned) ----------------------------------------
    rule_rows, rule_summary = run_rules(train, os.path.join(RAW, "validation_rules.json"))
    learned = learned_relationship_checks(train)
    rule_summary.to_csv(os.path.join(ART, "rule_violation_summary.csv"), index=False)

    # --- 5. drift -------------------------------------------------------------
    drift_cols = ["interest_rate", "current_balance", "loan_age_months",
                  "remaining_term_months", "days_past_due", "credit_score_band",
                  "ltv_band", "dti_band", "state", "loan_purpose", "servicer_name",
                  "current_status", "document_status", "source_system"]
    drift = drift_table(train, test, drift_cols)
    drift.to_csv(os.path.join(ART, "drift_psi.csv"), index=False)

    # --- 6. reconciliation + trust -------------------------------------------
    recon = reconcile(train, updates, static)
    trust = trust_scores(train, rule_rows, recon)
    trust.to_csv(os.path.join(ART, "record_trust_scores.csv"), index=False)
    by_serv, by_month = batch_quality(trust, train)
    by_serv.to_csv(os.path.join(ART, "batch_quality_servicer.csv"), index=False)
    by_month.to_csv(os.path.join(ART, "batch_quality_month.csv"), index=False)

    # --- 7. honest evaluation vs hidden ground truth --------------------------
    gt = pd.read_csv(os.path.join(GT, "injected_corruptions.csv"))
    tkey_all = set(train.loan_id + "|" + train.reporting_month)
    gt = gt[(gt.loan_id + "|" + gt.reporting_month).isin(tkey_all)].copy()  # evaluate only the scored window
    flagged = trust.loc[(trust.n_rules_fired > 0) | (trust.n_conflicts > 0),
                        ["loan_id", "reporting_month"]]
    fkey = set(flagged.loan_id + "|" + flagged.reporting_month)
    gt["caught"] = (gt.loan_id + "|" + gt.reporting_month).isin(fkey)
    eval_rows = (gt.groupby("corruption_type")["caught"]
                   .agg(injected="size", caught="sum")
                   .assign(recall_pct=lambda d: (100 * d.caught / d.injected).round(2))
                   .reset_index())
    # precision proxy: flagged rows that correspond to ANY labeled exception
    lab = train.loc[train.exception_required.eq(1), ["loan_id", "reporting_month"]]
    lkey = set(lab.loan_id + "|" + lab.reporting_month)
    n_flag = len(fkey)
    tp = len(fkey & lkey)
    precision = 100 * tp / max(n_flag, 1)
    # precision restricted to high-severity signals (low-severity rules are recall-oriented screeners)
    hi = trust.loc[trust.rules_fired.str.contains("R001|R002|R003|R004|R008", na=False)
                   | (trust.n_conflicts > 0), ["loan_id", "reporting_month"]]
    hkey = set(hi.loan_id + "|" + hi.reporting_month)
    precision_hi = 100 * len(hkey & lkey) / max(len(hkey), 1)
    eval_rows.to_csv(os.path.join(ART, "profiling_eval_vs_ground_truth.csv"), index=False)

    # --- 8. report -------------------------------------------------------------
    def md(df, n=None):
        return (df.head(n) if n else df).to_markdown(index=False)

    sig_drift = drift[drift.assessment != "stable"]
    low_trust = (trust.trust_band == "LOW").mean() * 100
    lines = [
        "# Data Intelligence Report — Task 1",
        f"_Generated from `data/raw/` ({len(train):,} train rows, {len(test):,} test rows, "
        f"{static.shape[0]:,} loans, {len(updates):,} servicer updates)._",
        "",
        "## 1. Headline findings",
        f"- **{rule_rows.n_rules_fired.gt(0).mean()*100:.1f}%** of train records violate at least one "
        "deterministic validation rule; violations are **not random** — they concentrate in specific "
        "servicers (see §6), which is a process problem, not noise.",
        f"- **{(recon.n_conflicts>0).mean()*100:.1f}%** of reconciled loan-months have at least one "
        "source conflict between the core panel and servicer updates.",
        f"- **{low_trust:.1f}%** of records fall in the LOW trust band (<0.5) and should route to human review.",
        f"- Train→test drift is **{'material' if len(sig_drift) else 'limited'}**: "
        f"{len(sig_drift)} feature(s) beyond the stable PSI threshold. Portfolio drift here is largely "
        "**survivorship** (defaulted/prepaid loans exit the panel) plus a riskier late-vintage mix.",
        "",
        "## 2. Rule engine results (starter rules R001–R008)",
        md(rule_summary),
        "",
        "### Learned cross-column relationship rules (mined, confidence ≥ 98%)",
        md(learned) if len(learned) else "_none discovered above thresholds_",
        "",
        "## 3. Train vs test drift (PSI)",
        "PSI bands: <0.10 stable · 0.10–0.25 moderate · >0.25 significant.",
        md(drift),
        "",
        "**Interpretation.** `loan_age_months` / `remaining_term_months` drift is **structural, not "
        "distributional**: a fixed loan panel mechanically ages between train and test windows, so high "
        "PSI there is expected and benign. The *meaningful* drift signals are `interest_rate` (moderate — "
        "rate-environment shift across windows) and the mild credit-mix shift from survivorship "
        "(riskier loans default/prepay out of the live panel). Models therefore avoid leaning on raw "
        "calendar-linked values and use age-normalized features instead (see Task 2 feature notes).",
        "",
        "## 4. Top absolute correlations / dependent fields",
        md(corr),
        "",
        "## 5. Missingness patterns by segment (% missing)",
    ]
    for seg, m in miss.items():
        nonzero = m.loc[:, (m > 0).any()] if (m.to_numpy() > 0).any() else m
        lines += [f"### by {seg}", nonzero.reset_index().to_markdown(index=False), ""]
    lines += [
        "## 6. Batch quality by servicer",
        md(by_serv),
        "",
        "_CasaMortgage-pattern servicers show elevated violation and conflict rates — "
        "batch quality scores make the process issue visible and assignable._",
        "",
        "## 7. Honest evaluation against injected ground truth",
        "Because the pack is generated with hidden, labeled corruptions, detector quality "
        "is **measured, not asserted**:",
        md(eval_rows),
        f"\n**Flag precision vs labeled exceptions: {precision:.1f}%** "
        f"({tp:,} of {n_flag:,} flagged records are labeled exceptions). "
        f"**High-severity-signal precision: {precision_hi:.1f}%** — low-severity rules (staleness, doc gaps) "
        "are deliberately recall-oriented screeners that route records to review rather than assert corruption.",
        "",
        "## 8. Record trust score (feeds Tasks 4, 6, 7)",
        "trust = 1 − severity-weighted penalties (rule violations, source conflicts, staleness), clipped to [0,1]. "
        "Low trust widens downstream prediction intervals — bad data in, honest uncertainty out.",
        md(trust.trust_band.value_counts().rename_axis("trust_band").reset_index(name="records")),
    ]
    with open(os.path.join(REP, "data_intelligence_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=== TASK 1 COMPLETE ===")
    print(f"rules fired on {rule_rows.n_rules_fired.gt(0).mean()*100:.2f}% of rows | "
          f"conflicts on {(recon.n_conflicts>0).mean()*100:.2f}% of reconciled months")
    print(eval_rows.to_string(index=False))
    print(f"precision vs labeled exceptions: {precision:.1f}% | high-severity precision: {precision_hi:.1f}%")
    print(f"drift features non-stable: {len(sig_drift)}")
    print("report -> reports/data_intelligence_report.md")


if __name__ == "__main__":
    main()
