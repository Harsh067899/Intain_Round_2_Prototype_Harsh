"""Task 1c — Servicer reconciliation + record trust scores.

servicer_updates.csv is a second source that may conflict with, lag, or duplicate
the core panel. This module detects balance/status/rate conflicts and stale
updates, then folds everything (rule violations, conflicts, staleness,
completeness) into a per-record trust score in [0, 1].

Design principle (carried over from prior entity-resolution work):
"wrong-but-confident is worse than honestly-empty" — a record with conflicting
sources keeps its data but loses trust, and that trust flows downstream into
prediction uncertainty (Task 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BALANCE_TOL = 0.02   # 2% relative tolerance between sources
RATE_TOL = 0.10      # 10bp tolerance on note rate


def reconcile(panel: pd.DataFrame, updates: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    up = updates.merge(
        panel[["loan_id", "reporting_month", "current_balance", "current_status"]],
        left_on=["loan_id", "update_month"], right_on=["loan_id", "reporting_month"],
        how="inner",
    ).merge(static[["loan_id", "interest_rate"]], on="loan_id", how="left")

    bal_rel = (up["reported_balance"] - up["current_balance"]).abs() / up["current_balance"].abs().clip(lower=1.0)
    up["balance_conflict"] = bal_rel > BALANCE_TOL
    up["status_conflict"] = up["reported_status"].ne(up["current_status"])
    up["rate_conflict"] = (up["reported_interest_rate"] - up["interest_rate"]).abs() > RATE_TOL
    up["any_conflict"] = up[["balance_conflict", "status_conflict", "rate_conflict"]].any(axis=1)
    up["balance_rel_diff"] = bal_rel.round(4)

    per_rec = (up.groupby(["loan_id", "update_month"])
                 .agg(n_updates=("any_conflict", "size"),
                      n_conflicts=("any_conflict", "sum"),
                      balance_conflict=("balance_conflict", "max"),
                      status_conflict=("status_conflict", "max"),
                      rate_conflict=("rate_conflict", "max"),
                      max_balance_rel_diff=("balance_rel_diff", "max"))
                 .reset_index()
                 .rename(columns={"update_month": "reporting_month"}))
    return per_rec


def trust_scores(panel: pd.DataFrame, rule_rows: pd.DataFrame,
                 recon: pd.DataFrame) -> pd.DataFrame:
    """Trust in [0,1]: 1 = fully consistent, fresh, complete, single-story record."""
    df = panel[["loan_id", "reporting_month", "last_updated_at", "source_system"]].copy()
    df = df.join(rule_rows[["rule_violation_score", "n_rules_fired", "rules_fired"]])
    df = df.merge(recon, on=["loan_id", "reporting_month"], how="left")
    df[["n_updates", "n_conflicts"]] = df[["n_updates", "n_conflicts"]].fillna(0)
    for c in ["balance_conflict", "status_conflict", "rate_conflict"]:
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)

    rep_ts = pd.PeriodIndex(df["reporting_month"], freq="M").to_timestamp()
    upd = pd.to_datetime(df["last_updated_at"], errors="coerce")
    stale_m = ((rep_ts - upd).dt.days / 30.44).clip(lower=0).fillna(6)

    # penalty model — transparent, severity-weighted, fully auditable
    penalty = (
        0.12 * df["rule_violation_score"]                      # deterministic violations
        + 0.25 * df["balance_conflict"].astype(float)          # source disagrees on money
        + 0.15 * df["status_conflict"].astype(float)           # source disagrees on state
        + 0.10 * df["rate_conflict"].astype(float)
        + 0.05 * stale_m.clip(upper=6) / 6 * 3                 # staleness up to 0.15
    )
    df["trust_score"] = (1.0 - penalty).clip(0.0, 1.0).round(4)
    df["trust_band"] = pd.cut(df["trust_score"], [-0.01, 0.5, 0.8, 1.0],
                              labels=["LOW", "MEDIUM", "HIGH"])
    return df[["loan_id", "reporting_month", "trust_score", "trust_band",
               "rule_violation_score", "n_rules_fired", "rules_fired",
               "n_conflicts", "balance_conflict", "status_conflict", "rate_conflict"]]


def batch_quality(trust: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    t = trust.merge(panel[["loan_id", "reporting_month", "servicer_name"]],
                    on=["loan_id", "reporting_month"], how="left")
    by_servicer = (t.groupby("servicer_name")
                     .agg(mean_trust=("trust_score", "mean"),
                          pct_low_trust=("trust_band", lambda s: 100 * (s == "LOW").mean()),
                          pct_any_rule=("n_rules_fired", lambda s: 100 * (s > 0).mean()),
                          pct_source_conflict=("n_conflicts", lambda s: 100 * (s > 0).mean()))
                     .round(3).reset_index())
    by_month = (t.groupby("reporting_month")
                  .agg(mean_trust=("trust_score", "mean"),
                       pct_low_trust=("trust_band", lambda s: 100 * (s == "LOW").mean()))
                  .round(3).reset_index())
    return by_servicer, by_month
