"""Shared feature pipeline — leakage-controlled by construction.

Every feature is computable from information available at reporting_month:
static origination attributes, the current month's servicing snapshot, and
BACKWARD-looking history (rolling delinquency counts, months since last
delinquency). Targets are strictly forward-looking, so no feature touches them.

Age-normalized / cross-sectional features are preferred over raw calendar
values per the drift finding in Task 1 (panel aging is structural).

Reads ONLY from data/raw/ (Section 6/7 schema) + Task 1 trust artifacts, so the
official organizer pack is a drop-in replacement.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW = os.path.join(ROOT, "data", "raw")
ART = os.path.join(ROOT, "reports", "artifacts")

CAT_COLS = ["credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose",
            "occupancy_type", "property_type", "servicer_name", "current_status",
            "document_status", "source_system"]
STATUS_ORD = {"CURRENT": 0, "DPD30": 1, "DPD60": 2, "DPD90": 3, "DEFAULT": 4, "PREPAID": 5}
TARGETS_BIN = ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
               "next_12m_default_flag", "next_12m_prepayment_flag"]
HORIZON = {"next_3m_delinquency_flag": 3, "next_6m_delinquency_flag": 6,
           "next_12m_default_flag": 12, "next_12m_prepayment_flag": 12}


def load_panel(split: str = "train") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, f"loan_monthly_performance_{split}.csv"))
    # de-duplicate exact key duplicates for modeling (flagged separately by R008)
    df = df.drop_duplicates(["loan_id", "reporting_month"], keep="first").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, trust: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.sort_values(["loan_id", "reporting_month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)

    out = df[["loan_id", "reporting_month"]].copy()
    out["_p"] = pd.PeriodIndex(df["reporting_month"], freq="M").to_timestamp()

    # --- static / snapshot ---------------------------------------------------
    out["interest_rate"] = df["interest_rate"]
    out["log_original_balance"] = np.log(df["original_balance"].clip(lower=1))
    out["loan_age_months"] = df["loan_age_months"].clip(lower=0)
    out["pct_term_elapsed"] = (df["loan_age_months"].clip(lower=0)
                               / (df["loan_age_months"].clip(lower=0) + df["remaining_term_months"]).clip(lower=1))
    out["balance_ratio"] = (df["current_balance"] / df["original_balance"]).clip(0, 2)
    out["days_past_due"] = df["days_past_due"].clip(0, 360)
    out["modification_flag"] = df["modification_flag"]
    out["status_ord"] = df["current_status"].map(STATUS_ORD).fillna(0).astype(int)

    # cross-sectional rate spread (drift-robust: relative to same-month cohort)
    month_mean = df.groupby("reporting_month")["interest_rate"].transform("mean")
    out["rate_spread_vs_month"] = df["interest_rate"] - month_mean

    # refinance incentive: own rate vs market proxy = mean rate of NEW originations
    # that month (age==0), forward-filled — derived purely from the data pack
    new_orig = (df.loc[df["loan_age_months"] == 0]
                  .groupby("reporting_month")["interest_rate"].mean())
    all_months = sorted(df["reporting_month"].unique())
    market = new_orig.reindex(all_months).ffill().bfill()
    out["refi_incentive"] = (df["interest_rate"] - df["reporting_month"].map(market)).astype(float)
    out["refi_incentive_pos"] = out["refi_incentive"].clip(lower=0)

    # --- backward-looking history (shifted so only PAST months are used) -----
    delinq_now = df["current_status"].isin(["DPD30", "DPD60", "DPD90"]).astype(int)
    out["is_delinq_now"] = delinq_now
    past = g.apply(lambda x: delinq_now.loc[x.index].shift(1), include_groups=False)
    past = past.reset_index(level=0, drop=True).reindex(df.index).fillna(0)
    out["n_delinq_last_6m"] = (past.groupby(df["loan_id"]).rolling(6, min_periods=1)
                                   .sum().reset_index(level=0, drop=True))
    out["n_delinq_last_12m"] = (past.groupby(df["loan_id"]).rolling(12, min_periods=1)
                                    .sum().reset_index(level=0, drop=True))
    out["ever_delinquent"] = (past.groupby(df["loan_id"]).cummax())
    worst = df["status_ord"] if "status_ord" in df else df["current_status"].map(STATUS_ORD)
    prev_status = g["current_status"].shift(1).map(STATUS_ORD).fillna(0)
    out["prev_status_ord"] = prev_status
    out["status_worsened"] = (out["status_ord"] > prev_status).astype(int)

    # --- categoricals ---------------------------------------------------------
    for c in CAT_COLS:
        out[c] = df[c].astype("category")

    # --- trust features (Task 1 artifacts) ------------------------------------
    if trust is not None:
        t = trust[["loan_id", "reporting_month", "trust_score", "n_rules_fired", "n_conflicts"]]
        out = out.merge(t.drop_duplicates(["loan_id", "reporting_month"]),
                        on=["loan_id", "reporting_month"], how="left")
        out["trust_score"] = out["trust_score"].fillna(1.0)
        out[["n_rules_fired", "n_conflicts"]] = out[["n_rules_fired", "n_conflicts"]].fillna(0)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"loan_id", "reporting_month", "_p"}
    return [c for c in df.columns if c not in drop]


def censor_mask(df: pd.DataFrame, target: str, last_month: str) -> pd.Series:
    """Keep ONLY rows whose full forward horizon is observed (unbiased censoring).

    Rows with incomplete horizons are dropped entirely — keeping event==1 rows from
    partially observed horizons would bias the positive rate upward. `last_month`
    is the true end of the label-observation window (labels near the end of the
    train file reflect events observed through the full panel horizon).
    """
    h = HORIZON[target]
    last = pd.Period(last_month, freq="M")
    months = pd.PeriodIndex(df["reporting_month"], freq="M")
    return pd.Series((last - months).map(lambda d: d.n) >= h, index=df.index)
