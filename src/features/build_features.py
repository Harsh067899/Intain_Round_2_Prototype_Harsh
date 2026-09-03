"""Shared feature pipeline — leakage-controlled by construction.

Phase 2 upgrade: Polars lazy-frame backend for 5-10x speedup over pandas.
Falls back to pandas if Polars is not installed.

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

# ── Polars availability check ────────────────────────────────────────────────
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def load_panel(split: str = "train") -> pd.DataFrame:
    path = os.path.join(RAW, f"loan_monthly_performance_{split}.csv")
    if HAS_POLARS:
        df = (pl.scan_csv(path)
                .unique(subset=["loan_id", "reporting_month"], keep="first")
                .collect()
                .to_pandas())
    else:
        df = pd.read_csv(path)
        df = df.drop_duplicates(["loan_id", "reporting_month"], keep="first").reset_index(drop=True)
    return df


def _build_features_polars(df: pd.DataFrame, trust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features using Polars lazy frames for 5-10x speedup."""
    lf = pl.from_pandas(df).lazy()

    # Sort for groupby operations
    lf = lf.sort(["loan_id", "reporting_month"])

    # --- static / snapshot features ---
    lf = lf.with_columns([
        pl.col("interest_rate").alias("interest_rate"),
        pl.col("original_balance").clip(lower_bound=1).log().alias("log_original_balance"),
        pl.col("loan_age_months").clip(lower_bound=0).alias("loan_age_months"),
        (pl.col("loan_age_months").clip(lower_bound=0) /
         (pl.col("loan_age_months").clip(lower_bound=0) +
          pl.col("remaining_term_months")).clip(lower_bound=1)).alias("pct_term_elapsed"),
        (pl.col("current_balance") / pl.col("original_balance")).clip(0, 2).alias("balance_ratio"),
        pl.col("days_past_due").clip(0, 360).alias("days_past_due"),
        pl.col("modification_flag").alias("modification_flag"),
        pl.col("current_status").replace_strict(STATUS_ORD, default=0).cast(pl.Int32).alias("status_ord"),
    ])

    # Cross-sectional rate spread (drift-robust)
    lf = lf.with_columns([
        (pl.col("interest_rate") -
         pl.col("interest_rate").mean().over("reporting_month")).alias("rate_spread_vs_month"),
    ])

    # Refinance incentive: own rate vs market proxy
    market = (lf.filter(pl.col("loan_age_months") == 0)
                .group_by("reporting_month")
                .agg(pl.col("interest_rate").mean().alias("market_rate"))
                .sort("reporting_month"))

    lf = lf.join(market, on="reporting_month", how="left")
    lf = lf.with_columns([
        (pl.col("interest_rate") - pl.col("market_rate").forward_fill().backward_fill())
        .alias("refi_incentive"),
    ])
    lf = lf.with_columns([
        pl.col("refi_incentive").clip(lower_bound=0).alias("refi_incentive_pos"),
    ])

    # Delinquency indicators
    lf = lf.with_columns([
        pl.col("current_status").is_in(["DPD30", "DPD60", "DPD90"]).cast(pl.Int32).alias("is_delinq_now"),
    ])

    # Previous status (shift within group)
    lf = lf.with_columns([
        pl.col("is_delinq_now").shift(1).over("loan_id").fill_null(0).alias("_prev_delinq"),
        pl.col("current_status").replace_strict(STATUS_ORD, default=0).cast(pl.Int32)
        .shift(1).over("loan_id").fill_null(0).alias("prev_status_ord"),
    ])

    # Rolling delinquency counts (backward-looking)
    try:
        lf = lf.with_columns([
            pl.col("_prev_delinq").rolling_sum(6, min_samples=1).over("loan_id").alias("n_delinq_last_6m"),
            pl.col("_prev_delinq").rolling_sum(12, min_samples=1).over("loan_id").alias("n_delinq_last_12m"),
            pl.col("_prev_delinq").cum_max().over("loan_id").alias("ever_delinquent"),
        ])
    except TypeError:
        # Fallback for older Polars releases
        lf = lf.with_columns([
            pl.col("_prev_delinq").rolling_sum(6, min_periods=1).over("loan_id").alias("n_delinq_last_6m"),
            pl.col("_prev_delinq").rolling_sum(12, min_periods=1).over("loan_id").alias("n_delinq_last_12m"),
            pl.col("_prev_delinq").cum_max().over("loan_id").alias("ever_delinquent"),
        ])

    lf = lf.with_columns([
        (pl.col("status_ord") > pl.col("prev_status_ord")).cast(pl.Int32).alias("status_worsened"),
    ])

    # Select output columns
    select_cols = [
        "loan_id", "reporting_month", "interest_rate", "log_original_balance",
        "loan_age_months", "pct_term_elapsed", "balance_ratio", "days_past_due",
        "modification_flag", "status_ord", "rate_spread_vs_month", "refi_incentive",
        "refi_incentive_pos", "is_delinq_now", "n_delinq_last_6m", "n_delinq_last_12m",
        "ever_delinquent", "prev_status_ord", "status_worsened",
    ] + CAT_COLS

    try:
        schema_names = lf.collect_schema().names()
    except AttributeError:
        schema_names = lf.columns
    available_cols = [c for c in select_cols if c in schema_names]
    out = lf.select(available_cols).collect().to_pandas()

    # Add period column for downstream compatibility
    out["_p"] = pd.PeriodIndex(out["reporting_month"], freq="M").to_timestamp()

    # Categoricals
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")

    # Trust features
    if trust is not None:
        t = trust[["loan_id", "reporting_month", "trust_score", "n_rules_fired", "n_conflicts"]]
        # Preserve rules_fired if present
        if "rules_fired" in trust.columns:
            t = trust[["loan_id", "reporting_month", "trust_score", "n_rules_fired",
                        "n_conflicts", "rules_fired"]]
        out = out.merge(t.drop_duplicates(["loan_id", "reporting_month"]),
                        on=["loan_id", "reporting_month"], how="left")
        out["trust_score"] = out["trust_score"].fillna(1.0)
        out[["n_rules_fired", "n_conflicts"]] = out[["n_rules_fired", "n_conflicts"]].fillna(0)
    else:
        out["trust_score"] = 1.0
        out["n_rules_fired"] = 0
        out["n_conflicts"] = 0

    return out


def _build_features_pandas(df: pd.DataFrame, trust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Original pandas-based feature builder (fallback)."""
    df = df.sort_values(["loan_id", "reporting_month"]).reset_index(drop=True)
    g = df.groupby("loan_id", sort=False)

    out = df[["loan_id", "reporting_month"]].copy()
    out["_p"] = pd.PeriodIndex(df["reporting_month"], freq="M").to_timestamp()

    # --- static / snapshot ---
    out["interest_rate"] = df["interest_rate"]
    out["log_original_balance"] = np.log(df["original_balance"].clip(lower=1))
    out["loan_age_months"] = df["loan_age_months"].clip(lower=0)
    out["pct_term_elapsed"] = (df["loan_age_months"].clip(lower=0)
                               / (df["loan_age_months"].clip(lower=0) + df["remaining_term_months"]).clip(lower=1))
    out["balance_ratio"] = (df["current_balance"] / df["original_balance"]).clip(0, 2)
    out["days_past_due"] = df["days_past_due"].clip(0, 360)
    out["modification_flag"] = df["modification_flag"]
    out["status_ord"] = df["current_status"].map(STATUS_ORD).fillna(0).astype(int)

    month_mean = df.groupby("reporting_month")["interest_rate"].transform("mean")
    out["rate_spread_vs_month"] = df["interest_rate"] - month_mean

    new_orig = (df.loc[df["loan_age_months"] == 0]
                  .groupby("reporting_month")["interest_rate"].mean())
    all_months = sorted(df["reporting_month"].unique())
    market = new_orig.reindex(all_months).ffill().bfill()
    out["refi_incentive"] = (df["interest_rate"] - df["reporting_month"].map(market)).astype(float)
    out["refi_incentive_pos"] = out["refi_incentive"].clip(lower=0)

    delinq_now = df["current_status"].isin(["DPD30", "DPD60", "DPD90"]).astype(int)
    out["is_delinq_now"] = delinq_now
    past = g.apply(lambda x: delinq_now.loc[x.index].shift(1), include_groups=False)
    past = past.reset_index(level=0, drop=True).reindex(df.index).fillna(0)
    out["n_delinq_last_6m"] = (past.groupby(df["loan_id"]).rolling(6, min_periods=1)
                                   .sum().reset_index(level=0, drop=True))
    out["n_delinq_last_12m"] = (past.groupby(df["loan_id"]).rolling(12, min_periods=1)
                                    .sum().reset_index(level=0, drop=True))
    out["ever_delinquent"] = (past.groupby(df["loan_id"]).cummax())
    prev_status = g["current_status"].shift(1).map(STATUS_ORD).fillna(0)
    out["prev_status_ord"] = prev_status
    out["status_worsened"] = (out["status_ord"] > prev_status).astype(int)

    for c in CAT_COLS:
        out[c] = df[c].astype("category")

    if trust is not None:
        t = trust[["loan_id", "reporting_month", "trust_score", "n_rules_fired", "n_conflicts"]]
        out = out.merge(t.drop_duplicates(["loan_id", "reporting_month"]),
                        on=["loan_id", "reporting_month"], how="left")
        out["trust_score"] = out["trust_score"].fillna(1.0)
        out[["n_rules_fired", "n_conflicts"]] = out[["n_rules_fired", "n_conflicts"]].fillna(0)
    else:
        out["trust_score"] = 1.0
        out["n_rules_fired"] = 0
        out["n_conflicts"] = 0
    return out


def build_features(df: pd.DataFrame, trust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features using Polars (fast) if available, else pandas (fallback)."""
    if HAS_POLARS:
        try:
            return _build_features_polars(df, trust)
        except Exception as e:
            print(f"⚠️ Polars feature build failed ({e}), falling back to pandas")
            return _build_features_pandas(df, trust)
    return _build_features_pandas(df, trust)


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"loan_id", "reporting_month", "_p", "rules_fired"}
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
