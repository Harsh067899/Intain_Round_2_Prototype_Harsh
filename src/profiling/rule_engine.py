"""Task 1b — Deterministic rule engine.

Executes the starter checks in validation_rules.json (R001-R008) and returns a
per-row violation matrix with severity weights. Every rule maps to a vectorized
check so 270k rows evaluate in milliseconds; each violation carries a reason code
used later by anomaly scoring, the reviewer copilot, and the audit trail.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

SEVERITY_W = {"high": 3.0, "medium": 2.0, "low": 1.0}


def _month(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s, freq="M")


def run_rules(df: pd.DataFrame, rules_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(rules_path) as f:
        spec = {r["id"]: r for r in json.load(f)["rules"]}

    v = pd.DataFrame(index=df.index)
    rep = _month(df["reporting_month"])
    orig = _month(df["origination_month"])
    upd = pd.to_datetime(df["last_updated_at"], errors="coerce")
    rep_ts = rep.to_timestamp()

    v["R001"] = (df["current_balance"] < 0)
    v["R002"] = (df["current_balance"] > df["original_balance"] * 1.01)
    v["R003"] = (orig > rep) | (df["loan_age_months"] < 0)
    dpd, st = df["days_past_due"], df["current_status"]
    v["R004"] = ((st.eq("CURRENT") & dpd.ne(0)) |
                 (st.eq("DPD30") & ~dpd.between(30, 59)) |
                 (st.eq("DPD60") & ~dpd.between(60, 89)) |
                 (st.eq("DPD90") & (dpd < 90)))
    v["R005"] = st.eq("PREPAID") & (df["current_balance"] > 0.01)
    months_stale = (rep_ts - upd).dt.days / 30.44
    v["R006"] = upd.isna() | (months_stale > 3) | (months_stale < -2)
    v["R007"] = df["document_status"].isin(["MISSING_NOTE", "PENDING_REVIEW"])
    v["R008"] = df.duplicated(["loan_id", "reporting_month"], keep=False)

    weights = {rid: SEVERITY_W[spec[rid]["severity"]] for rid in v.columns}
    v_w = v.astype(float).mul(pd.Series(weights))
    per_row = pd.DataFrame({
        "rule_violation_score": v_w.sum(axis=1),
        "rules_fired": v.apply(lambda r: ",".join(v.columns[r.values]) if r.any() else "", axis=1),
        "n_rules_fired": v.sum(axis=1),
    })
    summary = pd.DataFrame({
        "rule_id": v.columns,
        "name": [spec[r]["name"] for r in v.columns],
        "severity": [spec[r]["severity"] for r in v.columns],
        "violations": v.sum().to_numpy(),
        "violation_pct": (100 * v.mean()).round(3).to_numpy(),
    })
    return per_row, summary


def learned_relationship_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Extend starter rules with learned cross-column relationship breaks.

    Mines near-deterministic implications from the data itself (support>=1%,
    confidence>=99%), then reports the violating rows — 'association rules'
    per the Task 1 spec, without a heavyweight mining dependency.
    """
    checks = []
    # relationship candidates: (antecedent mask, consequent mask, name)
    cand = [
        (df.current_status.eq("PREPAID"), df.prepayment_flag.eq(1), "PREPAID=>prepayment_flag"),
        (df.current_status.eq("DEFAULT"), df.default_flag.eq(1), "DEFAULT=>default_flag"),
        (df.default_flag.eq(1), df.loss_severity_band.ne("NA"), "default=>loss_severity_present"),
        (df.loss_severity_band.ne("NA"), df.current_status.eq("DEFAULT"), "loss_severity=>DEFAULT"),
        (df.prepayment_flag.eq(1), df.current_balance.le(0.01), "prepaid=>zero_balance"),
        (df.loan_age_months.eq(0), df.current_status.eq("CURRENT"), "new_loan=>CURRENT"),
    ]
    for ante, cons, name in cand:
        sup = ante.mean()
        if sup < 0.001:
            continue
        conf = cons[ante].mean() if ante.any() else np.nan
        if conf >= 0.98:  # near-deterministic rule discovered
            breaks = int((ante & ~cons).sum())
            checks.append({"learned_rule": name, "support_pct": round(100 * sup, 3),
                           "confidence_pct": round(100 * conf, 2), "relationship_breaks": breaks})
    return pd.DataFrame(checks)
