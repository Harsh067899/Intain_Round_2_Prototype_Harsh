"""
Synthetic Loan Data Pack Generator
===================================
Generates all 8 files from Section 6 of the Intain AI Track problem statement,
schema-locked to Section 7 field names, so the entire downstream pipeline can be
swapped onto the real organizer data pack with zero refactoring.

Design notes
------------
- Loans evolve through a monthly state machine:
  CURRENT -> DPD30 -> DPD60 -> DPD90 -> DEFAULT (absorbing)
  with cure transitions back toward CURRENT, and CURRENT -> PREPAID (absorbing).
- Hazards depend on credit band, LTV band, DTI band, loan age (seasoning ramp),
  refinance incentive (note rate vs market rate), and a macro cycle.
- Targets (next_3m/6m delinquency, next_12m default/prepay, next_state) are
  computed from the simulated future, then the panel is truncated so labels
  never leak simulation internals.
- Messiness is injected AFTER clean simulation, with a hidden ground-truth file
  (data/ground_truth/) recording every corruption for honest anomaly-detector
  evaluation. Ground truth is NOT part of the shipped 8-file pack.
- servicer_updates.csv is generated as a genuinely conflicting second source.

Run: python src/datagen/generate.py
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ----------------------------- configuration ------------------------------

N_LOANS = 9000
SIM_START = pd.Period("2021-07", freq="M")   # earliest origination
TRAIN_END = pd.Period("2025-06", freq="M")   # last labeled reporting month
TEST_END = pd.Period("2025-12", freq="M")    # unlabeled test window end
CORRUPTION_RATE = 0.035                       # share of panel rows corrupted
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data")

STATES = ["CURRENT", "DPD30", "DPD60", "DPD90", "DEFAULT", "PREPAID"]
ABSORBING = {"DEFAULT", "PREPAID"}

CREDIT_BANDS = ["<620", "620-679", "680-739", "740-779", "780+"]
CREDIT_RISK = {"<620": 2.6, "620-679": 1.8, "680-739": 1.2, "740-779": 0.8, "780+": 0.5}
LTV_BANDS = ["<=60", "60-75", "75-85", "85-95", ">95"]
LTV_RISK = {"<=60": 0.7, "60-75": 0.9, "75-85": 1.1, "85-95": 1.4, ">95": 1.8}
DTI_BANDS = ["<=20", "20-35", "35-43", ">43"]
DTI_RISK = {"<=20": 0.8, "20-35": 1.0, "35-43": 1.2, ">43": 1.5}

STATES_US = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "MI", "PA", "AZ", "WA"]
STATE_W = [0.16, 0.12, 0.10, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.07, 0.06, 0.07]
STATE_RISK = {"FL": 1.25, "MI": 1.2, "OH": 1.15, "GA": 1.1, "AZ": 1.1}  # default 1.0

SERVICERS = ["AlphaServ", "BetaLoan", "CasaMortgage", "DeltaHome", "EagleTrust"]
SERVICER_W = [0.30, 0.25, 0.20, 0.15, 0.10]
# CasaMortgage is deliberately sloppy -> more stale/conflicting updates
SERVICER_MESS = {"AlphaServ": 0.6, "BetaLoan": 0.8, "CasaMortgage": 2.5, "DeltaHome": 1.0, "EagleTrust": 1.2}

PURPOSES = ["PURCHASE", "REFINANCE", "CASHOUT_REFI"]
OCCUPANCY = ["OWNER", "SECOND_HOME", "INVESTOR"]
PROPERTY = ["SFR", "CONDO", "TOWNHOUSE", "MULTI_2_4"]
DOC_STATUS = ["COMPLETE", "PARTIAL", "MISSING_NOTE", "PENDING_REVIEW"]

EXCEPTION_TYPES = [
    "BALANCE_MISMATCH", "DATE_INVALID", "STATUS_INCONSISTENT",
    "STALE_UPDATE", "DOC_GAP", "SOURCE_CONFLICT", "DUPLICATE_RECORD",
]


def market_rate(period: pd.Period) -> float:
    """Simple market mortgage-rate cycle: rises through 2022-23, eases 2025."""
    t = (period - SIM_START).n
    return 3.2 + 2.8 / (1 + np.exp(-(t - 14) / 5)) - 0.6 / (1 + np.exp(-(t - 44) / 4))


def macro_stress(period: pd.Period) -> float:
    """Mild credit-stress cycle peaking in 2023-24."""
    t = (period - SIM_START).n
    return 1.0 + 0.35 * np.exp(-((t - 30) ** 2) / 250)


def seasoning(age: np.ndarray) -> np.ndarray:
    """Delinquency seasoning ramp: peaks around 24-40 months."""
    return np.clip(age / 24.0, 0.15, 1.0) * (1 + 0.2 * np.exp(-((age - 32) ** 2) / 500))


# ----------------------------- static attributes ---------------------------

def gen_static() -> pd.DataFrame:
    n = N_LOANS
    orig_offset = RNG.integers(0, (TRAIN_END - SIM_START).n - 5, n)  # ensure >=6 obs months
    orig_month = np.array([SIM_START + int(o) for o in orig_offset])
    # later vintages skew slightly riskier -> real train/test drift signal
    late = orig_offset > np.quantile(orig_offset, 0.7)
    credit_p = np.where(
        late[:, None],
        np.array([[0.14, 0.24, 0.30, 0.20, 0.12]]),
        np.array([[0.08, 0.18, 0.30, 0.26, 0.18]]),
    )
    credit = np.array([RNG.choice(CREDIT_BANDS, p=p) for p in credit_p])
    df = pd.DataFrame({
        "loan_id": [f"LN{100000 + i}" for i in range(n)],
        "origination_month": [str(m) for m in orig_month],
        "original_balance": np.round(np.exp(RNG.normal(12.45, 0.45, n)).clip(40_000, 1_500_000), -2),
        "interest_rate": None,  # filled below vs market rate at origination
        "original_term_months": RNG.choice([180, 240, 360], n, p=[0.12, 0.08, 0.80]),
        "credit_score_band": credit,
        "ltv_band": RNG.choice(LTV_BANDS, n, p=[0.14, 0.24, 0.28, 0.24, 0.10]),
        "dti_band": RNG.choice(DTI_BANDS, n, p=[0.18, 0.42, 0.28, 0.12]),
        "state": RNG.choice(STATES_US, n, p=STATE_W),
        "loan_purpose": RNG.choice(PURPOSES, n, p=[0.55, 0.30, 0.15]),
        "occupancy_type": RNG.choice(OCCUPANCY, n, p=[0.82, 0.08, 0.10]),
        "property_type": RNG.choice(PROPERTY, n, p=[0.68, 0.16, 0.11, 0.05]),
        "servicer_name": RNG.choice(SERVICERS, n, p=SERVICER_W),
        "vintage": [str(m)[:4] for m in orig_month],
    })
    mr = np.array([market_rate(m) for m in orig_month])
    spread = {b: s for b, s in zip(CREDIT_BANDS, [1.6, 1.1, 0.7, 0.4, 0.2])}
    df["interest_rate"] = np.round(mr + df["credit_score_band"].map(spread).to_numpy()
                                   + RNG.normal(0, 0.18, n), 3)
    # latent risk multiplier used by the simulator (never shipped)
    risk = (df["credit_score_band"].map(CREDIT_RISK).to_numpy()
            * df["ltv_band"].map(LTV_RISK).to_numpy()
            * df["dti_band"].map(DTI_RISK).to_numpy()
            * df["state"].map(lambda s: STATE_RISK.get(s, 1.0)).to_numpy()
            * np.where(df["occupancy_type"] == "INVESTOR", 1.3, 1.0)
            * np.exp(RNG.normal(0, 0.25, n)))
    df["_risk"] = risk
    return df


# ----------------------------- panel simulation ----------------------------

@dataclass
class SimResult:
    panel: pd.DataFrame


def simulate_panel(static: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in static.rename(columns={"_risk":"risk_latent"}).itertuples(index=False):
        orig = pd.Period(rec.origination_month, freq="M")
        bal = rec.original_balance
        r_m = rec.interest_rate / 100 / 12
        term = rec.original_term_months
        pay = bal * (r_m * (1 + r_m) ** term) / ((1 + r_m) ** term - 1)
        state, dpd, mod = "CURRENT", 0, 0
        month = orig
        age = 0
        while month <= TEST_END and state not in ABSORBING:
            mstress = macro_stress(month)
            seas = float(seasoning(np.array([age]))[0])
            # hazards
            h_d30 = min(0.011 * rec.risk_latent * mstress * seas, 0.25)
            refi = max(rec.interest_rate - market_rate(month), 0.0)
            h_pre = min((0.004 + 0.006 * refi) * (0.3 if age < 6 else 1.0), 0.20)
            u = RNG.random()
            if state == "CURRENT":
                if u < h_pre:
                    nstate = "PREPAID"
                elif u < h_pre + h_d30:
                    nstate = "DPD30"
                else:
                    nstate = "CURRENT"
            elif state == "DPD30":
                p_worse, p_cure = min(0.32 * (rec.risk_latent ** 0.4) * mstress, 0.6), 0.34
                nstate = ("DPD60" if u < p_worse else
                          "CURRENT" if u < p_worse + p_cure else "DPD30")
            elif state == "DPD60":
                p_worse, p_back, p_cure = min(0.38 * (rec.risk_latent ** 0.4) * mstress, 0.65), 0.14, 0.08
                nstate = ("DPD90" if u < p_worse else
                          "DPD30" if u < p_worse + p_back else
                          "CURRENT" if u < p_worse + p_back + p_cure else "DPD60")
            else:  # DPD90
                p_def = min(0.16 * (rec.risk_latent ** 0.5) * mstress, 0.5)
                p_back, p_cure = 0.10, 0.04
                if mod == 0 and RNG.random() < 0.05:
                    mod = 1
                    p_def *= 0.6
                nstate = ("DEFAULT" if u < p_def else
                          "DPD60" if u < p_def + p_back else
                          "CURRENT" if u < p_def + p_back + p_cure else "DPD90")

            # bookkeeping for the CURRENT row (state during `month`)
            dpd = {"CURRENT": 0, "DPD30": 30 + int(RNG.integers(0, 25)),
                   "DPD60": 60 + int(RNG.integers(0, 25)),
                   "DPD90": 90 + int(RNG.integers(0, 60)),
                   "DEFAULT": 180, "PREPAID": 0}[state]
            rows.append((rec.loan_id, str(month), age, max(term - age, 0),
                         round(bal, 2), state, dpd, mod,
                         1 if state == "PREPAID" else 0,
                         1 if state == "DEFAULT" else 0,
                         nstate))
            # amortize if paying
            if state in ("CURRENT",):
                bal = max(bal * (1 + r_m) - pay, 0.0)
            if nstate == "PREPAID":
                bal = 0.0
            state = nstate
            month += 1
            age += 1
        # record absorbing entry row once
        if state in ABSORBING and month <= TEST_END:
            dpd = 180 if state == "DEFAULT" else 0
            rows.append((rec.loan_id, str(month), age, max(term - age, 0),
                         0.0 if state == "PREPAID" else round(bal, 2),
                         state, dpd, mod,
                         int(state == "PREPAID"), int(state == "DEFAULT"), state))
    panel = pd.DataFrame(rows, columns=[
        "loan_id", "reporting_month", "loan_age_months", "remaining_term_months",
        "current_balance", "current_status", "days_past_due", "modification_flag",
        "prepayment_flag", "default_flag", "_next_state",
    ])
    return panel


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["loan_id", "reporting_month"]).reset_index(drop=True)
    panel["_p"] = panel["reporting_month"].map(lambda s: pd.Period(s, freq="M"))
    delinq = panel["current_status"].isin(["DPD30", "DPD60", "DPD90"]).astype(int)
    panel["_is_delinq"] = delinq

    out = []
    for _, g in panel.groupby("loan_id", sort=False):
        g = g.sort_values("_p")
        d = g["_is_delinq"].to_numpy()
        dflt = g["default_flag"].to_numpy()
        pre = g["prepayment_flag"].to_numpy()
        n = len(g)

        def fwd_any(arr, k):
            res = np.zeros(n, dtype=int)
            for i in range(n):
                res[i] = int(arr[i + 1: min(i + 1 + k, n)].any())
            return res

        g = g.assign(
            next_3m_delinquency_flag=fwd_any(d, 3),
            next_6m_delinquency_flag=fwd_any(d, 6),
            next_12m_default_flag=fwd_any(dflt, 12),
            next_12m_prepayment_flag=fwd_any(pre, 12),
            next_state=g["_next_state"],
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------- messiness injection -------------------------

def inject_mess(panel: pd.DataFrame, static: pd.DataFrame):
    df = panel.copy()
    df["source_system"] = RNG.choice(["CORE_SVC", "LEGACY_BATCH", "API_FEED"],
                                     len(df), p=[0.6, 0.25, 0.15])
    # honest last_updated_at: reporting month end + small lag
    lag = RNG.integers(0, 6, len(df))
    df["last_updated_at"] = [
        str((pd.Period(m, freq="M") + 1).to_timestamp() + pd.Timedelta(days=int(l)))[:10]
        for m, l in zip(df["reporting_month"], lag)
    ]
    df["document_status"] = RNG.choice(DOC_STATUS, len(df), p=[0.86, 0.08, 0.03, 0.03])
    df["loss_severity_band"] = np.where(
        df["current_status"].eq("DEFAULT"),
        RNG.choice(["<10", "10-25", "25-40", ">40"], len(df), p=[0.25, 0.4, 0.25, 0.1]),
        "NA",
    )

    mess_w = static.set_index("loan_id")["servicer_name"].map(SERVICER_MESS)
    w = df["loan_id"].map(mess_w).to_numpy()
    w = w / w.sum()
    n_bad = int(CORRUPTION_RATE * len(df))
    bad_idx = RNG.choice(len(df), size=n_bad, replace=False, p=w)
    kinds = RNG.choice(
        ["BALANCE_MISMATCH", "DATE_INVALID", "STATUS_INCONSISTENT",
         "STALE_UPDATE", "DOC_GAP", "DUPLICATE_RECORD"],
        n_bad, p=[0.24, 0.14, 0.22, 0.18, 0.12, 0.10],
    )
    gt = []
    dup_rows = []
    orig_bal = static.set_index("loan_id")["original_balance"]
    for idx, kind in zip(bad_idx, kinds):
        i = int(idx)
        lid, month = df.at[i, "loan_id"], df.at[i, "reporting_month"]
        if kind == "BALANCE_MISMATCH":
            mode = RNG.random()
            if mode < 0.4:
                df.at[i, "current_balance"] = float(orig_bal[lid]) * float(RNG.uniform(1.05, 1.6))
            elif mode < 0.7:
                df.at[i, "current_balance"] = -abs(df.at[i, "current_balance"])
            else:
                df.at[i, "current_balance"] = df.at[i, "current_balance"] * 100  # unit error
        elif kind == "DATE_INVALID":
            df.at[i, "last_updated_at"] = str(
                pd.Period(month, freq="M").to_timestamp() - pd.Timedelta(days=int(RNG.integers(200, 900))))[:10]
            if RNG.random() < 0.5:
                df.at[i, "loan_age_months"] = -int(RNG.integers(1, 12))
        elif kind == "STATUS_INCONSISTENT":
            if df.at[i, "current_status"] == "CURRENT":
                df.at[i, "days_past_due"] = int(RNG.integers(45, 120))
            else:
                df.at[i, "days_past_due"] = 0
        elif kind == "STALE_UPDATE":
            df.at[i, "last_updated_at"] = str(
                (pd.Period(month, freq="M") - int(RNG.integers(4, 10))).to_timestamp())[:10]
        elif kind == "DOC_GAP":
            df.at[i, "document_status"] = RNG.choice(["MISSING_NOTE", "PENDING_REVIEW"])
        elif kind == "DUPLICATE_RECORD":
            r = df.iloc[i].copy()
            if RNG.random() < 0.5:
                r["current_balance"] = r["current_balance"] * float(RNG.uniform(0.97, 1.03))
            dup_rows.append(r)
        gt.append({"loan_id": lid, "reporting_month": month, "corruption_type": kind})

    if dup_rows:
        df = pd.concat([df, pd.DataFrame(dup_rows)], ignore_index=True)
    gt = pd.DataFrame(gt)

    # exception labels for TRAIN visibility
    key = df["loan_id"] + "|" + df["reporting_month"]
    gt_key = set(gt["loan_id"] + "|" + gt["reporting_month"])
    gt_type = dict(zip(gt["loan_id"] + "|" + gt["reporting_month"], gt["corruption_type"]))
    df["exception_required"] = key.isin(gt_key).astype(int)
    df["exception_type"] = key.map(gt_type).fillna("NONE")
    return df, gt


# ----------------------------- servicer updates -----------------------------

def gen_servicer_updates(panel: pd.DataFrame, static: pd.DataFrame):
    sample = panel.sample(frac=0.18, random_state=7).copy()
    conflict = RNG.random(len(sample)) < 0.30
    upd = pd.DataFrame({
        "loan_id": sample["loan_id"].to_numpy(),
        "update_month": sample["reporting_month"].to_numpy(),
        "reported_balance": np.where(
            conflict,
            sample["current_balance"].to_numpy() * RNG.uniform(0.85, 1.25, len(sample)),
            sample["current_balance"].to_numpy() * RNG.uniform(0.995, 1.005, len(sample)),
        ).round(2),
        "reported_status": np.where(
            conflict & (RNG.random(len(sample)) < 0.5),
            RNG.choice(["CURRENT", "DPD30", "DPD60"], len(sample)),
            sample["current_status"].to_numpy(),
        ),
        "reported_interest_rate": np.round(
            sample["loan_id"].map(static.set_index("loan_id")["interest_rate"]).to_numpy()
            + np.where(conflict, RNG.normal(0, 0.35, len(sample)), 0.0), 3),
        "update_source": RNG.choice(["SERVICER_PORTAL", "MONTHLY_TAPE", "MANUAL_ENTRY"],
                                    len(sample), p=[0.5, 0.35, 0.15]),
        "update_timestamp": [
            str((pd.Period(m, freq="M") - int(RNG.integers(0, 5))).to_timestamp()
                + pd.Timedelta(days=int(RNG.integers(0, 27))))[:10]
            for m in sample["reporting_month"]
        ],
    })
    gt_conf = pd.DataFrame({
        "loan_id": sample["loan_id"].to_numpy(), "update_month": sample["reporting_month"].to_numpy(),
        "is_true_conflict": conflict.astype(int),
    })
    return upd.reset_index(drop=True), gt_conf


# ----------------------------- small config files ---------------------------

DATA_DICTIONARY = """# Data Dictionary — Loan Performance Intelligence Engine (synthetic pack v1)

## loan_monthly_performance_train.csv / _test.csv (panel: one row per loan per month)
| Field | Definition |
|---|---|
| loan_id | Unique loan identifier (stable across files). |
| reporting_month | Month the record describes, YYYY-MM. |
| month_index | Months elapsed since panel start (0-based). |
| origination_month | Month the loan was originated, YYYY-MM. |
| loan_age_months | Age of the loan in months at reporting_month. Must be >= 0. |
| remaining_term_months | Contractual months remaining. |
| original_balance | Balance at origination (USD). |
| current_balance | Outstanding principal at reporting_month (USD). Must be >= 0 and <= original_balance for amortizing loans. |
| interest_rate | Note rate (%, fixed). |
| credit_score_band | Origination credit band: <620, 620-679, 680-739, 740-779, 780+. |
| ltv_band | Loan-to-value band at origination. |
| dti_band | Debt-to-income band at origination. |
| state | US state of the property. |
| loan_purpose | PURCHASE / REFINANCE / CASHOUT_REFI. |
| occupancy_type | OWNER / SECOND_HOME / INVESTOR. |
| property_type | SFR / CONDO / TOWNHOUSE / MULTI_2_4. |
| servicer_name | Current servicer. |
| current_status | CURRENT, DPD30, DPD60, DPD90, DEFAULT, PREPAID. |
| days_past_due | Days past due; must be consistent with current_status. |
| modification_flag | 1 if the loan has been modified. |
| prepayment_flag | 1 in the month the loan fully prepays (absorbing). |
| default_flag | 1 in the month the loan defaults (absorbing). |
| loss_severity_band | Loss severity band for defaulted loans, else NA. |
| last_updated_at | Date the record was last refreshed by the source system. |
| source_system | CORE_SVC / LEGACY_BATCH / API_FEED. |
| document_status | COMPLETE / PARTIAL / MISSING_NOTE / PENDING_REVIEW. |

## Targets (train only)
| Field | Definition |
|---|---|
| next_3m_delinquency_flag | 1 if the loan is 30+ DPD at any point in the next 3 months. |
| next_6m_delinquency_flag | 1 if 30+ DPD at any point in the next 6 months. |
| next_12m_default_flag | 1 if the loan defaults within the next 12 months. |
| next_12m_prepayment_flag | 1 if the loan fully prepays within the next 12 months. |
| next_state | current_status in the following month. |
| exception_required | 1 if the record needs human review (data-quality exception). |
| exception_type | BALANCE_MISMATCH, DATE_INVALID, STATUS_INCONSISTENT, STALE_UPDATE, DOC_GAP, SOURCE_CONFLICT, DUPLICATE_RECORD, NONE. |

## loan_static_attributes.csv
Origination-level record per loan: original_balance, interest_rate, original_term_months,
credit_score_band, ltv_band, dti_band, state, loan_purpose, occupancy_type, property_type,
servicer_name, vintage.

## servicer_updates.csv
Second-source updates. Fields: loan_id, update_month, reported_balance, reported_status,
reported_interest_rate, update_source, update_timestamp. May conflict with, lag, or duplicate
the core panel — use for source-conflict detection, staleness logic and reconciliation.

## Label caveat
Horizon labels near the end of the training window are right-censored (the future is not fully
observed). The survival/transition model handles censoring explicitly; classification models
should drop or down-weight rows whose full horizon is unobserved.
"""

VALIDATION_RULES = {
    "rules": [
        {"id": "R001", "name": "balance_non_negative", "severity": "high",
         "fields": ["current_balance"], "logic": "current_balance >= 0"},
        {"id": "R002", "name": "balance_le_original", "severity": "medium",
         "fields": ["current_balance", "original_balance"],
         "logic": "current_balance <= original_balance * 1.01 (amortizing, small accrual tolerance)"},
        {"id": "R003", "name": "date_order_valid", "severity": "high",
         "fields": ["origination_month", "reporting_month"],
         "logic": "origination_month <= reporting_month and loan_age_months >= 0"},
        {"id": "R004", "name": "status_dpd_consistent", "severity": "high",
         "fields": ["current_status", "days_past_due"],
         "logic": "CURRENT => dpd==0; DPD30 => 30<=dpd<60; DPD60 => 60<=dpd<90; DPD90 => dpd>=90"},
        {"id": "R005", "name": "closed_loan_balance", "severity": "medium",
         "fields": ["current_status", "current_balance"],
         "logic": "PREPAID => current_balance == 0"},
        {"id": "R006", "name": "stale_record", "severity": "low",
         "fields": ["last_updated_at", "reporting_month"],
         "logic": "last_updated_at within 3 months of reporting_month"},
        {"id": "R007", "name": "document_gap", "severity": "low",
         "fields": ["document_status"],
         "logic": "document_status in (MISSING_NOTE, PENDING_REVIEW) requires exception review"},
        {"id": "R008", "name": "duplicate_loan_month", "severity": "high",
         "fields": ["loan_id", "reporting_month"],
         "logic": "(loan_id, reporting_month) must be unique"},
    ]
}

MACRO_SCENARIOS = pd.DataFrame([
    {"scenario": "base", "delinquency_hazard_multiplier": 1.00, "default_hazard_multiplier": 1.00,
     "prepayment_multiplier": 1.00, "rate_shift_bps": 0, "unemployment_shift_pct": 0.0,
     "hpi_shift_pct": 0.0, "description": "Current macro conditions persist."},
    {"scenario": "adverse_credit", "delinquency_hazard_multiplier": 1.60, "default_hazard_multiplier": 1.80,
     "prepayment_multiplier": 0.70, "rate_shift_bps": 150, "unemployment_shift_pct": 2.5,
     "hpi_shift_pct": -8.0, "description": "Credit stress: unemployment up, home prices down, rates higher."},
    {"scenario": "high_prepayment", "delinquency_hazard_multiplier": 0.90, "default_hazard_multiplier": 0.85,
     "prepayment_multiplier": 2.20, "rate_shift_bps": -175, "unemployment_shift_pct": -0.5,
     "hpi_shift_pct": 4.0, "description": "Rate rally triggers a refinance wave."},
])


def submission_template(test: pd.DataFrame) -> pd.DataFrame:
    last = test.sort_values("reporting_month").groupby("loan_id").tail(1)
    return pd.DataFrame({
        "loan_id": last["loan_id"].to_numpy(),
        "reporting_month": last["reporting_month"].to_numpy(),
        "prob_delinq_3m": 0.0, "prob_delinq_6m": 0.0,
        "prob_default_12m": 0.0, "prob_prepay_12m": 0.0,
        "next_state_pred": "CURRENT",
        "anomaly_score": 0.0,
        "exception_required_prob": 0.0,
        "exception_type_pred": "NONE",
        "top_drivers": "",
        "recommended_action": "AUTO_ACCEPT",
        "confidence": 0.0,
    })


# ----------------------------- main -----------------------------------------

def main():
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ground_truth"), exist_ok=True)

    print("1/6 static attributes...")
    static = gen_static()

    print("2/6 simulating monthly panel (state machine)...")
    panel = simulate_panel(static)

    print("3/6 computing forward-looking targets...")
    panel = add_targets(panel)

    # join static context onto the panel (Section 7 says panel carries static too)
    keep_static = ["loan_id", "origination_month", "original_balance", "interest_rate",
                   "credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose",
                   "occupancy_type", "property_type", "servicer_name"]
    panel = panel.merge(static[keep_static], on="loan_id", how="left")
    panel["month_index"] = panel["_p"].map(lambda p: (p - SIM_START).n)

    print("4/6 injecting messiness + ground truth...")
    panel, gt_corrupt = inject_mess(panel, static)

    print("5/6 servicer updates (conflicting second source)...")
    upd, gt_conf = gen_servicer_updates(panel, static)
    # SOURCE_CONFLICT exceptions where a true conflict exists
    conf_key = set((gt_conf.loc[gt_conf.is_true_conflict == 1, "loan_id"]
                    + "|" + gt_conf.loc[gt_conf.is_true_conflict == 1, "update_month"]))
    key = panel["loan_id"] + "|" + panel["reporting_month"]
    mask = key.isin(conf_key) & panel["exception_type"].eq("NONE")
    panel.loc[mask, "exception_required"] = 1
    panel.loc[mask, "exception_type"] = "SOURCE_CONFLICT"

    print("6/6 splitting train/test and writing files...")
    cols = ["loan_id", "month_index", "reporting_month", "origination_month",
            "loan_age_months", "remaining_term_months", "original_balance",
            "current_balance", "interest_rate", "credit_score_band", "ltv_band",
            "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
            "servicer_name", "current_status", "days_past_due", "modification_flag",
            "prepayment_flag", "default_flag", "loss_severity_band",
            "last_updated_at", "source_system", "document_status"]
    tcols = ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
             "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
             "exception_required", "exception_type"]

    is_train = panel["_p"] <= TRAIN_END
    train = panel.loc[is_train, cols + tcols]
    test = panel.loc[~is_train, cols]

    raw = os.path.join(OUT, "raw")
    train.to_csv(os.path.join(raw, "loan_monthly_performance_train.csv"), index=False)
    test.to_csv(os.path.join(raw, "loan_monthly_performance_test.csv"), index=False)
    static.drop(columns=["_risk"]).to_csv(os.path.join(raw, "loan_static_attributes.csv"), index=False)
    upd.to_csv(os.path.join(raw, "servicer_updates.csv"), index=False)
    with open(os.path.join(raw, "data_dictionary.md"), "w", encoding="utf-8") as f:
        f.write(DATA_DICTIONARY)
    with open(os.path.join(raw, "validation_rules.json"), "w", encoding="utf-8") as f:
        json.dump(VALIDATION_RULES, f, indent=2)
    MACRO_SCENARIOS.to_csv(os.path.join(raw, "macro_scenarios.csv"), index=False)
    submission_template(test).to_csv(os.path.join(raw, "submission_template.csv"), index=False)

    gt_corrupt.to_csv(os.path.join(OUT, "ground_truth", "injected_corruptions.csv"), index=False)
    gt_conf.to_csv(os.path.join(OUT, "ground_truth", "servicer_true_conflicts.csv"), index=False)

    # dataset fingerprint for reproducibility / audit trail
    manifest = {}
    for fn in sorted(os.listdir(raw)):
        with open(os.path.join(raw, fn), "rb") as f:
            manifest[fn] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(OUT, "raw", "MANIFEST.sha256.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # quick sanity summary
    print("\n=== SANITY SUMMARY ===")
    print(f"loans: {static.shape[0]:,} | train rows: {len(train):,} | test rows: {len(test):,}")
    print(f"train window: {train.reporting_month.min()} .. {train.reporting_month.max()}")
    print(f"test window : {test.reporting_month.min()} .. {test.reporting_month.max()}")
    print("status mix (train):")
    print((train.current_status.value_counts(normalize=True) * 100).round(2).to_string())
    for t in tcols[:4]:
        print(f"{t}: {train[t].mean() * 100:.2f}% positive")
    print(f"exception_required: {train.exception_required.mean() * 100:.2f}% "
          f"| types: {train.loc[train.exception_type != 'NONE', 'exception_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
