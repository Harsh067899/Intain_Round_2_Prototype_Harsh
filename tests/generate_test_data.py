"""Generate a small, fast, self-contained test dataset for all 8 tasks.

Creates 50 loans × 12 months = 600 rows with KNOWN outcomes:
- 5 loans with injected corruptions (negative balance, date inversion, etc.)
- 5 loans that default (predictable: high DPD, low credit)
- 5 loans that prepay (predictable: high refi incentive)
- 5 loans with servicer conflicts (balance mismatch)
- 30 clean current loans

All data is written to tests/test_data/ so it doesn't interfere with
the real data in data/raw/. The test harness points ROOT at this directory.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

SEED = 999
N_LOANS = 50
MONTHS = pd.period_range("2024-01", "2024-12", freq="M")
TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
RAW = os.path.join(TEST_DIR, "raw")
GT = os.path.join(TEST_DIR, "ground_truth")


def _lid(i: int) -> str:
    return f"LNT{i:05d}"


def generate():
    rng = np.random.default_rng(SEED)
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(GT, exist_ok=True)

    # ---- Loan static attributes ------------------------------------------------
    statics = []
    for i in range(N_LOANS):
        lid = _lid(i)
        rate = round(rng.uniform(3.0, 7.5), 3)
        bal = int(rng.integers(100_000, 500_000))
        cr_band = ["<620", "620-660", "660-700", "700-740", "740+"][i % 5]
        ltv_band = ["<60%", "60-70%", "70-80%", "80-90%", "90%+"][i % 5]
        dti_band = ["<30%", "30-40%", "40-50%"][i % 3]
        state = ["CA", "TX", "FL", "NY", "IL"][i % 5]
        purpose = ["PURCHASE", "CASHOUT_REFI", "RATE_TERM_REFI"][i % 3]
        occ = ["PRIMARY", "SECOND", "INVESTOR"][i % 3]
        prop = ["SFR", "CONDO", "MULTI"][i % 3]
        svc = ["TestServicerA", "TestServicerB"][i % 2]
        statics.append({
            "loan_id": lid, "origination_month": "2023-06",
            "original_balance": bal, "interest_rate": rate,
            "original_term_months": 360, "credit_score_band": cr_band,
            "ltv_band": ltv_band, "dti_band": dti_band, "state": state,
            "loan_purpose": purpose, "occupancy_type": occ,
            "property_type": prop, "servicer_name": svc, "vintage": "2023",
        })
    static_df = pd.DataFrame(statics)
    static_df.to_csv(os.path.join(RAW, "loan_static_attributes.csv"), index=False)

    # ---- Monthly performance panel (train) -------------------------------------
    rows = []
    gt_rows = []         # corruption ground truth
    conflict_rows = []   # servicer conflict ground truth

    for i in range(N_LOANS):
        lid = _lid(i)
        s = statics[i]
        bal = s["original_balance"]
        rate = s["interest_rate"]
        status = "CURRENT"
        dpd = 0
        age = 6  # originated 2023-06, panel starts 2024-01

        for mi, month in enumerate(MONTHS):
            age_now = age + mi
            remaining = 360 - age_now
            cur_bal = max(bal * (1 - age_now / 360), 0)
            upd_ts = f"{month}-15T10:00:00"

            # ---- DEFAULT group: loans 0-4 start delinquent and default by month 8
            if i < 5:
                if mi < 3:
                    status, dpd = "DPD30", 42
                elif mi < 5:
                    status, dpd = "DPD60", 72
                elif mi < 7:
                    status, dpd = "DPD90", 120
                else:
                    status, dpd = "DEFAULT", 180

            # ---- PREPAY group: loans 5-9 prepay at month 6
            elif i < 10:
                if mi < 6:
                    status, dpd = "CURRENT", 0
                else:
                    status, dpd = "PREPAID", 0
                    cur_bal = 0

            # ---- CORRUPTION group: loans 10-14 have known data quality issues
            elif i < 15:
                status, dpd = "CURRENT", 0
                if i == 10 and mi == 4:
                    cur_bal = -5000  # R001: negative balance
                    gt_rows.append({"loan_id": lid, "reporting_month": str(month),
                                    "corruption_type": "BALANCE_MISMATCH"})
                elif i == 11 and mi == 5:
                    # R003: date inversion (origination AFTER reporting)
                    pass  # handled below in row construction
                    gt_rows.append({"loan_id": lid, "reporting_month": str(month),
                                    "corruption_type": "DATE_INVALID"})
                elif i == 12 and mi == 3:
                    # R004: status/DPD inconsistency
                    status, dpd = "CURRENT", 45
                    gt_rows.append({"loan_id": lid, "reporting_month": str(month),
                                    "corruption_type": "STATUS_INCONSISTENT"})
                elif i == 13 and mi == 6:
                    # R006: stale record
                    upd_ts = "2023-01-01T10:00:00"
                    gt_rows.append({"loan_id": lid, "reporting_month": str(month),
                                    "corruption_type": "STALE_UPDATE"})
                elif i == 14 and mi == 7:
                    # R007: document gap
                    pass  # handled via doc_status below

            # ---- CONFLICT group: loans 15-19 have servicer update conflicts
            elif i < 20:
                status, dpd = "CURRENT", 0

            # ---- CLEAN group: loans 20-49
            else:
                status, dpd = "CURRENT", 0
                # occasional mild delinquency for some loans to test model discrimination
                if i < 30 and mi == 8:
                    status, dpd = "DPD30", 35

            # Determine forward-looking targets
            default_ahead = 0
            delinq_3m = 0
            delinq_6m = 0
            prepay_12m = 0
            next_state = "CURRENT"

            if i < 5:  # will default
                if mi < 5:
                    delinq_3m = 1
                    delinq_6m = 1
                if mi < 7:
                    default_ahead = 1
                if mi + 1 < len(MONTHS):
                    next_status_list = ["DPD30", "DPD30", "DPD30", "DPD60", "DPD60",
                                        "DPD90", "DPD90", "DEFAULT", "DEFAULT", "DEFAULT",
                                        "DEFAULT", "DEFAULT"]
                    next_state = next_status_list[min(mi, len(next_status_list) - 1)]
            elif i < 10:  # will prepay
                if mi < 6:
                    prepay_12m = 1
                if mi + 1 < len(MONTHS) and mi < 5:
                    next_state = "CURRENT"
                elif mi + 1 < len(MONTHS):
                    next_state = "PREPAID"
            else:
                if i < 30 and mi == 7:  # loan about to hit DPD30 at mi=8
                    delinq_3m = 1
                    next_state = "DPD30"

            doc_status = "COMPLETE"
            if i == 14 and mi == 7:
                doc_status = "MISSING_NOTE"
                gt_rows.append({"loan_id": lid, "reporting_month": str(month),
                                "corruption_type": "DOC_GAP"})

            orig_month = "2023-06"
            if i == 11 and mi == 5:
                orig_month = "2025-12"  # date inversion

            exception_req = 0
            exception_type = "NONE"
            if i < 5 and mi >= 7:
                exception_req = 1
                exception_type = "STATUS_INCONSISTENT"
            elif 15 <= i < 20:
                exception_req = 1
                exception_type = "SOURCE_CONFLICT"

            row = {
                "loan_id": lid, "month_index": mi + 1,
                "reporting_month": str(month), "origination_month": orig_month,
                "loan_age_months": age_now, "remaining_term_months": remaining,
                "original_balance": s["original_balance"],
                "current_balance": round(cur_bal, 2),
                "interest_rate": rate,
                "credit_score_band": s["credit_score_band"],
                "ltv_band": s["ltv_band"], "dti_band": s["dti_band"],
                "state": s["state"], "loan_purpose": s["loan_purpose"],
                "occupancy_type": s["occupancy_type"],
                "property_type": s["property_type"],
                "servicer_name": s["servicer_name"],
                "current_status": status, "days_past_due": dpd,
                "modification_flag": 0,
                "prepayment_flag": 1 if status == "PREPAID" else 0,
                "default_flag": 1 if status == "DEFAULT" else 0,
                "loss_severity_band": "MODERATE" if status == "DEFAULT" else "NA",
                "last_updated_at": upd_ts,
                "source_system": "SYS_A",
                "document_status": doc_status,
                "next_3m_delinquency_flag": delinq_3m,
                "next_6m_delinquency_flag": delinq_6m,
                "next_12m_default_flag": default_ahead,
                "next_12m_prepayment_flag": prepay_12m,
                "next_state": next_state,
                "exception_required": exception_req,
                "exception_type": exception_type,
            }
            rows.append(row)

    panel_df = pd.DataFrame(rows)
    panel_df.to_csv(os.path.join(RAW, "loan_monthly_performance_train.csv"), index=False)

    # Test set: last 3 months only, all clean loans (20-49)
    test_rows = [r for r in rows if int(r["loan_id"][3:]) >= 20
                 and r["reporting_month"] in ["2024-10", "2024-11", "2024-12"]]
    pd.DataFrame(test_rows).to_csv(
        os.path.join(RAW, "loan_monthly_performance_test.csv"), index=False)

    # ---- Servicer updates (with conflicts for loans 15-19) ----------------------
    update_rows = []
    for i in range(N_LOANS):
        lid = _lid(i)
        s = statics[i]
        for mi, month in enumerate(MONTHS):
            if mi % 3 != 0:
                continue
            cur_bal = s["original_balance"] * (1 - (6 + mi) / 360)
            rep_bal = cur_bal
            rep_status = "CURRENT"
            rep_rate = s["interest_rate"]

            if 15 <= i < 20:
                rep_bal = cur_bal * 1.15  # >2% mismatch → conflict
                conflict_rows.append({
                    "loan_id": lid, "update_month": str(month),
                    "conflict_type": "balance_conflict"})

            update_rows.append({
                "loan_id": lid, "update_month": str(month),
                "reported_balance": round(rep_bal, 2),
                "reported_status": rep_status,
                "reported_interest_rate": rep_rate,
                "update_source": "SERVICER_PORTAL",
                "update_timestamp": f"{month}-20T14:00:00",
            })
    pd.DataFrame(update_rows).to_csv(
        os.path.join(RAW, "servicer_updates.csv"), index=False)

    # ---- Ground truth files -----------------------------------------------------
    pd.DataFrame(gt_rows).to_csv(
        os.path.join(GT, "injected_corruptions.csv"), index=False)
    pd.DataFrame(conflict_rows).to_csv(
        os.path.join(GT, "servicer_true_conflicts.csv"), index=False)

    # ---- Macro scenarios (copy from real or generate) ---------------------------
    scenarios = pd.DataFrame([
        {"scenario": "base", "delinquency_hazard_multiplier": 1.0,
         "default_hazard_multiplier": 1.0, "prepayment_multiplier": 1.0,
         "rate_shift_bps": 0, "unemployment_shift_pct": 0.0, "hpi_shift_pct": 0.0,
         "description": "Base scenario."},
        {"scenario": "adverse_credit", "delinquency_hazard_multiplier": 1.6,
         "default_hazard_multiplier": 1.8, "prepayment_multiplier": 0.7,
         "rate_shift_bps": 150, "unemployment_shift_pct": 2.5, "hpi_shift_pct": -8.0,
         "description": "Adverse credit."},
        {"scenario": "high_prepayment", "delinquency_hazard_multiplier": 0.9,
         "default_hazard_multiplier": 0.85, "prepayment_multiplier": 2.2,
         "rate_shift_bps": -175, "unemployment_shift_pct": -0.5, "hpi_shift_pct": 4.0,
         "description": "High prepayment."},
    ])
    scenarios.to_csv(os.path.join(RAW, "macro_scenarios.csv"), index=False)

    # ---- Submission template (test-set last month per loan) ---------------------
    test_df = pd.read_csv(os.path.join(RAW, "loan_monthly_performance_test.csv"))
    last = test_df.sort_values("reporting_month").drop_duplicates("loan_id", keep="last")
    template = last[["loan_id", "reporting_month"]].copy()
    for c in ["prob_delinq_3m", "prob_delinq_6m", "prob_default_12m", "prob_prepay_12m"]:
        template[c] = 0.0
    template["next_state_pred"] = "CURRENT"
    template["anomaly_score"] = 0.0
    template["exception_required_prob"] = 0.0
    template["exception_type_pred"] = "NONE"
    template["top_drivers"] = ""
    template["recommended_action"] = "AUTO_ACCEPT"
    template["confidence"] = 0.0
    template.to_csv(os.path.join(RAW, "submission_template.csv"), index=False)

    # ---- Validation rules (copy from project) -----------------------------------
    src_rules = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "validation_rules.json")
    if os.path.exists(src_rules):
        shutil.copy2(src_rules, os.path.join(RAW, "validation_rules.json"))
    else:
        rules = {"rules": [
            {"id": f"R00{i}", "name": n, "severity": s, "fields": [], "logic": ""}
            for i, (n, s) in enumerate([
                ("balance_non_negative", "high"), ("balance_le_original", "medium"),
                ("date_order_valid", "high"), ("status_dpd_consistent", "high"),
                ("closed_loan_balance", "medium"), ("stale_record", "low"),
                ("document_gap", "low"), ("duplicate_loan_month", "high"),
            ], 1)
        ]}
        with open(os.path.join(RAW, "validation_rules.json"), "w") as f:
            json.dump(rules, f, indent=2)

    # ---- Data dictionary --------------------------------------------------------
    with open(os.path.join(RAW, "data_dictionary.md"), "w") as f:
        f.write("| Field | Description |\n|---|---|\n")
        f.write("| loan_id | Unique loan identifier |\n")
        f.write("| current_balance | Outstanding principal balance |\n")
        f.write("| trust_score | Record quality score [0,1] |\n")
        f.write("| days_past_due | Number of days past due |\n")
        f.write("| current_status | Loan delinquency status |\n")
        f.write("| exception_type | Type of data exception |\n")

    print(f"[OK] Test data generated: {len(panel_df)} train rows, "
          f"{len(test_rows)} test rows, {len(gt_rows)} corruptions, "
          f"{len(conflict_rows)} conflicts")
    return TEST_DIR


if __name__ == "__main__":
    generate()
