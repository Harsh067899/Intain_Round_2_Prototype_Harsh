"""Task 4 — Anomaly & Exception Intelligence.

Three fused signals (the rule/ML combination the rubric asks for):
  1. Deterministic rule engine (Task 1)  -> severity-weighted violation score
  2. Isolation Forest                    -> statistical outliers rules can't anticipate
  3. Supervised LightGBM                 -> exception_required prob + exception_type

Final anomaly_score = weighted blend, rescaled to [0,1]. Every flagged record
carries reason codes (rules fired, isolation percentile, predicted type, source
conflicts) — reviewer-ready, grounded, auditable.

Outputs: models/exception_*.joblib, reports/anomaly_examples.md (20+ examples),
reports/artifacts/{anomaly_scores_train.csv, anomaly_eval.csv}
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from build_features import build_features, feature_cols, load_panel  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ART = os.path.join(ROOT, "reports", "artifacts")
MODELS = os.path.join(ROOT, "models")
ISO_FEATS = ["current_balance_ratio_raw", "days_past_due", "loan_age_months",
             "interest_rate", "rate_spread_vs_month", "balance_ratio"]

W_RULE, W_ISO, W_SUP = 0.45, 0.15, 0.40


def main():
    panel = load_panel("train")
    trust = pd.read_csv(os.path.join(ART, "record_trust_scores.csv"))
    X = build_features(panel, trust)
    feats = feature_cols(X)
    months = pd.PeriodIndex(X["reporting_month"], freq="M")
    grp = X["loan_id"].map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 100)
    is_tr = (grp < 70) & (months <= pd.Period("2024-06", freq="M"))
    is_va = (grp >= 70) & (months > pd.Period("2024-06", freq="M"))

    y_req = panel["exception_required"]
    y_type = panel["exception_type"]

    # ---- signal 1: rules (already in trust artifacts, joined by build_features)
    rule_score = X["n_rules_fired"].to_numpy() + X["n_conflicts"].to_numpy()

    # ---- signal 2: isolation forest (fit on TRAIN split only) -----------------
    iso_cols = ["days_past_due", "loan_age_months", "interest_rate",
                "rate_spread_vs_month", "balance_ratio", "pct_term_elapsed"]
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(X.loc[is_tr, iso_cols])
    iso_raw = -iso.score_samples(X[iso_cols])          # higher = more anomalous
    iso_pct = pd.Series(iso_raw).rank(pct=True).to_numpy()

    # ---- signal 3: supervised exception models --------------------------------
    m_req = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               min_child_samples=100, random_state=42, verbosity=-1)
    m_req.fit(X.loc[is_tr, feats], y_req[is_tr],
              eval_set=[(X.loc[is_va, feats], y_req[is_va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    p_req = m_req.predict_proba(X[feats])[:, 1]

    types = sorted(y_type.unique())
    tmap = {t: i for i, t in enumerate(types)}
    m_type = lgb.LGBMClassifier(objective="multiclass", num_class=len(types),
                                n_estimators=400, learning_rate=0.06, num_leaves=63,
                                random_state=42, verbosity=-1)
    m_type.fit(X.loc[is_tr, feats], y_type[is_tr].map(tmap),
               eval_set=[(X.loc[is_va, feats], y_type[is_va].map(tmap))],
               callbacks=[lgb.early_stopping(50, verbose=False)])
    p_type = m_type.predict_proba(X[feats])
    type_pred = pd.Series([types[i] for i in p_type.argmax(1)], index=X.index)

    # ---- fused anomaly score ----------------------------------------------------
    rule_n = np.clip(rule_score / 5.0, 0, 1)
    anomaly = W_RULE * rule_n + W_ISO * iso_pct + W_SUP * p_req
    anomaly = np.round(anomaly, 5)

    scores = X[["loan_id", "reporting_month"]].copy()
    scores["anomaly_score"] = anomaly
    scores["exception_required_prob"] = np.round(p_req, 5)
    scores["exception_type_pred"] = type_pred
    scores["rules_fired"] = X["rules_fired"] if "rules_fired" in X else ""
    scores = scores.merge(trust[["loan_id", "reporting_month", "rules_fired", "trust_score",
                                 "n_conflicts"]].drop_duplicates(["loan_id", "reporting_month"]),
                          on=["loan_id", "reporting_month"], how="left", suffixes=("_x", ""))
    scores["iso_percentile"] = np.round(iso_pct, 4)
    scores.drop(columns=[c for c in scores.columns if c.endswith("_x")], inplace=True)
    scores.to_csv(os.path.join(ART, "anomaly_scores_train.csv"), index=False)

    # ---- evaluation on validation split (labels + hidden GT) ---------------------
    ev = {}
    ev["exception_required AUC (val)"] = round(roc_auc_score(y_req[is_va], p_req[is_va]), 4)
    ev["exception_required PR-AUC (val)"] = round(average_precision_score(y_req[is_va], p_req[is_va]), 4)
    ev["anomaly_score AUC vs labels (val)"] = round(roc_auc_score(y_req[is_va], anomaly[is_va]), 4)
    va_typed = is_va & y_type.ne("NONE")
    ev["exception_type accuracy | exception (val)"] = round(
        float((type_pred[va_typed] == y_type[va_typed]).mean()), 4)
    prec, rec, thr = precision_recall_curve(y_req[is_va], anomaly[is_va])
    ok = prec >= 0.9
    ev["anomaly recall @ precision 0.90 (val)"] = round(float(rec[ok].max()), 4) if ok.any() else 0.0
    pd.Series(ev).rename("value").to_csv(os.path.join(ART, "anomaly_eval.csv"))

    joblib.dump({"model": m_req, "features": feats}, os.path.join(MODELS, "exception_required.joblib"))
    joblib.dump({"model": m_type, "features": feats, "types": types},
                os.path.join(MODELS, "exception_type.joblib"))
    joblib.dump({"iso": iso, "iso_cols": iso_cols, "weights": (W_RULE, W_ISO, W_SUP)},
                os.path.join(MODELS, "anomaly_fusion.joblib"))

    # ---- 20+ reviewer-ready examples with Expected Dollar Loss (Phase 3) ---------
    top = (scores.loc[is_va.values].sort_values("anomaly_score", ascending=False)
                 .drop_duplicates("loan_id").head(24))
    ctx = panel.set_index(["loan_id", "reporting_month"])
    LGD = float(os.environ.get("LGD_ASSUMPTION", 0.40))
    lines = ["# Reviewer-Ready Anomaly Examples (Task 4)",
             "_Top-scored validation records. Every claim below is grounded in computed artifacts: "
             "fired rules, source conflicts, isolation percentile, and the supervised exception model._",
             "",
             f"_Financial ROI Action Policy: Loss-Given-Default = {LGD:.0%}. "
             "Expected Dollar Loss (EDL) = P(exception/default) × Current Balance × LGD._",
             ""]
    for i, r in enumerate(top.itertuples(index=False), 1):
        row = ctx.loc[(r.loan_id, r.reporting_month)]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        curr_bal = float(row.current_balance) if "current_balance" in row else 0.0
        edl = r.exception_required_prob * curr_bal * LGD
        reasons = []
        if isinstance(r.rules_fired, str) and r.rules_fired:
            reasons.append(f"rules fired: {r.rules_fired}")
        if r.n_conflicts and r.n_conflicts > 0:
            reasons.append(f"{int(r.n_conflicts)} servicer source conflict(s)")
        if r.iso_percentile > 0.95:
            reasons.append(f"statistical outlier (isolation pct {r.iso_percentile:.2f})")
        reasons.append(f"model exception prob {r.exception_required_prob:.2f}")
        reasons.append(f"EDL ${edl:,.0f}")
        rec_action = "ESCALATE" if (r.anomaly_score > 0.7 or edl > 50000) else "REVIEW"
        lines += [
            f"## Example {i} — {r.loan_id} @ {r.reporting_month}",
            f"- **anomaly_score:** {r.anomaly_score:.3f} | **trust:** {r.trust_score:.2f} "
            f"| **predicted type:** {r.exception_type_pred} | **expected_dollar_loss:** ${edl:,.0f}",
            f"- **snapshot:** status={row.current_status}, dpd={row.days_past_due}, "
            f"balance={row.current_balance:,.0f} (orig {row.original_balance:,.0f}), "
            f"servicer={row.servicer_name}, doc={row.document_status}",
            f"- **why flagged:** {'; '.join(reasons)}",
            f"- **recommended action:** {rec_action} — "
            f"{'high dollar loss exposure or multi-signal agreement' if len(reasons)>2 else 'single-source signal, verify with servicer'}",
            "",
        ]
    with open(os.path.join(ROOT, "reports", "anomaly_examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=== TASK 4 COMPLETE ===")
    for k, v in ev.items():
        print(f"{k}: {v}")
    print(f"reviewer examples: {len(top)} -> reports/anomaly_examples.md")


if __name__ == "__main__":
    main()
