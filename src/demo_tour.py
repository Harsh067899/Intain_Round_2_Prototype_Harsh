"""MODELS TOUR -- run this on screen during a demo or finale Q&A.

Part 1 opens every file in models/ and shows what was trained and what's inside.
Part 2 takes ONE real loan from the test set and pushes it through every model
in sequence, printing each hand-off -- the end-to-end story on one screen.

Run: python src/demo_tour.py [loan_id]
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
RAW = os.path.join(ROOT, "data", "raw")
sys.path.insert(0, os.path.join(ROOT, "src", "features"))
sys.path.insert(0, os.path.join(ROOT, "src", "profiling"))
from build_features import build_features, feature_cols, load_panel  # noqa: E402
from rule_engine import run_rules  # noqa: E402
from reconcile import reconcile, trust_scores  # noqa: E402

BAR = "=" * 66


def describe(name):
    b = joblib.load(os.path.join(MODELS, name))
    print(f"\n--- models/{name}")
    if not isinstance(b, dict):
        print(f"    empirical Markov transition matrix, shape {getattr(b, 'shape', '?')} (counted, not trained)")
        return b
    m = b.get("model")
    kind = type(m).__name__ if m is not None else "-"
    if kind == "LGBMClassifier":
        nt = m.booster_.num_trees()
        champ = b.get("champion", "lightgbm")
        print(f"    TRAINED: LightGBM, {nt} trees, {len(b['features'])} features, champion={champ}")
    elif kind == "LogisticRegression":
        print(f"    TRAINED: LogisticRegression, {m.coef_.shape[1]} coefficients "
              f"(champion={b.get('champion')}: {b.get('reason', '')[:60]})")
    elif "iso" in b and "model" not in b and "q_glob" not in b:
        print(f"    Isolation Forest ({b['iso'].n_estimators} random trees, unsupervised) "
              f"+ fusion weights {b['weights']}")
    elif "q_glob" in b:
        print(f"    two numbers: q_glob={b['q_glob']:.4f}, lambda={b['lam']} "
              f"(conformal governance dial)")
    if "iso" in b and "model" in b:
        thresholds = getattr(b["iso"], "X_thresholds_", [])
        print(f"    + isotonic calibration staircase ({len(thresholds)} breakpoints)")
    if "states" in b:
        print(f"    states: {b['states']}")
    if "types" in b:
        print(f"    exception types: {b['types']}")
    return b


def main(loan_id=None):
    print(BAR + "\nPART 1 -- WHAT LIVES IN models/ (what we trained)\n" + BAR)
    tg = {}
    for f in sorted(os.listdir(MODELS)):
        if f.endswith(".joblib"):
            tg[f] = describe(f)

    print("\n" + BAR + "\nPART 2 -- ONE LOAN, END TO END\n" + BAR)
    test = load_panel("test")
    static = pd.read_csv(os.path.join(RAW, "loan_static_attributes.csv"))
    updates = pd.read_csv(os.path.join(RAW, "servicer_updates.csv"))

    # Task 1 on test (rules + reconcile + trust)
    rule_rows, _ = run_rules(test, os.path.join(RAW, "validation_rules.json"))
    recon = reconcile(test, updates, static)
    trust = trust_scores(test, rule_rows, recon)
    X = build_features(test, trust)
    feats = feature_cols(X)

    # pick an interesting loan: low trust at its last month
    last = X.sort_values("reporting_month").groupby("loan_id").tail(1)
    if loan_id is None:
        cand = last[(last.trust_score < 0.6)].sort_values("trust_score")
        row = cand.iloc[0] if len(cand) else last.sort_values("trust_score").iloc[0]
    else:
        row = last[last.loan_id == loan_id].iloc[0]
    lid, month = row["loan_id"], row["reporting_month"]
    trow = trust[(trust.loan_id == lid) & (trust.reporting_month == month)].iloc[0]
    snap = test[(test.loan_id == lid) & (test.reporting_month == month)].iloc[0]
    xrow = X[(X.loan_id == lid) & (X.reporting_month == month)]

    print(f"\nLOAN {lid} @ {month}")
    print(f"  snapshot     : status={snap.current_status}, dpd={snap.days_past_due}, "
          f"balance={snap.current_balance:,.0f} (orig {snap.original_balance:,.0f}), "
          f"servicer={snap.servicer_name}")

    print(f"\n[1] TRUST LAYER (no training -- rules + reconciliation + arithmetic)")
    rules_str = trow["rules_fired"] if isinstance(trow["rules_fired"], str) and trow["rules_fired"] else "none"
    print(f"  rules fired  : {rules_str} | source conflicts: {int(trow['n_conflicts'])}")
    print(f"  trust score  : {row['trust_score']:.2f}  -> this number now travels downstream")

    print(f"\n[2] FOUR CHAMPION MODELS (trained in Task 2) read the same feature row:")
    probs = {}
    for tgt, label in [("next_3m_delinquency_flag", "P(delinquent within 3m)"),
                       ("next_6m_delinquency_flag", "P(delinquent within 6m)"),
                       ("next_12m_default_flag",    "P(default within 12m) "),
                       ("next_12m_prepayment_flag", "P(prepay within 12m)  ")]:
        b = tg[f"{tgt}.joblib"]
        if b.get("champion") == "logistic":
            p = b["model"].predict_proba(b["scaler"].transform(xrow[b["features"]].astype(float)))[:, 1]
            who = "Logistic champion"
        else:
            p = b["model"].predict_proba(xrow[b["features"]])[:, 1]
            who = f"LightGBM ({b['model'].booster_.num_trees()} trees)"
        pc = float(np.clip(b["iso"].predict(p), 0, 1)[0])
        probs[tgt] = pc
        print(f"  {label} = {pc:.3f}   [{who} -> isotonic staircase]")

    print(f"\n[3] HAZARD ENGINE (trained multiclass LightGBM) -- next month's dice:")
    ns = tg["next_state.joblib"]
    dice = ns["model"].predict_proba(xrow[ns["features"]])[0]
    for s, p in sorted(zip(ns["states"], dice), key=lambda t: -t[1]):
        if p > 0.005:
            print(f"    -> {s:<8} {p:5.1%}")
    print(f"  chain these dice 12x  -> cumulative curves (Task 3); shock them -> scenarios (Task 5)")

    print(f"\n[4] ANOMALY FUSION (trained exception models + unsupervised forest + rules):")
    fus = tg["anomaly_fusion.joblib"]
    req = tg["exception_required.joblib"]
    typ = tg["exception_type.joblib"]
    iso_raw = -fus["iso"].score_samples(xrow[fus["iso_cols"]])
    p_req = float(req["model"].predict_proba(xrow[req["features"]])[:, 1][0])
    tpred = typ["types"][int(typ["model"].predict_proba(xrow[typ["features"]]).argmax(1)[0])]
    rule_n = float(np.clip((xrow["n_rules_fired"].iloc[0] + xrow["n_conflicts"].iloc[0]) / 5.0, 0, 1))
    w = fus["weights"]
    # iso percentile needs population context; use midpoint for single-row demo
    anom = w[0] * rule_n + w[1] * 0.5 + w[2] * p_req
    print(f"  exception prob={p_req:.2f}, predicted type={tpred}")
    print(f"  fused anomaly score ~= {anom:.2f}  (weights {w}: rules+forest+model)")

    print(f"\n[5] CONFORMAL GOVERNANCE (two frozen numbers) -- trust cashes in:")
    cf = tg["conformal_trust.joblib"]
    half = cf["q_glob"] * (1 + cf["lam"] * (1 - row["trust_score"]))
    lo = max(probs["next_12m_default_flag"] - half, 0)
    hi = min(probs["next_12m_default_flag"] + half, 1)
    print(f"  halfwidth = {cf['q_glob']:.3f} x (1 + {cf['lam']} x (1-{row['trust_score']:.2f})) = {half:.3f}")
    print(f"  12m default: {probs['next_12m_default_flag']:.3f} -> honest range [{lo:.3f}, {hi:.3f}]")
    print(f"  confidence column in submission.csv = {1 - half:.3f}")

    action = "ESCALATE" if anom > 0.7 else ("REVIEW" if (anom > 0.4 or row["trust_score"] < 0.5) else "AUTO_ACCEPT")
    print(f"\n[6] VERDICT -> recommended_action = {action}")
    print(f"    (copilot would now draft a note citing ONLY the numbers printed above,")
    print(f"     and the grounding checker verifies every claim against them)")
    print("\n" + BAR + "\nsame models, same order, 7,225 times = submission.csv\n" + BAR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
