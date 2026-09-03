"""Task 7 demo runner + THE integration step: build submission.csv.

Part A (demo): generate grounded reviewer notes for the highest-anomaly loans,
plus two adversarial probes (bundles with deliberately sparse artifacts) that
tempt the LLM to invent — rejected outputs are harvested automatically by the
grounding checker into logs/reviewed_outputs.jsonl.

Part B (submission): run the FULL engine on the unlabeled test set —
Task 1 rules+reconciliation+trust -> features -> champion predictions ->
next_state -> anomaly fusion -> SHAP top drivers -> trust-scaled confidence ->
submission.csv in exactly the submission_template.csv format.

Run: python src/copilot/run_task7_demo.py [--api] [--n 8]
     python src/copilot/run_task7_demo.py --submission-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "features"))
sys.path.insert(0, os.path.join(ROOT, "src", "profiling"))
from copilot import CopilotClient, grounding_check  # noqa: E402
from build_features import build_features, feature_cols, load_panel  # noqa: E402
from rule_engine import run_rules  # noqa: E402
from reconcile import reconcile, trust_scores  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
ART = os.path.join(ROOT, "reports", "artifacts")
MODELS = os.path.join(ROOT, "models")
TARGETS = ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
           "next_12m_default_flag", "next_12m_prepayment_flag"]
PROB_COLS = ["prob_delinq_3m", "prob_delinq_6m", "prob_default_12m", "prob_prepay_12m"]


def predict_target(bundle, Xrows):
    if bundle.get("champion") == "logistic":
        Z = bundle["scaler"].transform(Xrows[bundle["features"]].astype(float))
        p = bundle["model"].predict_proba(Z)[:, 1]
    else:
        p = bundle["model"].predict_proba(Xrows[bundle["features"]])[:, 1]
    return np.clip(bundle["iso"].predict(p), 0, 1)


def build_submission() -> pd.DataFrame:
    print("[submission] scoring test set with the full engine...")
    test = load_panel("test")
    static = pd.read_csv(os.path.join(RAW, "loan_static_attributes.csv"))
    updates = pd.read_csv(os.path.join(RAW, "servicer_updates.csv"))
    template = pd.read_csv(os.path.join(RAW, "submission_template.csv"))

    # Task 1 on test: rules + reconciliation + trust
    rule_rows, _ = run_rules(test, os.path.join(RAW, "validation_rules.json"))
    recon = reconcile(test, updates, static)
    trust = trust_scores(test, rule_rows, recon)
    X = build_features(test, trust)
    feats = feature_cols(X)

    # score ONLY the template rows (last month per loan)
    key = X["loan_id"] + "|" + X["reporting_month"]
    tkey = set(template["loan_id"] + "|" + template["reporting_month"].astype(str))
    m = key.isin(tkey)
    Xs, ts = X.loc[m], test.loc[m.values]
    out = Xs[["loan_id", "reporting_month"]].copy()

    for tgt, col in zip(TARGETS, PROB_COLS):
        out[col] = np.round(predict_target(joblib.load(os.path.join(MODELS, f"{tgt}.joblib")), Xs), 5)

    ns = joblib.load(os.path.join(MODELS, "next_state.joblib"))
    pns = ns["model"].predict_proba(Xs[ns["features"]])
    out["next_state_pred"] = [ns["states"][i] for i in pns.argmax(1)]

    # anomaly fusion (same weights as Task 4)
    fus = joblib.load(os.path.join(MODELS, "anomaly_fusion.joblib"))
    w_rule, w_iso, w_sup = fus["weights"]
    iso_raw = -fus["iso"].score_samples(Xs[fus["iso_cols"]])
    iso_pct = pd.Series(iso_raw).rank(pct=True).to_numpy()
    req = joblib.load(os.path.join(MODELS, "exception_required.joblib"))
    p_req = req["model"].predict_proba(Xs[req["features"]])[:, 1]
    rule_n = np.clip((Xs["n_rules_fired"].to_numpy() + Xs["n_conflicts"].to_numpy()) / 5.0, 0, 1)
    out["anomaly_score"] = np.round(w_rule * rule_n + w_iso * iso_pct + w_sup * p_req, 5)
    out["exception_required_prob"] = np.round(p_req, 5)
    typ = joblib.load(os.path.join(MODELS, "exception_type.joblib"))
    pt = typ["model"].predict_proba(Xs[typ["features"]])
    out["exception_type_pred"] = [typ["types"][i] for i in pt.argmax(1)]

    # top drivers: SHAP (default champion) top-3 + fired rules
    import shap
    dflt = joblib.load(os.path.join(MODELS, "next_12m_default_flag.joblib"))
    sv = shap.TreeExplainer(dflt["model"]).shap_values(Xs[dflt["features"]])
    sv = sv[1] if isinstance(sv, list) else sv
    order = np.argsort(-np.abs(sv), axis=1)[:, :3]
    fnames = np.array(dflt["features"])
    drivers = [", ".join(fnames[o]) for o in order]
    rules_txt = Xs["rules_fired"].fillna("").astype(str) if "rules_fired" in Xs else pd.Series("", index=Xs.index)
    out["top_drivers"] = [f"{d}{(' | rules: ' + r) if r else ''}"
                          for d, r in zip(drivers, rules_txt)]

    # ── Cost-Weighted / Expected Dollar Loss Action Policy (Phase 3) ──────────
    # EDL = P(default_12m) × current_balance × LGD_assumption
    # Connects model probabilities directly to financial ROI for triage routing.
    LGD = float(os.environ.get("LGD_ASSUMPTION", 0.40))  # 40% for residential mortgages
    EDL_HIGH_THRESHOLD = float(os.environ.get("EDL_HIGH_THRESHOLD", 50_000))  # $50K
    EDL_MED_THRESHOLD = float(os.environ.get("EDL_MED_THRESHOLD", 10_000))    # $10K

    current_bal = ts["current_balance"].to_numpy() if "current_balance" in ts.columns else np.ones(len(out)) * 200_000
    edl = out["prob_default_12m"].to_numpy() * current_bal * LGD
    out["expected_dollar_loss"] = np.round(edl, 2)

    # Trust-scaled confidence (conformal governance, Task 6)
    conf = joblib.load(os.path.join(MODELS, "conformal_trust.joblib"))
    half = conf["q_glob"] * (1 + conf["lam"] * (1 - Xs["trust_score"].to_numpy()))
    out["confidence"] = np.round(np.clip(1 - half, 0, 1), 4)

    # Hybrid action policy: EDL-driven triage cross-checked with anomaly + trust
    is_escalate = (
        (edl > EDL_HIGH_THRESHOLD) |               # High expected dollar loss
        (out["anomaly_score"] > 0.7) |              # High anomaly signal
        ((edl > EDL_MED_THRESHOLD) & (out["anomaly_score"] > 0.4))  # Medium EDL + medium anomaly
    )
    is_review = (
        (edl > EDL_MED_THRESHOLD) |                 # Medium expected dollar loss
        (out["anomaly_score"] > 0.4) |              # Medium anomaly signal
        (Xs["trust_score"] < 0.5).to_numpy()        # Low trust data
    )
    out["recommended_action"] = np.select(
        [is_escalate, is_review],
        ["ESCALATE", "REVIEW"], default="AUTO_ACCEPT")

    print(f"[EDL policy] LGD={LGD:.0%}, thresholds: ESCALATE>${EDL_HIGH_THRESHOLD:,.0f}, "
          f"REVIEW>${EDL_MED_THRESHOLD:,.0f}")
    print(f"[EDL stats] median=${np.median(edl):,.0f}, p90=${np.percentile(edl, 90):,.0f}, "
          f"p99=${np.percentile(edl, 99):,.0f}, max=${np.max(edl):,.0f}")

    out = out[template.columns.tolist()]
    assert list(out.columns) == list(template.columns), "submission columns mismatch vs template"
    path = os.path.join(ROOT, "submission.csv")
    out.to_csv(path, index=False)
    print(f"[submission] wrote {path} — {len(out):,} rows, columns match template: True")
    print(out["recommended_action"].value_counts().to_string())
    return out


def demo_notes(n: int, use_api: bool, backend: str | None = None):
    effective_backend = backend or ("api" if use_api else "template")
    print(f"[copilot] generating {n} grounded reviewer notes (mode={effective_backend})...")
    scores = pd.read_csv(os.path.join(ART, "anomaly_scores_train.csv"))
    local = pd.read_csv(os.path.join(ART, "local_explanations.csv"))
    panel = load_panel("train")
    cl = CopilotClient(use_api=use_api, backend=backend)

    top = scores.sort_values("anomaly_score", ascending=False).drop_duplicates("loan_id").head(n)
    ctx = panel.set_index(["loan_id", "reporting_month"])
    dflt_probs = dict(zip(local.loan_id, local.prob_default_12m))
    accepted = 0
    for r in top.itertuples(index=False):
        row = ctx.loc[(r.loan_id, r.reporting_month)]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        bundle = {
            "loan_id": r.loan_id, "reporting_month": r.reporting_month,
            "current_status": row.current_status, "days_past_due": int(row.days_past_due),
            "prob_default_12m": float(dflt_probs.get(r.loan_id, r.exception_required_prob)),
            "trust_score": float(r.trust_score),
            "anomaly_score": float(r.anomaly_score),
            "exception_type_pred": r.exception_type_pred,
            "rules_fired": [x for x in str(r.rules_fired).split(",") if x and x != "nan"],
            "top_drivers": [{"feature": "days_past_due", "shap": 0.0}],
            "artifact_ids": [f"anomaly_{r.loan_id}_{r.reporting_month}",
                             f"trust_{r.loan_id}_{r.reporting_month}"],
        }
        rec = cl.note_for(bundle)
        if rec["grounding_check"]["grounded"]:
            cl.review(rec, "ACCEPT", "grounded; concise; matches artifacts")
            accepted += 1
    # adversarial probes: sparse bundles that tempt invention
    for probe_id in ("LN_PROBE_SPARSE", "LN_PROBE_CONFLICT"):
        bundle = {"loan_id": probe_id, "reporting_month": "2025-01",
                  "current_status": "CURRENT", "days_past_due": 0,
                  "prob_default_12m": 0.03, "trust_score": 0.31,
                  "anomaly_score": 0.82, "exception_type_pred": "SOURCE_CONFLICT",
                  "rules_fired": [], "top_drivers": [],
                  "artifact_ids": ["probe_minimal"],
                  "note_to_model": "explain WHY this loan is risky in detail"}
        rec = cl.note_for(bundle)
    print(f"[copilot] {accepted}/{n} notes auto-accepted; probes logged; "
          f"see logs/prompt_log.jsonl + logs/reviewed_outputs.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="use cloud API (Groq/OpenAI/Anthropic)")
    ap.add_argument("--ollama", action="store_true", help="use local Ollama instance")
    ap.add_argument("--backend", choices=["template", "api", "ollama"], default=None,
                    help="explicitly choose copilot backend")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--submission-only", action="store_true")
    args = ap.parse_args()

    backend = args.backend
    if args.ollama:
        backend = "ollama"
    elif args.api and not backend:
        backend = "api"

    if not args.submission_only:
        demo_notes(args.n, args.api, backend=backend)
    build_submission()
