"""FastAPI application — Loan Performance Intelligence Engine API.

Endpoints:
  POST /api/v1/score       — Score a single loan
  POST /api/v1/batch-score — Score a batch of loans
  GET  /api/v1/loan/{id}   — Retrieve stored scores
  POST /api/v1/copilot/note — Generate a copilot reviewer note
  GET  /api/v1/health      — Health check
  GET  /api/v1/models      — List loaded models

Launch: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project modules are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "features"))
sys.path.insert(0, os.path.join(ROOT, "src", "profiling"))
sys.path.insert(0, os.path.join(ROOT, "src", "copilot"))

try:
    from .schemas import (
        LoanRecord, BatchScoreRequest, BatchScoreResponse, LoanScore,
        CopilotNoteRequest, CopilotNoteResponse,
        HealthResponse, ModelInfo, ModelsListResponse,
    )
except (ImportError, ValueError):
    from src.api.schemas import (
        LoanRecord, BatchScoreRequest, BatchScoreResponse, LoanScore,
        CopilotNoteRequest, CopilotNoteResponse,
        HealthResponse, ModelInfo, ModelsListResponse,
    )

MODELS_DIR = os.path.join(ROOT, "models")
TARGETS = ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
           "next_12m_default_flag", "next_12m_prepayment_flag"]
PROB_COLS = ["prob_delinq_3m", "prob_delinq_6m", "prob_default_12m", "prob_prepay_12m"]

# ── Model store (loaded at startup) ──────────────────────────────────────────
model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all joblib models at startup."""
    print("[api] Loading models...")
    for tgt in TARGETS:
        path = os.path.join(MODELS_DIR, f"{tgt}.joblib")
        if os.path.exists(path):
            model_store[tgt] = joblib.load(path)
            print(f"  [OK] {tgt}")

    for name in ["next_state", "anomaly_fusion", "exception_required",
                  "exception_type", "conformal_trust"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            model_store[name] = joblib.load(path)
            print(f"  [OK] {name}")

    print(f"[api] {len(model_store)} models loaded")
    yield
    model_store.clear()


app = FastAPI(
    title="Loan Performance Intelligence Engine API",
    description="ML-first engine for loan risk scoring, anomaly detection, and AI-assisted review.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper functions ─────────────────────────────────────────────────────────

def _predict_target(bundle: dict, X_row: pd.DataFrame) -> np.ndarray:
    """Score a single target using its champion model."""
    if bundle.get("champion") == "logistic":
        Z = bundle["scaler"].transform(X_row[bundle["features"]].astype(float))
        p = bundle["model"].predict_proba(Z)[:, 1]
    else:
        p = bundle["model"].predict_proba(X_row[bundle["features"]])[:, 1]
    return np.clip(bundle["iso"].predict(p), 0, 1)


def _score_records(records: list[LoanRecord], lgd: float = 0.40) -> list[LoanScore]:
    """Score a list of loan records through the full engine."""
    from build_features import build_features, feature_cols  # noqa: E402

    # Convert records to DataFrame
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)

    # Build features (no trust artifacts for API scoring — default trust=1.0)
    X = build_features(df, trust=None)
    feats = feature_cols(X)

    results = []
    for i, record in enumerate(records):
        X_row = X.iloc[[i]]

        # Predictions
        probs = {}
        for tgt, col in zip(TARGETS, PROB_COLS):
            if tgt in model_store:
                probs[col] = float(np.round(_predict_target(model_store[tgt], X_row)[0], 5))
            else:
                probs[col] = 0.0

        # Next state
        next_state = "CURRENT"
        if "next_state" in model_store:
            ns = model_store["next_state"]
            pns = ns["model"].predict_proba(X_row[ns["features"]])
            next_state = ns["states"][pns.argmax(1)[0]]

        # Anomaly scoring
        anomaly_score = 0.0
        exception_prob = 0.0
        exception_type = "NONE"

        if "anomaly_fusion" in model_store and "exception_required" in model_store:
            fus = model_store["anomaly_fusion"]
            req = model_store["exception_required"]
            w_rule, w_iso, w_sup = fus["weights"]

            iso_raw = -fus["iso"].score_samples(X_row[fus["iso_cols"]])
            iso_pct = 0.5  # single-record: no rank percentile, use median
            exception_prob = float(req["model"].predict_proba(X_row[req["features"]])[:, 1][0])

            n_rules = float(X_row["n_rules_fired"].iloc[0]) if "n_rules_fired" in X_row else 0
            n_conf = float(X_row["n_conflicts"].iloc[0]) if "n_conflicts" in X_row else 0
            rule_n = min((n_rules + n_conf) / 5.0, 1.0)

            anomaly_score = round(w_rule * rule_n + w_iso * iso_pct + w_sup * exception_prob, 5)

        if "exception_type" in model_store:
            typ = model_store["exception_type"]
            pt = typ["model"].predict_proba(X_row[typ["features"]])
            exception_type = typ["types"][pt.argmax(1)[0]]

        # EDL
        edl = probs.get("prob_default_12m", 0) * record.current_balance * lgd

        # Confidence
        trust_score = float(X_row["trust_score"].iloc[0]) if "trust_score" in X_row else 1.0
        confidence = 0.8
        if "conformal_trust" in model_store:
            conf = model_store["conformal_trust"]
            half = conf["q_glob"] * (1 + conf["lam"] * (1 - trust_score))
            confidence = round(max(min(1 - half, 1), 0), 4)

        # Action
        if edl > 50000 or anomaly_score > 0.7:
            action = "ESCALATE"
        elif edl > 10000 or anomaly_score > 0.4 or trust_score < 0.5:
            action = "REVIEW"
        else:
            action = "AUTO_ACCEPT"

        results.append(LoanScore(
            loan_id=record.loan_id,
            reporting_month=record.reporting_month,
            **probs,
            next_state_pred=next_state,
            anomaly_score=anomaly_score,
            exception_required_prob=round(exception_prob, 5),
            exception_type_pred=exception_type,
            expected_dollar_loss=round(edl, 2),
            confidence=confidence,
            recommended_action=action,
            trust_score=trust_score,
            top_drivers="",
        ))

    return results


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    backend = os.environ.get("LLM_BACKEND", "template")
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_store),
        llm_backend=backend,
    )


@app.get("/api/v1/models", response_model=ModelsListResponse)
async def list_models():
    """List all loaded models and their metadata."""
    models = []
    for name, bundle in model_store.items():
        if isinstance(bundle, dict):
            features = bundle.get("features", [])
            model_obj = bundle.get("model")
            model_type = type(model_obj).__name__ if model_obj else "unknown"
        else:
            features = []
            model_type = type(bundle).__name__
        models.append(ModelInfo(
            name=name,
            path=os.path.join(MODELS_DIR, f"{name}.joblib"),
            features_count=len(features),
            model_type=model_type,
        ))
    return ModelsListResponse(models=models, total=len(models))


@app.post("/api/v1/score", response_model=LoanScore)
async def score_single(record: LoanRecord):
    """Score a single loan record through the full engine."""
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded. Run the pipeline first.")
    try:
        results = _score_records([record])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/api/v1/batch-score", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest):
    """Score a batch of loans through the full engine."""
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded. Run the pipeline first.")
    try:
        results = _score_records(request.records, lgd=request.lgd_assumption)
        edls = [r.expected_dollar_loss for r in results]
        return BatchScoreResponse(
            scores=results,
            total=len(results),
            edl_summary={
                "mean": round(np.mean(edls), 2),
                "median": round(np.median(edls), 2),
                "p90": round(np.percentile(edls, 90), 2),
                "max": round(max(edls), 2),
                "total_portfolio_edl": round(sum(edls), 2),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")


@app.post("/api/v1/copilot/note", response_model=CopilotNoteResponse)
async def generate_note(request: CopilotNoteRequest):
    """Generate a grounded copilot reviewer note for a loan."""
    try:
        from copilot import CopilotClient  # noqa: E402

        bundle = {
            "loan_id": request.loan_id,
            "reporting_month": request.reporting_month,
            "current_status": request.current_status,
            "days_past_due": request.days_past_due,
            "prob_default_12m": request.prob_default_12m,
            "trust_score": request.trust_score,
            "anomaly_score": request.anomaly_score,
            "exception_type_pred": request.exception_type_pred,
            "rules_fired": request.rules_fired,
            "top_drivers": request.top_drivers or [{"feature": "N/A", "shap": 0.0}],
            "artifact_ids": [f"api_request_{request.loan_id}"],
        }

        client = CopilotClient(use_api=request.use_llm)
        rec = client.note_for(bundle)

        gc = rec.get("grounding_check", {})
        return CopilotNoteResponse(
            note=rec.get("output", ""),
            grounded=gc.get("grounded", False),
            mode=rec.get("mode", "unknown"),
            model=rec.get("model", "unknown"),
            unmatched_numbers=gc.get("unmatched_numbers", []),
            unmatched_rule_ids=gc.get("unmatched_rule_ids", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {str(e)}")


@app.get("/api/v1/loan/{loan_id}")
async def get_loan(loan_id: str):
    """Retrieve stored scores for a loan from submission.csv."""
    submission_path = os.path.join(ROOT, "submission.csv")
    if not os.path.exists(submission_path):
        raise HTTPException(status_code=404, detail="submission.csv not found. Run the pipeline first.")

    df = pd.read_csv(submission_path)
    loan_data = df[df["loan_id"] == loan_id]
    if loan_data.empty:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found in submission")

    return loan_data.to_dict(orient="records")
