"""Pydantic request/response schemas for the Loan Intelligence API."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ──────────────────────────────────────────────────────────

class LoanRecord(BaseModel):
    """A single loan record for scoring."""
    loan_id: str = Field(..., description="Unique loan identifier")
    reporting_month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM")
    origination_month: str = Field(..., pattern=r"^\d{4}-\d{2}$")
    loan_age_months: int = Field(..., ge=0)
    remaining_term_months: int = Field(..., ge=0)
    original_balance: float = Field(..., gt=0)
    current_balance: float = Field(..., ge=0)
    interest_rate: float = Field(..., gt=0, lt=30)
    credit_score_band: str
    ltv_band: str
    dti_band: str
    state: str
    loan_purpose: str
    occupancy_type: str
    property_type: str
    servicer_name: str
    current_status: str
    days_past_due: int = Field(..., ge=0)
    modification_flag: int = Field(..., ge=0, le=1)
    prepayment_flag: int = Field(0, ge=0, le=1)
    default_flag: int = Field(0, ge=0, le=1)
    loss_severity_band: Optional[str] = None
    last_updated_at: str = ""
    source_system: str = "API_FEED"
    document_status: str = "COMPLETE"


class BatchScoreRequest(BaseModel):
    """Batch scoring request."""
    records: list[LoanRecord] = Field(..., min_length=1, max_length=5000)
    lgd_assumption: float = Field(0.40, ge=0, le=1, description="Loss-given-default assumption")


class CopilotNoteRequest(BaseModel):
    """Request to generate a copilot reviewer note."""
    loan_id: str
    reporting_month: str
    current_status: str
    days_past_due: int
    prob_default_12m: float
    trust_score: float
    anomaly_score: float
    exception_type_pred: str
    rules_fired: list[str] = []
    top_drivers: list[dict] = []
    use_llm: bool = Field(False, description="Use LLM backend (Ollama/Groq) vs template")


# ── Response schemas ─────────────────────────────────────────────────────────

class LoanScore(BaseModel):
    """Scoring result for a single loan."""
    loan_id: str
    reporting_month: str
    prob_delinq_3m: float
    prob_delinq_6m: float
    prob_default_12m: float
    prob_prepay_12m: float
    next_state_pred: str
    anomaly_score: float
    exception_required_prob: float
    exception_type_pred: str
    expected_dollar_loss: float
    confidence: float
    recommended_action: str
    trust_score: float
    top_drivers: str


class BatchScoreResponse(BaseModel):
    """Batch scoring response."""
    scores: list[LoanScore]
    total: int
    edl_summary: dict = Field(default_factory=dict, description="EDL statistics")


class CopilotNoteResponse(BaseModel):
    """Copilot note response."""
    note: str
    grounded: bool
    mode: str
    model: str
    unmatched_numbers: list[str] = []
    unmatched_rule_ids: list[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    version: str = "1.0.0"
    llm_backend: str = "template"


class ModelInfo(BaseModel):
    """Model metadata."""
    name: str
    path: str
    features_count: int
    model_type: str


class ModelsListResponse(BaseModel):
    """List of loaded models."""
    models: list[ModelInfo]
    total: int
