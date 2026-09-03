"""Pandera schema definitions for data validation at ingestion.

Enforces column types, nullable constraints, value ranges, and cross-column
checks before any processing begins — catches schema violations that rules
cannot (wrong dtype, unexpected nulls, out-of-range values).

Used at the top of run_task1.py to validate raw data pack integrity.
"""
from __future__ import annotations

try:
    import pandera.pandas as pa
    from pandera.pandas import Column, Check, DataFrameSchema
except (ImportError, AttributeError):
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema

# ── Shared enumerations ──────────────────────────────────────────────────────
CREDIT_BANDS = {"<620", "620-679", "680-739", "740-779", "780+"}
LTV_BANDS = {"<60", "60-70", "70-80", "80-90", "90+"}
DTI_BANDS = {"<20", "20-30", "30-40", "40-50", "50+"}
STATUSES = {"CURRENT", "DPD30", "DPD60", "DPD90", "DEFAULT", "PREPAID"}
DOC_STATUSES = {"COMPLETE", "PARTIAL", "MISSING_NOTE", "PENDING_REVIEW"}
LOAN_PURPOSES = {"PURCHASE", "REFINANCE", "CASHOUT_REFI"}
OCCUPANCY_TYPES = {"OWNER", "SECOND_HOME", "INVESTOR"}
PROPERTY_TYPES = {"SFR", "CONDO", "TOWNHOUSE", "MULTI_2_4"}
SOURCE_SYSTEMS = {"CORE_SVC", "LEGACY_BATCH", "API_FEED"}
EXCEPTION_TYPES = {
    "BALANCE_MISMATCH", "DATE_INVALID", "STATUS_INCONSISTENT",
    "STALE_UPDATE", "DOC_GAP", "SOURCE_CONFLICT", "DUPLICATE_RECORD", "NONE",
}


# ── Loan Monthly Performance Schema (train) ──────────────────────────────────
loan_monthly_train_schema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False, coerce=True,
                          description="Unique loan identifier"),
        "reporting_month": Column(str, nullable=False, coerce=True,
                                  checks=Check.str_matches(r"^\d{4}-\d{2}$"),
                                  description="YYYY-MM format"),
        "month_index": Column(int, nullable=False, coerce=True,
                              checks=Check.ge(0),
                              description="Months elapsed since panel start"),
        "origination_month": Column(str, nullable=False, coerce=True,
                                    checks=Check.str_matches(r"^\d{4}-\d{2}$")),
        "loan_age_months": Column(int, nullable=False, coerce=True,
                                  checks=Check.ge(0)),
        "remaining_term_months": Column(int, nullable=False, coerce=True,
                                        checks=Check.ge(0)),
        "original_balance": Column(float, nullable=False, coerce=True,
                                   checks=Check.gt(0)),
        "current_balance": Column(float, nullable=False, coerce=True,
                                  description="Must be >= 0"),
        "interest_rate": Column(float, nullable=False, coerce=True,
                                checks=[Check.gt(0), Check.lt(30)],
                                description="Note rate %"),
        "credit_score_band": Column(str, nullable=False, coerce=True,
                                    checks=Check.isin(CREDIT_BANDS)),
        "ltv_band": Column(str, nullable=False, coerce=True,
                           checks=Check.isin(LTV_BANDS)),
        "dti_band": Column(str, nullable=False, coerce=True,
                           checks=Check.isin(DTI_BANDS)),
        "state": Column(str, nullable=False, coerce=True),
        "loan_purpose": Column(str, nullable=False, coerce=True,
                               checks=Check.isin(LOAN_PURPOSES)),
        "occupancy_type": Column(str, nullable=False, coerce=True,
                                 checks=Check.isin(OCCUPANCY_TYPES)),
        "property_type": Column(str, nullable=False, coerce=True,
                                checks=Check.isin(PROPERTY_TYPES)),
        "servicer_name": Column(str, nullable=False, coerce=True),
        "current_status": Column(str, nullable=False, coerce=True,
                                 checks=Check.isin(STATUSES)),
        "days_past_due": Column(int, nullable=False, coerce=True,
                                checks=Check.ge(0)),
        "modification_flag": Column(int, nullable=False, coerce=True,
                                    checks=Check.isin([0, 1])),
        "prepayment_flag": Column(int, nullable=False, coerce=True,
                                  checks=Check.isin([0, 1])),
        "default_flag": Column(int, nullable=False, coerce=True,
                               checks=Check.isin([0, 1])),
        "loss_severity_band": Column(str, nullable=True, coerce=True),
        "last_updated_at": Column(str, nullable=False, coerce=True),
        "source_system": Column(str, nullable=False, coerce=True,
                                checks=Check.isin(SOURCE_SYSTEMS)),
        "document_status": Column(str, nullable=False, coerce=True,
                                  checks=Check.isin(DOC_STATUSES)),
        # Target columns (train only)
        "next_3m_delinquency_flag": Column(int, nullable=False, coerce=True,
                                           checks=Check.isin([0, 1])),
        "next_6m_delinquency_flag": Column(int, nullable=False, coerce=True,
                                           checks=Check.isin([0, 1])),
        "next_12m_default_flag": Column(int, nullable=False, coerce=True,
                                        checks=Check.isin([0, 1])),
        "next_12m_prepayment_flag": Column(int, nullable=False, coerce=True,
                                           checks=Check.isin([0, 1])),
        "next_state": Column(str, nullable=False, coerce=True,
                             checks=Check.isin(STATUSES)),
        "exception_required": Column(int, nullable=False, coerce=True,
                                     checks=Check.isin([0, 1])),
        "exception_type": Column(str, nullable=False, coerce=True,
                                 checks=Check.isin(EXCEPTION_TYPES)),
    },
    coerce=True,
    strict=False,  # allow extra columns
    name="LoanMonthlyPerformanceTrain",
    description="Schema for the monthly loan performance training panel",
)


# ── Loan Monthly Performance Schema (test — no target columns) ───────────────
_test_cols = {k: v for k, v in loan_monthly_train_schema.columns.items()
              if k not in {"next_3m_delinquency_flag", "next_6m_delinquency_flag",
                           "next_12m_default_flag", "next_12m_prepayment_flag",
                           "next_state", "exception_required", "exception_type"}}

loan_monthly_test_schema = DataFrameSchema(
    columns=_test_cols,
    coerce=True,
    strict=False,
    name="LoanMonthlyPerformanceTest",
    description="Schema for the monthly loan performance test panel (no targets)",
)


# ── Loan Static Attributes Schema ────────────────────────────────────────────
loan_static_schema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False, coerce=True),
        "original_balance": Column(float, nullable=False, coerce=True,
                                   checks=Check.gt(0)),
        "interest_rate": Column(float, nullable=False, coerce=True,
                                checks=[Check.gt(0), Check.lt(30)]),
        "original_term_months": Column(int, nullable=False, coerce=True,
                                       checks=Check.gt(0)),
        "credit_score_band": Column(str, nullable=False, coerce=True,
                                    checks=Check.isin(CREDIT_BANDS)),
        "ltv_band": Column(str, nullable=False, coerce=True,
                           checks=Check.isin(LTV_BANDS)),
        "dti_band": Column(str, nullable=False, coerce=True,
                           checks=Check.isin(DTI_BANDS)),
        "state": Column(str, nullable=False, coerce=True),
        "loan_purpose": Column(str, nullable=False, coerce=True,
                               checks=Check.isin(LOAN_PURPOSES)),
        "occupancy_type": Column(str, nullable=False, coerce=True,
                                 checks=Check.isin(OCCUPANCY_TYPES)),
        "property_type": Column(str, nullable=False, coerce=True,
                                checks=Check.isin(PROPERTY_TYPES)),
        "servicer_name": Column(str, nullable=False, coerce=True),
        "vintage": Column(str, nullable=False, coerce=True),
    },
    coerce=True,
    strict=False,
    name="LoanStaticAttributes",
)


# ── Servicer Updates Schema ───────────────────────────────────────────────────
servicer_updates_schema = DataFrameSchema(
    columns={
        "loan_id": Column(str, nullable=False, coerce=True),
        "update_month": Column(str, nullable=False, coerce=True,
                               checks=Check.str_matches(r"^\d{4}-\d{2}$")),
        "reported_balance": Column(float, nullable=False, coerce=True),
        "reported_status": Column(str, nullable=False, coerce=True,
                                  checks=Check.isin(STATUSES)),
        "reported_interest_rate": Column(float, nullable=False, coerce=True,
                                        checks=[Check.gt(0), Check.lt(30)]),
        "update_source": Column(str, nullable=False, coerce=True),
        "update_timestamp": Column(str, nullable=False, coerce=True),
    },
    coerce=True,
    strict=False,
    name="ServicerUpdates",
)


# ── Convenience validation runners ───────────────────────────────────────────

def validate_train(df, lazy: bool = True):
    """Validate the training panel. Returns (validated_df, errors_or_None)."""
    try:
        validated = loan_monthly_train_schema.validate(df, lazy=lazy)
        return validated, None
    except pa.errors.SchemaErrors as e:
        return df, e


def validate_test(df, lazy: bool = True):
    """Validate the test panel."""
    try:
        validated = loan_monthly_test_schema.validate(df, lazy=lazy)
        return validated, None
    except pa.errors.SchemaErrors as e:
        return df, e


def validate_static(df, lazy: bool = True):
    """Validate static attributes."""
    try:
        validated = loan_static_schema.validate(df, lazy=lazy)
        return validated, None
    except pa.errors.SchemaErrors as e:
        return df, e


def validate_updates(df, lazy: bool = True):
    """Validate servicer updates."""
    try:
        validated = servicer_updates_schema.validate(df, lazy=lazy)
        return validated, None
    except pa.errors.SchemaErrors as e:
        return df, e
