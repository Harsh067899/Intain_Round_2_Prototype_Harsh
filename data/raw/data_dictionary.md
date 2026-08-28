# Data Dictionary — Loan Performance Intelligence Engine (synthetic pack v1)

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
