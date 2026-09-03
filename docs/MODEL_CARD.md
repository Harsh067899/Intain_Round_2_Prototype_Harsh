# Model Card — Loan Performance Intelligence Engine
_Tasks 2–3 core models. Updated as tasks complete._

## Objective
Predict loan-level performance from a monthly servicing panel: 3m/6m delinquency,
12m default, 12m prepayment (binary, calibrated probabilities), next-month state
(multinomial — doubles as the discrete-time hazard engine), and data-quality
exceptions (Task 4).

## Data
Schema-locked synthetic pack matching the Section 6/7 spec (organizer pack pending;
support ticket raised Aug 27 — drop-in swap when received). 9,000 loans; 227k
labeled monthly rows 2021-07..2025-06; second-source servicer updates; injected
corruptions with hidden ground truth. SHA-256 manifest: data/raw/MANIFEST.sha256.json.

## Features
Origination attributes; current servicing snapshot; backward-looking delinquency
history (rolling 6/12m counts, prior status); data-derived refinance incentive
(rate vs same-month new-origination mean); Task 1 trust features (trust_score,
rule counts, source conflicts). No feature reads any forward-looking value —
enforced by construction and verified by a 3-seed label-permutation test
(mean AUC 0.483 ≈ 0.5).

## Validation method
Out-of-time AND out-of-loan: loans hashed into disjoint 70/30 groups; train =
group A, months ≤ 2024-06 with label horizons fully observed before cutoff;
validation = group B, months 2024-07..2025-06. **Loan overlap = 0 (asserted in
code).** Unbiased right-censoring: rows with incomplete forward horizons are
dropped entirely. Calibration fit on the first half of validation months,
metrics reported on the held-out second half.

## Models & headline metrics (held-out validation)
| target | champion | ROC-AUC | notes |
|---|---|---|---|
| 3m delinquency | LightGBM (isotonic) | 0.805 | beats logistic 0.776 |
| 6m delinquency | LightGBM (isotonic) | 0.749 | beats logistic 0.706 |
| 12m default | LightGBM (isotonic) | 0.787 | ≈ logistic 0.792 (parity, disclosed) |
| 12m prepayment | **Logistic** (isotonic) | 0.660 | beats all LGBM variants (regime shift) |
| next_state | LightGBM multinomial | macro-F1 0.582 | unweighted → honest hazards |

Calibration improves Brier on every target (e.g. default 0.0579 → 0.0474).

## Known limitations & failure modes
- Prepayment is the weakest target: driven by rate regime; tree models cannot
  extrapolate incentive beyond the training range (documented finding; linear
  champion shipped).
- Transition-model forward simulation uses representative DPD values per state and
  ages features mechanically → mild over-projection vs observed (8.3% vs 6.3%
  12m default on the validation cohort); disclosed on the curve figure.
- Labels near the panel end are right-censored; handled by strict horizon filtering.
- Synthetic data: relationships are realistic in shape and ordering but absolute
  levels are generator-chosen; all rates are calibration-anchored, not market claims.

## Action policy: Expected Dollar Loss (EDL)
Connects calibrated machine learning default probabilities directly to portfolio financial ROI:
$$\text{EDL} = P(\text{default}_{12m}) \times \text{Current Balance} \times \text{LGD}$$
- **Assumptions**: Baseline Loss Given Default ($\text{LGD} = 40\%$) for first-lien amortizing residential mortgages.
- **Triage thresholds**:
  - `ESCALATE`: $\text{EDL} > \$50,000$ OR $\text{anomaly\_score} > 0.70$ OR $(\text{EDL} > \$10,000 \land \text{anomaly\_score} > 0.40)$
  - `REVIEW`: $\text{EDL} > \$10,000$ OR $\text{anomaly\_score} > 0.40$ OR $\text{trust\_score} < 0.50$
  - `AUTO_ACCEPT`: Compliant portfolio records below risk/exposure limits.

## Deployment & execution interfaces
- **Streamlit Web UI** (`dashboard/app.py`): Multi-page interactive application featuring portfolio KPI overviews, loan risk radar charts, anomaly queues, calibration plots, scenario simulations, and copilot governance consoles.
- **FastAPI REST Service** (`src/api/main.py`): Low-latency production microservice with pre-loaded models on startup exposing single-loan (`/api/v1/score`) and vectorized batch scoring (`/api/v1/batch-score`).
- **Local On-Prem Copilot**: Ollama local inference (`http://localhost:11434/api/chat` with `llama3.1:8b`) with two-stream JSONL audit logging and automatic template fallback.

## Leakage controls
Strict horizon censoring; disjoint loan groups; backward-only history features;
permutation test; "too-good-to-be-true" review gate on every metric. Pandera
schema validation guarantees ingestion data contracts.
