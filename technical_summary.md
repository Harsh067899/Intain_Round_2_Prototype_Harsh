# 🧠 Technical Summary — Loan Performance Intelligence Engine

## What We Built
An end-to-end ML pipeline that takes messy loan-level data (9,000 loans × 36 months = ~270K records) and produces:
- **Trust scores** per record (data quality as a signal)
- **Predictions** for 4 outcomes (delinquency, default, prepayment)
- **State transition curves** & stress scenarios
- **Anomaly detection** with 24 reviewer-ready examples
- **AI-generated reviewer notes** with hallucination guardrails

**One command:** `python run_all.py` → ~5 minutes → `submission.csv` + all reports

---

## The Core Idea
> **Trust score at ingestion, propagated everywhere.**
> A per-record quality score (0 to 1) computed from rule violations, source conflicts, and staleness.
> It doesn't just flag — it changes downstream behavior:

| Where trust flows | What changes |
|---|---|
| Model features | Models learn low-trust records behave differently |
| Anomaly fusion | Low trust + outlier + rule hit = escalation |
| Prediction intervals | Intervals **widen** as trust drops (conformal scaling) |
| Review routing | AUTO_ACCEPT / REVIEW / ESCALATE per record |
| Copilot | Must cite computed facts; ungrounded claims auto-rejected |

---

## Task-by-Task Breakdown

### Task 1 — Data Intelligence & Trust Scoring
**What:** Profile data quality, detect corruption, reconcile conflicting sources, assign trust scores.

**Algorithms & Techniques:**
| Component | Method |
|---|---|
| Rule engine | 8 deterministic rules (R001–R008): balance checks, date ordering, status-DPD consistency, staleness, duplicates, doc gaps |
| Learned rules | Association rule mining (confidence ≥98%) — e.g., `PREPAID ⇒ prepayment_flag` (100% confidence) |
| Source reconciliation | Panel vs servicer_updates join on (loan_id, reporting_month); flag field-level conflicts (balance, status, DPD) |
| Trust scoring | `trust = 1 − Σ(severity_weight × penalty)` clipped to [0,1]. Severity weights: high=0.3, medium=0.15, low=0.05 |
| Drift detection | Population Stability Index (PSI) per feature, train vs test |
| Missingness | Segment-level missing% by servicer and source_system |

**Key Results:**
- 99.5–100% recall per corruption type (measured vs hidden ground truth)
- 100% precision on high-severity signals
- CasaMortgage identified as problematic servicer (15.1% violation rate vs ~9% peers)

---

### Task 2 — Loan Performance Prediction (4 targets)
**What:** Predict 4 binary outcomes per loan-month.

**Targets:**
1. `next_3m_delinquency_flag` — will this loan become delinquent within 3 months?
2. `next_6m_delinquency_flag` — within 6 months?
3. `next_12m_default_flag` — will it default within 12 months?
4. `next_12m_prepayment_flag` — will it prepay within 12 months?

**Algorithms:**
| Component | Method | Details |
|---|---|---|
| **Baseline** | Logistic Regression | L2-regularized, StandardScaler preprocessing |
| **Improved** | LightGBM (Gradient Boosted Trees) | `num_leaves=31`, `learning_rate=0.05`, early stopping on validation AUC |
| **Champion selection** | Per-target best on validation AUC | Logistic wins prepayment (regime-shift robustness) |
| **Calibration** | Isotonic Regression | Train on first half of validation, evaluate on second half |
| **Leakage test** | Label permutation test | 3-seed random label shuffle → mean AUC 0.483 (~0.5 = no leakage) |

**Feature Engineering (31 features, all backward-looking only):**
- `credit_score_band`, `ltv_band`, `dti_band` — static risk factors
- `interest_rate`, `rate_spread_vs_month` — rate environment (loan rate − monthly median)
- `loan_age_months`, `pct_term_elapsed`, `remaining_term_months` — lifecycle
- `days_past_due`, `is_delinq_now`, `status_ord` — current delinquency state
- `n_delinq_last_12m`, `worst_status_last_12m` — delinquency history
- `balance_ratio` — current_balance / original_balance
- `log_original_balance` — loan size
- `trust_score`, `n_rules_fired`, `n_conflicts` — **trust features** (the innovation)
- `modification_flag`, `loan_purpose`, `occupancy_type`, `state` — categorical

**Validation:** Out-of-time (calendar cutoff) AND out-of-loan (no loan in both sets). Overlap asserted = 0.

**Key Results:**
| Target | Champion | AUC | Brier (calibrated) |
|---|---|---|---|
| 3m delinquency | LightGBM | **0.805** | 0.061 |
| 6m delinquency | LightGBM | **0.749** | 0.099 |
| 12m default | LightGBM | **0.787** | 0.047 |
| 12m prepayment | **Logistic** | **0.660** | 0.067 |

---

### Task 3 — Transition / Survival Model
**What:** Model the loan state machine (Current → DPD30 → DPD60 → DPD90 → Default, with Prepaid as competing exit) and produce cumulative incidence curves.

**Algorithms:**
| Component | Method | Details |
|---|---|---|
| **Transition model** | LightGBM Multiclass | 6-class next-state prediction (CURRENT, DPD30, DPD60, DPD90, DEFAULT, PREPAID), 372 trees |
| **Baseline** | Empirical Markov matrix | 4×6 transition matrix counted from training data |
| **Curve generation** | Discrete-time hazard chaining | Chain monthly next-state probabilities over 12 months → cumulative default/prepay curves |
| **Censoring** | Natural observation | Each loan contributes only its observed transitions, then stops. No imputation. |

**Why this approach:**
- Monthly panel IS person-period data → next-state probabilities ARE discrete-time hazards
- Competing risks (default vs prepay) live in one multinomial head
- Same model powers Task 2 (next-state target), Task 3 (curves), and Task 5 (scenarios)
- Trained **WITHOUT class weighting** — weighting inflated rare-state hazards and corrupted simulation (14.6% vs 6.3% observed default; unweighted: 8.3%)

**Key Results:**
- Model macro-F1: **0.420** vs baseline 0.379
- 12m default: model 8.26% vs observed 6.32% (~2pp over-projection, disclosed)
- Monotone by credit band (higher risk bands → higher default)

---

### Task 4 — Anomaly & Exception Detection
**What:** Score every record for anomaly/exception likelihood, classify exception types, generate reason codes.

**Algorithms:**
| Component | Method | Details |
|---|---|---|
| **Unsupervised** | Isolation Forest | 200 trees, contamination=auto; catches statistical outliers rules can't anticipate |
| **Rule signal** | Normalized rule count | `(n_rules_fired + n_conflicts) / 5.0`, clipped to [0,1] |
| **Supervised: exception_required** | LightGBM Binary | 127 trees, 31 features; predicts P(this record needs exception handling) |
| **Supervised: exception_type** | LightGBM Multiclass | 432 trees; predicts which of 8 exception types (BALANCE_MISMATCH, DATE_INVALID, DOC_GAP, DUPLICATE_RECORD, NONE, SOURCE_CONFLICT, STALE_UPDATE, STATUS_INCONSISTENT) |
| **Fusion** | Weighted combination | `anomaly = 0.45×rule_signal + 0.15×isolation_pct + 0.40×exception_prob` |
| **Routing** | Threshold policy | >0.7 → ESCALATE, >0.4 or trust<0.5 → REVIEW, else → AUTO_ACCEPT |

**Key Results:**
- Exception AUC: **0.995**
- Recall@p90: **0.956**
- 24 grounded reviewer-ready examples with reason codes

---

### Task 5 — Scenario & Stress Simulation
**What:** Project portfolio outcomes under base, adverse credit, and high prepayment scenarios.

**Algorithms:**
| Component | Method | Details |
|---|---|---|
| **Shock application** | Hazard multipliers | Scale monthly transition hazards from `macro_scenarios.csv`, then re-normalize rows to sum to 1 |
| **Propagation** | 12-month hazard chaining | Shocked hazards compound month-over-month (delinquency shock → later default escalation) |
| **Uncertainty** | Monte Carlo simulation | 300 paths × 2,000 sampled loans; multinomial draws per month per loan |
| **Segments** | Stratified projections | By credit_score_band, vintage, state, servicer |

**Why hazard multiplication, not output multiplication:**
- Multiplying final outputs ignores dynamics
- Scaling monthly hazards lets shocks **compound** — CCAR-style stress logic

**Key Results:**
| Scenario | 12m Default | 12m Prepay |
|---|---|---|
| Base | 8.26% | 7.55% |
| Adverse Credit | **15.00%** | 5.04% |
| High Prepayment | 6.92% | **15.38%** |

Monte Carlo 90% band (adverse): **13.75% – 16.30%** (brackets expected value ✓)

---

### Task 6 — Explainability & Uncertainty
**What:** Explain predictions globally and locally, quantify uncertainty, analyze errors.

**Algorithms:**
| Component | Method | Details |
|---|---|---|
| **Global importance** | TreeSHAP | Mean absolute SHAP values across validation sample |
| **Local explanations** | TreeSHAP per-loan | Top-5 drivers for each loan (e.g., `is_delinq_now=1 (+0.819)`) |
| **Calibration** | Isotonic regression | Reliability diagrams showing predicted vs actual probabilities |
| **Uncertainty intervals** | Trust-scaled normalized conformal | `halfwidth = q_global × (1 + λ × (1 − trust_score))` where q_global=0.095, λ=0.6 |
| **Error analysis** | FP/FN deep-dive | Top-10 false positives and false negatives with feature profiles |

**Top Global Drivers (12m default):**
1. `credit_score_band` (SHAP 0.251) — dominant
2. `pct_term_elapsed` (0.074)
3. `loan_age_months` (0.073)
4. `interest_rate` (0.067)
5. `is_delinq_now` (0.055)

**Trust-scaled conformal intervals:**
| Trust Band | Halfwidth | Coverage |
|---|---|---|
| LOW | 0.126 | 92.4% |
| MEDIUM | 0.114 | 92.2% |
| HIGH | 0.095 | 90.7% |

Coverage ≥90% everywhere. LOW trust → wider intervals → honest uncertainty.

---

### Task 7 — LLM Reviewer Copilot
**What:** Generate grounded reviewer notes for flagged loans; govern AI output.

**Algorithms & Architecture:**
| Component | Method | Details |
|---|---|---|
| **Retrieval** | Mini-RAG (dependency-free) | Select dictionary entries + rules for fields present in the record; corpus ~3k tokens |
| **Generation** | Groq API (LLaMA) or template fallback | Temperature=0.0 to prevent hallucination; artifact bundle as context |
| **Grounding checker** | Programmatic claim extraction | Extracts every number and rule-ID from generated note; rejects if ANY claim not in artifact bundle |
| **Logging** | Two-stream JSONL | `prompt_log.jsonl` (prompt, model, timestamp, output, grounding verdict) + `reviewed_outputs.jsonl` (human decision + reason) |
| **Labeling** | All outputs marked | "RECOMMENDATION — human decision required" |

**Key Results:**
- 10/10 notes grounded after checker tuning
- Live rejection example: LLM editorialized 0.03 as "relatively high" and invented "3%" → auto-rejected
- Template fallback mode keeps pipeline reproducible without API keys

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML framework | **LightGBM** (gradient boosted trees) — tabular SOTA, SHAP-compatible |
| Baseline | **Scikit-learn** LogisticRegression, StandardScaler |
| Calibration | **Scikit-learn** IsotonicRegression |
| Anomaly (unsupervised) | **Scikit-learn** IsolationForest |
| Explainability | **SHAP** (TreeSHAP for LightGBM) |
| Feature engineering | **Pandas** + **NumPy** |
| Model persistence | **Joblib** |
| LLM | **Groq API** (OpenAI SDK compatible) with template fallback |
| Uncertainty | **Conformal prediction** (split conformal, trust-scaled) |
| Containerization | **Docker** |
| Reproducibility | Seeded RNG (numpy default_rng(42)), SHA-256 manifest |

---

## Data Flow in One Picture

```
8 CSV files (data/raw/)
    ↓
[Rule Engine] 8 rules + 5 learned rules
    ↓
[Reconciliation] panel ↔ servicer_updates conflicts
    ↓
[Trust Scoring] 0-1 per record
    ↓
[Feature Builder] 31 features (backward-only + trust features)
    ↓
┌─────────────────────────────────────────────────┐
│ [LightGBM × 3]     → 3m/6m delinq, 12m default │
│ [Logistic × 1]     → 12m prepayment             │
│ [LightGBM 6-class] → next-state hazard           │
│ [Isolation Forest]  → unsupervised anomaly        │
│ [LightGBM binary]  → exception_required           │
│ [LightGBM multi]   → exception_type               │
└─────────────────────────────────────────────────┘
    ↓
[Isotonic Calibration] honest probabilities
    ↓
[Anomaly Fusion] 0.45×rules + 0.15×forest + 0.40×model
    ↓
[Hazard Chaining] 12-month curves + scenario shocks + Monte Carlo
    ↓
[Conformal] trust-scaled intervals (q=0.095, λ=0.6)
    ↓
[Copilot] mini-RAG + grounding checker
    ↓
submission.csv (7,225 rows × 13 columns)
```

---

## Model Inventory (what lives in `models/`)

| File | Type | What It Does |
|---|---|---|
| `next_3m_delinquency_flag.joblib` | LightGBM (4 trees) + isotonic | Predict 3m delinquency |
| `next_6m_delinquency_flag.joblib` | LightGBM (5 trees) + isotonic | Predict 6m delinquency |
| `next_12m_default_flag.joblib` | LightGBM (2 trees) + isotonic | Predict 12m default |
| `next_12m_prepayment_flag.joblib` | LogisticRegression (8 coefs) + isotonic | Predict 12m prepayment |
| `next_state.joblib` | LightGBM multiclass (372 trees) | 6-state transition model |
| `transition_baseline.joblib` | Empirical Markov (4×6 matrix) | Baseline for comparison |
| `anomaly_fusion.joblib` | Isolation Forest (200 trees) + weights | Unsupervised anomaly |
| `exception_required.joblib` | LightGBM (127 trees) | P(exception needed) |
| `exception_type.joblib` | LightGBM (432 trees) | 8-class exception type |
| `conformal_trust.joblib` | 2 numbers: q=0.095, λ=0.6 | Trust-scaled intervals |
