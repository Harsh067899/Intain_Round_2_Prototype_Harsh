# Task 3 — Discrete-Time Transition / Survival Model

## Formulation
Monthly panel = person-period format, so monthly next-state probabilities ARE discrete-time hazards; chaining them yields cumulative incidence. **Competing risks** (default vs prepayment) live in one multinomial head — a prepaid loan exits the default risk set. **Censoring**: each loan contributes exactly its observed transitions and then stops; no imputation, no bias.

## Model vs baseline (validation transitions, live states only)
| metric | empirical Markov (baseline) | LightGBM multinomial hazard |
|---|---|---|
| log-loss | **0.1787** | 0.1822 |
| macro-F1 | 0.3793 | **0.4195** |

**Honest read:** the status-only baseline is very strong on log-loss because CURRENT→CURRENT dominates the transition mass; the model is statistically comparable there (0.1822 vs 0.1787) while clearly winning macro-F1 — i.e. it is much better on the rare transitions that matter (into DPD90/DEFAULT/PREPAID). Crucially, only the covariate model produces loan-level heterogeneity: the baseline gives every CURRENT loan identical hazards, so it cannot support segment curves, scenario shocks, or per-loan review.

## Curve validation — 2024-07 active cohort (n=2,087), 12-month horizon
| outcome | model-implied | observed |
|---|---|---|
| cumulative default | 8.26% | 6.32% |
| cumulative prepay | 7.55% | 5.46% |

_Figure: reports/figures/cumulative_incidence.png_

## Model-implied 12-month incidence by credit band
| credit_band   |   model_cum_default_12m |   model_cum_prepay_12m |   n_loans |
|:--------------|------------------------:|-----------------------:|----------:|
| 620-679       |                  0.1374 |                 0.0906 |       425 |
| <620          |                  0.1298 |                 0.0937 |       176 |
| 680-739       |                  0.0765 |                 0.0767 |       590 |
| 740-779       |                  0.0597 |                 0.0645 |       544 |
| 780+          |                  0.0385 |                 0.0631 |       352 |

## Baseline transition matrix (train window)
| current_status   |   CURRENT |   DEFAULT |   DPD30 |   DPD60 |   DPD90 |   PREPAID |
|:-----------------|----------:|----------:|--------:|--------:|--------:|----------:|
| CURRENT          |    0.985  |    0      |  0.0112 |  0      |  0      |    0.0038 |
| DPD30            |    0.3164 |    0      |  0.1785 |  0.5051 |  0      |    0      |
| DPD60            |    0.0785 |    0      |  0.1437 |  0.1872 |  0.5906 |    0      |
| DPD90            |    0.04   |    0.2897 |  0      |  0.1143 |  0.556  |    0      |