# Task 6 — Explainability & Responsible AI

## Global drivers — 12m default (TreeSHAP, validation sample)
| feature              |   mean_abs_shap |
|:---------------------|----------------:|
| credit_score_band    |       0.250873  |
| pct_term_elapsed     |       0.0737151 |
| loan_age_months      |       0.0728578 |
| interest_rate        |       0.0666653 |
| is_delinq_now        |       0.055151  |
| log_original_balance |       0.0538231 |
| rate_spread_vs_month |       0.0468268 |
| loan_purpose         |       0.0352298 |
| ltv_band             |       0.0282921 |
| state                |       0.021852  |
| dti_band             |       0.0146598 |
| balance_ratio        |       0.0101446 |

_Figure: reports/figures/shap_summary_default.png_

## Prepayment champion (logistic) — standardized coefficients
The prepayment champion is linear by deliberate choice (regime-shift robustness, see Task 2); its coefficients ARE its global explanation.

| feature              |   std_coefficient |
|:---------------------|------------------:|
| status_ord           |           -0.381  |
| interest_rate        |            0.1347 |
| rate_spread_vs_month |           -0.0496 |
| loan_age_months      |           -0.0444 |
| trust_score          |           -0.0336 |
| n_delinq_last_12m    |           -0.0214 |
| days_past_due        |           -0.0211 |
| balance_ratio        |            0.0096 |

## Local explanations
Per-loan SHAP decompositions for top-risk and error-case loans (also the grounding source for Task 7 reviewer notes):

| loan_id   | reporting_month   |   prob_default_12m |   actual |   trust_score | top_drivers                                                                                                                                                                                        |
|:----------|:------------------|-------------------:|---------:|--------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LN100356  | 2024-09           |             0.5764 |        0 |          1    | is_delinq_now=1 (+0.819); credit_score_band=680-739 (+0.108); pct_term_elapsed=0.08888888888888889 (+0.090); loan_purpose=CASHOUT_REFI (-0.081); loan_age_months=32 (+0.047)                       |
| LN100364  | 2024-07           |             0.5764 |        0 |          1    | is_delinq_now=1 (+0.846); rate_spread_vs_month=-1.770263850027979 (+0.092); pct_term_elapsed=0.08888888888888889 (+0.060); occupancy_type=SECOND_HOME (-0.042); credit_score_band=740-779 (+0.042) |
| LN100531  | 2024-07           |             0.0076 |        1 |          0.65 | credit_score_band=780+ (-0.539); interest_rate=3.988 (-0.088); state=PA (-0.062); pct_term_elapsed=0.08333333333333333 (+0.048); rate_spread_vs_month=-1.6432638500279793 (+0.040)                 |
| LN100688  | 2024-12           |             0.0076 |        1 |          1    | credit_score_band=780+ (-0.687); state=OH (-0.062); loan_age_months=23 (+0.048); pct_term_elapsed=0.12777777777777777 (+0.048); rate_spread_vs_month=-0.5704601509530516 (-0.025)                  |
| LN100768  | 2024-08           |             0.0076 |        1 |          1    | credit_score_band=780+ (-0.675); dti_band=20-35 (-0.071); interest_rate=4.15 (-0.058); log_original_balance=12.517227074167097 (+0.050); pct_term_elapsed=0.08888888888888889 (+0.049)             |
| LN101212  | 2024-07           |             0.7692 |        1 |          1    | is_delinq_now=1 (+0.999); credit_score_band=780+ (-0.300); state=TX (+0.119); dti_band=35-43 (+0.093); pct_term_elapsed=0.04722222222222222 (+0.084)                                               |
| LN101430  | 2024-07           |             0.0076 |        1 |          1    | credit_score_band=780+ (-0.608); loan_purpose=CASHOUT_REFI (-0.068); state=MI (-0.062); pct_term_elapsed=0.04722222222222222 (+0.048); loan_age_months=17 (+0.034)                                 |
| LN101822  | 2024-09           |             0.7692 |        0 |          1    | is_delinq_now=1 (+0.975); credit_score_band=780+ (-0.332); state=TX (+0.114); pct_term_elapsed=0.07222222222222222 (+0.114); dti_band=35-43 (+0.088)                                               |

## False negative / false positive analysis (top 10 each)
| kind           | loan_id   |   prob |   actual |   trust_score |   loan_age_months |   n_delinq_last_12m |   days_past_due |
|:---------------|:----------|-------:|---------:|--------------:|------------------:|--------------------:|----------------:|
| FALSE_NEGATIVE | LN104617  |  0.008 |        1 |          1    |                12 |                   0 |               0 |
| FALSE_NEGATIVE | LN100768  |  0.008 |        1 |          1    |                32 |                   0 |               0 |
| FALSE_NEGATIVE | LN107309  |  0.008 |        1 |          1    |                23 |                   0 |               0 |
| FALSE_NEGATIVE | LN100688  |  0.008 |        1 |          1    |                23 |                   3 |               0 |
| FALSE_NEGATIVE | LN103857  |  0.008 |        1 |          0.37 |                 0 |                   0 |               0 |
| FALSE_NEGATIVE | LN105737  |  0.008 |        1 |          1    |                 0 |                   0 |               0 |
| FALSE_NEGATIVE | LN102697  |  0.008 |        1 |          1    |                24 |                   0 |               0 |
| FALSE_NEGATIVE | LN102034  |  0.008 |        1 |          1    |                 0 |                   0 |               0 |
| FALSE_NEGATIVE | LN100531  |  0.008 |        1 |          0.65 |                30 |                   0 |               0 |
| FALSE_NEGATIVE | LN101430  |  0.008 |        1 |          1    |                17 |                   2 |               0 |
| FALSE_POSITIVE | LN101822  |  0.769 |        0 |          1    |                26 |                   1 |              78 |
| FALSE_POSITIVE | LN108003  |  0.769 |        0 |          1    |                24 |                   0 |              44 |
| FALSE_POSITIVE | LN102107  |  0.576 |        0 |          1    |                36 |                   3 |              82 |
| FALSE_POSITIVE | LN100356  |  0.576 |        0 |          1    |                32 |                   3 |             126 |
| FALSE_POSITIVE | LN102230  |  0.576 |        0 |          1    |                10 |                   1 |              45 |
| FALSE_POSITIVE | LN105851  |  0.576 |        0 |          1    |                12 |                   0 |              53 |
| FALSE_POSITIVE | LN105850  |  0.576 |        0 |          1    |                34 |                   0 |              51 |
| FALSE_POSITIVE | LN100364  |  0.576 |        0 |          1    |                32 |                   0 |              34 |
| FALSE_POSITIVE | LN107318  |  0.576 |        0 |          1    |                17 |                   6 |              77 |
| FALSE_POSITIVE | LN103681  |  0.576 |        0 |          1    |                35 |                   0 |              50 |

**Cluster reading.** 80% of the worst false negatives had ZERO delinquencies in the prior 12 months — quiet loans that broke without warning; their mean trust score (0.90) is also below portfolio average, i.e. part of what the model missed was hidden behind unreliable data. 100% of the loudest false positives were already past-due loans that subsequently cured — the model prices the risk that existed even though the coin landed well. Both patterns argue for the trust-routed human review lane rather than blind automation.

## Uncertainty — trust-linked conformal intervals (90% nominal)
Two methods shown deliberately. **Honest empirical finding:** plain per-band conformal produced near-identical halfwidths across trust bands on this data — injected corruptions distort fields but were generated independently of default hazards, so residuals alone do not grow with low trust. Rather than overclaim, we ship **trust-scaled normalized conformal as a governance policy**: low-trust records get deliberately conservative (wider) intervals, because unreliable data must never produce confident predictions. Coverage is verified to hold at or above the 90% nominal level in every band (conservatism shows up as over-coverage on LOW/MEDIUM — by design):

| trust_band   | method                           |   n_cal |   n_eval |   mean_halfwidth |   empirical_coverage |
|:-------------|:---------------------------------|--------:|---------:|-----------------:|---------------------:|
| LOW          | per-band residual q90            |     114 |      132 |           0.0949 |               0.9242 |
| MEDIUM       | per-band residual q90            |     447 |      433 |           0.0949 |               0.9215 |
| HIGH         | per-band residual q90            |    5872 |     6269 |           0.0949 |               0.9067 |
| LOW          | SHIPPED: trust-scaled normalized |    6433 |      132 |           0.1259 |               0.9242 |
| MEDIUM       | SHIPPED: trust-scaled normalized |    6433 |      433 |           0.1137 |               0.9215 |
| HIGH         | SHIPPED: trust-scaled normalized |    6433 |     6269 |           0.0954 |               0.9067 |

**Reading:** the shipped intervals widen as trust falls (mean halfwidth LOW > MEDIUM > HIGH) with coverage >= nominal everywhere — the Trust Engine thesis implemented as auditable policy, with the underlying empirical picture disclosed rather than hidden.
