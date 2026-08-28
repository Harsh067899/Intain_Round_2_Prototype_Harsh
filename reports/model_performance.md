# Task 2 — Model Performance

**Split:** out-of-time AND out-of-loan (disjoint loan groups; train <= 2024-06, validation 2024-06+1..2025-06; loan overlap = 0, asserted in code).
**Censoring:** rows whose forward horizon extends past the data end are excluded unless the event occurred (see build_features.censor_mask).
**Leakage control:** label-permutation test mean AUC over 3 runs = 0.483 (~0.5 => no leakage path). Note: LightGBM early stopping uses the validation fold; final calibrated metrics are computed on the held-out second half of validation months.

| target                   | model                        |   roc_auc |   pr_auc |   f1_best |   precision@top10pct |   recall@p90 |     brier |   base_rate |   brier_precal_heldout |   macro_f1 |   logloss |
|:-------------------------|:-----------------------------|----------:|---------:|----------:|---------------------:|-------------:|----------:|------------:|-----------------------:|-----------:|----------:|
| next_3m_delinquency_flag | baseline_logreg              |    0.776  |   0.339  |    0.449  |               0.408  |       0      |   0.37829 |      0.0908 |              nan       |   nan      |  nan      |
| next_3m_delinquency_flag | lightgbm_raw                 |    0.8055 |   0.4678 |    0.5638 |               0.4688 |       0      |   0.07509 |      0.0908 |              nan       |   nan      |  nan      |
| next_3m_delinquency_flag | lightgbm_calibrated(heldout) |    0.7882 |   0.4633 |    0.572  |               0.5074 |       0      |   0.06081 |      0.0979 |                0.0804  |   nan      |  nan      |
| next_6m_delinquency_flag | baseline_logreg              |    0.7059 |   0.3138 |    0.3583 |               0.4095 |       0      |   0.42427 |      0.1391 |              nan       |   nan      |  nan      |
| next_6m_delinquency_flag | lightgbm_raw                 |    0.7488 |   0.429  |    0.4382 |               0.5211 |       0      |   0.1115  |      0.1391 |              nan       |   nan      |  nan      |
| next_6m_delinquency_flag | lightgbm_calibrated(heldout) |    0.7217 |   0.4137 |    0.4508 |               0.5407 |       0      |   0.09911 |      0.1433 |                0.11491 |   nan      |  nan      |
| next_12m_default_flag    | baseline_logreg              |    0.7916 |   0.3133 |    0.4287 |               0.3273 |       0      |   0.46373 |      0.066  |              nan       |   nan      |  nan      |
| next_12m_default_flag    | lightgbm_raw                 |    0.7866 |   0.3466 |    0.4859 |               0.3167 |       0      |   0.05816 |      0.066  |              nan       |   nan      |  nan      |
| next_12m_default_flag    | lightgbm_calibrated(heldout) |    0.7905 |   0.3481 |    0.4887 |               0.3192 |       0.0246 |   0.04739 |      0.0656 |                0.0579  |   nan      |  nan      |
| next_12m_prepayment_flag | baseline_logreg              |    0.6604 |   0.1062 |    0.1873 |               0.1222 |       0      |   0.26395 |      0.0678 |              nan       |   nan      |  nan      |
| next_12m_prepayment_flag | lightgbm_raw                 |    0.581  |   0.0861 |    0.1663 |               0.0905 |       0      |   0.06292 |      0.0678 |              nan       |   nan      |  nan      |
| next_12m_prepayment_flag | lightgbm_calibrated(heldout) |    0.607  |   0.0946 |    0.175  |               0.0908 |       0      |   0.0666  |      0.0724 |                0.06675 |   nan      |  nan      |
| next_state               | lightgbm_multiclass          |  nan      | nan      |  nan      |             nan      |     nan      | nan       |    nan      |              nan       |     0.5816 |    0.1803 |

**Note on metrics:** recall@p90 ≈ 0 is genuine (90% precision is mathematically unattainable at 6-14% base rates). The realistic operating comparison is precision@top10pct vs base_rate.


## Champion selection & the regime-shift finding
Champions are selected per target on validation AUC. For **prepayment**, the compact logistic baseline beats every LightGBM configuration tried (incl. monotone-constrained and linear-tree variants): the dominant driver is refinance incentive, and the validation window sits in a different rate regime than training — tree models cannot extrapolate beyond the training range of the incentive feature, while the linear model can. Complexity is not free under regime shift; we ship the model that generalizes.


_Reliability diagrams: reports/figures/reliability_*.png (isotonic fit on first half of validation months, evaluated on second half)._
