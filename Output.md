PS G:\loan-intel> python run_all.py

1/6 static attributes...
2/6 simulating monthly panel (state machine)...
3/6 computing forward-looking targets...
4/6 injecting messiness + ground truth...
5/6 servicer updates (conflicting second source)...
6/6 splitting train/test and writing files...

=== SANITY SUMMARY ===
loans: 9,000 | train rows: 227,446 | test rows: 42,078
train window: 2021-07 .. 2025-06
test window : 2025-07 .. 2025-12
status mix (train):
current_status
CURRENT    95.12
DPD30       1.65
DPD90       1.30
DPD60       1.13
PREPAID     0.43
DEFAULT     0.36
next_3m_delinquency_flag: 6.98% positive
next_6m_delinquency_flag: 11.13% positive
next_12m_default_flag: 5.32% positive
next_12m_prepayment_flag: 5.48% positive
exception_required: 9.09% | types: {'SOURCE_CONFLICT': 11956, 'BALANCE_MISMATCH': 1893, 'STATUS_INCONSISTENT': 1770, 'DUPLICATE_RECORD': 1458, 'STALE_UPDATE': 1431, 'DATE_INVALID': 1142, 'DOC_GAP': 1016}
   (64s)

== Task 1: profiling + rules + trust ==
=== TASK 1 COMPLETE ===
rules fired on 9.64% of rows | conflicts on 30.00% of reconciled months
    corruption_type  injected  caught  recall_pct
   BALANCE_MISMATCH      1893    1888       99.74
       DATE_INVALID      1142    1142      100.00
            DOC_GAP      1016    1016      100.00
   DUPLICATE_RECORD       729     729      100.00
       STALE_UPDATE      1431    1431      100.00
STATUS_INCONSISTENT      1770    1762       99.55
precision vs labeled exceptions: 61.3% | high-severity precision: 100.0%
drift features non-stable: 3
report -> reports/data_intelligence_report.md
   (10s)

== Task 2: multi-target prediction ==
split OK — train rows 96,801 | val rows 26,792 | loan overlap 0
next_3m_delinquency_flag: LR auc=0.776 | LGBM auc=0.8055 | cal brier 0.0804->0.06081
next_6m_delinquency_flag: LR auc=0.7059 | LGBM auc=0.7488 | cal brier 0.11491->0.09911
next_12m_default_flag: LR auc=0.7916 | LGBM auc=0.7866 | cal brier 0.0579->0.04739
  champion[next_12m_prepayment_flag] = LOGISTIC (0.6604 vs 0.581)
next_12m_prepayment_flag: LR auc=0.6604 | LGBM auc=0.581 | cal brier 0.06675->0.0666
next_state: macro_f1=0.5816 logloss=0.1803
permutation test (3 shuffled-label runs) mean AUC = 0.4833 (runs: [0.529, 0.434, 0.487]; expect ~0.50)
wrote reports/model_performance.md
   (39s)

== Task 3: transition/survival model ==
=== TASK 3 COMPLETE ===
model logloss 0.1822 vs baseline 0.1787 | macro-F1 0.4195 vs 0.3793
12m cum default: model 8.26% vs observed 6.32%
12m cum prepay : model 7.55% vs observed 5.46%
credit_band  model_cum_default_12m  model_cum_prepay_12m  n_loans
    620-679                 0.1374                0.0906      425
       <620                 0.1298                0.0937      176
    680-739                 0.0765                0.0767      590
    740-779                 0.0597                0.0645      544
       780+                 0.0385                0.0631      352
   (11s)

== Task 4: anomaly + exceptions ==
=== TASK 4 COMPLETE ===
exception_required AUC (val): 0.9947
exception_required PR-AUC (val): 0.9845
anomaly_score AUC vs labels (val): 0.9946
exception_type accuracy | exception (val): 0.9034
anomaly recall @ precision 0.90 (val): 0.9557
reviewer examples: 24 -> reports/anomaly_examples.md
   (16s)

== Task 5: scenario simulation ==
=== TASK 5 COMPLETE ===
                 cum_default  cum_prepay  pct_delinquent
scenario
base                    8.26        7.55            7.43
adverse_credit         15.00        5.04            9.76
high_prepayment         6.92       15.38            6.46
MC 90% band adverse 12m default: 13.75%–16.30%
   (12s)

== Task 6: explainability + uncertainty ==
=== TASK 6 COMPLETE ===
             feature  mean_abs_shap
   credit_score_band       0.250873
    pct_term_elapsed       0.073715
     loan_age_months       0.072858
       interest_rate       0.066665
       is_delinq_now       0.055151
log_original_balance       0.053823
trust_band                           method  n_cal  n_eval  mean_halfwidth  empirical_coverage 
       LOW            per-band residual q90    114     132          0.0949              0.9242 
    MEDIUM            per-band residual q90    447     433          0.0949              0.9215 
      HIGH            per-band residual q90   5872    6269          0.0949              0.9067 
       LOW SHIPPED: trust-scaled normalized   6433     132          0.1259              0.9242 
    MEDIUM SHIPPED: trust-scaled normalized   6433     433          0.1137              0.9215 
      HIGH SHIPPED: trust-scaled normalized   6433    6269          0.0954              0.9067 
FN with clean 12m history: 80% | FP already past-due: 100%
   (14s)

== Task 7 + submission.csv ==
[copilot] generating 8 grounded reviewer notes (mode=template)...
[copilot] 8/8 notes auto-accepted; probes logged; see logs/prompt_log.jsonl + logs/reviewed_outputs.jsonl
[submission] scoring test set with the full engine...
[submission] wrote G:\loan-intel\src\copilot\..\..\submission.csv — 7,225 rows, columns match template: True
recommended_action
AUTO_ACCEPT    6651
REVIEW          557
ESCALATE         17
   (9s)
