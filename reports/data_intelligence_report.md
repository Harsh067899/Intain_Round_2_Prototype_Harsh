# Data Intelligence Report — Task 1
_Generated from `data/raw/` (227,446 train rows, 42,078 test rows, 9,000 loans, 48,514 servicer updates)._

## 1. Headline findings
- **9.6%** of train records violate at least one deterministic validation rule; violations are **not random** — they concentrate in specific servicers (see §6), which is a process problem, not noise.
- **30.0%** of reconciled loan-months have at least one source conflict between the core panel and servicer updates.
- **1.8%** of records fall in the LOW trust band (<0.5) and should route to human review.
- Train→test drift is **material**: 3 feature(s) beyond the stable PSI threshold. Portfolio drift here is largely **survivorship** (defaulted/prepaid loans exit the panel) plus a riskier late-vintage mix.

## 2. Rule engine results (starter rules R001–R008)
| rule_id   | name                  | severity   |   violations |   violation_pct |
|:----------|:----------------------|:-----------|-------------:|----------------:|
| R001      | balance_non_negative  | high       |          572 |           0.251 |
| R002      | balance_le_original   | medium     |         1351 |           0.594 |
| R003      | date_order_valid      | high       |          589 |           0.259 |
| R004      | status_dpd_consistent | high       |         1761 |           0.774 |
| R005      | closed_loan_balance   | medium     |            3 |           0.001 |
| R006      | stale_record          | low        |         2573 |           1.131 |
| R007      | document_gap          | low        |        14681 |           6.455 |
| R008      | duplicate_loan_month  | high       |         1458 |           0.641 |

### Learned cross-column relationship rules (mined, confidence ≥ 98%)
| learned_rule                   |   support_pct |   confidence_pct |   relationship_breaks |
|:-------------------------------|--------------:|-----------------:|----------------------:|
| PREPAID=>prepayment_flag       |         0.428 |           100    |                     0 |
| DEFAULT=>default_flag          |         0.356 |           100    |                     0 |
| default=>loss_severity_present |         0.356 |           100    |                     0 |
| prepaid=>zero_balance          |         0.428 |            99.69 |                     3 |
| new_loan=>CURRENT              |         3.962 |           100    |                     0 |

## 3. Train vs test drift (PSI)
PSI bands: <0.10 stable · 0.10–0.25 moderate · >0.25 significant.
| feature               |    psi | assessment   |
|:----------------------|-------:|:-------------|
| remaining_term_months | 2.6285 | SIGNIFICANT  |
| loan_age_months       | 2.5423 | SIGNIFICANT  |
| interest_rate         | 0.1828 | moderate     |
| current_status        | 0.0179 | stable       |
| current_balance       | 0.0034 | stable       |
| dti_band              | 0.0007 | stable       |
| ltv_band              | 0.0005 | stable       |
| state                 | 0.0004 | stable       |
| credit_score_band     | 0.0002 | stable       |
| loan_purpose          | 0.0002 | stable       |
| servicer_name         | 0.0001 | stable       |
| document_status       | 0.0001 | stable       |
| source_system         | 0.0001 | stable       |
| days_past_due         | 0      | stable       |

**Interpretation.** `loan_age_months` / `remaining_term_months` drift is **structural, not distributional**: a fixed loan panel mechanically ages between train and test windows, so high PSI there is expected and benign. The *meaningful* drift signals are `interest_rate` (moderate — rate-environment shift across windows) and the mild credit-mix shift from survivorship (riskier loans default/prepay out of the live panel). Models therefore avoid leaning on raw calendar-linked values and use age-normalized features instead (see Task 2 feature notes).

## 4. Top absolute correlations / dependent fields
| feature_a                | feature_b                |   abs_corr |
|:-------------------------|:-------------------------|-----------:|
| next_3m_delinquency_flag | next_6m_delinquency_flag |   0.774009 |
| next_6m_delinquency_flag | next_12m_default_flag    |   0.539375 |
| month_index              | loan_age_months          |   0.524854 |
| next_3m_delinquency_flag | next_12m_default_flag    |   0.513079 |
| days_past_due            | default_flag             |   0.50662  |
| month_index              | interest_rate            |   0.450304 |
| loan_age_months          | interest_rate            |   0.406141 |
| days_past_due            | next_3m_delinquency_flag |   0.379085 |
| days_past_due            | next_12m_default_flag    |   0.377514 |
| days_past_due            | next_6m_delinquency_flag |   0.287208 |
| days_past_due            | modification_flag        |   0.26741  |
| loan_age_months          | next_6m_delinquency_flag |   0.168676 |
| loan_age_months          | remaining_term_months    |   0.168634 |
| loan_age_months          | next_3m_delinquency_flag |   0.161426 |
| loan_age_months          | days_past_due            |   0.146781 |

## 5. Missingness patterns by segment (% missing)
### by servicer_name
| servicer_name   |   miss_loss_severity_band |
|:----------------|--------------------------:|
| AlphaServ       |                     99.65 |
| BetaLoan        |                     99.67 |
| CasaMortgage    |                     99.65 |
| DeltaHome       |                     99.61 |
| EagleTrust      |                     99.63 |

### by source_system
| source_system   |   miss_loss_severity_band |
|:----------------|--------------------------:|
| API_FEED        |                     99.63 |
| CORE_SVC        |                     99.62 |
| LEGACY_BATCH    |                     99.71 |

## 6. Batch quality by servicer
| servicer_name   |   mean_trust |   pct_low_trust |   pct_any_rule |   pct_source_conflict |
|:----------------|-------------:|----------------:|---------------:|----------------------:|
| AlphaServ       |        0.966 |           1.683 |          8.226 |                 5.456 |
| BetaLoan        |        0.963 |           1.729 |          8.892 |                 5.575 |
| CasaMortgage    |        0.942 |           2.451 |         15.107 |                 5.611 |
| DeltaHome       |        0.961 |           1.74  |          9.805 |                 5.386 |
| EagleTrust      |        0.959 |           1.778 |          9.973 |                 5.453 |

_CasaMortgage-pattern servicers show elevated violation and conflict rates — batch quality scores make the process issue visible and assignable._

## 7. Honest evaluation against injected ground truth
Because the pack is generated with hidden, labeled corruptions, detector quality is **measured, not asserted**:
| corruption_type     |   injected |   caught |   recall_pct |
|:--------------------|-----------:|---------:|-------------:|
| BALANCE_MISMATCH    |       1893 |     1888 |        99.74 |
| DATE_INVALID        |       1142 |     1142 |       100    |
| DOC_GAP             |       1016 |     1016 |       100    |
| DUPLICATE_RECORD    |        729 |      729 |       100    |
| STALE_UPDATE        |       1431 |     1431 |       100    |
| STATUS_INCONSISTENT |       1770 |     1762 |        99.55 |

**Flag precision vs labeled exceptions: 61.3%** (19,745 of 32,223 flagged records are labeled exceptions). **High-severity-signal precision: 100.0%** — low-severity rules (staleness, doc gaps) are deliberately recall-oriented screeners that route records to review rather than assert corruption.

## 8. Record trust score (feeds Tasks 4, 6, 7)
trust = 1 − severity-weighted penalties (rule violations, source conflicts, staleness), clipped to [0,1]. Low trust widens downstream prediction intervals — bad data in, honest uncertainty out.
| trust_band   |   records |
|:-------------|----------:|
| HIGH         |    208588 |
| MEDIUM       |     14818 |
| LOW          |      4040 |