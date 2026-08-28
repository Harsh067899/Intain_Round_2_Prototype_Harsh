# Reviewer-Ready Anomaly Examples (Task 4)
_Top-scored validation records. Every claim below is grounded in computed artifacts: fired rules, source conflicts, isolation percentile, and the supervised exception model._

## Example 1 — LN101237 @ 2025-03
- **anomaly_score:** 0.818 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=164,325 (orig 196,900), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); statistical outlier (isolation pct 0.99); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 2 — LN100171 @ 2024-07
- **anomaly_score:** 0.816 | **trust:** 0.17 | **predicted type:** STATUS_INCONSISTENT
- **snapshot:** status=CURRENT, dpd=111, balance=174,563 (orig 183,300), servicer=CasaMortgage, doc=PENDING_REVIEW
- **why flagged:** rules fired: R004,R007; 1 servicer source conflict(s); statistical outlier (isolation pct 0.98); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 3 — LN105548 @ 2024-12
- **anomaly_score:** 0.812 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=252,613 (orig 278,500), servicer=AlphaServ, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); statistical outlier (isolation pct 0.95); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 4 — LN101662 @ 2024-10
- **anomaly_score:** 0.808 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=113,036 (orig 124,100), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 5 — LN106367 @ 2025-06
- **anomaly_score:** 0.807 | **trust:** 0.27 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=650,014 (orig 681,400), servicer=CasaMortgage, doc=MISSING_NOTE
- **why flagged:** rules fired: R007,R008; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 6 — LN102233 @ 2024-07
- **anomaly_score:** 0.789 | **trust:** 0.25 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=155,917 (orig 163,400), servicer=CasaMortgage, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 1.00
- **recommended action:** ESCALATE — single-source signal, verify with servicer

## Example 7 — LN106268 @ 2024-11
- **anomaly_score:** 0.788 | **trust:** 0.25 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=217,860 (orig 227,100), servicer=BetaLoan, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 1.00
- **recommended action:** ESCALATE — single-source signal, verify with servicer

## Example 8 — LN104312 @ 2025-01
- **anomaly_score:** 0.782 | **trust:** 0.00 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=149,597 (orig 155,300), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 9 — LN101761 @ 2025-05
- **anomaly_score:** 0.780 | **trust:** 0.12 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=155,918 (orig 160,700), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 10 — LN106649 @ 2025-05
- **anomaly_score:** 0.775 | **trust:** 0.14 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=153,991 (orig 157,300), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 11 — LN102065 @ 2025-06
- **anomaly_score:** 0.773 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=98,230 (orig 101,200), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 12 — LN103411 @ 2025-03
- **anomaly_score:** 0.772 | **trust:** 0.25 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=250,584 (orig 252,100), servicer=EagleTrust, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 1.00
- **recommended action:** ESCALATE — single-source signal, verify with servicer

## Example 13 — LN105326 @ 2024-07
- **anomaly_score:** 0.772 | **trust:** 0.27 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=189,214 (orig 195,100), servicer=EagleTrust, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 14 — LN101897 @ 2025-03
- **anomaly_score:** 0.771 | **trust:** 0.14 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=503,865 (orig 507,200), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 15 — LN103541 @ 2024-07
- **anomaly_score:** 0.771 | **trust:** 0.25 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=306,168 (orig 309,800), servicer=BetaLoan, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 1.00
- **recommended action:** ESCALATE — single-source signal, verify with servicer

## Example 16 — LN101909 @ 2025-06
- **anomaly_score:** 0.770 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=264,037 (orig 265,800), servicer=AlphaServ, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 17 — LN104329 @ 2025-03
- **anomaly_score:** 0.769 | **trust:** 0.02 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=355,964 (orig 358,700), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 18 — LN107940 @ 2025-06
- **anomaly_score:** 0.768 | **trust:** 0.02 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=211,582 (orig 216,500), servicer=DeltaHome, doc=PARTIAL
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 19 — LN106081 @ 2024-07
- **anomaly_score:** 0.767 | **trust:** 0.24 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=170,048 (orig 179,300), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 20 — LN102173 @ 2025-06
- **anomaly_score:** 0.766 | **trust:** 0.29 | **predicted type:** DUPLICATE_RECORD
- **snapshot:** status=CURRENT, dpd=0, balance=196,238 (orig 199,000), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 21 — LN100027 @ 2024-07
- **anomaly_score:** 0.758 | **trust:** 0.00 | **predicted type:** DATE_INVALID
- **snapshot:** status=CURRENT, dpd=0, balance=628,200 (orig 629,400), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 22 — LN108298 @ 2025-04
- **anomaly_score:** 0.729 | **trust:** 0.29 | **predicted type:** STATUS_INCONSISTENT
- **snapshot:** status=CURRENT, dpd=55, balance=422,748 (orig 514,300), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R004; 1 servicer source conflict(s); statistical outlier (isolation pct 1.00); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 23 — LN106193 @ 2025-06
- **anomaly_score:** 0.729 | **trust:** 0.53 | **predicted type:** SOURCE_CONFLICT
- **snapshot:** status=DPD90, dpd=148, balance=135,004 (orig 142,100), servicer=AlphaServ, doc=PENDING_REVIEW
- **why flagged:** rules fired: R007; 1 servicer source conflict(s); statistical outlier (isolation pct 1.00); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree

## Example 24 — LN100126 @ 2024-07
- **anomaly_score:** 0.729 | **trust:** 0.29 | **predicted type:** STATUS_INCONSISTENT
- **snapshot:** status=CURRENT, dpd=89, balance=341,101 (orig 361,200), servicer=BetaLoan, doc=PARTIAL
- **why flagged:** rules fired: R004; 1 servicer source conflict(s); statistical outlier (isolation pct 0.99); model exception prob 1.00
- **recommended action:** ESCALATE — multiple independent signals agree
