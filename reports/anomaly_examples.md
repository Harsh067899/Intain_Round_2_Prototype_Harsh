# Reviewer-Ready Anomaly Examples (Task 4)
_Top-scored validation records. Every claim below is grounded in computed artifacts: fired rules, source conflicts, isolation percentile, and the supervised exception model._

_Financial ROI Action Policy: Loss-Given-Default = 40%. Expected Dollar Loss (EDL) = P(exception/default) × Current Balance × LGD._

## Example 1 — LN101237 @ 2025-03
- **anomaly_score:** 0.454 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $5,819
- **snapshot:** status=CURRENT, dpd=0, balance=164,325 (orig 196,900), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); statistical outlier (isolation pct 0.99); model exception prob 0.09; EDL $5,819
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 2 — LN100171 @ 2024-07
- **anomaly_score:** 0.452 | **trust:** 0.17 | **predicted type:** NONE | **expected_dollar_loss:** $6,182
- **snapshot:** status=CURRENT, dpd=111, balance=174,563 (orig 183,300), servicer=CasaMortgage, doc=PENDING_REVIEW
- **why flagged:** rules fired: R004,R007; 1 servicer source conflict(s); statistical outlier (isolation pct 0.98); model exception prob 0.09; EDL $6,182
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 3 — LN105548 @ 2024-12
- **anomaly_score:** 0.449 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $9,206
- **snapshot:** status=CURRENT, dpd=0, balance=252,613 (orig 278,500), servicer=AlphaServ, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); statistical outlier (isolation pct 0.95); model exception prob 0.09; EDL $9,206
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 4 — LN101662 @ 2024-10
- **anomaly_score:** 0.443 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $3,987
- **snapshot:** status=CURRENT, dpd=0, balance=113,036 (orig 124,100), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $3,987
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 5 — LN106367 @ 2025-06
- **anomaly_score:** 0.443 | **trust:** 0.27 | **predicted type:** NONE | **expected_dollar_loss:** $23,018
- **snapshot:** status=CURRENT, dpd=0, balance=650,014 (orig 681,400), servicer=CasaMortgage, doc=MISSING_NOTE
- **why flagged:** rules fired: R007,R008; 1 servicer source conflict(s); model exception prob 0.09; EDL $23,018
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 6 — LN102233 @ 2024-07
- **anomaly_score:** 0.425 | **trust:** 0.25 | **predicted type:** NONE | **expected_dollar_loss:** $5,484
- **snapshot:** status=CURRENT, dpd=0, balance=155,917 (orig 163,400), servicer=CasaMortgage, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 0.09; EDL $5,484
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 7 — LN106268 @ 2024-11
- **anomaly_score:** 0.423 | **trust:** 0.25 | **predicted type:** NONE | **expected_dollar_loss:** $7,663
- **snapshot:** status=CURRENT, dpd=0, balance=217,860 (orig 227,100), servicer=BetaLoan, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 0.09; EDL $7,663
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 8 — LN104312 @ 2025-01
- **anomaly_score:** 0.418 | **trust:** 0.00 | **predicted type:** NONE | **expected_dollar_loss:** $5,262
- **snapshot:** status=CURRENT, dpd=0, balance=149,597 (orig 155,300), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $5,262
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 9 — LN101761 @ 2025-05
- **anomaly_score:** 0.416 | **trust:** 0.12 | **predicted type:** NONE | **expected_dollar_loss:** $5,484
- **snapshot:** status=CURRENT, dpd=0, balance=155,918 (orig 160,700), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $5,484
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 10 — LN106649 @ 2025-05
- **anomaly_score:** 0.411 | **trust:** 0.14 | **predicted type:** NONE | **expected_dollar_loss:** $5,424
- **snapshot:** status=CURRENT, dpd=0, balance=153,991 (orig 157,300), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $5,424
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 11 — LN102065 @ 2025-06
- **anomaly_score:** 0.409 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $3,477
- **snapshot:** status=CURRENT, dpd=0, balance=98,230 (orig 101,200), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $3,477
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 12 — LN103411 @ 2025-03
- **anomaly_score:** 0.408 | **trust:** 0.25 | **predicted type:** NONE | **expected_dollar_loss:** $8,952
- **snapshot:** status=CURRENT, dpd=0, balance=250,584 (orig 252,100), servicer=EagleTrust, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 0.09; EDL $8,952
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 13 — LN105326 @ 2024-07
- **anomaly_score:** 0.408 | **trust:** 0.27 | **predicted type:** NONE | **expected_dollar_loss:** $6,655
- **snapshot:** status=CURRENT, dpd=0, balance=189,214 (orig 195,100), servicer=EagleTrust, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $6,655
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 14 — LN101897 @ 2025-03
- **anomaly_score:** 0.407 | **trust:** 0.14 | **predicted type:** NONE | **expected_dollar_loss:** $18,000
- **snapshot:** status=CURRENT, dpd=0, balance=503,865 (orig 507,200), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $18,000
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 15 — LN103541 @ 2024-07
- **anomaly_score:** 0.407 | **trust:** 0.25 | **predicted type:** NONE | **expected_dollar_loss:** $10,827
- **snapshot:** status=CURRENT, dpd=0, balance=306,168 (orig 309,800), servicer=BetaLoan, doc=MISSING_NOTE
- **why flagged:** rules fired: R003,R006,R007; model exception prob 0.09; EDL $10,827
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 16 — LN101909 @ 2025-06
- **anomaly_score:** 0.406 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $9,432
- **snapshot:** status=CURRENT, dpd=0, balance=264,037 (orig 265,800), servicer=AlphaServ, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $9,432
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 17 — LN104329 @ 2025-03
- **anomaly_score:** 0.405 | **trust:** 0.02 | **predicted type:** NONE | **expected_dollar_loss:** $12,520
- **snapshot:** status=CURRENT, dpd=0, balance=355,964 (orig 358,700), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $12,520
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 18 — LN107940 @ 2025-06
- **anomaly_score:** 0.404 | **trust:** 0.02 | **predicted type:** NONE | **expected_dollar_loss:** $7,442
- **snapshot:** status=CURRENT, dpd=0, balance=211,582 (orig 216,500), servicer=DeltaHome, doc=PARTIAL
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $7,442
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 19 — LN106081 @ 2024-07
- **anomaly_score:** 0.403 | **trust:** 0.24 | **predicted type:** NONE | **expected_dollar_loss:** $6,018
- **snapshot:** status=CURRENT, dpd=0, balance=170,048 (orig 179,300), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $6,018
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 20 — LN102173 @ 2025-06
- **anomaly_score:** 0.402 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $7,110
- **snapshot:** status=CURRENT, dpd=0, balance=196,238 (orig 199,000), servicer=BetaLoan, doc=COMPLETE
- **why flagged:** rules fired: R008; 2 servicer source conflict(s); model exception prob 0.09; EDL $7,110
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 21 — LN100027 @ 2024-07
- **anomaly_score:** 0.393 | **trust:** 0.00 | **predicted type:** NONE | **expected_dollar_loss:** $22,095
- **snapshot:** status=CURRENT, dpd=0, balance=628,200 (orig 629,400), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R003,R006; 1 servicer source conflict(s); model exception prob 0.09; EDL $22,095
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 22 — LN106193 @ 2025-06
- **anomaly_score:** 0.366 | **trust:** 0.53 | **predicted type:** NONE | **expected_dollar_loss:** $4,987
- **snapshot:** status=DPD90, dpd=148, balance=135,004 (orig 142,100), servicer=AlphaServ, doc=PENDING_REVIEW
- **why flagged:** rules fired: R007; 1 servicer source conflict(s); statistical outlier (isolation pct 1.00); model exception prob 0.09; EDL $4,987
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 23 — LN100579 @ 2024-08
- **anomaly_score:** 0.366 | **trust:** 0.63 | **predicted type:** NONE | **expected_dollar_loss:** $11,867
- **snapshot:** status=DPD90, dpd=144, balance=321,287 (orig 335,800), servicer=DeltaHome, doc=PENDING_REVIEW
- **why flagged:** rules fired: R007; 1 servicer source conflict(s); statistical outlier (isolation pct 0.99); model exception prob 0.09; EDL $11,867
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement

## Example 24 — LN108298 @ 2025-04
- **anomaly_score:** 0.365 | **trust:** 0.29 | **predicted type:** NONE | **expected_dollar_loss:** $15,109
- **snapshot:** status=CURRENT, dpd=55, balance=422,748 (orig 514,300), servicer=CasaMortgage, doc=COMPLETE
- **why flagged:** rules fired: R004; 1 servicer source conflict(s); statistical outlier (isolation pct 1.00); model exception prob 0.09; EDL $15,109
- **recommended action:** REVIEW — high dollar loss exposure or multi-signal agreement
