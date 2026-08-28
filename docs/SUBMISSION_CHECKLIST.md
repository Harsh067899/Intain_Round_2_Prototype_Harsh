# Intain FinTech Challenge 2026 — AI Track Submission Checklist
**Deadline: Aug 31, 2026, 11:59 PM IST (HackerEarth)** · Target: win the track

## 0. Logistics (do first)
- [ ] Confirm organizer data pack received (7 files per Section 6)
- [ ] Resolve duplicate team registration ("Harsh's Team" vs "Harsh's Team 2")
- [ ] Team formation finalized on HackerEarth
- [ ] Claude/LLM API key ready for copilot module
- [ ] GitHub repo created (private until submission if allowed)

## Task 1 — Data Intelligence & Profiling (15 pts)
- [ ] Column distribution profiles (numeric + categorical)
- [ ] Missing-value pattern analysis (matrix/heatmap, MCAR vs systematic)
- [ ] Outlier detection + invalid date relationship checks
- [ ] Correlation / dependent-field analysis
- [ ] Cross-column relationship breaks (association rules)
- [ ] Train vs test drift comparison (PSI / KS per feature)
- [ ] Record-level AND batch-level data-quality scores
- [ ] Run + extend validation_rules.json
- [ ] servicer_updates.csv conflict detection, staleness, reconciliation (trust scores)

## Task 2 — Loan Performance Prediction (20 pts)
- [ ] Targets: next_3m_delinq, next_6m_delinq, next_12m_default, next_12m_prepayment
- [ ] Time-aware split: calendar cutoff + grouped by loan_id (NO loan in both sets)
- [ ] Leakage audit: features use only data ≤ reporting_month (document controls)
- [ ] Baseline model (logistic regression) vs improved (LightGBM) comparison
- [ ] Class imbalance handling (weights/focal; report PR-AUC, recall@precision)
- [ ] Calibration: isotonic/Platt + reliability diagrams + Brier score
- [ ] Metrics table: ROC-AUC, PR-AUC, F1, recall@fixed precision, Brier, macro-F1

## Task 3 — Time-to-Event / Transition Model (15 pts)
- [ ] Discrete-time hazard model OR monthly Markov next_state transition model
- [ ] Competing risks handled (prepayment vs default) — explain censoring
- [ ] Cumulative event curves / survival projections plotted
- [ ] Baseline comparison (e.g., Kaplan-Meier or constant-hazard vs model)

## Task 4 — Anomaly & Exception Detection (10 pts)
- [ ] Record-level anomaly score (rules + isolation forest combined)
- [ ] exception_required probability + exception_type classifier
- [ ] Anomaly driver explanations (reason codes)
- [ ] ≥ 20 numbered reviewer-ready anomaly examples

## Task 5 — Scenario & Stress Simulation (10 pts)
- [ ] Base / adverse-credit / high-prepayment scenarios from macro_scenarios.csv
- [ ] Projected delinquency, default, prepayment rates per scenario
- [ ] Segment-level impacts: vintage, credit band, state, servicer
- [ ] Top scenario driver explanation

## Task 6 — Explainability (10 pts)
- [ ] Global feature importance (SHAP summary)
- [ ] Local explanations for individual loans
- [ ] Model confidence / uncertainty shown (conformal intervals ↔ trust score)
- [ ] False positive / false negative analysis

## Task 7 — LLM Reviewer Copilot (10 pts)
- [ ] Grounded reviewer notes (every sentence traces to computed artifacts)
- [ ] Prompt log: prompt, model, timestamp, output (JSONL)
- [ ] All outputs labeled "recommendation, not decision"
- [ ] ≥ 2 examples of wrong/vague/overconfident LLM output + how caught
- [ ] (Advanced) RAG over data_dictionary.md + validation_rules.json

## Task 8 — Agentic Coding Evidence (5 pts)
- [ ] AI Development Log started Day 1, updated daily
- [ ] Tools used, representative prompts, accepted vs rejected outputs
- [ ] Human review process + approx AI-generated code share + lessons

## Deliverables (Section 11)
- [ ] GitHub repo — clean, runnable, README with setup steps (5 pts w/ repro)
- [ ] Reproducible end-to-end notebook/scripts
- [ ] submission.csv matching submission_template.csv EXACTLY
- [ ] Model card (objective, data, features, validation, metrics, limitations, leakage controls, failure modes)
- [ ] Data intelligence report
- [ ] Explainability report
- [ ] Scenario report
- [ ] LLM copilot demo
- [ ] AI Development Log
- [ ] 5-minute demo video following Section 14's 15-beat script

## Disqualification guards (Section 13)
- [ ] Non-LLM models do ALL prediction (LLM = explanation only)
- [ ] No random splits leaking loans; justification documented
- [ ] No target leakage into features
- [ ] All metrics reported honestly; nothing fabricated
- [ ] Every LLM narrative grounded in computed outputs

## Day plan (Aug 27 → 31)
- **Aug 27**: Logistics, repo, schema/mock data, profiling + trust scoring, AI log started
- **Aug 28**: Feature pipeline, time-aware split, baseline + LightGBM multi-target, calibration
- **Aug 29**: Transition/hazard model + curves, anomaly + exception detection, 20 examples
- **Aug 30**: Scenarios, SHAP + conformal, LLM copilot + prompt logs, all reports drafted
- **Aug 31**: submission.csv validation, model card, video recording, buffer, SUBMIT EARLY (by evening, not 11:58 PM)
