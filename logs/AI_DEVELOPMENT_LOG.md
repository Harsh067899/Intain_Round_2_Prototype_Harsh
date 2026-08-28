
# AI Development Log — Loan Performance Intelligence Engine
Required deliverable (Task 8). Updated daily. Human = Harsh Sahu; AI assistant = Gemini 3.1 Pro and Claude Sonnet 5(Anthropic).

## Tools used
- Claude (claude.ai) — architecture design, code generation, critical review of plan vs rubric free tier
- Antigravity IDE and CLI and Open AI Codex 

## Approx AI-generated code share
- Day 1 (Aug 27): ~85% AI-generated, 100% human-reviewed and executed/verified against sanity checks

*Note from Harsh: This log is my personal auditing trail. While I relied on Claude to accelerate boilerplate code and architecture prototyping, I maintained strict human-in-the-loop governance. Every design decision, evaluation metric, and AI-generated script was manually executed, scrutinized, and corrected by me before shipping.*

## Entry — Aug 27, 2026
**Goal:** Unblock the build while the organizer data pack is pending.
**Representative prompts:**
1. "Critically analyze the Intain AI-track problem statement, map plan to judging rubric, identify what most teams will miss."
2. "The organizer data pack is missing — generate a schema-locked synthetic data pack matching Section 6/7 exactly, with realistic hazards and injected corruptions with ground truth."

**Accepted output:** Repo skeleton; src/datagen/generate.py (monthly state-machine
simulator, forward-looking targets, messiness injection, servicer conflict source,
8-file pack + SHA-256 manifest).

**Rejected/corrected AI output (examples — keep collecting these!):**
- First generator version crashed: AI used itertuples on a DataFrame with an
  underscore-prefixed column (_risk), which pandas silently renames. Caught at
  runtime by human-in-the-loop execution; fixed by renaming before iteration.
  Lesson: AI code that "looks right" still needs execution verification.
- AI initially planned to source Fannie/Freddie raw data; human review of Section 5
  ("should not require students to register with data portals") + the license-violation
  disqualification condition led to the synthetic-generator decision instead.

**Human review process:** every generated file executed; sanity checks verified
(monotone default rates by credit/LTV band, refi-incentive prepayment pattern,
train/test drift, injected-violation counts). Design decisions (state machine,
censoring caveat, ground-truth separation) reviewed and approved by human.

**Lessons learned (running list):**
- Schema-lock early: building against the spec's exact field names makes the real
  data pack a drop-in swap.

## Entry — Aug 28, 2026 (Task 1: Data Intelligence & Profiling)
**Goal:** Complete profiling engine, rule engine, servicer reconciliation, trust scores, drift analysis, and the generated data intelligence report.

**Accepted output:** `src/profiling/` (profile_engine.py, rule_engine.py, reconcile.py, run_task1.py);
`reports/data_intelligence_report.md` + 8 artifact CSVs including record_trust_scores.csv (feeds Tasks 4/6/7).

**Rejected/corrected AI output:**
- AI's first evaluation reported ~84–86% corruption recall. Human-in-the-loop investigation
  showed the *evaluation* was wrong, not the detector: ground truth included test-window
  corruptions the train-only run could never see (84.9% of GT rows were in-window — matching
  the "recall" exactly). Fixed the eval to score only the evaluated window → true recall 99.5–100%.
  Lesson: suspicious metrics deserve investigation in BOTH directions — an unfairly low score
  is as misleading as an inflated one.
- pandas API mismatch (`Series.reset_index(names=...)`) caught at runtime and fixed.
- AI's drift table initially flagged loan_age/remaining_term PSI as "SIGNIFICANT" without
  context; human review added the structural-vs-distributional interpretation (panel aging
  is mechanical, not drift) to avoid a naive claim in front of judges.

**Verified results (vs hidden ground truth):** per-type corruption recall 99.5–100%;
high-severity-signal precision 100%; low-severity screeners framed honestly as recall-oriented.
Profiler independently surfaced the planted sloppy-servicer pattern (CasaMortgage: 15.1%
violation rate vs ~9% peers) — discovery, not assertion.

## Entry — Aug 28, 2026 (Tasks 2 & 3: Prediction + Transition Modeling)
**Accepted output:** src/features/build_features.py, src/models/run_task2.py,
src/models/run_task3.py; models/*.joblib; model_performance.md, transition_model_report.md,
reliability figures, cumulative incidence curves; docs/MODEL_CARD.md.

**Rejected/corrected AI output (chronological — all caught by executing and reading results):**
1. Censoring bug: AI's first censor_mask used the wrong observation window (train-file end
   instead of the label-generation horizon) AND kept event==1 rows from partial horizons —
   12m targets had ONE class in validation. Fixed to strict full-horizon censoring (unbiased).
2. Fixed calibration cutoff left an empty evaluation half for 12m targets after the censoring
   fix; replaced with a per-target median-month cutoff.
3. Prepayment: AI's LightGBM lost to the logistic baseline (0.58 vs 0.66). Diagnosis, not
   tuning-by-faith: the refi-incentive feature is real (monotone 4.0%→9.1% across quintiles)
   but the validation window sits in a different rate regime — trees cannot extrapolate.
   Tested monotone constraints and linear_tree; logistic still won. Shipped per-target
   champion selection with the regime-shift finding documented.
4. Transition model trained with class_weight="balanced" destroyed probability calibration:
   hazard chaining over-predicted 12m default 14.6% vs 6.3% observed and lost to the empirical
   baseline on log-loss. Retrained unweighted → 8.3% vs 6.3%, monotone credit-band curves.
   Lesson: classification-metric training choices can silently corrupt downstream simulation.
5. AI's report template bolded the model's metrics unconditionally; human review made the
   comparison table honest (baseline wins log-loss narrowly; model wins macro-F1) — judges
   punish overclaiming.

**Verified:** loan overlap asserted 0; permutation mean AUC 0.483 (3 seeds); calibration
improves Brier on all targets; simulated cumulative incidence within ~2pp of observed.

## Entry — Aug 29, 2026 (Tasks 4-7 + full integration)
**Accepted output:** src/anomaly/run_task4.py, src/scenarios/run_task5.py,
src/explain/run_task6.py, src/copilot/{copilot.py,run_task7_demo.py}, run_all.py,
Dockerfile, submission.csv builder. Full pipeline verified end-to-end in 4.5 min.

**Rejected/corrected AI output:**
1. Monte Carlo scenario bands (18.6-21.2%) failed to contain the expected-value
   estimate (15.0%) — a designed consistency check caught it. Root cause isolated
   with a minimal toy-chain reproduction: in-place state updates let loans take
   TWO transitions per month (CURRENT->DPD30 then DPD30->... in the same step).
   Fixed with snapshot-then-assign; band now 13.8-16.3%, correctly bracketing 15.0%.
2. Conformal intervals came out IDENTICAL across trust bands — the "intervals widen
   with low trust" thesis was NOT empirically present (injected corruptions are
   independent of default hazards by construction). Rather than fake it, we ship
   trust-scaled normalized conformal as an explicit governance policy with the flat
   empirical finding disclosed in the report, and per-band coverage verified >= nominal.
3. Grounding checker rejected 10/10 notes on first run — over-strict claim extraction
   treated loan-id digits, date fragments, "12m" horizon tokens, and rule-id digits as
   quantitative claims. Fixed by scrubbing identifiers before extraction; 10/10 grounded.

**Design notes:** Task 4's high AUC (0.995) is by design, not leakage — the fusion
feeds rule outputs to the supervised model, so it learns the rule->exception mapping
plus residual patterns. Mini-RAG is deliberately dependency-free (the corpus is ~3k
tokens; retrieval = auditable selection + logged ids), rejected LlamaIndex as
overkill that would bloat the judges' install.
4. Switched LLM integration from Anthropic to Groq via the OpenAI SDK to 
   leverage free-tier inference. The model hallucinated under temperature > 0.0, which 
   the grounding checker immediately caught. Setting temperature=0.0 locked the output 
   to pure extraction and allowed the pipeline to pass.
