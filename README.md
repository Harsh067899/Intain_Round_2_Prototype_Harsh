# Loan Performance Intelligence Engine
**Intain FinTech Challenge 2026 — AI Track | Team: Harsh Sahu**

An ML-first engine that profiles messy loan-level data, predicts multi-outcome loan
performance with time-aware validation, models state transitions, detects anomalies,
runs macro scenarios, and explains everything to a human reviewer through a governed
LLM copilot.

## Core Documentation (Start Here)
- **[Solution Story & Engineering Decisions](SOLUTION_STORY.md)** — The narrative of how we solved the problem.
- **[System Architecture](docs/ARCHITECTURE.md)** — End-to-end mermaid flow diagram and component overview.
- **[AI Development Log](logs/AI_DEVELOPMENT_LOG.md)** — Human-in-the-loop tracking, AI guardrails, and lessons learned.

## End-to-End Pipeline Flow

```mermaid
flowchart TD
    A["8-file data pack<br/>(schema-locked, SHA-256 manifest)"] --> B["Trust layer — Task 1<br/>rule engine + learned rules<br/>servicer reconciliation → trust score"]
    B --> C["Leakage-safe feature pipeline<br/>backward-only history, refi incentive,<br/>trust features"]
    C --> D["4 champion models — Task 2<br/>per-target selection, isotonic calibration"]
    C --> E["Multinomial hazard engine — Task 3<br/>discrete-time next-state model"]
    D --> F["Anomaly fusion — Task 4<br/>rules + isolation forest + supervised"]
    E --> G["Curves & scenarios — Tasks 3+5<br/>hazard chaining, shock multipliers, Monte Carlo"]
    D --> H["Explainability — Task 6<br/>SHAP global/local, FP/FN analysis,<br/>trust-scaled conformal intervals"]
    F --> I["Governed copilot — Task 7<br/>artifact-only bundles, mini-RAG,<br/>grounding checker, two-stream logs"]
    G --> I
    H --> I
    I --> J["submission.csv + reports + audit logs<br/>one command, ~5 min, reproducible"]

## Quickstart
```bash
pip install -r requirements.txt
python run_all.py                     # full pipeline: data -> Tasks 1-7 -> submission.csv (~5 min)
python run_all.py --skip-datagen      # rerun on existing data/raw (e.g. official organizer pack)
python run_all.py --api               # Task 7 notes via Groq API (set GROQ_API_KEY in .env)
# or: docker build -t loan-intel . && docker run loan-intel
```

## Results snapshot (held-out validation; full details in reports/)
| Component | Result |
|---|---|
| Corruption detection (vs hidden ground truth) | 99.5-100% recall per type; 100% high-severity precision |
| 3m/6m delinquency AUC | 0.805 / 0.749 (LGBM, calibrated) |
| 12m default AUC | 0.787 (calibrated Brier 0.058->0.047) |
| 12m prepayment | logistic champion 0.660 (regime-shift finding, documented) |
| Transition model | macro-F1 0.420 vs 0.379 baseline; 12m curves within ~2pp of observed |
| Leakage | permutation test mean AUC 0.483; loan overlap 0 (asserted) |
| Anomaly engine | exception AUC 0.995; recall@p90 0.956; 24 grounded reviewer examples |
| Scenarios | adverse 15.0% / base 8.3% / high-prepay 6.9% 12m default; MC 90% band 13.8-16.3% |
| Uncertainty | trust-scaled conformal: LOW 0.126 > MED 0.114 > HIGH 0.095 halfwidth, coverage >= 90% nominal |
| Copilot governance | 10/10 grounded notes; two-stream JSONL logs; auto-reject on ungrounded claims |

## Data provenance (IMPORTANT)
Section 6 of the problem statement describes an organizer-provided data pack. As of
Aug 27 the pack was not yet distributed (HackerEarth support ticket raised; escalation
confirmed by HackerEarth on Aug 27). To avoid losing build time, this repo ships a
**schema-locked synthetic generator** (`src/datagen/generate.py`) that produces all 8
files with the exact Section 6/7 field names, realistic hazard dynamics, injected
data-quality issues with hidden ground truth, and a conflicting second source.
**If/when the official pack arrives, drop its files into `data/raw/` — every
downstream stage reads only from there and requires zero code changes.**

Synthetic-data stress testing is also a listed Advanced Feature (Section 10); the
hidden ground truth in `data/ground_truth/` lets us report honest precision/recall
for the anomaly detector instead of unverifiable claims.

## Repo layout
```
data/raw/            8-file data pack (generated or official)
data/ground_truth/   injected-corruption labels (evaluation only, never features)
src/datagen/         synthetic pack generator (seeded, reproducible)
src/profiling/       Task 1: distributions, missingness, drift, quality scores
src/features/        feature pipeline (leakage-controlled)
src/models/          Task 2+3: multi-target models, calibration, transition model
src/anomaly/         Task 4: rules + ML anomaly and exception detection
src/scenarios/       Task 5: base / adverse / high-prepayment projections
src/explain/         Task 6: SHAP global/local, error analysis, uncertainty
src/copilot/         Task 7: grounded LLM reviewer notes + prompt logs
reports/             generated reports (data intelligence, explainability, scenario)
logs/                LLM prompt logs (JSONL) + AI Development Log
```

## Reproducibility
All randomness is seeded (numpy default_rng(42)). `data/raw/MANIFEST.sha256.json`
fingerprints every data file for audit traceability.

