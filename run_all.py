#!/usr/bin/env python3
"""One-command reproducible pipeline: data -> Task 1..7 -> submission.csv.

Usage: python run_all.py [--skip-datagen] [--api]
  --skip-datagen  reuse existing data/raw (e.g. the official organizer pack)
  --api           Task 7 uses the Anthropic API (needs ANTHROPIC_API_KEY)
"""
import argparse, subprocess, sys, time

PY = sys.executable

STEPS = [
    ("data pack (synthetic, seeded)", [PY, "src/datagen/generate.py"]),
    ("Task 1: profiling + rules + trust", [PY, "src/profiling/run_task1.py"]),
    ("Task 2: multi-target prediction", [PY, "src/models/run_task2.py"]),
    ("Task 3: transition/survival model", [PY, "src/models/run_task3.py"]),
    ("Task 4: anomaly + exceptions", [PY, "src/anomaly/run_task4.py"]),
    ("Task 5: scenario simulation", [PY, "src/scenarios/run_task5.py"]),
    ("Task 6: explainability + uncertainty", [PY, "src/explain/run_task6.py"]),
    ("Task 7 + submission.csv", [PY, "src/copilot/run_task7_demo.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-datagen", action="store_true")
    ap.add_argument("--api", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for name, cmd in STEPS:
        if a.skip_datagen and "datagen" in cmd[1]:
            print(f"== SKIP {name}"); continue
        if a.api and "task7" in cmd[1]:
            cmd = cmd + ["--api"]
        print(f"\n== {name} =="); s = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"FAILED at: {name}")
        print(f"   ({time.time()-s:.0f}s)")
    print(f"\nALL STEPS COMPLETE in {(time.time()-t0)/60:.1f} min -> submission.csv")
