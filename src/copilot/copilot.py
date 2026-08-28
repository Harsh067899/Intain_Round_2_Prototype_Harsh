"""Task 7 — Governed LLM Reviewer Copilot.

Governance contract (what makes this "smart LLM usage", not a wrapper):
  1. The LLM sees ONLY a computed-artifact bundle: predictions, SHAP drivers,
     fired rules, source conflicts, trust score, plus data-dictionary entries and
     validation rules retrieved for exactly the fields present (mini-RAG — small
     by design: the corpus is ~3k tokens, so retrieval is selection + logging of
     retrieved ids, fully auditable, no vector-index dependency).
  2. Every output is labeled RECOMMENDATION — the human decides.
  3. A grounding checker extracts every number and rule-id from the note and
     verifies it exists in the bundle; unmatched claims => automatic REJECT.
  4. Two-stream logging: logs/prompt_log.jsonl (prompt, model, timestamp, bundle
     artifact ids, output, grounding result) and logs/reviewed_outputs.jsonl
     (human accept/reject/correct + reason).
  5. Runs in two modes: --api (Anthropic API, needs ANTHROPIC_API_KEY) for real
     LLM notes, or template fallback (deterministic, clearly labeled) so the
     pipeline stays reproducible without secrets.

Demo runner: python src/copilot/run_task7_demo.py [--api] [--n 8]
Includes two adversarial probes (sparse-artifact loans) designed to tempt the
LLM into invention — harvested rejections are exactly the "LLM was wrong"
evidence the rubric asks for.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW = os.path.join(ROOT, "data", "raw")
ART = os.path.join(ROOT, "reports", "artifacts")
LOGS = os.path.join(ROOT, "logs")

MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
BASE_URL = os.environ.get("BASE_URL", "https://api.groq.com/openai/v1")


# ---------------------------- mini-RAG retriever -----------------------------

def load_dictionary() -> dict[str, str]:
    entries = {}
    with open(os.path.join(RAW, "data_dictionary.md")) as f:
        for line in f:
            m = re.match(r"\|\s*([a-z_0-9]+)\s*\|\s*(.+?)\s*\|", line)
            if m and m.group(1) not in ("Field",):
                entries[m.group(1)] = m.group(2)
    return entries


def load_rules() -> dict[str, dict]:
    with open(os.path.join(RAW, "validation_rules.json")) as f:
        return {r["id"]: r for r in json.load(f)["rules"]}


def retrieve(bundle: dict, dictionary: dict, rules: dict) -> dict:
    """Select dictionary entries for fields in the bundle + fired rules. Logged."""
    fields = set()
    for d in bundle.get("top_drivers", []):
        fields.add(d["feature"].split("=")[0])
    fields |= {"trust_score", "exception_type", "current_status", "days_past_due"}
    dict_hits = {k: v for k, v in dictionary.items() if k in fields}
    rule_hits = {rid: rules[rid] for rid in bundle.get("rules_fired", []) if rid in rules}
    return {"dictionary": dict_hits, "rules": rule_hits,
            "retrieved_ids": sorted(dict_hits) + sorted(rule_hits)}


# ---------------------------- grounding checker ------------------------------

def grounding_check(note: str, bundle: dict) -> dict:
    """Every number and rule-id in the note must exist in the bundle."""
    ground_nums = set()

    def collect(o):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                collect(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}", "{:g}"):
                try:
                    ground_nums.add(fmt.format(o))
                except Exception:
                    pass
    collect(bundle)
    # strip identifiers before extracting quantitative claims:
    # normalize non-breaking hyphens/en-dashes and spaces, then strip identifiers
    norm = note.replace(",", "").replace("\u2011", "-").replace("\u2013", "-").replace("\u202f", " ").replace("\u00a0", " ")
    scrub = re.sub(r"R0\d\d|LN[\w]+|\d{4}-\d{2}|\b\d+[- ]?m(?:o(?:nth)?s?)?\b", " ", norm)
    claimed = re.findall(r"\d+(?:\.\d+)?", scrub)
    unmatched_nums = [c for c in claimed
                      if not any(abs(float(c) - float(g)) < 1e-6 or c == g.lstrip("0") or c == g
                                 for g in ground_nums if _is_num(g))]
    claimed_rules = set(re.findall(r"R0\d\d", note))
    unmatched_rules = sorted(claimed_rules - set(bundle.get("rules_fired", [])))
    ok = bool(note.strip()) and not unmatched_nums and not unmatched_rules
    if not note.strip():
        unmatched_nums = ["<EMPTY_OUTPUT>"]
    return {"grounded": ok, "unmatched_numbers": unmatched_nums[:10],
            "unmatched_rule_ids": unmatched_rules}


def _is_num(s):
    try:
        float(s)
        return True
    except Exception:
        return False


# ---------------------------- copilot client ---------------------------------

SYSTEM = (
    "You are a loan-review copilot. Write a 4-6 sentence reviewer note using ONLY "
    "the facts in the provided JSON bundle. Cite rule ids exactly as given. Use only "
    "numbers that appear in the bundle. If the bundle lacks information for a claim, "
    "say the data is insufficient instead of guessing. End with: 'RECOMMENDATION — "
    "human decision required.'"
)


class CopilotClient:
    def __init__(self, use_api: bool):
        self.use_api = use_api
        os.makedirs(LOGS, exist_ok=True)
        self.dictionary = load_dictionary()
        self.rules = load_rules()

    def note_for(self, bundle: dict) -> dict:
        retrieved = retrieve(bundle, self.dictionary, self.rules)
        prompt = (f"Bundle:\n{json.dumps(bundle, default=str, indent=1)}\n\n"
                  f"Retrieved definitions:\n{json.dumps(retrieved['dictionary'], indent=1)}\n"
                  f"Fired rule details:\n{json.dumps(retrieved['rules'], indent=1)}\n\n"
                  "Write the reviewer note now.")
        if self.use_api:
            note, mode = self._call_api(prompt), "anthropic_api"
        else:
            note, mode = self._template(bundle), "template_fallback"
        check = grounding_check(note, {**bundle, "retrieved": retrieved})
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME if self.use_api else "deterministic_template",
            "mode": mode,
            "prompt_template": "reviewer_note_v1",
            "system": SYSTEM if self.use_api else None,
            "prompt": prompt,
            "bundle_artifact_ids": bundle.get("artifact_ids", []),
            "retrieved_ids": retrieved["retrieved_ids"],
            "output": note,
            "grounding_check": check,
            "label": "RECOMMENDATION — human decision required",
        }
        with open(os.path.join(LOGS, "prompt_log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not check["grounded"]:
            self.review(rec, "REJECT", f"grounding checker: unmatched {check['unmatched_numbers']}"
                                       f"{check['unmatched_rule_ids']}")
        return rec

    def review(self, rec: dict, decision: str, reason: str):
        with open(os.path.join(LOGS, "reviewed_outputs.jsonl"), "a") as f:
            f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                "output_ts": rec["timestamp"], "decision": decision,
                                "reason": reason, "output": rec["output"]}) + "\n")

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI
        import os
        
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY or run without --api")

        # The fix: explicitly point the OpenAI client to Groq's API URL
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key
        )
        
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        
        return response.choices[0].message.content

    def _template(self, b: dict) -> str:
        drivers = "; ".join(f"{d['feature']} ({d['shap']:+0.3f})" for d in b.get("top_drivers", [])[:3])
        rules = ", ".join(b.get("rules_fired", [])) or "none"
        return (f"Loan {b['loan_id']} at {b['reporting_month']}: 12m default probability "
                f"{b['prob_default_12m']:.2f} with trust score {b['trust_score']:.2f} "
                f"(interval widens accordingly). Key model drivers: {drivers}. "
                f"Deterministic rules fired: {rules}. Anomaly score {b['anomaly_score']:.2f}; "
                f"predicted exception type {b['exception_type_pred']}. "
                f"[template_fallback — run with --api for LLM-generated notes] "
                f"RECOMMENDATION — human decision required.")
