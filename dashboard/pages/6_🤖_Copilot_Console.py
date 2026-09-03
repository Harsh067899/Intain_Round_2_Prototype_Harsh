"""🤖 Copilot Console — LLM governance audit trail and note browser."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from utils import (
    load_prompt_log, load_reviewed_outputs, metric_card, styled_figure,
    donut_chart, COLORS,
)

st.set_page_config(page_title="Copilot Console", page_icon="🤖", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .note-card {
        background: linear-gradient(135deg, #1a1f2edd, #232a3ddd);
        padding: 16px 20px; border-radius: 12px; border: 1px solid #2a3042;
        margin-bottom: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .grounded-pass { border-left: 4px solid #69f0ae; }
    .grounded-fail { border-left: 4px solid #ff5252; }
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '🤖 Copilot Console</div>', unsafe_allow_html=True)

st.markdown("""
<div style="color:#9e9e9e;font-size:1.0em;margin-bottom:1.5em;line-height:1.6">
    <strong>Governance Contract:</strong> Every LLM output is grounded against computed artifacts.
    Unmatched numbers or rule IDs trigger automatic REJECT. Two-stream audit logging
    captures both the prompt/output and the human review decision.
</div>
""", unsafe_allow_html=True)

prompt_log = load_prompt_log()
reviewed = load_reviewed_outputs()

if not prompt_log:
    st.warning("⚠️ No prompt log data available. Run the pipeline with `python run_all.py` first.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total = len(prompt_log)
grounded = sum(1 for r in prompt_log if r.get("grounding_check", {}).get("grounded", False))
rejected = sum(1 for r in reviewed if r.get("decision") == "REJECT")
accepted = sum(1 for r in reviewed if r.get("decision") == "ACCEPT")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Total Notes", str(total)), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Grounded ✅", str(grounded),
                            f"{grounded/max(total,1)*100:.0f}%"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Auto-Rejected ❌", str(rejected),
                            f"{rejected/max(total,1)*100:.0f}%", "inverse"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Accepted 🎯", str(accepted)), unsafe_allow_html=True)

st.markdown("---")

# ── Grounding Pass/Fail Donut ─────────────────────────────────────────────────
col_donut, col_timeline = st.columns(2)

with col_donut:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                'Grounding Check Results</div>', unsafe_allow_html=True)
    fig = donut_chart(
        ["Grounded", "Ungrounded"],
        [grounded, total - grounded],
        "Grounding Checks",
        [COLORS["accent_green"], COLORS["accent_red"]],
    )
    st.plotly_chart(fig, use_container_width=True)

with col_timeline:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                'Review Decisions</div>', unsafe_allow_html=True)
    if reviewed:
        decisions = pd.Series([r.get("decision", "UNKNOWN") for r in reviewed]).value_counts()
        dec_colors = {"ACCEPT": COLORS["accent_green"], "REJECT": COLORS["accent_red"],
                      "CORRECT": COLORS["accent_amber"]}
        fig = donut_chart(
            decisions.index.tolist(), decisions.values.tolist(),
            "Decisions",
            [dec_colors.get(d, COLORS["text_secondary"]) for d in decisions.index],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No review decisions recorded yet.")

st.markdown("---")

# ── Prompt Log Browser ────────────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '📜 Prompt Log Browser</div>', unsafe_allow_html=True)

# Filters
col_f1, col_f2 = st.columns(2)
with col_f1:
    filter_grounding = st.selectbox("Filter by Grounding", ["All", "Grounded Only", "Ungrounded Only"])
with col_f2:
    max_display = st.slider("Max Notes to Display", 5, 50, 10)

filtered_log = prompt_log.copy()
if filter_grounding == "Grounded Only":
    filtered_log = [r for r in filtered_log if r.get("grounding_check", {}).get("grounded", False)]
elif filter_grounding == "Ungrounded Only":
    filtered_log = [r for r in filtered_log if not r.get("grounding_check", {}).get("grounded", True)]

for i, rec in enumerate(filtered_log[:max_display]):
    gc = rec.get("grounding_check", {})
    is_grounded = gc.get("grounded", False)
    css_class = "grounded-pass" if is_grounded else "grounded-fail"
    badge_color = COLORS["accent_green"] if is_grounded else COLORS["accent_red"]
    badge_text = "✅ GROUNDED" if is_grounded else "❌ UNGROUNDED"

    mode = rec.get("mode", "unknown")
    model = rec.get("model", "N/A")
    timestamp = rec.get("timestamp", "N/A")
    output = rec.get("output", "(no output)")
    retrieved_ids = rec.get("retrieved_ids", [])

    # Ungrounded details
    unmatched_nums = gc.get("unmatched_numbers", [])
    unmatched_rules = gc.get("unmatched_rule_ids", [])

    unmatched_html = ""
    if not is_grounded:
        if unmatched_nums:
            unmatched_html += (f'<div style="color:#ff5252;font-size:0.85em;margin-top:6px">'
                               f'⚠️ Unmatched numbers: {", ".join(str(n) for n in unmatched_nums[:5])}</div>')
        if unmatched_rules:
            unmatched_html += (f'<div style="color:#ff5252;font-size:0.85em">'
                               f'⚠️ Unmatched rules: {", ".join(unmatched_rules)}</div>')

    st.markdown(f"""
    <div class="note-card {css_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="color:{COLORS['accent_cyan']};font-weight:700;font-size:0.95em">
                Note #{i+1}</span>
            <span style="background:{badge_color}22;color:{badge_color};padding:2px 10px;
                         border-radius:10px;font-size:0.8em;font-weight:600;
                         border:1px solid {badge_color}44">{badge_text}</span>
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.8em;margin-bottom:6px">
            {timestamp} · Mode: {mode} · Model: {model}
        </div>
        <div style="color:{COLORS['text_primary']};font-size:0.92em;line-height:1.6;
                    background:rgba(0,0,0,0.15);padding:12px;border-radius:8px;margin-bottom:6px">
            {output[:500]}{'...' if len(output) > 500 else ''}
        </div>
        <div style="color:{COLORS['text_secondary']};font-size:0.8em">
            Retrieved IDs: {', '.join(retrieved_ids[:8]) if retrieved_ids else 'none'}
        </div>
        {unmatched_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Reviewed Outputs Browser ─────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '🔍 Reviewed Outputs</div>', unsafe_allow_html=True)

if reviewed:
    for i, rev in enumerate(reviewed[:15]):
        decision = rev.get("decision", "UNKNOWN")
        dec_color = {"ACCEPT": COLORS["accent_green"], "REJECT": COLORS["accent_red"],
                     "CORRECT": COLORS["accent_amber"]}.get(decision, COLORS["text_secondary"])
        reason = rev.get("reason", "N/A")
        output_snippet = str(rev.get("output", ""))[:200]

        st.markdown(f"""
        <div class="note-card" style="border-left:4px solid {dec_color}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="color:{COLORS['accent_cyan']};font-weight:600;font-size:0.9em">
                    Review #{i+1}</span>
                <span style="background:{dec_color}22;color:{dec_color};padding:2px 10px;
                             border-radius:10px;font-size:0.8em;font-weight:600;
                             border:1px solid {dec_color}44">{decision}</span>
            </div>
            <div style="color:{COLORS['text_secondary']};font-size:0.85em;margin-bottom:4px">
                <strong>Reason:</strong> {reason}</div>
            <div style="color:{COLORS['text_primary']};font-size:0.85em;
                        background:rgba(0,0,0,0.1);padding:8px;border-radius:6px">
                {output_snippet}{'...' if len(str(rev.get('output', ''))) > 200 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No reviewed outputs recorded yet.")
