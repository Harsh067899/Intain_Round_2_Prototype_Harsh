"""🔍 Loan Inspector — Drill into any loan's full risk profile."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    load_trust_scores, load_anomaly_scores, load_submission,
    load_local_explanations, metric_card, styled_figure, radar_chart,
    shap_waterfall_chart, trust_badge, action_badge, COLORS,
)

st.set_page_config(page_title="Loan Inspector", page_icon="🔍", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '🔍 Loan Inspector</div>', unsafe_allow_html=True)

trust = load_trust_scores()
anomaly = load_anomaly_scores()
submission = load_submission()
local_exp = load_local_explanations()

if trust.empty:
    st.warning("⚠️ No data available. Run the pipeline first: `python run_all.py`")
    st.stop()

# ── Loan selector ─────────────────────────────────────────────────────────────
loan_ids = sorted(trust["loan_id"].unique())

col_sel, col_info = st.columns([1, 3])
with col_sel:
    selected = st.selectbox("🔎 Select Loan ID", loan_ids, index=0)

# ── Gather data for selected loan ────────────────────────────────────────────
loan_trust = trust[trust["loan_id"] == selected].sort_values("reporting_month")
loan_anom = anomaly[anomaly["loan_id"] == selected] if not anomaly.empty else pd.DataFrame()
loan_sub = submission[submission["loan_id"] == selected] if not submission.empty else pd.DataFrame()
loan_exp = local_exp[local_exp["loan_id"] == selected] if not local_exp.empty else pd.DataFrame()

if loan_trust.empty:
    st.warning(f"No data found for loan {selected}")
    st.stop()

latest = loan_trust.iloc[-1]

with col_info:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {COLORS['bg_card']}dd, {COLORS['bg_card_hover']}dd);
                padding:16px 24px;border-radius:16px;border:1px solid {COLORS['border']};
                display:flex;align-items:center;gap:24px;flex-wrap:wrap">
        <div><span style="color:{COLORS['text_secondary']};font-size:0.8em;text-transform:uppercase;
              letter-spacing:1px">Loan</span><br>
             <span style="color:{COLORS['accent_cyan']};font-size:1.3em;font-weight:700">{selected}</span></div>
        <div><span style="color:{COLORS['text_secondary']};font-size:0.8em;text-transform:uppercase;
              letter-spacing:1px">Latest Month</span><br>
             <span style="color:{COLORS['text_primary']};font-weight:600">{latest['reporting_month']}</span></div>
        <div><span style="color:{COLORS['text_secondary']};font-size:0.8em;text-transform:uppercase;
              letter-spacing:1px">Trust</span><br>
             {trust_badge(latest['trust_score'], latest.get('trust_band'))}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(metric_card("Trust Score", f"{latest['trust_score']:.3f}"), unsafe_allow_html=True)
with c2:
    n_rules = int(latest.get("n_rules_fired", 0))
    st.markdown(metric_card("Rules Fired", str(n_rules)), unsafe_allow_html=True)
with c3:
    n_conf = int(latest.get("n_conflicts", 0))
    st.markdown(metric_card("Conflicts", str(n_conf)), unsafe_allow_html=True)
with c4:
    if not loan_anom.empty:
        anom_latest = loan_anom.iloc[-1]
        st.markdown(metric_card("Anomaly Score", f"{anom_latest['anomaly_score']:.3f}"), unsafe_allow_html=True)
    else:
        st.markdown(metric_card("Anomaly Score", "N/A"), unsafe_allow_html=True)
with c5:
    if not loan_sub.empty:
        action = loan_sub.iloc[-1]["recommended_action"]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {COLORS['bg_card']}dd, {COLORS['bg_card_hover']}dd);
                    padding:20px;border-radius:16px;border:1px solid {COLORS['border']};text-align:center;
                    box-shadow:0 4px 20px rgba(0,0,0,0.3)">
            <div style="color:{COLORS['text_secondary']};font-size:0.85em;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:8px">Action</div>
            {action_badge(action)}
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Row: Predictions + Radar ─────────────────────────────────────────────────
col_pred, col_radar = st.columns([3, 2])

with col_pred:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '📊 Prediction Probabilities</div>', unsafe_allow_html=True)
    if not loan_sub.empty:
        sub_row = loan_sub.iloc[-1]
        prob_cols = {
            "3m Delinquency": "prob_delinq_3m",
            "6m Delinquency": "prob_delinq_6m",
            "12m Default": "prob_default_12m",
            "12m Prepayment": "prob_prepay_12m",
        }
        fig = go.Figure()
        names, vals, colors = [], [], []
        bar_colors = [COLORS["accent_amber"], COLORS["accent_amber"],
                      COLORS["accent_red"], COLORS["accent_cyan"]]
        for i, (name, col) in enumerate(prob_cols.items()):
            if col in sub_row.index:
                names.append(name)
                vals.append(float(sub_row[col]))
                colors.append(bar_colors[i])
        fig.add_trace(go.Bar(
            x=names, y=vals, marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=13, color=COLORS["text_primary"]),
        ))
        fig.update_layout(yaxis_title="Probability", yaxis=dict(range=[0, max(vals) * 1.3 if vals else 1]))
        st.plotly_chart(styled_figure(fig, 350), use_container_width=True)
    else:
        st.info("No submission predictions available for this loan.")

with col_radar:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '🎯 Risk Radar</div>', unsafe_allow_html=True)
    radar_data = {}
    radar_data["Trust"] = float(latest["trust_score"])
    if not loan_anom.empty:
        radar_data["Anomaly"] = float(loan_anom.iloc[-1]["anomaly_score"])
    if not loan_sub.empty:
        sr = loan_sub.iloc[-1]
        if "prob_default_12m" in sr.index:
            radar_data["Default Risk"] = min(float(sr["prob_default_12m"]) * 3, 1.0)
        if "prob_delinq_3m" in sr.index:
            radar_data["Delinq Risk"] = min(float(sr["prob_delinq_3m"]) * 3, 1.0)
        if "confidence" in sr.index:
            radar_data["Confidence"] = float(sr["confidence"])
    if len(radar_data) >= 3:
        st.plotly_chart(radar_chart(radar_data, f"Risk Profile — {selected}"), use_container_width=True)
    else:
        st.info("Insufficient data for radar chart.")

# ── SHAP Drivers ──────────────────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '🧠 SHAP Risk Drivers</div>', unsafe_allow_html=True)
if not loan_exp.empty and "top_drivers" in loan_exp.columns:
    drivers_str = str(loan_exp.iloc[-1]["top_drivers"])
    fig = shap_waterfall_chart(drivers_str, selected)
    st.plotly_chart(fig, use_container_width=True)
elif not loan_sub.empty and "top_drivers" in loan_sub.columns:
    st.code(str(loan_sub.iloc[-1]["top_drivers"]), language=None)
else:
    st.info("No SHAP driver data available for this loan.")

# ── Trust Score Timeline ──────────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '📈 Trust Score Over Time</div>', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=loan_trust["reporting_month"], y=loan_trust["trust_score"],
    mode="lines+markers",
    line=dict(color=COLORS["accent_cyan"], width=2),
    marker=dict(size=6, color=COLORS["accent_cyan"]),
    fill="tozeroy", fillcolor="rgba(79, 195, 247, 0.08)",
))
fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["accent_red"],
              annotation_text="LOW threshold", annotation_position="top left")
fig.update_layout(xaxis_title="Month", yaxis_title="Trust Score",
                  yaxis=dict(range=[0, 1.05]))
st.plotly_chart(styled_figure(fig, 300), use_container_width=True)

# ── Rules Fired Detail ────────────────────────────────────────────────────────
if "rules_fired" in loan_trust.columns:
    fired = loan_trust[loan_trust["rules_fired"].notna() & (loan_trust["rules_fired"] != "")]
    if not fired.empty:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                    'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                    '🚨 Rules Fired History</div>', unsafe_allow_html=True)
        st.dataframe(
            fired[["reporting_month", "rules_fired", "trust_score", "n_rules_fired", "n_conflicts"]],
            use_container_width=True, hide_index=True,
        )
