"""⚠️ Anomaly Triage — Prioritized queue of flagged loans for review."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_anomaly_scores, load_trust_scores, load_submission,
    metric_card, styled_figure, donut_chart, action_badge, trust_badge,
    COLORS, ACTION_COLORS,
)

st.set_page_config(page_title="Anomaly Triage", page_icon="⚠️", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '⚠️ Anomaly Triage Queue</div>', unsafe_allow_html=True)

anomaly = load_anomaly_scores()
trust = load_trust_scores()
submission = load_submission()

if anomaly.empty:
    st.warning("⚠️ No anomaly data available. Run the pipeline first: `python run_all.py`")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
latest = anomaly.drop_duplicates("loan_id", keep="last")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Total Scored", f"{len(latest):,}"), unsafe_allow_html=True)
with c2:
    hi = (latest["anomaly_score"] > 0.7).sum()
    st.markdown(metric_card("High Anomaly (>0.7)", f"{hi:,}",
                            f"{hi/len(latest)*100:.1f}%", "inverse"), unsafe_allow_html=True)
with c3:
    med = ((latest["anomaly_score"] > 0.4) & (latest["anomaly_score"] <= 0.7)).sum()
    st.markdown(metric_card("Medium (0.4-0.7)", f"{med:,}"), unsafe_allow_html=True)
with c4:
    lo = (latest["anomaly_score"] <= 0.4).sum()
    st.markdown(metric_card("Low (≤0.4)", f"{lo:,}"), unsafe_allow_html=True)

st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    min_score = st.slider("Minimum Anomaly Score", 0.0, 1.0, 0.4, 0.05)
with col_ctrl2:
    sort_by = st.selectbox("Sort By", ["anomaly_score", "exception_required_prob",
                                        "trust_score", "iso_percentile"])
with col_ctrl3:
    top_n = st.slider("Show Top N", 10, 500, 50, 10)

# ── Filtered Table ────────────────────────────────────────────────────────────
filtered = latest[latest["anomaly_score"] >= min_score].sort_values(sort_by, ascending=False).head(top_n)

st.markdown(f'<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            f'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            f'🔍 Flagged Loans ({len(filtered)} shown)</div>', unsafe_allow_html=True)

display_cols = ["loan_id", "reporting_month", "anomaly_score", "exception_required_prob",
                "exception_type_pred", "trust_score", "iso_percentile", "n_conflicts"]
avail_cols = [c for c in display_cols if c in filtered.columns]

if not filtered.empty:
    def highlight_anomaly(row):
        styles = [""] * len(row)
        if "anomaly_score" in row.index:
            score = row["anomaly_score"]
            idx = row.index.get_loc("anomaly_score")
            if score > 0.7:
                styles[idx] = "color: #ff5252; font-weight: 700"
            elif score > 0.4:
                styles[idx] = "color: #ffd740; font-weight: 600"
            else:
                styles[idx] = "color: #69f0ae"
        return styles

    styled_df = filtered[avail_cols].style.apply(highlight_anomaly, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)

st.markdown("---")

# ── Signal Decomposition ─────────────────────────────────────────────────────
col_sig, col_type = st.columns(2)

with col_sig:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '📊 Signal Decomposition (Top 50)</div>', unsafe_allow_html=True)
    top50 = latest.sort_values("anomaly_score", ascending=False).head(50)
    fig = go.Figure()

    if "iso_percentile" in top50.columns and "exception_required_prob" in top50.columns:
        # Infer rule component: anomaly - (iso * 0.15 + sup * 0.40) / 0.45
        rule_comp = np.clip(top50["anomaly_score"] - 0.15 * top50["iso_percentile"]
                            - 0.40 * top50["exception_required_prob"], 0, 1)
        fig.add_trace(go.Bar(name="Rules (45%)", x=top50["loan_id"], y=rule_comp,
                             marker_color=COLORS["accent_red"]))
        fig.add_trace(go.Bar(name="Isolation (15%)", x=top50["loan_id"],
                             y=top50["iso_percentile"] * 0.15,
                             marker_color=COLORS["accent_purple"]))
        fig.add_trace(go.Bar(name="Supervised (40%)", x=top50["loan_id"],
                             y=top50["exception_required_prob"] * 0.40,
                             marker_color=COLORS["accent_amber"]))
        fig.update_layout(barmode="stack", xaxis_title="Loan ID", yaxis_title="Anomaly Score",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                          legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(styled_figure(fig, 400), use_container_width=True)

with col_type:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                'Exception Type Distribution</div>', unsafe_allow_html=True)
    if "exception_type_pred" in latest.columns:
        type_counts = latest["exception_type_pred"].value_counts()
        colors_list = [COLORS["accent_red"], COLORS["accent_amber"], COLORS["accent_cyan"],
                       COLORS["accent_purple"], COLORS["accent_green"], "#e0e0e0", "#78909c"]
        fig = donut_chart(
            type_counts.index.tolist(), type_counts.values.tolist(),
            "Predicted Exception Types", colors_list,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Anomaly Score Distribution ────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            'Anomaly Score Distribution</div>', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=latest["anomaly_score"], nbinsx=60,
    marker=dict(color=COLORS["accent_amber"], line=dict(width=0.5, color=COLORS["bg_primary"])),
))
fig.add_vline(x=0.4, line_dash="dash", line_color=COLORS["accent_amber"],
              annotation_text="REVIEW threshold")
fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["accent_red"],
              annotation_text="ESCALATE threshold")
fig.update_layout(xaxis_title="Anomaly Score", yaxis_title="Count")
st.plotly_chart(styled_figure(fig, 300), use_container_width=True)
