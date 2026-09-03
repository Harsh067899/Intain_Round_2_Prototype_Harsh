"""📊 Portfolio Overview — Trust, Actions, Data Quality at a glance."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_trust_scores, load_anomaly_scores, load_submission, load_drift,
    load_batch_quality_servicer, load_batch_quality_month,
    load_rule_violation_summary, metric_card, styled_figure,
    donut_chart, COLORS, TRUST_COLORS, ACTION_COLORS,
)

st.set_page_config(page_title="Portfolio Overview", page_icon="📊", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '📊 Portfolio Overview</div>', unsafe_allow_html=True)

trust = load_trust_scores()
anomaly = load_anomaly_scores()
submission = load_submission()
drift = load_drift()
bq_serv = load_batch_quality_servicer()
bq_month = load_batch_quality_month()
rules = load_rule_violation_summary()

# ── KPI Row ───────────────────────────────────────────────────────────────────
if not trust.empty:
    c1, c2, c3, c4, c5 = st.columns(5)
    n_loans = trust["loan_id"].nunique()
    with c1:
        st.markdown(metric_card("Total Loans", f"{n_loans:,}"), unsafe_allow_html=True)
    with c2:
        avg = trust["trust_score"].mean()
        st.markdown(metric_card("Avg Trust", f"{avg:.3f}"), unsafe_allow_html=True)
    with c3:
        low = (trust["trust_band"] == "LOW").mean() * 100
        st.markdown(metric_card("Low Trust %", f"{low:.1f}%"), unsafe_allow_html=True)
    with c4:
        if not anomaly.empty:
            hi_anom = (anomaly["anomaly_score"] > 0.7).mean() * 100
            st.markdown(metric_card("High Anomaly %", f"{hi_anom:.1f}%"), unsafe_allow_html=True)
    with c5:
        if not submission.empty:
            esc = (submission["recommended_action"] == "ESCALATE").sum()
            st.markdown(metric_card("Escalations", f"{esc:,}"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Trust Distribution + Action Breakdown ──────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                    'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                    'Trust Band Distribution</div>', unsafe_allow_html=True)
        band_counts = trust["trust_band"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
        fig = donut_chart(
            band_counts.index.tolist(), band_counts.values.tolist(),
            "Trust Bands",
            colors=[TRUST_COLORS.get(b, "#9e9e9e") for b in band_counts.index]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if not submission.empty:
            st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                        'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                        'Recommended Actions</div>', unsafe_allow_html=True)
            act_counts = submission["recommended_action"].value_counts()
            fig = donut_chart(
                act_counts.index.tolist(), act_counts.values.tolist(),
                "Action Distribution",
                colors=[ACTION_COLORS.get(a, "#9e9e9e") for a in act_counts.index]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Trust Score Histogram ──────────────────────────────────────────
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                'Trust Score Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trust["trust_score"], nbinsx=50,
        marker=dict(color=COLORS["accent_cyan"], line=dict(width=0.5, color=COLORS["bg_primary"])),
    ))
    fig.update_layout(
        xaxis_title="Trust Score", yaxis_title="Count",
        bargap=0.05,
    )
    st.plotly_chart(styled_figure(fig, 350), use_container_width=True)

    # ── Row 3: Rule Violations + Drift ────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                    'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                    'Rule Violation Summary</div>', unsafe_allow_html=True)
        if not rules.empty:
            st.dataframe(rules, use_container_width=True, hide_index=True)
        else:
            st.info("No rule violation data available.")

    with col_d:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                    'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                    'Train → Test Drift (PSI)</div>', unsafe_allow_html=True)
        if not drift.empty:
            def psi_color(val):
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    return ""
                if v > 0.25:
                    return "color: #ff5252; font-weight: 700"
                elif v > 0.10:
                    return "color: #ffd740; font-weight: 600"
                return "color: #69f0ae"

            if "psi" in drift.columns:
                styled = drift.style.applymap(psi_color, subset=["psi"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(drift, use_container_width=True, hide_index=True)
        else:
            st.info("No drift data available.")

    # ── Row 4: Batch Quality by Servicer ──────────────────────────────────────
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '🏢 Batch Quality by Servicer</div>', unsafe_allow_html=True)
    if not bq_serv.empty:
        st.dataframe(bq_serv, use_container_width=True, hide_index=True)
    else:
        st.info("No batch quality data available.")

    # ── Row 5: Batch Quality Over Time ────────────────────────────────────────
    if not bq_month.empty and "reporting_month" in bq_month.columns:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                    'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                    '📅 Batch Quality Over Time</div>', unsafe_allow_html=True)
        num_cols = bq_month.select_dtypes(include="number").columns.tolist()
        if num_cols:
            fig = go.Figure()
            palette = [COLORS["accent_cyan"], COLORS["accent_amber"],
                       COLORS["accent_red"], COLORS["accent_purple"]]
            for i, col in enumerate(num_cols[:4]):
                fig.add_trace(go.Scatter(
                    x=bq_month["reporting_month"], y=bq_month[col],
                    mode="lines+markers", name=col,
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=5),
                ))
            fig.update_layout(xaxis_title="Month", yaxis_title="Score",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(styled_figure(fig, 350), use_container_width=True)
else:
    st.warning("⚠️ No trust score data available. Run the pipeline first: `python run_all.py`")
