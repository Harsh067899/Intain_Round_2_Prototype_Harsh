"""🌪️ Scenario Simulator — Explore stress scenarios on the portfolio."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    load_scenario_curves, load_scenario_segments, metric_card,
    styled_figure, COLORS,
)

st.set_page_config(page_title="Scenario Simulator", page_icon="🌪️", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '🌪️ Scenario Simulator</div>', unsafe_allow_html=True)

curves = load_scenario_curves()
segments = load_scenario_segments()

if curves.empty:
    st.warning("⚠️ No scenario data available. Run the pipeline first: `python run_all.py`")
    st.stop()

# ── Scenario Overview ─────────────────────────────────────────────────────────
st.markdown("""
<div style="color:#9e9e9e;font-size:1.0em;margin-bottom:1.5em;line-height:1.6">
    Macro scenario shocks are applied to monthly transition hazards and propagated forward —
    delinquency compounds into later defaults (CCAR-style stress logic). Monte Carlo bands
    validate that the expected-value estimate lies within the simulated range.
</div>
""", unsafe_allow_html=True)

# ── Cumulative Incidence Curves ───────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '📉 Cumulative Default Incidence by Scenario</div>', unsafe_allow_html=True)

# Detect column structure
month_col = [c for c in curves.columns if "month" in c.lower() or "horizon" in c.lower()]
scenario_col = [c for c in curves.columns if "scenario" in c.lower()]

if scenario_col and month_col:
    sc = scenario_col[0]
    mc = month_col[0]
    value_cols = [c for c in curves.columns if c not in [sc, mc]]

    if value_cols:
        fig = go.Figure()
        scenario_colors = {
            "base": COLORS["accent_green"],
            "adverse_credit": COLORS["accent_red"],
            "high_prepayment": COLORS["accent_amber"],
        }
        for vc in value_cols:
            for scen in curves[sc].unique():
                sub = curves[curves[sc] == scen].sort_values(mc)
                color = scenario_colors.get(scen, COLORS["accent_cyan"])
                dash = "solid" if "default" in vc.lower() else "dash"
                fig.add_trace(go.Scatter(
                    x=sub[mc], y=sub[vc], mode="lines+markers",
                    name=f"{scen} — {vc}",
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=5),
                ))
        fig.update_layout(xaxis_title="Horizon (months)", yaxis_title="Cumulative Rate",
                          yaxis=dict(tickformat=".1%"),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(styled_figure(fig, 450), use_container_width=True)
else:
    # Fallback: show the raw table with a simple line plot
    fig = go.Figure()
    num_cols = curves.select_dtypes(include="number").columns
    palette = [COLORS["accent_green"], COLORS["accent_red"], COLORS["accent_amber"],
               COLORS["accent_cyan"], COLORS["accent_purple"]]
    for i, col in enumerate(num_cols):
        fig.add_trace(go.Scatter(
            x=curves.index, y=curves[col], mode="lines+markers",
            name=col, line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(xaxis_title="Row Index", yaxis_title="Value")
    st.plotly_chart(styled_figure(fig, 400), use_container_width=True)

# ── Scenario Curves Figure (if available) ─────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
scenario_img = os.path.join(FIG_DIR, "scenario_curves.png")
cum_img = os.path.join(FIG_DIR, "cumulative_incidence.png")

if os.path.exists(scenario_img) or os.path.exists(cum_img):
    st.markdown("---")
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '📊 Pre-generated Scenario Figures</div>', unsafe_allow_html=True)
    img_cols = st.columns(2)
    if os.path.exists(cum_img):
        with img_cols[0]:
            st.image(cum_img, caption="Cumulative Incidence", use_container_width=True)
    if os.path.exists(scenario_img):
        with img_cols[1]:
            st.image(scenario_img, caption="Scenario Curves", use_container_width=True)

st.markdown("---")

# ── Segment Impact Table ─────────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '🎯 Segment Impact Analysis</div>', unsafe_allow_html=True)

if not segments.empty:
    # Filter to a specific scenario
    if scenario_col and scenario_col[0] in segments.columns:
        sc = scenario_col[0]
        scenarios = segments[sc].unique().tolist()
        selected_scenario = st.selectbox("Select Scenario", scenarios)
        display = segments[segments[sc] == selected_scenario]
    else:
        display = segments

    st.dataframe(display, use_container_width=True, hide_index=True, height=400)

    # Visual: impact by segment
    num_cols = display.select_dtypes(include="number").columns.tolist()
    seg_cols = [c for c in display.columns if c not in num_cols and "scenario" not in c.lower()]

    if seg_cols and num_cols:
        seg_col = seg_cols[0]
        val_col = num_cols[0]
        fig = go.Figure(go.Bar(
            x=display[seg_col].astype(str),
            y=display[val_col],
            marker=dict(
                color=display[val_col],
                colorscale=[[0, COLORS["accent_green"]], [0.5, COLORS["accent_amber"]],
                            [1, COLORS["accent_red"]]],
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in display[val_col]],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title=seg_col, yaxis_title=val_col,
                          xaxis=dict(tickangle=-45))
        st.plotly_chart(styled_figure(fig, 350), use_container_width=True)
else:
    st.info("No segment impact data available.")

# ── Raw Data Explorer ─────────────────────────────────────────────────────────
with st.expander("📋 Raw Scenario Curves Data"):
    st.dataframe(curves, use_container_width=True, hide_index=True)
