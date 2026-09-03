"""📈 Model Performance — Metrics, calibration, SHAP, and conformal coverage."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    load_metrics, load_shap_global, load_reliability_curves,
    load_fpfn, load_conformal, metric_card, styled_figure, COLORS,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;font-weight:800;font-size:2em;margin-bottom:0.5em">'
            '📈 Model Performance</div>', unsafe_allow_html=True)

metrics = load_metrics()
shap_global = load_shap_global()
reliability = load_reliability_curves()
fpfn = load_fpfn()
conformal = load_conformal()

if metrics.empty:
    st.warning("⚠️ No metrics data available. Run the pipeline first: `python run_all.py`")
    st.stop()

# ── Metrics Table ─────────────────────────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '🏆 Champion Model Metrics (Validation Set)</div>', unsafe_allow_html=True)

# Highlight champion rows
def style_metrics(row):
    styles = [""] * len(row)
    if "champion" in metrics.columns and row.get("champion", False):
        return ["background: rgba(79, 195, 247, 0.08); font-weight: 600"] * len(row)
    return styles

st.dataframe(metrics.style.apply(style_metrics, axis=1),
             use_container_width=True, hide_index=True, height=400)

st.markdown("---")

# ── KPI Cards: Best AUCs ─────────────────────────────────────────────────────
if "roc_auc" in metrics.columns and "target" in metrics.columns:
    targets = metrics["target"].unique()
    cols = st.columns(min(len(targets), 4))
    for i, tgt in enumerate(targets[:4]):
        subset = metrics[metrics["target"] == tgt]
        best = subset.loc[subset["roc_auc"].idxmax()] if not subset.empty else None
        if best is not None:
            with cols[i]:
                label = tgt.replace("next_", "").replace("_flag", "").replace("_", " ").title()
                st.markdown(metric_card(label, f"AUC {best['roc_auc']:.3f}",
                                        f"Model: {best.get('model', 'N/A')}"),
                            unsafe_allow_html=True)
    st.markdown("---")

# ── AUC Comparison Bar Chart ─────────────────────────────────────────────────
col_auc, col_shap = st.columns(2)

with col_auc:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                'ROC-AUC by Target & Model</div>', unsafe_allow_html=True)
    if "roc_auc" in metrics.columns:
        fig = go.Figure()
        palette = [COLORS["accent_cyan"], COLORS["accent_amber"],
                   COLORS["accent_red"], COLORS["accent_purple"],
                   COLORS["accent_green"]]
        for i, model in enumerate(metrics["model"].unique()):
            sub = metrics[metrics["model"] == model]
            fig.add_trace(go.Bar(
                name=model,
                x=[t.replace("next_", "").replace("_flag", "") for t in sub["target"]],
                y=sub["roc_auc"],
                marker_color=palette[i % len(palette)],
                text=[f"{v:.3f}" for v in sub["roc_auc"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
        fig.update_layout(barmode="group", yaxis_title="ROC-AUC",
                          yaxis=dict(range=[0.5, 1.0]),
                          legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(styled_figure(fig, 400), use_container_width=True)

# ── SHAP Global Importance ────────────────────────────────────────────────────
with col_shap:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '🧠 SHAP Global Importance (12m Default)</div>', unsafe_allow_html=True)
    if not shap_global.empty:
        feat_col = shap_global.columns[0]
        val_col = shap_global.columns[1] if len(shap_global.columns) > 1 else None
        if val_col:
            top = shap_global.nlargest(15, val_col)
            fig = go.Figure(go.Bar(
                x=top[val_col].values[::-1],
                y=top[feat_col].values[::-1],
                orientation="h",
                marker=dict(
                    color=top[val_col].values[::-1],
                    colorscale=[[0, COLORS["accent_cyan"]], [1, COLORS["accent_red"]]],
                    line=dict(width=0),
                ),
                text=[f"{v:.4f}" for v in top[val_col].values[::-1]],
                textposition="outside",
            ))
            fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
            st.plotly_chart(styled_figure(fig, 400), use_container_width=True)
    else:
        st.info("No SHAP data available.")

st.markdown("---")

# ── Reliability / Calibration Curves ──────────────────────────────────────────
st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
            'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
            '📐 Calibration / Reliability Curves</div>', unsafe_allow_html=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "reports", "figures")

reliability_images = [f for f in os.listdir(FIG_DIR) if f.startswith("reliability_")] if os.path.isdir(FIG_DIR) else []

if reliability_images:
    cols = st.columns(min(len(reliability_images), 4))
    for i, img_name in enumerate(sorted(reliability_images)):
        with cols[i % len(cols)]:
            target_name = img_name.replace("reliability_", "").replace(".png", "").replace("_", " ").title()
            st.markdown(f'<div style="color:{COLORS["text_secondary"]};font-size:0.85em;'
                        f'text-align:center;margin-bottom:4px">{target_name}</div>',
                        unsafe_allow_html=True)
            st.image(os.path.join(FIG_DIR, img_name), use_container_width=True)
elif not reliability.empty:
    if "target" in reliability.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color=COLORS["text_secondary"]),
                                 name="Perfect"))
        for tgt in reliability["target"].unique():
            sub = reliability[reliability["target"] == tgt]
            if "mean_predicted" in sub.columns and "fraction_positive" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["mean_predicted"], y=sub["fraction_positive"],
                    mode="lines+markers", name=tgt.replace("next_", "").replace("_flag", ""),
                ))
        fig.update_layout(xaxis_title="Mean Predicted", yaxis_title="Fraction Positive")
        st.plotly_chart(styled_figure(fig, 400), use_container_width=True)
else:
    st.info("No calibration data available.")

st.markdown("---")

# ── Conformal Coverage by Trust Band ──────────────────────────────────────────
col_conf, col_fpfn = st.columns(2)

with col_conf:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '🎯 Conformal Coverage by Trust Band</div>', unsafe_allow_html=True)
    if not conformal.empty:
        st.dataframe(conformal, use_container_width=True, hide_index=True)
    else:
        st.info("No conformal coverage data available.")

# ── FP/FN Analysis ────────────────────────────────────────────────────────────
with col_fpfn:
    st.markdown('<div style="color:#4fc3f7;font-weight:700;font-size:1.15em;'
                'border-bottom:2px solid #2a3042;padding-bottom:8px;margin-bottom:1em">'
                '❌ False Positive / False Negative Analysis</div>', unsafe_allow_html=True)
    if not fpfn.empty:
        st.dataframe(fpfn, use_container_width=True, hide_index=True)
    else:
        st.info("No FP/FN data available.")
