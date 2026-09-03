"""Shared utilities for the Loan Performance Intelligence Engine dashboard.

Provides cached data loading, badge formatting, and chart helpers used across all pages.
"""
from __future__ import annotations

import os
import json
from functools import lru_cache

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "reports", "artifacts")
FIG = os.path.join(ROOT, "reports", "figures")
MODELS = os.path.join(ROOT, "models")
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "logs")
REPORTS = os.path.join(ROOT, "reports")

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0e1117",
    "bg_card": "#1a1f2e",
    "bg_card_hover": "#232a3d",
    "accent_blue": "#4fc3f7",
    "accent_cyan": "#00e5ff",
    "accent_green": "#69f0ae",
    "accent_amber": "#ffd740",
    "accent_red": "#ff5252",
    "accent_purple": "#b388ff",
    "text_primary": "#e0e0e0",
    "text_secondary": "#9e9e9e",
    "border": "#2a3042",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

ACTION_COLORS = {
    "ESCALATE": "#ff5252",
    "REVIEW": "#ffd740",
    "AUTO_ACCEPT": "#69f0ae",
}

TRUST_COLORS = {
    "LOW": "#ff5252",
    "MEDIUM": "#ffd740",
    "HIGH": "#69f0ae",
}


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_artifact(name: str) -> pd.DataFrame:
    """Load a CSV artifact from reports/artifacts/."""
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_trust_scores() -> pd.DataFrame:
    return load_artifact("record_trust_scores.csv")


@st.cache_data(ttl=600)
def load_anomaly_scores() -> pd.DataFrame:
    return load_artifact("anomaly_scores_train.csv")


@st.cache_data(ttl=600)
def load_local_explanations() -> pd.DataFrame:
    return load_artifact("local_explanations.csv")


@st.cache_data(ttl=600)
def load_metrics() -> pd.DataFrame:
    return load_artifact("metrics_task2.csv")


@st.cache_data(ttl=600)
def load_scenario_curves() -> pd.DataFrame:
    return load_artifact("scenario_curves.csv")


@st.cache_data(ttl=600)
def load_scenario_segments() -> pd.DataFrame:
    return load_artifact("scenario_segments.csv")


@st.cache_data(ttl=600)
def load_conformal() -> pd.DataFrame:
    return load_artifact("conformal_by_trust.csv")


@st.cache_data(ttl=600)
def load_shap_global() -> pd.DataFrame:
    return load_artifact("shap_global_default.csv")


@st.cache_data(ttl=600)
def load_reliability_curves() -> pd.DataFrame:
    return load_artifact("reliability_curves.csv")


@st.cache_data(ttl=600)
def load_fpfn() -> pd.DataFrame:
    return load_artifact("fpfn_analysis.csv")


@st.cache_data(ttl=600)
def load_batch_quality_servicer() -> pd.DataFrame:
    return load_artifact("batch_quality_servicer.csv")


@st.cache_data(ttl=600)
def load_batch_quality_month() -> pd.DataFrame:
    return load_artifact("batch_quality_month.csv")


@st.cache_data(ttl=600)
def load_rule_violation_summary() -> pd.DataFrame:
    return load_artifact("rule_violation_summary.csv")


@st.cache_data(ttl=600)
def load_drift() -> pd.DataFrame:
    return load_artifact("drift_psi.csv")


@st.cache_data(ttl=600)
def load_submission() -> pd.DataFrame:
    path = os.path.join(ROOT, "submission.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_prompt_log() -> list[dict]:
    path = os.path.join(LOGS, "prompt_log.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


@st.cache_data(ttl=600)
def load_reviewed_outputs() -> list[dict]:
    path = os.path.join(LOGS, "reviewed_outputs.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ── Badge formatters ──────────────────────────────────────────────────────────

def trust_badge(score: float, band: str = None) -> str:
    """Returns an HTML badge for a trust score."""
    if band is None:
        if score < 0.5:
            band = "LOW"
        elif score < 0.8:
            band = "MEDIUM"
        else:
            band = "HIGH"
    color = TRUST_COLORS.get(band, "#9e9e9e")
    return f'<span style="background:{color}22;color:{color};padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85em;border:1px solid {color}44">{band} ({score:.2f})</span>'


def action_badge(action: str) -> str:
    """Returns an HTML badge for an action recommendation."""
    color = ACTION_COLORS.get(action, "#9e9e9e")
    icons = {"ESCALATE": "🚨", "REVIEW": "⚠️", "AUTO_ACCEPT": "✅"}
    icon = icons.get(action, "")
    return f'<span style="background:{color}22;color:{color};padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85em;border:1px solid {color}44">{icon} {action}</span>'


def metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    """Returns styled HTML for a KPI metric card."""
    delta_html = ""
    if delta:
        d_color = COLORS["accent_green"] if delta_color == "normal" else COLORS["accent_red"]
        delta_html = f'<div style="color:{d_color};font-size:0.8em;margin-top:2px">{delta}</div>'
    return f"""
    <div style="background:linear-gradient(135deg, {COLORS['bg_card']}dd, {COLORS['bg_card_hover']}dd);
                backdrop-filter:blur(10px);padding:20px;border-radius:16px;
                border:1px solid {COLORS['border']};text-align:center;
                box-shadow:0 4px 20px rgba(0,0,0,0.3)">
        <div style="color:{COLORS['text_secondary']};font-size:0.85em;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:8px">{label}</div>
        <div style="color:{COLORS['accent_cyan']};font-size:1.8em;font-weight:700">{value}</div>
        {delta_html}
    </div>
    """


# ── Chart helpers ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=40, r=20, t=40, b=40),
)


def styled_figure(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent dark theme styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    return fig


def radar_chart(data: dict, title: str = "Risk Profile") -> go.Figure:
    """Create a radar chart for a loan's risk profile."""
    categories = list(data.keys())
    values = list(data.values())
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor=f"rgba(79, 195, 247, 0.15)",
        line=dict(color=COLORS["accent_cyan"], width=2),
        marker=dict(size=6, color=COLORS["accent_cyan"]),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"],
                            tickfont=dict(size=10, color=COLORS["text_secondary"])),
            angularaxis=dict(gridcolor=COLORS["border"],
                             tickfont=dict(size=11, color=COLORS["text_primary"])),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=14)),
    )
    return styled_figure(fig, height=380)


def donut_chart(labels: list, values: list, title: str, colors: list = None) -> go.Figure:
    """Create a donut chart."""
    if colors is None:
        colors = [COLORS["accent_green"], COLORS["accent_amber"], COLORS["accent_red"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors[:len(labels)],
                    line=dict(color=COLORS["bg_primary"], width=2)),
        textinfo="label+percent", textfont=dict(size=12),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)),
                      showlegend=True,
                      legend=dict(font=dict(size=11)))
    return styled_figure(fig, height=350)


def shap_waterfall_chart(drivers_str: str, loan_id: str) -> go.Figure:
    """Parse SHAP driver string and create a horizontal bar waterfall chart."""
    if not isinstance(drivers_str, str) or not drivers_str.strip():
        fig = go.Figure()
        fig.add_annotation(text="No SHAP data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        return styled_figure(fig, height=300)

    entries = drivers_str.split("; ")
    features, shap_vals = [], []
    for entry in entries:
        try:
            parts = entry.rsplit(" (", 1)
            feat = parts[0].strip()
            val = float(parts[1].rstrip(")"))
            features.append(feat)
            shap_vals.append(val)
        except (IndexError, ValueError):
            continue

    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        return styled_figure(fig, height=300)

    colors_bar = [COLORS["accent_red"] if v > 0 else COLORS["accent_green"] for v in shap_vals]
    features.reverse()
    shap_vals.reverse()
    colors_bar.reverse()

    fig = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker=dict(color=colors_bar, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=f"SHAP Drivers — {loan_id}", font=dict(size=14)),
        xaxis_title="SHAP value (impact on default probability)",
        yaxis_title="",
    )
    return styled_figure(fig, height=max(250, len(features) * 40))
