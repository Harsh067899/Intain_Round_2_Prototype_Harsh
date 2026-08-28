"""Task 1a — Data profiling: distributions, missingness, correlations, drift (PSI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_HINTS = ["balance", "rate", "months", "days", "index", "flag"]


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        base = {
            "column": c, "dtype": str(s.dtype),
            "n": len(s), "missing_pct": round(100 * s.isna().mean(), 3),
            "n_unique": s.nunique(dropna=True),
        }
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            base.update(min=s.min(), p01=q[0.01], p25=q[0.25], median=q[0.5],
                        p75=q[0.75], p99=q[0.99], max=s.max(),
                        mean=round(float(s.mean()), 4), std=round(float(s.std()), 4),
                        pct_negative=round(100 * (s < 0).mean(), 3))
        else:
            vc = s.value_counts(dropna=True).head(5)
            base.update(top_values="; ".join(f"{k}:{v}" for k, v in vc.items()))
        rows.append(base)
    return pd.DataFrame(rows)


def missingness_patterns(df: pd.DataFrame, by: list[str]) -> dict[str, pd.DataFrame]:
    """Is missingness random, or concentrated in specific segments?"""
    out = {}
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    for seg in by:
        if seg not in df.columns or not miss_cols:
            continue
        out[seg] = (df.assign(**{f"miss_{c}": df[c].isna() for c in miss_cols})
                      .groupby(seg)[[f"miss_{c}" for c in miss_cols]]
                      .mean().mul(100).round(2))
    return out


def top_correlations(df: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True).abs()
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
                 .stack().sort_values(ascending=False).head(k))
    df = pairs.rename("abs_corr").reset_index()
    df.columns = ["feature_a", "feature_b", "abs_corr"]
    return df


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable | 0.1-0.25 moderate | >0.25 significant."""
    e, a = expected.dropna(), actual.dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan
    if pd.api.types.is_numeric_dtype(e) and e.nunique() > bins:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        e_bin = pd.cut(e, edges, include_lowest=True)
        a_bin = pd.cut(a, edges, include_lowest=True)
    else:
        e_bin, a_bin = e.astype(str), a.astype(str)
    ep = e_bin.value_counts(normalize=True)
    ap = a_bin.value_counts(normalize=True).reindex(ep.index).fillna(0)
    ep, ap = ep.clip(lower=1e-6), ap.clip(lower=1e-6)
    return float(((ap - ep) * np.log(ap / ep)).sum())


def drift_table(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c in train.columns and c in test.columns:
            v = psi(train[c], test[c])
            band = "stable" if v < 0.1 else "moderate" if v < 0.25 else "SIGNIFICANT"
            rows.append({"feature": c, "psi": round(v, 4), "assessment": band})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
