# core/A12_traitement_nan.py
from __future__ import annotations
from typing import Dict, Optional, Literal
import pandas as pd

Strategy = Literal[
    "none", "zero", "mean", "median",
    "interpolate", "drop_rows", "drop_cols"
]

INTERP_METHODS = {
    "linear", "time", "index", "nearest",
    "polynomial", "spline", "pad", "bfill"
}

def compute_nan_stats(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retourne un petit tableau (nom_df, total_vals, nb_nan, pct_nan)."""
    rows = []
    for name, df in (df_dict or {}).items():
        total = int(df.size)
        nb_nan = int(df.isna().sum().sum())
        pct = (nb_nan / total * 100) if total else 0.0
        rows.append(dict(df=name, total_vals=total, nb_nan=nb_nan, pct_nan=round(pct, 2)))
    return pd.DataFrame(rows)

def _apply_strategy_one(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    interp_method: Optional[str] = None,
    interp_order: Optional[int] = None,
) -> pd.DataFrame:
    d = df.copy()

    if strategy == "none":
        return d

    if strategy == "zero":
        return d.fillna(0)

    if strategy == "mean":
        return d.fillna(d.mean(numeric_only=True))

    if strategy == "median":
        return d.fillna(d.median(numeric_only=True))

    if strategy == "interpolate":
        method = (interp_method or "linear").lower()
        if method not in INTERP_METHODS:
            method = "linear"
        if method in {"polynomial", "spline"}:
            order = int(interp_order or 2)
            return d.interpolate(method=method, order=order)
        return d.interpolate(method=method)

    if strategy == "drop_rows":
        return d.dropna(axis=0)

    if strategy == "drop_cols":
        return d.dropna(axis=1)

    return d

def apply_nan_strategy_all(
    df_dict: Dict[str, pd.DataFrame],
    *,
    strategy: Strategy,
    interp_method: Optional[str] = None,
    interp_order: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Applique la même stratégie à tous les DF."""
    out = {}
    for name, df in (df_dict or {}).items():
        out[name] = _apply_strategy_one(
            df, strategy,
            interp_method=interp_method,
            interp_order=interp_order,
        )
    return out

def apply_nan_strategy_per_df(
    df_dict: Dict[str, pd.DataFrame],
    per_df: Dict[str, dict],
) -> Dict[str, pd.DataFrame]:
    """
    per_df: { df_name: {strategy:..., interp_method:..., interp_order:...}, ... }
    """
    out = {}
    for name, df in (df_dict or {}).items():
        params = per_df.get(name, {}) or {}
        out[name] = _apply_strategy_one(
            df,
            params.get("strategy", "none"),
            interp_method=params.get("interp_method"),
            interp_order=params.get("interp_order"),
        )
    return out
