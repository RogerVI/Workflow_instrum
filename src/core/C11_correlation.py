# core/C11_correlation.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, max_error

TempRange = Tuple[float, float, int]
TempRangeSpec = Union[TempRange, Dict[str, TempRange]]
PolySpec = Union[Tuple[bool, int], Dict[str, Tuple[bool, int]]]

def _filter_period(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    d = df.copy()
    try:
        d.index = pd.to_datetime(d.index)
    except Exception:
        pass
    if start is not None:
        d = d[d.index >= pd.to_datetime(start)]
    if end is not None:
        d = d[d.index <= pd.to_datetime(end)]
    return d

def _clean_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x.flatten()) & ~np.isnan(y)
    return x[mask], y[mask]

def _make_temp_range(tmin: float, tmax: float, steps: int = 8) -> np.ndarray:
    return np.linspace(float(tmin), float(tmax), int(max(2, steps))).reshape(-1,1)

def _get_tr_for(df_name: str, spec: TempRangeSpec) -> TempRange:
    if isinstance(spec, tuple):
        return spec
    # dict
    return spec.get(df_name, next(iter(spec.values())))

def _get_poly_for(df_name: str, spec: PolySpec) -> Tuple[bool, int]:
    if isinstance(spec, tuple):
        return spec
    # dict
    return spec.get(df_name, next(iter(spec.values())))

def analyse_correlation_core(
    df_dict: Dict[str, pd.DataFrame],
    selections: Dict[str, Dict[str, List[str]]],
    # selections = { df_name: { "temp": "TempCol", "depl": ["colA","colB", ...] } }
    t_ranges: TempRangeSpec,
    min_abs_corr: float = 0.5,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    poly_spec: PolySpec = (False, 2),
) -> Dict[str, Dict]:
    """
    Retourne un dict riche par DF :
    {
      df_name: {
        "rows": [ {Capteur, Température, Type, Degree, Pearson, R2, MAE, ErrMax}, ... ],
        "series": {
           capteur: {
             "x": np.ndarray, "y": np.ndarray, "temp_col": str,
             "t_range": np.ndarray (Nx1), "tmin": float, "tmax": float,
             "lin": {...},           # résultats linéaire
             "poly": {...} (opt)     # résultats polynomiaux si demandés
           },
           ...
        }
      },
      ...
    }
    """
    out: Dict[str, Dict] = {}

    for name, df in df_dict.items():
        if name not in selections:
            continue
        if df is None or df.empty:
            continue

        sel = selections[name]
        temp_col = sel.get("temp")
        depl_cols = sel.get("depl", [])
        if not temp_col or not depl_cols:
            continue

        d = _filter_period(df, start, end)
        if d.empty or temp_col not in d.columns:
            continue

        do_poly, poly_degree = _get_poly_for(name, poly_spec)
        tmin, tmax, steps = _get_tr_for(name, t_ranges)
        temp_range = _make_temp_range(tmin, tmax, steps)

        rows: List[Dict] = []
        series: Dict[str, Dict] = {}

        X_full = d[temp_col].values.reshape(-1,1)
        for col in depl_cols:
            if col not in d.columns:
                continue
            y_full = d[col].values
            x, y = _clean_xy(X_full, y_full)
            if x.size < 3:
                continue

            # Pearson (sur données filtrées)
            corr = float(np.corrcoef(x.flatten(), y)[0,1])

            if abs(corr) < float(min_abs_corr):
                # on ignore cette courbe (trop faible corrélation)
                continue

            # Régression linéaire
            reg = LinearRegression()
            reg.fit(x, y)
            y_pred = reg.predict(x)
            y_pred_temp = reg.predict(temp_range)

            resid = y - y_pred
            err_std = float(resid.std())
            err_max = float(np.abs(resid).max())
            r2 = float(r2_score(y, y_pred))
            mae = float(mean_absolute_error(y, y_pred))

            info = {
                "x": x, "y": y,
                "temp_col": temp_col,
                "t_range": temp_range, "tmin": float(temp_range[0][0]), "tmax": float(temp_range[-1][0]),
                "lin": {
                    "coef": float(reg.coef_.flatten()[0]),
                    "intercept": float(reg.intercept_),
                    "y_pred": y_pred,
                    "y_pred_temp": y_pred_temp,
                    "err_std": err_std,
                    "err_max": err_max,
                    "r2": r2,
                    "mae": mae,
                }
            }
            series[col] = info

            rows.append({
                "Capteur": col,
                "Température": temp_col,
                "Type": "Lin",
                "Degree": np.nan,
                "Pearson": round(corr, 3),
                "R2": round(r2, 3),
                "MAE": round(mae, 3),
                "Erreur max": round(err_max, 3),
            })

            # Régression polynomiale (si demandée)
            if do_poly:
                deg = int(max(2, poly_degree))
                poly = PolynomialFeatures(degree=deg, include_bias=False)
                Xp = poly.fit_transform(x)
                regp = LinearRegression()
                regp.fit(Xp, y)
                y_pp = regp.predict(Xp)
                tp = poly.transform(temp_range)
                y_tp = regp.predict(tp)

                resid_p = y - y_pp
                err_std_p = float(resid_p.std())
                err_max_p = float(np.abs(resid_p).max())
                r2_p = float(r2_score(y, y_pp))
                mae_p = float(mean_absolute_error(y, y_pp))

                info["poly"] = {
                    "degree": deg,
                    "y_pred": y_pp,
                    "y_pred_temp": y_tp,
                    "err_std": err_std_p,
                    "err_max": err_max_p,
                    "r2": r2_p,
                    "mae": mae_p,
                }

                rows.append({
                    "Capteur": col,
                    "Température": temp_col,
                    "Type": "Poly",
                    "Degree": deg,
                    "Pearson": round(corr, 3),
                    "R2": round(r2_p, 3),
                    "MAE": round(mae_p, 3),
                    "Erreur max": round(err_max_p, 3),
                })

        out[name] = {"rows": rows, "series": series}

    return out

def export_correlation_to_excel(
    results: Dict[str, Dict],
    excel_basename: str = "resultats_regression.xlsx"
) -> bytes:
    """
    Excel en mémoire :
    - une feuille par DF avec les lignes 'rows'
    - une feuille 'Predictions' (Lin + Poly) avec valeurs vs température
    """
    import io
    bio = io.BytesIO()

    pred_rows = []
    for df_name, block in results.items():
        series = block.get("series", {})
        for col, info in series.items():
            tr = info["t_range"].flatten().tolist()

            lin = info.get("lin")
            if lin:
                for t, yp in zip(tr, lin["y_pred_temp"]):
                    pred_rows.append({
                        "DataFrame": df_name,
                        "Capteur": col,
                        "Type": "Lin",
                        "Degree": np.nan,
                        "Température": float(t),
                        "Prediction": float(yp),
                        "Inf95": float(yp - 1.96*lin["err_std"]),
                        "Sup95": float(yp + 1.96*lin["err_std"]),
                        "InfMax": float(yp - lin["err_max"]),
                        "SupMax": float(yp + lin["err_max"]),
                        "R2": float(lin["r2"]),
                        "MAE": float(lin["mae"]),
                    })

            poly = info.get("poly")
            if poly:
                for t, yp in zip(tr, poly["y_pred_temp"]):
                    pred_rows.append({
                        "DataFrame": df_name,
                        "Capteur": col,
                        "Type": "Poly",
                        "Degree": poly["degree"],
                        "Température": float(t),
                        "Prediction": float(yp),
                        "Inf95": float(yp - 1.96*poly["err_std"]),
                        "Sup95": float(yp + 1.96*poly["err_std"]),
                        "InfMax": float(yp - poly["err_max"]),
                        "SupMax": float(yp + poly["err_max"]),
                        "R2": float(poly["r2"]),
                        "MAE": float(poly["mae"]),
                    })

    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for df_name, block in results.items():
            rows = block.get("rows", [])
            df_rows = pd.DataFrame(rows)
            if df_rows.empty:
                df_rows = pd.DataFrame(columns=["Capteur","Température","Type","Degree","Pearson","R2","MAE","Erreur max"])
            df_rows.to_excel(w, sheet_name=str(df_name)[:31], index=False)

        if pred_rows:
            pd.DataFrame(pred_rows).to_excel(w, sheet_name="Predictions", index=False)

    bio.seek(0)
    return bio.read()
