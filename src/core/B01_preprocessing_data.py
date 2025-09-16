import pandas as pd
import numpy as np
from typing import Dict, Optional, Iterable

def _is_dt_index(df: pd.DataFrame) -> bool:
    return pd.api.types.is_datetime64_any_dtype(df.index)

def _rename_numeric_to_brut(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return df
    ren = {c: (f"{c}_brut" if not c.endswith("_brut") else c) for c in num_cols}
    # éviter collisions (rare mais safe)
    ren = {k: v for k, v in ren.items() if v not in df.columns or v == k}
    return df.rename(columns=ren)

def _add_moving_avg(df: pd.DataFrame, window: str) -> pd.DataFrame:
    df = df.copy()
    brut_cols = [c for c in df.columns if c.endswith("_brut")]
    for col in brut_cols:
        base = col[:-5]  # enlève '_brut'
        df[f"{base}_moyenne glissante_{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df

def _add_speed(df: pd.DataFrame, mean_window: str, smooth: bool, smooth_window: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if df.index.size < 2:
        return df
    # pas de timezone pour éviter surprises
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    dt_hours = df.index.to_series().diff().dt.total_seconds() / 3600.0
    brut_cols = [c for c in df.columns if c.endswith("_brut")]
    for col in brut_cols:
        delta = df[col].diff()
        speed = delta / dt_hours
        name_v = f"{col}_vitesse_{mean_window}"
        df[name_v] = speed.rolling(mean_window, min_periods=1).mean()
        if smooth and smooth_window:
            name_s = f"{name_v}_smoothed_{smooth_window}"
            df[name_s] = df[name_v].rolling(smooth_window, min_periods=1).mean()
    return df

def appliquer_transformations_temporelles(
    df_dict: Dict[str, pd.DataFrame],
    config_global: Optional[dict] = None,
    configs_par_df: Optional[Dict[str, dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Transforme chaque DF :
      - renomme les colonnes numériques -> *_brut
      - moyennes glissantes (option)
      - vitesses moyennes + lissage (option)

    config_global / configs_par_df – clés possibles :
      {
        "appliquer_moyenne": bool,
        "periode_moy": "24H" | "7D" | "15T" | ...  (rolling time-based)
        "appliquer_vitesse": bool,
        "delta_duree": "6H" | "24H" | ...
        "lisser": bool,
        "periode_lissage": "48H" | ...
      }
    """
    out = {}
    for name, df in df_dict.items():
        if df is None or df.empty or not _is_dt_index(df):
            continue

        cfg = (configs_par_df or {}).get(name, config_global or {})
        df2 = _rename_numeric_to_brut(df)

        if cfg.get("appliquer_moyenne") and cfg.get("periode_moy"):
            df2 = _add_moving_avg(df2, cfg["periode_moy"])

        if cfg.get("appliquer_vitesse") and cfg.get("delta_duree"):
            df2 = _add_speed(
                df2,
                mean_window=cfg["delta_duree"],
                smooth=bool(cfg.get("lisser")),
                smooth_window=cfg.get("periode_lissage"),
            )

        out[name] = df2
    return out

def exclure_dates(
    df_dict: Dict[str, pd.DataFrame],
    dates_exclure: Iterable[pd.Timestamp],
    apply_all: bool = True,
    per_df_cols: Optional[Dict[str, "all | list[str]"]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Exclut des dates (index datetime) :
      - apply_all=True  -> supprime ces lignes pour TOUS les DF (toutes colonnes)
      - apply_all=False -> per_df_cols précise par DF soit "all" (supprimer lignes),
                           soit une liste de colonnes (mettre NaN uniquement sur ces colonnes)
    Retourne un nouveau dict (copie défensive).
    """
    ix_excl = pd.to_datetime(list(dates_exclure))
    out: Dict[str, pd.DataFrame] = {}

    for name, df in df_dict.items():
        if df is None or df.empty:
            out[name] = df
            continue
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            # on ne touche pas si l'index n'est pas temporel
            out[name] = df
            continue

        d = df.copy()
        if apply_all:
            d = d[~d.index.isin(ix_excl)]
        else:
            spec = (per_df_cols or {}).get(name)
            if spec is None:
                # rien à faire pour ce DF
                out[name] = d
                continue
            if spec == "all":
                d = d[~d.index.isin(ix_excl)]
            else:
                # colonnes spécifiques -> mettre NaN aux dates exclues
                cols = [c for c in spec if c in d.columns]
                if cols:
                    mask = d.index.isin(ix_excl)
                    d.loc[mask, cols] = np.nan
        out[name] = d
    return out
