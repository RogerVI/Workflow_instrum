# core/A20_stats.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

def _period_mask(idx: pd.DatetimeIndex, resume_par: str) -> pd.Series:
    """Retourne un masque booléen pour la dernière semaine (ISO) ou le dernier mois du dernier timestamp."""
    if not isinstance(idx, pd.DatetimeIndex) or idx.size == 0:
        return pd.Series(False, index=idx)

    last = idx[-1]
    if resume_par == "semaine":
        iso = idx.isocalendar()
        last_iso = last.isocalendar()
        return (iso.week == last_iso.week) & (iso.year == last_iso.year)
    # défaut: mois
    return (idx.month == last.month) & (idx.year == last.year)

def stats_df_dict_all_with_temp_corr_only_for_others(
    df_dict: Dict[str, pd.DataFrame],
    temp_col_contains: str = "temp",
    arrondi: int = 3,
    resume_par: str = "mois",
) -> Dict[str, pd.DataFrame]:
    """
    Stats pour toutes les colonnes numériques, y compris température,
    mais la corrélation température n'est calculée QUE pour les autres capteurs.
    resume_par ∈ {"mois","semaine"}
    """
    results: Dict[str, pd.DataFrame] = {}

    for nom, df in df_dict.items():
        if df is None or df.empty:
            results[nom] = pd.DataFrame()
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            # on tente une conversion prudente
            try:
                d2 = df.copy()
                d2.index = pd.to_datetime(d2.index)
                d2 = d2.sort_index()
                df = d2
            except Exception:
                # pas d'index temporel: on saute
                results[nom] = pd.DataFrame()
                continue

        temp_cols = [c for c in df.columns if temp_col_contains.lower() in str(c).lower()]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            results[nom] = pd.DataFrame()
            continue

        mask = _period_mask(df.index, resume_par=resume_par)
        df_periode = df.loc[mask] if mask.any() else df.iloc[0:0]

        rows = []
        for col in numeric_cols:
            s_all = df[col].dropna()
            s_per = df_periode[col].dropna()

            if s_all.empty:
                continue

            min_global = s_all.min()
            max_global = s_all.max()
            mean_global = s_all.mean()
            median_global = s_all.median()
            std_global = s_all.std()

            if s_per.empty:
                min_p = max_p = mean_p = median_p = inc_p = std_p = np.nan
            else:
                min_p = s_per.min()
                max_p = s_per.max()
                mean_p = s_per.mean()
                median_p = s_per.median()
                inc_p = s_per.iloc[-1] - s_per.iloc[0]
                std_p = s_per.std()

            # corrélation avec la température (uniquement pour capteurs ≠ temp)
            if temp_cols and (col not in temp_cols):
                tcol = temp_cols[0]  # première trouvée
                t_all = df[tcol].dropna()
                inter = s_all.index.intersection(t_all.index)
                if len(inter) > 2:
                    corr_temp = float(np.corrcoef(s_all.loc[inter], t_all.loc[inter])[0, 1])
                else:
                    corr_temp = np.nan
            else:
                corr_temp = np.nan

            rows.append({
                "Capteur": col,
                "Min Global": round(min_global, arrondi),
                "Max Global": round(max_global, arrondi),
                "Moyenne Globale": round(mean_global, arrondi),
                "Médiane Globale": round(median_global, arrondi),
                "Écart-type Global": round(std_global, arrondi),
                f"Min {resume_par.title()}": round(min_p, arrondi) if pd.notna(min_p) else np.nan,
                f"Max {resume_par.title()}": round(max_p, arrondi) if pd.notna(max_p) else np.nan,
                f"Moyenne {resume_par.title()}": round(mean_p, arrondi) if pd.notna(mean_p) else np.nan,
                f"Médiane {resume_par.title()}": round(median_p, arrondi) if pd.notna(median_p) else np.nan,
                f"Incrément {resume_par.title()}": round(inc_p, arrondi) if pd.notna(inc_p) else np.nan,
                f"Écart-type {resume_par.title()}": round(std_p, arrondi) if pd.notna(std_p) else np.nan,
                "Corrélation Temp": round(corr_temp, arrondi) if pd.notna(corr_temp) else np.nan,
            })

        results[nom] = pd.DataFrame(rows)

    return results

def stats_to_excel_bytes(stats: Dict[str, pd.DataFrame]) -> bytes:
    """Exporte un dict de DataFrames en un seul XLSX (feuilles par DF)."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for name, df in stats.items():
            if df is None or df.empty:
                continue
            sheet = str(name)[:31]
            df.to_excel(w, sheet_name=sheet, index=False)
    bio.seek(0)
    return bio.read()


def plot_stat_histogram(
    stats: Dict[str, pd.DataFrame],
    column: str,
    kind: str = "hist",
    bins: int = 20,
    figsize: tuple = (8, 5)
):
    """
    Crée un histogramme ou boxplot pour une statistique donnée
    (ex: 'Min Global', 'Moyenne Globale') sur l'ensemble des DF.

    kind : "hist" ou "box"
    """
    import pandas as pd

    vals = []
    labels = []
    for df_name, df in stats.items():
        if df is None or df.empty or column not in df.columns:
            continue
        col_vals = pd.to_numeric(df[column], errors="coerce").dropna()
        if col_vals.empty:
            continue
        vals.append(col_vals)
        labels.append(df_name)

    if not vals:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "hist":
        ax.hist(vals, bins=bins, stacked=True, label=labels, alpha=0.7)
        ax.set_title(f"Histogramme de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Fréquence")
        ax.legend()
    elif kind == "box":
        ax.boxplot(vals, labels=labels, vert=True, patch_artist=True)
        ax.set_title(f"Boxplot de {column}")
        ax.set_ylabel(column)

    return fig


def plot_stat_bars_per_df(
    stats: dict[str, pd.DataFrame],
    column: str,
    label_col: str = "Capteur",
    sort: str = "desc",      # "asc", "desc", or "none"
    top_n: int | None = None,
    figsize: tuple = (10, 5),
):
    """
    Crée un bar chart par DF : X = label_col (ex: 'Capteur'), Y = column (ex: 'Max Global').
    Retourne un dict {df_name: fig}
    """
    figs: dict[str, plt.Figure] = {}
    for df_name, df in stats.items():
        if df is None or df.empty or column not in df.columns:
            continue
        if label_col not in df.columns:
            # si pas de colonne 'Capteur', on utilise l'index numérique
            df_plot = df.reset_index().rename(columns={"index": label_col})
        else:
            df_plot = df.copy()

        s = df_plot[[label_col, column]].dropna()
        if s.empty:
            continue

        if sort == "asc":
            s = s.sort_values(by=column, ascending=True)
        elif sort == "desc":
            s = s.sort_values(by=column, ascending=False)

        if top_n is not None and top_n > 0:
            s = s.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(s[label_col].astype(str), s[column].astype(float))
        ax.set_title(f"{df_name} — {column}")
        ax.set_xlabel(label_col)
        ax.set_ylabel(column)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        figs[df_name] = fig

    return figs

def plot_stat_bars_combined(
    stats: dict[str, pd.DataFrame],
    column: str,
    label_col: str = "Capteur",
    add_df_prefix: bool = True,
    sort: str = "none",
    top_n: int | None = None,
    figsize: tuple = (12, 6),
):
    """
    Combine tous les DF dans un seul bar chart :
    X = concaténation des labels (optionnellement préfixés par le nom du DF),
    Y = column.
    """
    import pandas as pd

    rows: list[tuple[str, float]] = []
    for df_name, df in stats.items():
        if df is None or df.empty or column not in df.columns:
            continue
        if label_col not in df.columns:
            df_plot = df.reset_index().rename(columns={"index": label_col})
        else:
            df_plot = df.copy()

        sub = df_plot[[label_col, column]].dropna()
        if sub.empty:
            continue

        for _, r in sub.iterrows():
            label = str(r[label_col])
            if add_df_prefix:
                label = f"{df_name} | {label}"
            try:
                y = float(r[column])
            except Exception:
                continue
            rows.append((label, y))

    if not rows:
        return None

    df_all = pd.DataFrame(rows, columns=[label_col, column])

    if sort == "asc":
        df_all = df_all.sort_values(by=column, ascending=True)
    elif sort == "desc":
        df_all = df_all.sort_values(by=column, ascending=False)

    if top_n is not None and top_n > 0:
        df_all = df_all.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df_all[label_col].astype(str), df_all[column].astype(float))
    ax.set_title(f"Barres combinées — {column}")
    ax.set_xlabel(label_col + (" (préfixé par DF)" if add_df_prefix else ""))
    ax.set_ylabel(column)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig
