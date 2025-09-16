# core/C01_graph.py
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn (optionnel) pour palettes & thèmes
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# Couleurs par défaut des seuils (HEX uniquement)
DEFAULT_THRESHOLD_COLORS: Dict[str, str] = {
    "vigilance": "#FFA500",        # orange
    "vigilance accrue": "#FF8C00", # darkorange
    "alerte": "#FF0000",           # red
}

# =========================
#  Utils seuils (purs)
# =========================
def normalize_thresholds(
    seuils: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Normalise la structure des seuils:
      { df_name: { type: {"min": float, "max": float} | {"val": float}, ... }, ... }
    - {"val": x} -> {"min": x, "max": x}
    - lève ValueError si incohérent
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for df_name, per_type in (seuils or {}).items():
        if not isinstance(per_type, dict):
            raise ValueError(f"Seuils pour '{df_name}' doit être un dict.")
        out[df_name] = {}
        for typ, val in per_type.items():
            if not isinstance(val, dict):
                raise ValueError(f"Seuil '{typ}' pour '{df_name}' doit être un dict.")
            if "min" in val or "max" in val:
                if "min" not in val or "max" not in val:
                    raise ValueError(f"Seuil '{typ}' pour '{df_name}' doit avoir 'min' ET 'max'.")
                vmin, vmax = float(val["min"]), float(val["max"])
            elif "val" in val:
                vmin = vmax = float(val["val"])
            else:
                raise ValueError(f"Seuil '{typ}' pour '{df_name}' doit contenir 'val' ou ('min' et 'max').")
            out[df_name][typ] = {"min": vmin, "max": vmax}
    return out


def build_uniform_thresholds(
    df_names: list[str],
    types: list[str],
    value_or_minmax: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Construit un mapping identique pour tous les DF.
    value_or_minmax: { type: {"val": x} OU {"min": x, "max": y} }
    """
    base = normalize_thresholds({"__all__": value_or_minmax})["__all__"]
    return {name: {t: base[t] for t in types if t in base} for name in df_names}


def validate_colors(colors: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Retourne colors si fourni, sinon DEFAULT_THRESHOLD_COLORS (copie).
    """
    if not colors:
        return DEFAULT_THRESHOLD_COLORS.copy()
    return {k: str(v) for k, v in colors.items()}


def merge_thresholds(
    a: Dict[str, Dict[str, Dict[str, float]]],
    b: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Merge simple (b écrase a).
    """
    out = {k: v.copy() for k, v in a.items()}
    for df_name, per_type in (b or {}).items():
        out.setdefault(df_name, {})
        out[df_name].update(per_type or {})
    return out


def expand_single_or_minmax(
    single: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Dict[str, float]:
    """
    Utilitaire: retourne {"val": single} ou {"min": vmin, "max": vmax}.
    (normalize_thresholds s’occupera de valider)
    """
    if single is not None:
        return {"val": float(single)}
    return {"min": float(vmin), "max": float(vmax)}


# =========================
#  Unités (pur)
# =========================
def demander_unites_df_core(
    df_dict: Dict[str, pd.DataFrame],
    unites_par_df: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Pure: si unites_par_df est fourni, on le normalise et on complète avec ''.
    """
    out = {name: "" for name in df_dict.keys()}
    if unites_par_df:
        for k, v in unites_par_df.items():
            if k in out:
                out[k] = str(v or "")
    return out


# =========================
#  Tracé Matplotlib (core)
# =========================
def tracer_colonnes_separees_core(
    df_dict: Dict[str, pd.DataFrame],
    # --- data & sélection ---
    palette: str = "tab10",
    tracer_brut: bool = True,
    tracer_moyenne: bool = True,
    tracer_vitesse: bool = True,
    tracer_temp: bool = True,
    activer_seuils: bool = True,
    seuils_par_df: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    couleurs_seuils: Optional[Dict[str, str]] = None,
    tracer_toutes_colonnes: bool = True,
    colonnes_selectionnees: Optional[Dict[str, list[int]]] = None,  # {df: [indices]}
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    unites_par_df: Optional[Dict[str, str]] = None,
    # --- style & rendu ---
    fig_width: float = 24.0,
    fig_height: float = 10.0,
    use_seaborn_theme: bool = False,     # active sns.set_theme(...)
    palette_source: str = "auto",        # "auto" -> sns si dispo, sinon mpl
    line_width: float = 2.2,             # épaisseur des courbes principales
    temp_color: str = "black",
    temp_alpha: float = 0.4,
    temp_linewidth: float = 1.2,
    grid_alpha: float = 0.6,
    grid_style: str = "--",
    legend_loc: str = "upper center",    # ex: "upper center", "upper right", ...
    legend_ncol: int = 3,
    title_size: int = 16,
    axis_size: int = 13,
    legend_size: int = 13,
) -> Dict[str, "plt.Figure"]:
    """
    Version robuste :
    - Détection température souple ('temp', 'temperature', 'température', '°C')
    - Fallback: si rien ne matche, trace toutes les colonnes numériques
    - Intègre les seuils dans l'échelle Y
    - Légende unique combinée (temp + principales + seuils)
    - Tous les paramètres de rendu sont pilotables depuis l’UI
    """
    figs: Dict[str, "plt.Figure"] = {}
    seuils_par_df = seuils_par_df or {}
    couleurs_seuils = validate_colors(couleurs_seuils)

    # Thème seaborn global (optionnel)
    if use_seaborn_theme and _HAS_SNS:
        try:
            sns.set_theme(context="notebook", style="whitegrid")
        except Exception:
            pass

    def _is_temp(col: str) -> bool:
        c = col.lower()
        return ("temp" in c) or ("température" in c) or ("temperature" in c) or ("°c" in c)

    for nom_df, df in (df_dict or {}).items():
        if df is None or df.empty:
            continue

        d = df.copy()

        # index datetime + tri
        try:
            d.index = pd.to_datetime(d.index)
        except Exception:
            pass
        d = d.sort_index()

        # filtres temporels
        if start_date is not None:
            d = d[d.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            d = d[d.index <= pd.to_datetime(end_date)]
        if d.empty:
            continue

        # colonnes à utiliser
        cols_sorted = sorted(d.columns)
        if tracer_toutes_colonnes or not colonnes_selectionnees:
            cols_to_use = cols_sorted
        else:
            idxs = colonnes_selectionnees.get(nom_df, []) or []
            cols_to_use = [cols_sorted[i] for i in idxs if 0 <= i < len(cols_sorted)]

        # split types
        y_main_cols, y_temp_cols = [], []
        for col in cols_to_use:
            cu = col.upper()
            if _is_temp(col) and tracer_temp:
                y_temp_cols.append(col)
            elif "_VITESSE_" in cu and tracer_vitesse:
                y_main_cols.append(col)
            elif "_MOYENNE GLISSANTE_" in cu and tracer_moyenne:
                y_main_cols.append(col)
            elif col.endswith("_brut") and tracer_brut:
                y_main_cols.append(col)

        # Fallback: si tout vide → toutes colonnes numériques
        if not y_main_cols and not y_temp_cols:
            num_cols = d.select_dtypes(include="number").columns.tolist()
            if num_cols:
                y_main_cols = num_cols
            else:
                continue  # rien à tracer

        # figure
        fig, (ax_temp, ax_main) = plt.subplots(
            2, 1,
            figsize=(fig_width, fig_height),
            sharex=True,
            gridspec_kw={'height_ratios': [1, 2]}
        )

        # --- TEMP ---
        if y_temp_cols:
            ax_temp.set_ylabel("Température (°C)", fontsize=axis_size)
            for col in y_temp_cols:
                ax_temp.plot(
                    d.index, d[col],
                    linestyle=':',
                    color=temp_color,
                    alpha=temp_alpha,
                    linewidth=temp_linewidth,
                    label=col
                )
            ax_temp.tick_params(axis='y', labelsize=axis_size)
            ax_temp.grid(True, linestyle=':', alpha=0.25)
        else:
            ax_temp.axis('off')

        # --- Données principales ---
        # Génère une palette de couleurs robuste
        colors = []
        if len(y_main_cols) > 0:
            want_sns = (palette_source == "seaborn") or (palette_source == "auto" and _HAS_SNS)
            if want_sns and _HAS_SNS:
                try:
                    colors = sns.color_palette(palette, n_colors=len(y_main_cols))
                except Exception:
                    colors = []
            if not colors:
                try:
                    cmap = plt.get_cmap(palette)
                    denom = max(1, len(y_main_cols) - 1)
                    colors = [cmap(i / denom) for i in range(len(y_main_cols))]
                except Exception:
                    cyl = plt.rcParams.get("axes.prop_cycle", None)
                    base = (cyl.by_key()["color"] if cyl else ["#1f77b4"])
                    while len(base) < len(y_main_cols):
                        base = base + base
                    colors = base[:len(y_main_cols)]

        unite = (unites_par_df or {}).get(nom_df, "")
        for i, col in enumerate(y_main_cols):
            label = f"{col} [{unite}]" if unite else col
            ax_main.plot(d.index, d[col], label=label, color=colors[i], linewidth=line_width)

        ax_main.set_ylabel(
            f"Déplacements cumulés ({unite})" if unite else "Déplacements cumulés",
            fontsize=axis_size
        )
        ax_main.grid(True, linestyle=grid_style, alpha=grid_alpha)

        # --- Seuils ---
        if activer_seuils and seuils_par_df and couleurs_seuils:
            labels_seen = set()
            per_df_thr = seuils_par_df.get(nom_df, {}) or {}
            for typ, color in (couleurs_seuils or {}).items():
                sd = per_df_thr.get(typ)
                if not sd:
                    continue
                for extrem in ("min", "max"):
                    val = sd.get(extrem)
                    if val is None:
                        continue
                    label = f"{typ.title()} ({extrem})"
                    show_label = label if label not in labels_seen else "_nolegend_"
                    labels_seen.add(label)
                    ax_main.axhline(val, color=color, linestyle='--', linewidth=1.5, label=show_label)

        # --- Autoscale incluant seuils ---
        y_min = y_max = None
        if y_main_cols:
            s = d[y_main_cols].stack()
            if not s.empty:
                y_min, y_max = float(s.min()), float(s.max())

        thr_vals = []
        if activer_seuils and seuils_par_df and couleurs_seuils:
            per_df_thr = seuils_par_df.get(nom_df, {}) or {}
            for typ in (couleurs_seuils or {}):
                sd = per_df_thr.get(typ)
                if sd:
                    if sd.get('min') is not None:
                        thr_vals.append(float(sd['min']))
                    if sd.get('max') is not None:
                        thr_vals.append(float(sd['max']))

        if y_min is not None and y_max is not None:
            if thr_vals:
                y_min = min([y_min] + thr_vals)
                y_max = max([y_max] + thr_vals)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            ax_main.set_ylim(y_min - margin, y_max + margin)

        # --- Légende unique combinée ---
        h1, l1 = ax_main.get_legend_handles_labels()
        h2, l2 = ax_temp.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        if handles:
            if legend_loc == "upper center":
                ax_main.legend(
                    handles, labels,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=legend_ncol,
                    fontsize=legend_size,
                    frameon=True
                )
            else:
                ax_main.legend(
                    handles, labels,
                    loc=legend_loc,
                    ncol=legend_ncol,
                    fontsize=legend_size,
                    frameon=True
                )

        ax_main.set_xlabel("Temps", fontsize=axis_size)
        fig.suptitle(f"Tracé : {nom_df}", fontsize=title_size)
        fig.tight_layout()

        figs[nom_df] = fig

    return figs
