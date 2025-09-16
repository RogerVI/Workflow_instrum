# core/C02_plotly.py
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import plotly.graph_objs as go


def _legend_dict(position: str) -> dict:
    """Positionne proprement la légende selon 'bottom' | 'top' | 'right' | 'none'."""
    pos = (position or "").lower()
    if pos == "bottom":
        return dict(orientation="h", x=0.5, xanchor="center", yanchor="top", y=-0.25)
    if pos == "top":
        return dict(orientation="h", x=0.5, xanchor="center", yanchor="bottom", y=1.1)
    if pos == "right":
        return dict(orientation="v", x=1.02, xanchor="left", y=1.0)
    return dict()  # invisible si non utilisé par l'appelant


def tracer_colonnes_plotly_core(
    df_dict: Dict[str, pd.DataFrame],
    *,
    # quoi tracer
    tracer_brut: bool = True,
    tracer_moyenne: bool = True,
    tracer_vitesse: bool = True,
    tracer_temp: bool = True,
    # sélection de colonnes
    tracer_toutes_colonnes: bool = True,
    colonnes_selectionnees: Optional[Dict[str, list[int]]] = None,  # {df_name: [indices]}
    # filtres temporels
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    # style/affichage
    legend_position: str = "bottom",  # 'bottom' | 'top' | 'right' | 'none'
    fig_width: int = 1100,
    fig_height: int = 550,
    template: str = "plotly_white",
    # labels unités
    unite_label: Optional[Dict[str, str]] = None,  # {df_name: "mm" ...}
) -> Dict[str, go.Figure]:
    """
    Construit pour chaque DataFrame un graphique Plotly interactif :
      - axe principal y (déplacements / mesures) + axe secondaire y2 (températures)
      - boutons de zoom temporel (6h, 1j, 7j, 1mois, All)
      - couleurs par défaut Plotly
      - width/height paramétrables

    Retourne: { nom_df: plotly.graph_objs.Figure }
    """
    figs: Dict[str, go.Figure] = {}

    def _is_temp(col: str) -> bool:
        c = col.lower()
        return ("temp" in c) or ("température" in c) or ("temperature" in c) or ("°c" in c)

    for nom_df, df in (df_dict or {}).items():
        if df is None or df.empty:
            continue

        d = df.copy()
        # index → datetime si possible
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

        # split
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

        # fallback : si rien ne matche → toutes colonnes numériques en principal
        if not y_main_cols and not y_temp_cols:
            num_cols = d.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                continue
            y_main_cols = num_cols

        traces = []

        # axe principal
        unite = (unite_label or {}).get(nom_df, "")
        for col in y_main_cols:
            label = f"{col} [{unite}]" if unite else col
            traces.append(
                go.Scatter(
                    x=d.index,
                    y=d[col],
                    mode="lines",
                    name=label,
                    yaxis="y1",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
                )
            )

        # axe secondaire (températures)
        for col in y_temp_cols:
            traces.append(
                go.Scatter(
                    x=d.index,
                    y=d[col],
                    mode="lines",
                    name=col,
                    line=dict(dash="dot", color="black"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} °C<extra>%{fullData.name}</extra>",
                )
            )

        # légende
        legend_cfg = _legend_dict(legend_position)
        if legend_position == "none":
            legend_cfg["visible"] = False

        # axes
        y_title = f"Déplacements cumulés ({unite})" if unite else "Déplacements cumulés"

        layout = go.Layout(
            title=f"Tracé interactif : {nom_df}",
            width=int(fig_width),
            height=int(fig_height),
            template=template,
            xaxis=dict(
                title="Temps",
                type="date",
                rangeselector=dict(
                    buttons=[
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1j", step="day", stepmode="backward"),
                        dict(count=7, label="7j", step="day", stepmode="backward"),
                        dict(count=1, label="1mois", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title=y_title, fixedrange=False),
            yaxis2=dict(title="Température (°C)", overlaying="y", side="right", showgrid=False, fixedrange=False),
            hovermode="x unified",
            legend=legend_cfg,
            margin=dict(l=60, r=60, t=60, b=80),
        )

        fig = go.Figure(data=traces, layout=layout)
        figs[nom_df] = fig

    return figs
