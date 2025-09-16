# services/C11_streamlit_correlation.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.C11_correlation import (
    analyse_correlation_core,
    export_correlation_to_excel,
    TempRangeSpec,  # alias: Union[Tuple[float,float,int], Dict[str, Tuple[float,float,int]]]
    PolySpec,       # alias: Union[Tuple[bool,int], Dict[str, Tuple[bool,int]]]
)

from services.Z00_global_toolbar import render_global_toolbar
from core.Z00_prefs import (
    apply_unified_prefs,
    merge_prefs,
    prefs_to_bytes,
    collect_all_prefs,
)




# ---------- helpers ----------
def _parse_indices(expr: str, n: int) -> list[int]:
    """
    Parse "1,3-5,10" -> [0,2,3,4,9] (indices 0-based, bornés dans [0, n-1])
    """
    out = []
    expr = (expr or "").strip()
    if not expr:
        return out
    for p in expr.replace(" ", "").split(","):
        if not p:
            continue
        if "-" in p:
            try:
                a, b = [int(x) for x in p.split("-", 1)]
            except Exception:
                continue
            a, b = max(1, a), max(1, b)
            for k in range(min(a, b), max(a, b) + 1):
                if 1 <= k <= n:
                    out.append(k - 1)
        else:
            try:
                k = int(p)
                if 1 <= k <= n:
                    out.append(k - 1)
            except Exception:
                pass
    return sorted(set(out))


def _scatter_lin(name: str, capteur: str, info: dict):
    x = info["x"].flatten()
    y = info["y"]
    tr = info["t_range"].flatten()
    lin = info["lin"]

    order = np.argsort(x)
    x_s = x[order]
    y_lin_s = lin["y_pred"][order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.35, label="Points")
    ax.plot(x_s, y_lin_s, color="red", label="Régression linéaire")
    ax.plot(tr, lin["y_pred_temp"], "ro", ms=4)
    ax.fill_between(
        tr,
        lin["y_pred_temp"] - 1.96 * lin["err_std"],
        lin["y_pred_temp"] + 1.96 * lin["err_std"],
        color="orange",
        alpha=0.2,
        label="Intervalle 95% (lin)",
    )
    ax.plot(tr, lin["y_pred_temp"] + lin["err_max"], "k--", alpha=0.6, label="Erreur max ±")
    ax.plot(tr, lin["y_pred_temp"] - lin["err_max"], "k--", alpha=0.6)
    ax.set_xlabel(info["temp_col"])
    ax.set_ylabel(capteur)
    ax.set_title(f"{name} | Lin | {capteur} vs {info['temp_col']}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def _scatter_poly(name: str, capteur: str, info: dict):
    if "poly" not in info:
        return None
    x = info["x"].flatten()
    y = info["y"]
    tr = info["t_range"].flatten()
    poly = info["poly"]

    order = np.argsort(x)
    x_s = x[order]
    y_poly_s = poly["y_pred"][order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.35, label="Points")
    ax.plot(x_s, y_poly_s, color="purple", label=f"Poly deg {poly['degree']}")
    ax.plot(tr, poly["y_pred_temp"], "mo", ms=4)
    ax.fill_between(
        tr,
        poly["y_pred_temp"] - 1.96 * poly["err_std"],
        poly["y_pred_temp"] + 1.96 * poly["err_std"],
        color="violet",
        alpha=0.15,
        label="Intervalle 95% (poly)",
    )
    ax.plot(tr, poly["y_pred_temp"] + poly["err_max"], "k--", alpha=0.4, label="Erreur max ±")
    ax.plot(tr, poly["y_pred_temp"] - poly["err_max"], "k--", alpha=0.4)
    ax.set_xlabel(info["temp_col"])
    ax.set_ylabel(capteur)
    ax.set_title(f"{name} | Poly deg {poly['degree']} | {capteur} vs {info['temp_col']}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- page ----------
def render_correlation_page():
    st.title("Corrélation & Régressions")

    # Bouton pour sauvegarder les réglages (globaux) de l’utilisateur
    prefs = collect_all_prefs()
    st.download_button(
        "💾 Sauvegarder mes réglages",
        data=prefs_to_bytes(prefs),
        file_name="mes_reglages.json",
        mime="application/json",
        key="save_prefs_this_page",
    )


    # Source: prétraités si présents, sinon regroupements
    df_ns = st.session_state.get("df", {})
    data = df_ns.get("preprocessed") or df_ns.get("groups")
    if not data:
        st.warning("Aucune donnée. Va d’abord dans **Créer DF** / **Pré-traitement**.")
        return

    # -------- Période d’analyse --------
    st.subheader("Période d’analyse")
    colp1, colp2 = st.columns(2)
    with colp1:
        start_s = st.text_input("Début (YYYY-MM-DD ou YYYY-MM-DD HH:MM)", value="", key="c11_start")
    with colp2:
        end_s = st.text_input("Fin (YYYY-MM-DD ou YYYY-MM-DD HH:MM)", value="", key="c11_end")
    start = pd.to_datetime(start_s) if start_s.strip() else None
    end = pd.to_datetime(end_s) if end_s.strip() else None
    if start and end and start > end:
        st.error("⛔ La date de début est après la date de fin.")
        return

    # -------- Sélection colonnes par DF --------
    st.subheader("Sélection des colonnes")
    selections: Dict[str, Dict[str, list[str]]] = {}
    for name, df in data.items():
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        temp_guess = next((c for c in cols if any(k in c.lower() for k in ["temp", "°c", "température"])), None)

        with st.expander(f"{name}", expanded=False):
            temp_col = st.selectbox(
                f"{name} — colonne température",
                cols,
                index=cols.index(temp_guess) if temp_guess in cols else 0,
                key=f"c11_temp_{name}",
            )
            st.caption(
                "Colonnes déplacement — saisis une plage d’indices, ex: 1,3-5,10 (indices visibles ci-dessous)."
            )
            st.code("\n".join([f"{i+1}: {c}" for i, c in enumerate(cols)]), language="text")
            expr = st.text_input(f"{name} — indices à prendre", value="", key=f"c11_depl_expr_{name}")
            idxs = _parse_indices(expr, n=len(cols))
            depl_cols = [cols[i] for i in idxs if cols[i] != temp_col]
            if depl_cols:
                selections[name] = {"temp": temp_col, "depl": depl_cols}

    if not selections:
        st.info("Sélectionne au moins une colonne température et des colonnes de déplacement (via plages d’indices).")
        return

    # -------- Plage de températures --------
    st.subheader("Plage de températures pour les prédictions")
    same_tr = st.checkbox("Appliquer la même plage à tous les DF", value=True, key="c11_same_tr")
    if same_tr:
        c1, c2, c3 = st.columns(3)
        with c1:
            tmin_g = st.number_input("Température min", value=-5.0, step=0.5, key="c11_tmin_g")
        with c2:
            tmax_g = st.number_input("Température max", value=35.0, step=0.5, key="c11_tmax_g")
        with c3:
            steps_g = st.number_input("Nb de pas", min_value=2, value=8, step=1, key="c11_steps_g")
        t_ranges: TempRangeSpec = (float(tmin_g), float(tmax_g), int(steps_g))
    else:
        tr_map: Dict[str, Tuple[float, float, int]] = {}
        for name in selections.keys():
            c1, c2, c3 = st.columns(3)
            with c1:
                tmin = st.number_input(f"{name} — Tmin", value=-5.0, step=0.5, key=f"c11_tmin_{name}")
            with c2:
                tmax = st.number_input(f"{name} — Tmax", value=35.0, step=0.5, key=f"c11_tmax_{name}")
            with c3:
                steps = st.number_input(f"{name} — pas", min_value=2, value=8, step=1, key=f"c11_steps_{name}")
            tr_map[name] = (float(tmin), float(tmax), int(steps))
        t_ranges = tr_map

    # -------- Régression polynomiale --------
    st.subheader("Paramètres de régression")
    cR1, cR2 = st.columns(2)
    with cR1:
        min_abs_corr = st.number_input(
            "Seuil |corr| pour lancer la régression",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="c11_thr",
        )
    with cR2:
        same_poly = st.checkbox("Appliquer le même réglage poly à tous les DF", value=True, key="c11_same_poly")

    if same_poly:
        cP1, cP2 = st.columns(2)
        with cP1:
            do_poly_g = st.checkbox("Activer régression polynomiale", value=False, key="c11_poly_g")
        with cP2:
            poly_degree_g = st.number_input(
                "Degré du polynôme", min_value=2, max_value=6, value=2, step=1, key="c11_deg_g"
            )
        poly_spec: PolySpec = (bool(do_poly_g), int(poly_degree_g))
    else:
        poly_map: Dict[str, Tuple[bool, int]] = {}
        for name in selections.keys():
            cP1, cP2 = st.columns(2)
            with cP1:
                dp = st.checkbox(f"{name} — Poly ?", value=False, key=f"c11_poly_{name}")
            with cP2:
                dg = st.number_input(
                    f"{name} — Degré", min_value=2, max_value=6, value=2, step=1, key=f"c11_deg_{name}"
                )
            poly_map[name] = (bool(dp), int(dg))
        poly_spec = poly_map

    # ... après avoir construit les widgets (seuil corr, same_tr, t_ranges inputs, same_poly, etc.)
    if st.button("💾 Sauvegarder mes réglages (cette page)"):
        # Exemple minimal : tu peux étendre selon tes widgets exacts
        st.session_state["corr_cfg"] = {
            "start": st.session_state.get("corr_start", ""),
            "end": st.session_state.get("corr_end", ""),
            "min_abs_corr": st.session_state.get("corr_thr", 0.5),
            "same_tr": st.session_state.get("corr_same_tr", True),
            # pour same_tr True on stocke tmin/tmax/steps globaux
            "tmin_g": st.session_state.get("corr_tmin_g", None),
            "tmax_g": st.session_state.get("corr_tmax_g", None),
            "steps_g": st.session_state.get("corr_steps_g", None),
            "same_poly": st.session_state.get("corr_same_poly", True),
            "do_poly_g": st.session_state.get("corr_poly_g", False),
            "poly_degree_g": st.session_state.get("corr_deg_g", 2),
            # tu peux aussi stocker les sélections par DF (temp col + expr indices),
            # par ex. {"DF1": {"temp":"Température", "expr":"1,3-5"}, ...}
            # si tu as les valeurs sous la main :
            # "selections_raw": selections_raw_dict
        }
        st.success("Réglages Corrélation mémorisés ✅ (inclus dans l’export global depuis Import)")


    # -------- Lancement --------
    if st.button("Analyser", key="c11_run"):
        results = analyse_correlation_core(
            df_dict=data,
            selections=selections,
            t_ranges=t_ranges,
            min_abs_corr=float(min_abs_corr),
            start=start,
            end=end,
            poly_spec=poly_spec,
        )

        # Tables + Graphiques
        any_row = False
        for name, block in results.items():
            rows = block.get("rows", [])
            if rows:
                any_row = True
                st.markdown(f"### {name} — Résumé")
                df_rows = pd.DataFrame(rows)
                st.dataframe(df_rows)

            series = block.get("series", {})
            if series:
                for capteur, info in series.items():
                    # Linéaire
                    fig_lin = _scatter_lin(name, capteur, info)
                    st.pyplot(fig_lin)
                    # Poly sur image séparée (si activé)
                    fig_poly = _scatter_poly(name, capteur, info)
                    if fig_poly is not None:
                        st.pyplot(fig_poly)

        if not any_row:
            st.warning("Aucun couple n’a dépassé le seuil de corrélation. Ajuste le seuil, la période ou les sélections.")

        # Export Excel (inclut lin + poly avec degré dans les colonnes)
        xlsx = export_correlation_to_excel(results, excel_basename="resultats_regression.xlsx")
        st.download_button(
            "📥 Exporter résultats (Excel)",
            data=xlsx,
            file_name="resultats_regression.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="c11_dl_xlsx",
        )
