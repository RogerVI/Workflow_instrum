# services/A12_streamlit_traitement_nan.py
from __future__ import annotations
import json
import streamlit as st
import pandas as pd
from core.A12_traitement_nan import (
    compute_nan_stats,
    apply_nan_strategy_all,
    apply_nan_strategy_per_df,
    INTERP_METHODS,
)

# libellés -> codes internes
_STRATS = {
    "Ne rien faire": "none",
    "Remplacer par zéro": "zero",
    "Remplacer par la moyenne": "mean",
    "Remplacer par la médiane": "median",
    "Interpoler": "interpolate",
    "Supprimer lignes avec NaN": "drop_rows",
    "Supprimer colonnes avec NaN": "drop_cols",
}
_STRATS_INV = {v: k for k, v in _STRATS.items()}


def _render_interp_controls(prefix: str, default_method: str = "linear", default_order: int | None = None):
    c1, c2 = st.columns([2, 1])
    with c1:
        method = st.selectbox(
            f"{prefix} • Méthode d’interpolation",
            options=sorted(INTERP_METHODS),
            index=sorted(INTERP_METHODS).index(default_method) if default_method in INTERP_METHODS else 0,
            key=f"{prefix}_interp_method",
        )
    order = None
    with c2:
        if method in {"polynomial", "spline"}:
            order = st.number_input(
                f"{prefix} • Ordre",
                min_value=1, max_value=7,
                value=int(default_order) if default_order else 2,
                step=1,
                key=f"{prefix}_interp_order",
            )
    return method, order


def _dump_page_prefs(cfg: dict) -> bytes:
    """JSON (bytes) des réglages de CETTE page uniquement."""
    payload = {"nan_cfg": cfg}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _apply_loaded_page_prefs(payload: dict):
    """Applique un JSON de réglages de page → st.session_state['nan_cfg'] + préremplissage UI si possible."""
    if not isinstance(payload, dict) or "nan_cfg" not in payload:
        st.warning("JSON invalide: clé 'nan_cfg' absente.")
        return
    st.session_state["nan_cfg"] = payload["nan_cfg"]


def render_nan_block():
    st.subheader("Traitement des valeurs manquantes (NaN)")

    # --- Données sources ---
    df_ns = st.session_state.get("df", {})
    data = df_ns.get("preprocessed") or df_ns.get("groups")
    if not data:
        st.info("Aucune donnée. Va d’abord dans **Créer DF** / **Pré-traitement**.")
        return

    # --- Import/Export réglages de page ---
    st.markdown("##### ⚙️ Réglages (cette page)")
    col_cfgA, col_cfgB = st.columns(2)
    with col_cfgA:
        up_cfg = st.file_uploader("📤 Importer réglages (JSON)", type=["json"], key="nan_cfg_upload")
        if up_cfg:
            try:
                payload = json.load(up_cfg)
                _apply_loaded_page_prefs(payload)
                st.success("Réglages NaN importés ✅")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"JSON invalide : {e}")

    with col_cfgB:
        # bouton qui propose un download après clic
        if st.button("💾 Sauvegarder mes réglages (cette page)", key="nan_save_cfg_btn"):
            cfg = st.session_state.get("nan_cfg", {
                "scope": "all",         # all | per_df
                "strategy": "none",     # si scope == all
                "interp_method": None,
                "interp_order": None,
                "per_df": {},           # si scope == per_df
            })
            st.download_button(
                "📥 Télécharger le JSON NaN",
                data=_dump_page_prefs(cfg),
                file_name="reglages_nan.json",
                mime="application/json",
                key="nan_save_cfg_dl",
            )

    # --- Statistiques NaN ---
    stats = compute_nan_stats(data)
    st.markdown("**Taux de NaN**")
    st.dataframe(stats)

    st.markdown("---")

    # --- Charger cfg en mémoire s'il n'existe pas ---
    st.session_state.setdefault("nan_cfg", {
        "scope": "all",         # "all" ou "per_df"
        "strategy": "none",     # globale
        "interp_method": "linear",
        "interp_order": None,
        "per_df": {},           # {df_name: {strategy, interp_method, interp_order}}
    })
    cfg = st.session_state["nan_cfg"]

    # --- UI : Portée ---
    scope = st.radio(
        "Portée",
        ["Appliquer à tous", "Configurer par DataFrame"],
        index=0 if cfg.get("scope", "all") == "all" else 1,
        key="nan_scope_radio",
    )
    cfg["scope"] = "all" if scope.startswith("Appliquer") else "per_df"

    # === MODE GLOBAL ===
    if cfg["scope"] == "all":
        default_label = _STRATS_INV.get(cfg.get("strategy", "none"), "Ne rien faire")
        strat_label = st.selectbox(
            "Stratégie",
            list(_STRATS.keys()),
            index=list(_STRATS.keys()).index(default_label),
            key="nan_all_strategy_sel",
        )
        strat_code = _STRATS[strat_label]
        cfg["strategy"] = strat_code

        interp_method = None
        interp_order = None
        if strat_code == "interpolate":
            interp_method, interp_order = _render_interp_controls(
                "Global",
                default_method=cfg.get("interp_method", "linear"),
                default_order=cfg.get("interp_order"),
            )
        cfg["interp_method"] = interp_method
        cfg["interp_order"] = interp_order

        # Appliquer (global)
        if st.button("Appliquer le traitement (tous les DF)", key="nan_apply_all_btn"):
            out = apply_nan_strategy_all(
                data,
                strategy=strat_code,
                interp_method=interp_method,
                interp_order=interp_order,
            )
            st.session_state.setdefault("df", {})
            st.session_state["df"]["preprocessed"] = out
            st.success("Traitement appliqué à tous les DataFrames ✅")

    # === MODE PAR DF ===
    else:
        per_df_cfg_ui = {}
        for name, df in data.items():
            with st.expander(f"{name}", expanded=False):
                # valeurs par défaut depuis cfg
                prev = (cfg.get("per_df") or {}).get(name, {})
                prev_strat = prev.get("strategy", "none")
                prev_label = _STRATS_INV.get(prev_strat, "Ne rien faire")

                strat_label = st.selectbox(
                    f"{name} • Stratégie",
                    list(_STRATS.keys()),
                    index=list(_STRATS.keys()).index(prev_label),
                    key=f"nan_strat_sel_{name}",
                )
                strat_code = _STRATS[strat_label]

                method = None
                order = None
                if strat_code == "interpolate":
                    method, order = _render_interp_controls(
                        f"{name}",
                        default_method=prev.get("interp_method", "linear"),
                        default_order=prev.get("interp_order"),
                    )

                per_df_cfg_ui[name] = dict(
                    strategy=strat_code,
                    interp_method=method,
                    interp_order=order,
                )

        # Persiste dans le cfg en session
        cfg["per_df"] = per_df_cfg_ui

        if st.button("Appliquer le traitement (par DF)", key="nan_apply_perdf_btn"):
            out = apply_nan_strategy_per_df(data, per_df_cfg_ui)
            st.session_state.setdefault("df", {})
            st.session_state["df"]["preprocessed"] = out
            st.success("Traitement appliqué selon la configuration par DF ✅")

    # Sauvegarde en session du cfg mis à jour
    st.session_state["nan_cfg"] = cfg
