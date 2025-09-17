# services/A11_streamlit_preprocessing_data.py
from __future__ import annotations
import re
import streamlit as st
import pandas as pd

from core.B01_preprocessing_data import appliquer_transformations_temporelles, exclure_dates
from services.B02_streamlit_traitement_nan import render_nan_block

from core.Z00_prefs import (
    apply_unified_prefs,
    merge_prefs,
    prefs_to_bytes,
    collect_all_prefs,
)

_PRESETS = [
    ("15 minutes", "15T"),
    ("1 heure", "1H"),
    ("6 heures", "6H"),
    ("12 heures", "12H"),
    ("24 heures", "24H"),
    ("48 heures", "48H"),
    ("72 heures", "72H"),
    ("7 jours", "7D"),
]

def _select_time_window(label: str, key: str, default_code: str | None = None) -> str:
    preset_labels = [p[0] for p in _PRESETS] + ["Personnalisé…"]
    # préremplissage si default_code connu
    default_index = 0
    if default_code:
        for i, (lab, code) in enumerate(_PRESETS):
            if code == default_code:
                default_index = i
                break
    choice = st.selectbox(label, options=preset_labels, index=default_index, key=f"{key}_preset")
    if choice == "Personnalisé…":
        return st.text_input("Fenêtre (ex: 90T, 2H, 10D)", value=default_code or "", key=f"{key}_custom")
    else:
        return dict(_PRESETS)[choice]

def _parse_date_ranges(s: str) -> pd.DatetimeIndex:
    s = (s or "").strip()
    if not s:
        return pd.DatetimeIndex([])
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        # intervalle 'start:end' (sans heure à droite -> 23:00:00)
        if ":" in part and re.search(r"\d:\d|\d{2}:\d{2}", part) is None and part.count(":") == 1:
            start, end = [x.strip() for x in part.split(":", 1)]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
                end = end + " 23:00:00"
            rng = pd.date_range(start, end, freq="H")
            out.extend(list(rng))
        elif ":" in part and part.count(":") >= 2:
            out.append(pd.to_datetime(part))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            day = pd.to_datetime(part)
            out.extend(list(pd.date_range(day, day + pd.Timedelta(hours=23), freq="H")))
        else:
            out.append(pd.to_datetime(part))
    return pd.DatetimeIndex(pd.to_datetime(out))

def render_preprocessing_page():
    st.title("Pré-traitement temporel")

    # Précharge cfg
    st.session_state.setdefault("preproc_cfg", {})
    cfg_prev = st.session_state["preproc_cfg"]

    df_ns = st.session_state.get("df", {})
    groups = df_ns.get("groups")
    if not groups:
        st.warning("Aucun regroupement. Va d’abord dans **Créer DF**.")
        return

    st.info("Les transformations s’appliquent sur chaque DataFrame, indexés par `Timestamp` (datetime).")

    st.subheader("Options globales")
    col1, col2 = st.columns(2)
    with col1:
        apply_avg = st.checkbox("Appliquer des moyennes glissantes", value=cfg_prev.get("apply_avg", True))
        avg_win = _select_time_window("Fenêtre de moyenne glissante", key="avg",
                                      default_code=cfg_prev.get("avg_win")) if apply_avg else ""
    with col2:
        apply_spd = st.checkbox("Calculer des vitesses moyennes", value=cfg_prev.get("apply_spd", True))
        spd_win = _select_time_window("Fenêtre de moyenne (vitesse)", key="spd",
                                      default_code=cfg_prev.get("spd_win")) if apply_spd else ""
        smooth   = st.checkbox("Lisser les vitesses", value=cfg_prev.get("smooth", True)) if apply_spd else False
        smooth_win = _select_time_window("Fenêtre de lissage", key="smooth",
                                         default_code=cfg_prev.get("smooth_win")) if (apply_spd and smooth) else ""

    use_per_df = st.checkbox("Configurer différemment par DataFrame",
                             value=cfg_prev.get("use_per_df", False))

    configs_par_df = {}
    if use_per_df:
        st.subheader("Configurations par DataFrame")
        prev_map: dict = cfg_prev.get("configs_par_df", {})
        for name in groups.keys():
            prev = prev_map.get(name, {})
            st.markdown(f"**{name}**")
            c1, c2 = st.columns(2)
            with c1:
                a_avg = st.checkbox(f"[{name}] Moyenne glissante", value=prev.get("appliquer_moyenne", apply_avg), key=f"avg_{name}")
                a_avg_win = _select_time_window(f"[{name}] Fenêtre moyenne", key=f"avg_{name}",
                                                default_code=prev.get("periode_moy") or avg_win) if a_avg else ""
            with c2:
                a_spd = st.checkbox(f"[{name}] Vitesse moyenne", value=prev.get("appliquer_vitesse", apply_spd), key=f"spd_{name}")
                a_spd_win = _select_time_window(f"[{name}] Fenêtre vitesse", key=f"spd_{name}",
                                                default_code=prev.get("delta_duree") or spd_win) if a_spd else ""
                a_smooth = st.checkbox(f"[{name}] Lisser vitesse", value=prev.get("lisser", smooth), key=f"smooth_{name}") if a_spd else False
                a_smooth_win = _select_time_window(f"[{name}] Fenêtre lissage", key=f"smooth_{name}",
                                                   default_code=prev.get("periode_lissage") or smooth_win) if (a_spd and a_smooth) else ""

            configs_par_df[name] = {
                "appliquer_moyenne": bool(a_avg),
                "periode_moy": a_avg_win or None,
                "appliquer_vitesse": bool(a_spd),
                "delta_duree": a_spd_win or None,
                "lisser": bool(a_smooth),
                "periode_lissage": a_smooth_win or None,
            }

    # === Enregistrement des réglages page ===
    if st.button("💾 Sauvegarder mes réglages (cette page)"):
        st.session_state["preproc_cfg"] = {
            "apply_avg": bool(apply_avg),
            "avg_win": avg_win or None,
            "apply_spd": bool(apply_spd),
            "spd_win": spd_win or None,
            "smooth": bool(smooth),
            "smooth_win": smooth_win or None,
            "use_per_df": bool(use_per_df),
            "configs_par_df": configs_par_df if use_per_df else {},
        }
        st.success("Réglages pré-traitement mémorisés ✅ (inclus dans l’export global depuis Import)")

    # === Appliquer les transformations ===
    if st.button("Appliquer les transformations", key="apply_transformations"):
        config_global = {
            "appliquer_moyenne": bool(apply_avg),
            "periode_moy": avg_win or None,
            "appliquer_vitesse": bool(apply_spd),
            "delta_duree": spd_win or None,
            "lisser": bool(smooth),
            "periode_lissage": smooth_win or None,
        }
        transformed = appliquer_transformations_temporelles(
            groups,
            config_global=None if use_per_df else config_global,
            configs_par_df=configs_par_df if use_per_df else None,
        )
        st.session_state.df["preprocessed"] = transformed
        st.success("Transformations appliquées ✅")

        with st.expander("Aperçu (premières lignes)"):
            for name, df in transformed.items():
                st.write(f"**{name}** — {df.shape[0]}x{df.shape[1]}")
                st.dataframe(df.head(100))

    # === Exclusion de dates ===
    st.subheader("Exclure des dates")
    st.session_state.setdefault("excluded_dates_cfg", {})
    ex_prev = st.session_state["excluded_dates_cfg"]

    base_key = "preprocessed" if st.session_state.df.get("preprocessed") else "groups"
    base_dict = st.session_state.df.get(base_key, {})

    st.caption("Format: '2024-06-01, 2024-06-03:2024-06-05, 2024-06-10 14:00:00'")
    dates_str = st.text_input("Dates / intervalles à exclure", value=ex_prev.get("dates_str", ""), key="excl_dates_input")

    apply_all = st.checkbox("Appliquer à tous les DataFrames et à toutes les colonnes",
                            value=ex_prev.get("apply_all", True),
                            key="excl_all")

    per_df_cols = {}
    if not apply_all:
        st.info("Choisis, pour chaque DF, si tu supprimes les lignes (toutes colonnes) ou si tu cibles des colonnes.")
        prev_map = ex_prev.get("per_df_cols", {})
        for name, df in base_dict.items():
            with st.expander(f"{name}"):
                mode_default = prev_map.get(name, "all")
                mode = st.radio(
                    f"Mode pour {name}",
                    options=["Supprimer les lignes (toutes colonnes)", "Colonnes spécifiques → NaN"],
                    index=0 if mode_default == "all" else 1,
                    key=f"excl_mode_{name}",
                )
                if mode.startswith("Supprimer"):
                    per_df_cols[name] = "all"
                else:
                    prev_cols = prev_map.get(name, [])
                    cols = st.multiselect(
                        f"Colonnes à mettre à NaN aux dates exclues ({name})",
                        options=list(df.columns),
                        default=prev_cols if isinstance(prev_cols, list) else [],
                        key=f"excl_cols_{name}",
                    )
                    if cols:
                        per_df_cols[name] = cols

    if st.button("Appliquer l'exclusion", key="apply_exclusion"):
        dates_ix = _parse_date_ranges(dates_str)
        if dates_ix.empty:
            st.warning("Aucune date valide à exclure.")
        else:
            result = exclure_dates(
                base_dict,
                dates_exclure=dates_ix,
                apply_all=apply_all,
                per_df_cols=per_df_cols if not apply_all else None,
            )
            st.session_state.df[base_key] = result
            # mémorise les réglages d’exclusion
            st.session_state["excluded_dates_cfg"] = {
                "dates_str": dates_str,
                "apply_all": bool(apply_all),
                "per_df_cols": per_df_cols if not apply_all else {},
            }
            st.success(f"Exclusion appliquée sur '{base_key}' ✅")
            st.experimental_rerun()

    # === Bloc NaN (contient son propre bouton de sauvegarde) ===
    render_nan_block()
