# services/Z00_global_toolbar.py
from __future__ import annotations
import json, re
import streamlit as st
import pandas as pd

PREF_KEYS = {
    "g_use_sns","g_pal_src","g_palette","g_fig_w","g_fig_h","g_lw",
    "g_grid_style","g_grid_alpha","g_temp_col","g_temp_alpha","g_temp_lw",
    "g_leg_loc","g_leg_ncol","g_title_size","g_axis_size","g_legend_size",
    "viz_units","preproc_cfg","nan_cfg","excluded_dates_cfg",
    "thresholds_colors","viz","corr_cfg","plotly_style",
}

def _is_primitive(x): return isinstance(x, (type(None), bool, int, float, str))

def _prune(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)): return None
    if isinstance(obj, (bytes, bytearray, memoryview)): return None
    if _is_primitive(obj): return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = _prune(v)
            if pv is not None:
                out[str(k)] = pv
        return out
    if isinstance(obj, (list, tuple, set)):
        out = []
        for v in obj:
            pv = _prune(v)
            if pv is not None:
                out.append(pv)
        return out
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return None

def _collect_current_prefs() -> dict:
    out = {}
    for k in PREF_KEYS:
        if k in st.session_state:
            out[k] = st.session_state[k]
    if "viz" in out and isinstance(out["viz"], dict):
        out["viz"] = {"thresholds": out["viz"].get("thresholds")}
    return _prune(out) or {}

def _merge_into_session(prefs: dict):
    if not isinstance(prefs, dict): return
    for k, v in prefs.items():
        st.session_state[k] = v

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def render_global_toolbar(title: str = "Réglages (import/export)"):
    safe = _slug(title) or "global"
    with st.expander(title, expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            up = st.file_uploader(
                "📤 Importer réglages (JSON)",
                type=["json"],
                key=f"prefs_upload_{safe}",   # ✅ unique key per page
            )
            if up is not None:
                try:
                    content = up.read().decode("utf-8", errors="ignore")
                    prefs = json.loads(content)
                    _merge_into_session(prefs)
                    st.success("Réglages importés ✅ (rechargement)")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Fichier JSON invalide : {e}")

        with c2:
            prefs = _collect_current_prefs()
            st.download_button(
                "💾 Exporter mes réglages (JSON)",
                data=json.dumps(prefs, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="mes_reglages.json",
                mime="application/json",
                key=f"prefs_download_{safe}",  # ✅ unique key per page
            )

        st.caption("ℹ️ Seuls les **réglages UI** sont exportés (pas les données).")
