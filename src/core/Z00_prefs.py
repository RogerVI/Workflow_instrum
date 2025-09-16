# core/Z00_prefs.py
from __future__ import annotations
import json
import streamlit as st

# --- Liste blanche de clés qu'on sauvegarde (évite d'embarquer les DataFrames) ---
_WHITELIST_PREFIXES = ("*_cfg",)  # tout ce qui finit par _cfg
_WHITELIST_EXACT = {
    "viz",                # ex: {"thresholds": {...}}
    "viz_units",          # unités par DF
    "thresholds_colors",  # couleurs des seuils
    # ajoute ici d’autres clés de config si besoin
}

def _is_pref_key(k: str) -> bool:
    if k in _WHITELIST_EXACT:
        return True
    # suffixes (_cfg)
    for suf in (_p.replace("*", "") for _p in _WHITELIST_PREFIXES):
        if k.endswith(suf):
            return True
    return False


def collect_all_prefs() -> dict:
    """
    Récupère uniquement les préférences utiles dans st.session_state.
    ⚠️ N’inclut pas les DataFrames (df/sources/groups/preprocessed).
    """
    prefs = {}
    for k, v in st.session_state.items():
        if _is_pref_key(k):
            prefs[k] = v
    return prefs


def prefs_to_bytes(prefs: dict | None = None) -> bytes:
    """
    Retourne un JSON (bytes) des préférences.
    - Si `prefs` est fourni (dict), on sérialise ce dict.
    - Sinon, on sérialise collect_all_prefs().
    Permet donc les 2 usages: prefs_to_bytes() et prefs_to_bytes(prefs).
    """
    if prefs is None:
        prefs = collect_all_prefs()
    txt = json.dumps(prefs, ensure_ascii=False, indent=2)
    return txt.encode("utf-8")


def merge_prefs(prefs: dict) -> None:
    """
    Merge *non destructif* des préférences importées dans st.session_state.
    N’écrase pas les données si elles ne sont pas dans le JSON.
    """
    if not isinstance(prefs, dict):
        return
    for k, v in prefs.items():
        st.session_state[k] = v


# Aliases de compatibilité (si certaines pages importent ces noms)
def get_unified_prefs() -> dict:
    """Alias -> collect_all_prefs()"""
    return collect_all_prefs()

def export_unified_prefs_bytes() -> bytes:
    """Alias -> prefs_to_bytes()"""
    return prefs_to_bytes()

def apply_unified_prefs(prefs: dict) -> None:
    """Alias -> merge_prefs(prefs)"""
    merge_prefs(prefs)
