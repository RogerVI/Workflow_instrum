# services/A00_streamlit_import_data.py
from __future__ import annotations

import os
import json
from io import BytesIO

import pandas as pd
import streamlit as st

from core.A00_import_data import convertir_colonne_timestamp  # si tu l'utilises
from core.Z00_prefs import (
    merge_prefs,       # applique un dict de prefs dans session_state
    prefs_to_bytes,    # exporte TOUTES les prefs connues (en JSON bytes)
)

# ---------- Helpers ----------
def dict_to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for name, df in dfs.items():
            df.to_excel(w, sheet_name=str(name)[:31], index=(df.index.name is not None))
    bio.seek(0)
    return bio.read()


# ---------- Page ----------
def render_import_page():
    st.title("Importer & Réglages globaux")

    # État anti-boucle pour l’import de réglages
    st.session_state.setdefault("_prefs_state", {"pending": None, "applied_once": False})

    # On prépare le namespace des dataFrames en mémoire
    st.session_state.setdefault("df", {})
    df_ns = st.session_state["df"]

    # ============== 1) Import des données Excel ==============
    st.subheader("1) Importer des données")

    fmt = st.selectbox(
        "Format date/heure du fichier",
        options=[
            ("%Y-%m-%d %H:%M", "2025-09-10 14:35"),
            ("%Y-%m-%d %H:%M:%S", "2025-09-10 14:35:12"),
            ("%d/%m/%Y %H:%M", "10/09/2025 14:35"),
            ("%d/%m/%Y %H:%M:%S", "10/09/2025 14:35:12"),
        ],
        format_func=lambda x: f"{x[1]}  →  {x[0]}",
        key="import_fmt",
    )[0]

    files = st.file_uploader(
        "Dépose un ou plusieurs .xlsx",
        type=["xlsx"],
        accept_multiple_files=True,
        key="import_files",
    )

    if files:
        merged: dict[str, pd.DataFrame] = {}
        one_file = len(files) == 1

        for uf in files:
            base = os.path.splitext(uf.name)[0]
            xls = pd.ExcelFile(uf, engine="openpyxl")

            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")

                # Normalisation éventuelle des colonnes de dates
                if "Date (UTC)" in df.columns:
                    df = df.drop(columns=["Date (UTC)"])
                if "Date (Europe/Paris)" in df.columns:
                    df = df.rename(columns={"Date (Europe/Paris)": "Timestamp"})

                key = (sheet if one_file else f"{base}_{sheet}")[:31]

                # Conversion de la colonne Timestamp si besoin
                try:
                    df = convertir_colonne_timestamp(df, "Timestamp", fmt)
                except Exception:
                    # Si ta fonction n'est pas nécessaire partout, on ignore l'erreur
                    pass

                merged[key] = df

        df_ns["sources"] = merged
        st.success(f"{len(merged)} feuille(s) chargée(s) en mémoire ✅")

    # État courant (vitre de contrôle)
    nb_sources = len(df_ns.get("sources", {}))
    st.caption(f"📦 Données en mémoire : **{nb_sources}** feuille(s).")
    if nb_sources:
        ex_name, ex_df = next(iter(df_ns["sources"].items()))
        st.write(f"Extrait de **{ex_name}**")
        st.dataframe(ex_df.head(50))
        # Export rapide des sources chargées
        st.download_button(
            "📥 Exporter les sources importées (Excel)",
            data=dict_to_excel_bytes(df_ns["sources"]),
            file_name="sources_importees.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_sources_xlsx",
        )

    st.markdown("---")

    # ============== 2) Réglages globaux ==============
    st.subheader("2) Réglages globaux (toutes les pages)")

    colA, colB = st.columns(2)

    # ---- IMPORT de réglages globaux ----
    with colA:
        up = st.file_uploader(
            "📤 Importer réglages globaux (JSON)",
            type=["json"],
            key="prefs_global_upload",
        )

        # On lit le fichier une seule fois et on stocke en "pending"
        if (
            up is not None
            and st.session_state["_prefs_state"]["pending"] is None
            and not st.session_state["_prefs_state"]["applied_once"]
        ):
            try:
                payload = json.load(up)
                st.session_state["_prefs_state"]["pending"] = payload
                st.success("Réglages chargés en mémoire. Clique **Appliquer les réglages importés**.")
            except Exception as e:
                st.error(f"JSON invalide : {e}")

        # Bouton d’application (évite la boucle de rerun)
        if st.session_state["_prefs_state"]["pending"] is not None:
            if st.button("✅ Appliquer les réglages importés", key="prefs_apply_once"):
                merge_prefs(st.session_state["_prefs_state"]["pending"])
                st.session_state["_prefs_state"]["pending"] = None
                st.session_state["_prefs_state"]["applied_once"] = True
                st.experimental_rerun()

        # Permettre un nouvel import (si on a déjà appliqué une fois)
        if st.session_state["_prefs_state"]["applied_once"]:
            if st.button("♻️ Autoriser un nouvel import", key="prefs_allow_new_import"):
                st.session_state["_prefs_state"]["applied_once"] = False
                st.success("Tu peux importer un nouveau fichier de réglages.")

    # ---- EXPORT de réglages globaux ----
    with colB:
        st.download_button(
            "💾 Exporter mes réglages globaux (JSON)",
            data=prefs_to_bytes(),
            file_name="reglages_globaux.json",
            mime="application/json",
            key="prefs_global_export",
        )

    st.info(
        "Conseil : charge tes Excel ici, puis importe ton **JSON global** de réglages. "
        "Ensuite, va dans les autres onglets : les champs seront pré-remplis. "
        "Tu peux aussi exporter un nouveau JSON global à tout moment."
    )
