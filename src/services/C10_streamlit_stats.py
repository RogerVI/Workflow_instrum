# services/C10_streamlit_stats.py
from __future__ import annotations
import io, zipfile
import streamlit as st
import pandas as pd
from core.C10_stats import (
    stats_df_dict_all_with_temp_corr_only_for_others,
    stats_to_excel_bytes,
    plot_stat_bars_per_df,
    plot_stat_bars_combined
)

from core.Z00_prefs import (
    apply_unified_prefs,
    merge_prefs,
    prefs_to_bytes,
    collect_all_prefs,
)

def render_stats_page():
    st.title("Analyse statistique (par DF)")

    st.session_state.setdefault("stats_cfg", {})
    prev = st.session_state["stats_cfg"]

    df_ns = st.session_state.get("df", {})
    data = df_ns.get("preprocessed") or df_ns.get("groups")
    if not data:
        st.warning("Aucune donnée. Va d’abord dans **Créer DF** / **Pré-traitement**.")
        return

    st.subheader("Paramètres")
    c1, c2, c3 = st.columns(3)
    with c1:
        resume_par = st.selectbox("Résumé par", ["mois", "semaine"], index=(0 if prev.get("resume_par","mois")=="mois" else 1))
    with c2:
        temp_contains = st.text_input("Mot-clé température", value=prev.get("temp_contains","temp"))
    with c3:
        arrondi = st.number_input("Arrondi", min_value=0, max_value=6, value=int(prev.get("arrondi",3)), step=1)

    st.subheader("Sélection des DataFrames")
    all_names = list(data.keys())
    default_sel = prev.get("selected_dfs", all_names)
    choix = st.multiselect("DF à analyser", options=all_names, default=default_sel)
    if not choix:
        st.info("Sélectionne au moins un DataFrame.")
        return

    # Save page cfg
    if st.button("💾 Sauvegarder mes réglages (cette page)"):
        st.session_state["stats_cfg"] = {
            "resume_par": resume_par,
            "temp_contains": temp_contains,
            "arrondi": int(arrondi),
            "selected_dfs": choix,
        }
        st.success("Réglages Statistiques mémorisés ✅ (inclus dans l’export global depuis Import)")

    subset = {k: data[k] for k in choix}

    if st.button("Calculer les statistiques", key="stats_compute"):
        stats = stats_df_dict_all_with_temp_corr_only_for_others(
            subset,
            temp_col_contains=temp_contains,
            arrondi=int(arrondi),
            resume_par=resume_par,
        )
        st.session_state["last_stats"] = stats
        st.success("Statistiques calculées ✅")

    stats = st.session_state.get("last_stats")
    if not stats:
        return

    st.subheader("Résultats")
    for name, df_stats in stats.items():
        st.markdown(f"**{name}**")
        if df_stats is None or df_stats.empty:
            st.info("Aucune donnée statistique.")
        else:
            st.dataframe(df_stats)

    if any((v is not None and not v.empty) for v in stats.values()):
        st.download_button(
            "📥 Exporter tout (XLSX)",
            data=stats_to_excel_bytes(stats),
            file_name="stats_par_df.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="stats_dl_xlsx",
        )

    st.subheader("Visualisation des statistiques")
    example_df = next((v for v in stats.values() if v is not None and not v.empty), None)
    if example_df is None:
        st.info("Aucune statistique disponible pour tracer.")
        return

    label_col = "Capteur" if "Capteur" in example_df.columns else example_df.columns[0]
    value_cols = [c for c in example_df.columns if c != label_col]
    if not value_cols:
        st.info("Aucune colonne de valeurs à tracer.")
        return

    c1, c2 = st.columns(2)
    with c1:
        col_choice = st.selectbox("Statistique (Y)", value_cols, index=0, key="stat_col_choice")
    with c2:
        mode = st.radio("Mode", ["Barres par DF", "Barres combinées"], horizontal=True, key="stat_mode")

    c3, c4, c5 = st.columns(3)
    with c3:
        sort = st.selectbox("Tri", ["desc", "asc", "none"], index=0, key="stat_sort")
    with c4:
        top_n = st.number_input("Top N (optionnel)", min_value=0, value=0, step=1, key="stat_topn")
        top_n = int(top_n) if top_n > 0 else None
    with c5:
        width = st.number_input("Largeur figure", min_value=6, value=12, step=1, key="stat_w")
    height = st.number_input("Hauteur figure", min_value=4, value=6, step=1, key="stat_h")

    def _fig_to_png(fig) -> bytes:
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=200, bbox_inches="tight")
        bio.seek(0)
        return bio.read()

    if st.button("Tracer la statistique", key="plot_stat_bars"):
        if mode == "Barres par DF":
            figs = plot_stat_bars_per_df(
                stats, column=col_choice, label_col=label_col,
                sort=sort, top_n=top_n, figsize=(width, height)
            )
            if not figs:
                st.warning("Aucun graphe à afficher.")
            else:
                for name, fig in figs.items():
                    st.markdown(f"**{name}**")
                    st.pyplot(fig)
                    st.download_button(
                        f"📥 Exporter « {name} » (PNG)",
                        data=_fig_to_png(fig),
                        file_name=f"{name.replace('/', '_')[:80]}_{col_choice}.png",
                        mime="image/png",
                        key=f"dl_stat_png_{name}",
                    )
                bio = io.BytesIO()
                with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, fig in figs.items():
                        zf.writestr(f"{name.replace('/', '_')[:80]}_{col_choice}.png", _fig_to_png(fig))
                bio.seek(0)
                st.download_button(
                    "🗂️ Exporter TOUT (ZIP)",
                    data=bio.read(),
                    file_name=f"stats_{col_choice}_all.zip",
                    mime="application/zip",
                    key="dl_stat_zip_all",
                )
        else:
            fig = plot_stat_bars_combined(
                stats, column=col_choice, label_col=label_col,
                add_df_prefix=True, sort=sort, top_n=top_n, figsize=(width, height)
            )
            if fig is None:
                st.warning("Aucun graphe à afficher.")
            else:
                st.pyplot(fig)
                st.download_button(
                    "📥 Exporter (PNG)",
                    data=_fig_to_png(fig),
                    file_name=f"stats_combined_{col_choice}.png",
                    mime="image/png",
                    key="dl_stat_combined_png",
                )
