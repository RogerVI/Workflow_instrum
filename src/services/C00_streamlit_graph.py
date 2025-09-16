# services/C01_streamlit_graph.py
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import streamlit as st
from matplotlib.colors import to_hex as _mpl_to_hex

from core.C01_graph import (
    tracer_colonnes_separees_core,
    DEFAULT_THRESHOLD_COLORS,
)
from core.C02_plotly import tracer_colonnes_plotly_core
from services.D00_streamlit_doc_export import render_docx_export_block

from services.Z00_global_toolbar import render_global_toolbar
from core.Z00_prefs import (
    apply_unified_prefs,
    merge_prefs,
    prefs_to_bytes,
    collect_all_prefs,
)




# ===========================
# Palettes disponibles
# ===========================
PALETTES = [
    "Paired", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c", "Dark2",
    "pastel1", "pastel2", "mako", "rocket", "crest", "flare", "chroma",
    "bright", "Spectral", "coolwarm", "RdYlBu", "RdYlGn", "RdBu", "PuOr", "PRGn",
    "twilight", "twilight_shifted", "hsv",
    "Blues", "Reds", "Greens", "Purples", "Oranges", "Greys",
    "Accent", "RdGy", "PuRd", "BuPu", "YlGnBu", "YlGn", "GnBu", "OrRd",
    "cubehelix", "viridis", "magma", "plasma", "inferno", "cividis", "icefire", "vlag",
]

# ===========================
# Helpers couleurs / exports
# ===========================
def _as_hex(c: str) -> str:
    """Assure une couleur hex valide pour color_picker (convertit 'orange' -> '#FFA500')."""
    if isinstance(c, str) and c.startswith("#") and (len(c) in (4, 7)):
        return c
    try:
        return _mpl_to_hex(c)
    except Exception:
        return "#FF8C00"

def _normalize_color_dict(d: dict | None) -> dict:
    d = d or {}
    return {k: _as_hex(v) for k, v in d.items()}

def _mpl_fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=200)
    bio.seek(0)
    return bio.read()

def _mpl_zip(figs: dict) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs.items():
            safe = f"{name}".replace("/", "_")[:80]
            zf.writestr(f"{safe}.png", _mpl_fig_to_png_bytes(fig))
    bio.seek(0)
    return bio.read()

def _plotly_fig_to_png_bytes(fig) -> bytes:
    # nécessite kaleido pour le PNG côté serveur
    try:
        return fig.to_image(format="png", scale=2)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(
            "Export PNG Plotly indisponible (kaleido manquant ? `pip install -U kaleido`)."
        ) from e

def _plotly_zip(figs: dict) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figs.items():
            safe = f"{name}".replace("/", "_")[:80]
            zf.writestr(f"{safe}.png", _plotly_fig_to_png_bytes(fig))
    bio.seek(0)
    return bio.read()


# ===========================
# Preset "Notebook"
# ===========================
NOTEBOOK_PRESET = {
    "g_use_sns": False,       # Notebook original: pas de style seaborn forcé
    "g_pal_src": "auto",      # auto: seaborn si dispo, sinon matplotlib
    "g_palette": "bright",
    "g_fig_w": 24.0,
    "g_fig_h": 10.0,
    "g_lw": 2.2,
    "g_grid_style": "--",
    "g_grid_alpha": 0.6,
    "g_temp_col": "#000000",
    "g_temp_alpha": 0.4,
    "g_temp_lw": 1.2,
    "g_leg_loc": "upper center",
    "g_leg_ncol": 3,
    "g_title_size": 16,
    "g_axis_size": 13,
    "g_legend_size": 13,
}

def _apply_notebook_preset():
    for k, v in NOTEBOOK_PRESET.items():
        st.session_state[k] = v
    st.success("Preset « Comme mon notebook » appliqué ✅")


# ===========================
# Page principale
# ===========================
def render_graph_page():
    st.title("Graphiques + Seuils (même page)")

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

    # Seuils en mémoire (n’empêchent jamais le tracé)
    thr = st.session_state.get("viz", {}).get("thresholds")
    colors_saved = _normalize_color_dict(
        st.session_state.get("thresholds_colors") or DEFAULT_THRESHOLD_COLORS.copy()
    )

    # ------------------ Options d’affichage ------------------
    st.subheader("Options d’affichage")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tracer_brut = st.checkbox("Brut", True, key="g_brut")
    with c2:
        tracer_moy = st.checkbox("Moyenne glissante", True, key="g_moy")
    with c3:
        tracer_vit = st.checkbox("Vitesse", True, key="g_vit")
    with c4:
        tracer_temp = st.checkbox("Température", True, key="g_temp")

    show_thresholds = st.checkbox("Afficher les seuils", value=bool(thr), key="g_show_seuils")
    if show_thresholds and not thr:
        st.info("ℹ️ Aucun seuil chargé. Les graphes s’afficheront **sans** lignes de seuils.")

    st.subheader("Moteur de tracé")
    engine = st.radio(
        "Choisir le moteur",
        ["Matplotlib", "Plotly (interactif)"],
        index=0,
        key="g_engine",
    )

    plotly_w, plotly_h = 1100, 550
    if engine == "Plotly (interactif)":
        cW, cH = st.columns(2)
        with cW:
            plotly_w = st.number_input(
                "Largeur (Plotly)", min_value=600, max_value=3000, value=1100, step=50, key="g_ply_w"
            )
        with cH:
            plotly_h = st.number_input(
                "Hauteur (Plotly)", min_value=300, max_value=2000, value=550, step=25, key="g_ply_h"
            )

    # ------------------ Filtres temporels ------------------
    st.subheader("Filtre temporel")
    colA, colB = st.columns(2)
    with colA:
        start_date = st.text_input(
            "Début (YYYY-MM-DD ou YYYY-MM-DD HH:MM:SS)",
            value="",
            key="g_start",
        )
    with colB:
        end_date = st.text_input(
            "Fin (YYYY-MM-DD ou YYYY-MM-DD HH:MM:SS)",
            value="",
            key="g_end",
        )
    start = pd.to_datetime(start_date) if start_date.strip() else None
    end = pd.to_datetime(end_date) if end_date.strip() else None
    if start and end and start > end:
        st.error("⛔ La date de début est après la date de fin.")
        return

    # ------------------ Unités ------------------
    st.subheader("Unités par DataFrame")
    saved_units = st.session_state.get("viz_units") or {}
    units_user = {k: saved_units.get(k, "") for k in data.keys()}

    with st.expander("Assigner des unités en lot", expanded=False):
        df_names = list(data.keys())
        selection_grp = st.multiselect(
            "Sélectionne un groupe de DataFrames",
            options=df_names,
            key="units_batch_select",
        )
        unit_grp = st.text_input(
            "Unité à attribuer au groupe (ex: mm, °C…)",
            value="",
            key="units_batch_value",
        )
        if st.button("Attribuer au groupe", key="units_apply_batch"):
            if not selection_grp or not unit_grp.strip():
                st.warning("Sélectionne au moins un DF et saisis une unité.")
            else:
                for name in selection_grp:
                    units_user[name] = unit_grp.strip()
                st.success(f"Unité « {unit_grp.strip()} » attribuée à {len(selection_grp)} DF.")
                st.session_state["viz_units"] = units_user
                st.experimental_rerun()

    for name in data.keys():
        units_user[name] = st.text_input(
            f"Unité pour {name}",
            value=units_user.get(name, ""),
            key=f"unit_{name}",
        )
    st.session_state["viz_units"] = units_user

    # ------------------ Colonnes ------------------
    st.subheader("Colonnes à tracer")
    tracer_all = st.checkbox("Tracer toutes les colonnes", True, key="g_all")
    selection = {}
    if not tracer_all:
        for name, df in data.items():
            cols_sorted = sorted(df.columns)
            opts = [f"{i}: {c}" for i, c in enumerate(cols_sorted)]
            choix = st.multiselect(f"Sélection pour {name}", options=opts, key=f"sel_{name}")
            idx = [int(x.split(":")[0]) for x in choix]
            selection[name] = idx

    # ------------------ Diagnostic (optionnel) ------------------
    with st.expander("Diagnostic colonnes (optionnel)"):
        for name, df in data.items():
            st.write(f"**{name}**", list(df.columns)[:120], "…")

    # ------------------ Seuils ------------------
    st.subheader("Seuils")
    with st.expander("Configurer / importer des seuils", expanded=False):
        # Import JSON seuils
        up = st.file_uploader("Importer seuils (JSON)", type=["json"], key="g_thr_upl")
        if up:
            try:
                st.session_state.setdefault("viz", {})
                st.session_state["viz"]["thresholds"] = json.load(up)
                thr = st.session_state["viz"]["thresholds"]  # refresh local
                st.success("Seuils importés ✅")
            except Exception as e:
                st.error(f"JSON invalide : {e}")

        # Types + couleurs
        types_dispo = ["vigilance", "vigilance accrue", "alerte"]
        default_types = []
        if isinstance(thr, dict) and thr:
            first_df = next(iter(thr.keys()), None)
            if first_df:
                default_types = list(thr.get(first_df, {}).keys())
        if not default_types:
            default_types = ["vigilance"]

        selected_types = st.multiselect(
            "Types de seuils actifs",
            types_dispo,
            default=default_types,
        )

        cols_cfg = {}
        for t in selected_types:
            cols_cfg[t] = st.color_picker(
                f"Couleur pour « {t} »",
                _as_hex(colors_saved.get(t, "#FF8C00")),
                key=f"thr_color_{t}",
            )
        if st.button("Enregistrer couleurs", key="g_thr_colors"):
            st.session_state["thresholds_colors"] = _normalize_color_dict(cols_cfg)
            colors_saved = st.session_state["thresholds_colors"]
            st.success("Couleurs enregistrées ✅")

        # Valeurs par DataFrame via TABS
        st.markdown("### Valeurs des seuils (par DataFrame)")
        df_names_thr = list(data.keys())
        if not df_names_thr:
            st.info("Aucun DataFrame chargé.")
        else:
            tabs = st.tabs(df_names_thr)
            new_thr = {}
            for tab, df_name in zip(tabs, df_names_thr):
                with tab:
                    per_df = (thr or {}).get(df_name, {})
                    per_df_out = {}
                    for t in selected_types:
                        mode = st.radio(
                            f"{t} — mode",
                            ["Valeur unique", "Min/Max"],
                            horizontal=True,
                            key=f"thr_mode_{df_name}_{t}",
                        )
                        if mode == "Valeur unique":
                            val_default = 0.0
                            if per_df.get(t):
                                val_default = float(
                                    per_df[t].get("min", per_df[t].get("max", 0.0))
                                )
                            val = st.number_input(
                                f"{t} — valeur",
                                value=val_default,
                                key=f"thr_val_{df_name}_{t}",
                            )
                            per_df_out[t] = {"min": val, "max": val}
                        else:
                            vmin_default = float(per_df.get(t, {}).get("min", 0.0))
                            vmax_default = float(per_df.get(t, {}).get("max", 0.0))
                            vmin = st.number_input(
                                f"{t} — min",
                                value=vmin_default,
                                key=f"thr_min_{df_name}_{t}",
                            )
                            vmax = st.number_input(
                                f"{t} — max",
                                value=vmax_default,
                                key=f"thr_max_{df_name}_{t}",
                            )
                            per_df_out[t] = {"min": vmin, "max": vmax}

                    if per_df_out:
                        new_thr[df_name] = per_df_out

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Enregistrer seuils en mémoire", key="g_thr_save"):
                    st.session_state.setdefault("viz", {})
                    st.session_state["viz"]["thresholds"] = new_thr
                    thr = new_thr
                    st.success("Seuils enregistrés ✅")
            with c2:
                st.download_button(
                    "📥 Exporter seuils (JSON)",
                    data=json.dumps(new_thr, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="seuils.json",
                    mime="application/json",
                    key="g_thr_dl",
                )

    # ------------------ Style & rendu (Matplotlib) ------------------
    st.subheader("🎛️ Style & rendu (Matplotlib)")
    if st.button("🎨 Preset : Comme mon notebook"):
        _apply_notebook_preset()

    with st.expander("Options d’apparence", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            use_sns = st.checkbox(
                "Forcer style Seaborn",
                value=st.session_state.get("g_use_sns", NOTEBOOK_PRESET["g_use_sns"]),
                key="g_use_sns",
            )
            pal_src = st.selectbox(
                "Source de palette",
                ["auto", "seaborn", "matplotlib"],
                index=["auto", "seaborn", "matplotlib"].index(
                    st.session_state.get("g_pal_src", NOTEBOOK_PRESET["g_pal_src"])
                ),
                key="g_pal_src",
            )
            palette_choice = st.selectbox(
                "Palette",
                options=PALETTES,
                index=PALETTES.index(
                    st.session_state.get("g_palette", NOTEBOOK_PRESET["g_palette"])
                ),
                key="g_palette",
            )
        with colB:
            fig_w = st.number_input(
                "Largeur figure",
                min_value=8.0,
                max_value=64.0,
                value=float(st.session_state.get("g_fig_w", NOTEBOOK_PRESET["g_fig_w"])),
                step=0.5,
                key="g_fig_w",
            )
            fig_h = st.number_input(
                "Hauteur figure",
                min_value=4.0,
                max_value=36.0,
                value=float(st.session_state.get("g_fig_h", NOTEBOOK_PRESET["g_fig_h"])),
                step=0.5,
                key="g_fig_h",
            )
            line_w = st.slider(
                "Épaisseur courbes",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state.get("g_lw", NOTEBOOK_PRESET["g_lw"])),
                step=0.1,
                key="g_lw",
            )
        with colC:
            grid_opts = ["--", "-", "-.", ":"]
            grid_style = st.selectbox(
                "Style de grille",
                grid_opts,
                index=grid_opts.index(st.session_state.get("g_grid_style", NOTEBOOK_PRESET["g_grid_style"])),
                key="g_grid_style",
            )
            grid_alpha = st.slider(
                "Intensité grille",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("g_grid_alpha", NOTEBOOK_PRESET["g_grid_alpha"])),
                step=0.05,
                key="g_grid_alpha",
            )
            temp_col = st.color_picker(
                "Couleur température",
                value=st.session_state.get("g_temp_col", NOTEBOOK_PRESET["g_temp_col"]),
                key="g_temp_col",
            )

        colD, colE, colF = st.columns(3)
        with colD:
            temp_alpha = st.slider(
                "Opacité température",
                0.0, 1.0,
                float(st.session_state.get("g_temp_alpha", NOTEBOOK_PRESET["g_temp_alpha"])),
                0.05,
                key="g_temp_alpha",
            )
            temp_lw = st.slider(
                "Épaisseur température",
                0.5, 4.0,
                float(st.session_state.get("g_temp_lw", NOTEBOOK_PRESET["g_temp_lw"])),
                0.1,
                key="g_temp_lw",
            )
        with colE:
            leg_loc = st.selectbox(
                "Position légende",
                [
                    "upper center","upper right","upper left","lower right","lower left",
                    "right","center right","center left","lower center","center"
                ],
                index=[
                    "upper center","upper right","upper left","lower right","lower left",
                    "right","center right","center left","lower center","center"
                ].index(st.session_state.get("g_leg_loc", NOTEBOOK_PRESET["g_leg_loc"])),
                key="g_leg_loc",
            )
            leg_ncol = st.number_input(
                "Colonnes légende",
                min_value=1, max_value=6,
                value=int(st.session_state.get("g_leg_ncol", NOTEBOOK_PRESET["g_leg_ncol"])),
                step=1,
                key="g_leg_ncol",
            )
        with colF:
            title_size = st.number_input(
                "Taille titre", 8, 40,
                int(st.session_state.get("g_title_size", NOTEBOOK_PRESET["g_title_size"])),
                1, key="g_title_size",
            )
            axis_size = st.number_input(
                "Taille axes", 8, 30,
                int(st.session_state.get("g_axis_size", NOTEBOOK_PRESET["g_axis_size"])),
                1, key="g_axis_size",
            )
            legend_size = st.number_input(
                "Taille légende", 8, 30,
                int(st.session_state.get("g_legend_size", NOTEBOOK_PRESET["g_legend_size"])),
                1, key="g_legend_size",
            )


    # ... juste après tes contrôles UI de style/units/seuils ...
    if st.button("💾 Sauvegarder mes réglages (cette page)"):
        # Style matplotlib
        st.session_state["graph_style"] = {
            "use_sns": st.session_state.get("g_use_sns"),
            "pal_src": st.session_state.get("g_pal_src"),
            "palette": st.session_state.get("g_palette"),
            "fig_w": st.session_state.get("g_fig_w"),
            "fig_h": st.session_state.get("g_fig_h"),
            "lw": st.session_state.get("g_lw"),
            "grid_style": st.session_state.get("g_grid_style"),
            "grid_alpha": st.session_state.get("g_grid_alpha"),
            "temp_col": st.session_state.get("g_temp_col"),
            "temp_alpha": st.session_state.get("g_temp_alpha"),
            "temp_lw": st.session_state.get("g_temp_lw"),
            "leg_loc": st.session_state.get("g_leg_loc"),
            "leg_ncol": st.session_state.get("g_leg_ncol"),
            "title_size": st.session_state.get("g_title_size"),
            "axis_size": st.session_state.get("g_axis_size"),
            "legend_size": st.session_state.get("g_legend_size"),
        }
        # Unités
        st.session_state["viz_units"] = st.session_state.get("viz_units", {})
        # Seuils + couleurs
        st.session_state.setdefault("viz", {})
        st.session_state["viz"]["thresholds"] = st.session_state.get("viz", {}).get("thresholds", {})
        st.session_state["thresholds_colors"] = st.session_state.get("thresholds_colors", {})

        # (optionnel) plotly style
        st.session_state["plotly_style"] = {
            "width": st.session_state.get("g_ply_w", 1100),
            "height": st.session_state.get("g_ply_h", 550),
            "template": "plotly_white",
        }

        st.success("Réglages Graphiques/Seuils mémorisés ✅ (inclus dans l’export global depuis Import)")


    # ------------------ Tracé + export ------------------
    active_seuils = bool(st.session_state.get("g_show_seuils") and isinstance(thr, dict) and len(thr) > 0)

    if st.button("Tracer", key="g_plot"):
        if engine == "Matplotlib":
            figs = tracer_colonnes_separees_core(
                data,
                # data & sélection
                palette=st.session_state["g_palette"],
                tracer_brut=tracer_brut,
                tracer_moyenne=tracer_moy,
                tracer_vitesse=tracer_vit,
                tracer_temp=tracer_temp,
                activer_seuils=active_seuils,
                seuils_par_df=thr,
                couleurs_seuils=colors_saved,
                tracer_toutes_colonnes=tracer_all,
                colonnes_selectionnees=selection if not tracer_all else None,
                start_date=start, end_date=end,
                unites_par_df=st.session_state.get("viz_units", {}),
                # style & rendu
                fig_width=st.session_state["g_fig_w"],
                fig_height=st.session_state["g_fig_h"],
                use_seaborn_theme=st.session_state["g_use_sns"],
                palette_source=st.session_state["g_pal_src"],
                line_width=st.session_state["g_lw"],
                temp_color=st.session_state["g_temp_col"],
                temp_alpha=st.session_state["g_temp_alpha"],
                temp_linewidth=st.session_state["g_temp_lw"],
                grid_alpha=st.session_state["g_grid_alpha"],
                grid_style=st.session_state["g_grid_style"],
                legend_loc=st.session_state["g_leg_loc"],
                legend_ncol=int(st.session_state["g_leg_ncol"]),
                title_size=int(st.session_state["g_title_size"]),
                axis_size=int(st.session_state["g_axis_size"]),
                legend_size=int(st.session_state["g_legend_size"]),
            )

            if not figs:
                st.warning("Rien à afficher avec ces options. (Vérifie les colonnes ou la plage de dates)")
            else:
                # Affichage + export unitaire
                for name, fig in figs.items():
                    st.subheader(name)
                    st.pyplot(fig)
                    st.download_button(
                        f"📥 Exporter « {name} » (PNG)",
                        data=_mpl_fig_to_png_bytes(fig),
                        file_name=f"{name.replace('/', '_')[:80]}.png",
                        mime="image/png",
                        key=f"dl_png_{name}",
                    )

                # Stocker les PNG (pour export Word)
                st.session_state["last_fig_pngs"] = {
                    name: _mpl_fig_to_png_bytes(fig) for name, fig in figs.items()
                }

                # Export ZIP global
                st.download_button(
                    "🗂️ Exporter TOUT (ZIP)",
                    data=_mpl_zip(figs),
                    file_name="graphs_export.zip",
                    mime="application/zip",
                    key="dl_zip_all",
                )
        else:
            figs_p = tracer_colonnes_plotly_core(
                data,
                tracer_brut=tracer_brut,
                tracer_moyenne=tracer_moy,
                tracer_vitesse=tracer_vit,
                tracer_temp=tracer_temp,
                tracer_toutes_colonnes=tracer_all,
                colonnes_selectionnees=selection if not tracer_all else None,
                start_date=start, end_date=end,
                legend_position="bottom",
                fig_width=plotly_w,
                fig_height=plotly_h,
                unite_label=units_user,
                template="plotly_white",
            )
            if not figs_p:
                st.warning("Rien à afficher avec ces options. (Vérifie les colonnes ou la plage de dates)")
            else:
                kaleido_ok = True
                try:
                    import kaleido  # noqa: F401
                except Exception:
                    kaleido_ok = False
                    st.info("ℹ️ Pour l’export PNG Plotly, installe **kaleido** : `pip install -U kaleido`")

                for name, fig in figs_p.items():
                    st.subheader(name)
                    st.plotly_chart(fig, use_container_width=True)
                    if kaleido_ok:
                        try:
                            st.download_button(
                                f"📥 Exporter « {name} » (PNG)",
                                data=_plotly_fig_to_png_bytes(fig),
                                file_name=f"{name.replace('/', '_')[:80]}.png",
                                mime="image/png",
                                key=f"dl_png_plotly_{name}",
                            )
                        except Exception as e:
                            st.warning(f"Export PNG pour « {name} » indisponible : {e}")

                if kaleido_ok:
                    st.download_button(
                        "🗂️ Exporter TOUT (ZIP)",
                        data=_plotly_zip(figs_p),
                        file_name="graphs_plotly_export.zip",
                        mime="application/zip",
                        key="dl_zip_all_plotly",
                    )

                # Pour l’export Word, on ne gère que les PNG Matplotlib ici.
                # On vide la mémoire pour éviter d’insérer de vieilles images.
                st.session_state.pop("last_fig_pngs", None)

    # ------------------ Export Word (s’il y a des figures) ------------------
    if st.session_state.get("last_fig_pngs"):
        render_docx_export_block()
    else:
        st.info("Trace d’abord des graphiques Matplotlib pour activer l’export Word.")
