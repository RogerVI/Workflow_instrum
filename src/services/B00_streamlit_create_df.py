# services/A10_streamlit_create_df.py
from __future__ import annotations
import json, re, ast
from io import BytesIO
import streamlit as st
import pandas as pd

from core.B00_create_df import (
    detecter_gabarits_colonnes,
    extraire_et_fusionner_par_multi_gabarits_interactif,
    renommer_dataframes,
    renommer_colonnes_df_dict,
    filtrer_colonnes_par_numero,
)

from core.Z00_prefs import (
    apply_unified_prefs,
    merge_prefs,
    prefs_to_bytes,
    collect_all_prefs,
)

# ---------- Helpers ----------

def _parse_index_ranges(s: str) -> list[int]:
    """Ex: '1,3-5, 8' -> [1,3,4,5,8] (indices entiers)."""
    s = (s or "").strip()
    if not s:
        return []
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)

def _load_mapping_any(uploaded_file):
    """Charge un mapping depuis un fichier uploadé (JSON pur ou dict Python)."""
    txt = uploaded_file.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise
        body = m.group(0)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return ast.literal_eval(body)

def _export_xlsx(d: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for k, df in d.items():
            df.to_excel(w, sheet_name=str(k)[:31], index=(df.index.name is not None))
    bio.seek(0)
    return bio.read()

# ====== Préférences (uniquement pour CETTE page) ======

def _dump_page_prefs(cfg: dict) -> bytes:
    """Retourne un JSON bytes uniquement des réglages de cette page."""
    payload = {"create_df_cfg": cfg}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

def _apply_loaded_page_prefs(payload: dict):
    """Applique un JSON de réglages de page à st.session_state['create_df_cfg'] et re-préremplit les champs UI."""
    if not isinstance(payload, dict) or "create_df_cfg" not in payload:
        st.warning("JSON invalide : clé 'create_df_cfg' absente.")
        return
    st.session_state["create_df_cfg"] = payload["create_df_cfg"]

    # Pré-remplissage UI (texte des combinaisons)
    combos = st.session_state["create_df_cfg"].get("combos", [])
    if combos:
        lines = []
        for grp in combos:
            try:
                lines.append(",".join(str(int(x)) for x in grp))
            except Exception:
                pass
        st.session_state["combos_text"] = "\n".join(lines)

# ---------- Page principale ----------

def render_create_df_page():
    st.title("Créer / Regrouper / Renommer / Nettoyer")

    # INIT état de page
    st.session_state.setdefault("create_df_cfg", {
        "combos": [],              # [[0,2], [1,3,4], ...]
        "rename_df_map": {},       # {old_df: new_df}
        "rename_cols_map": {},     # {df_name: {old_col: new_col}}
        "delete_cols": {},         # {df_name: [indices]}
    })

    # SOURCES (données chargées dans l’onglet Import)
    st.session_state.setdefault("df", {})
    dfs_clean = st.session_state["df"].get("sources")
    if not dfs_clean:
        st.warning("Commence par l’onglet **Importer** pour charger des données.")
        return

    # ====== Barre d’actions (import/export de réglages de page) ======
    st.markdown("### Réglages de cette page")
    colA, colB = st.columns(2)
    with colA:
        up_cfg = st.file_uploader("📤 Importer réglages (JSON)", type=["json"], key="create_df_cfg_upload")
        if up_cfg:
            try:
                payload = json.load(up_cfg)
                _apply_loaded_page_prefs(payload)
                st.success("Réglages de page importés ✅")
            except Exception as e:
                st.error(f"JSON invalide : {e}")
    with colB:
        if st.button("💾 Sauvegarder mes réglages (cette page)", key="save_create_df_cfg"):
            data = _dump_page_prefs(st.session_state["create_df_cfg"])
            st.download_button(
                "📥 Télécharger le JSON de réglages",
                data=data,
                file_name="reglages_create_df.json",
                mime="application/json",
                key="dl_create_df_cfg",
            )

    # ------------------------- 1) FUSION PAR GABARITS -------------------------
    st.subheader("1) Fusion par gabarits")

    gabs = detecter_gabarits_colonnes(dfs_clean)
    g_sorted = sorted(gabs.items(), key=lambda x: x[0])  # liste[(gabarit, colonnes)]

    with st.expander("Gabarits détectés"):
        for i, (g, cols) in enumerate(g_sorted):
            st.write(f"{i} — **{g}** ({len(cols)} cols)")
            st.code(", ".join(cols), language="text")

    # zone texte → combos
    saisies = st.text_area(
        "Combinaisons d’indices (une par ligne)",
        placeholder="0,2\n1,3,4",
        key="combos_text",
        value=st.session_state.get("combos_text", ""),
    )

    # MAJ immédiate de l’état (même sans cliquer fusionner)
    combos_preview = []
    for ligne in (saisies or "").splitlines():
        try:
            idxs = sorted(set(int(x) for x in re.split(r"[,\s]+", ligne.strip()) if x != ""))
            if idxs:
                # clamp indices dans l'intervalle des gabarits existants
                idxs = [i for i in idxs if 0 <= i < len(g_sorted)]
                if idxs:
                    combos_preview.append(idxs)
        except Exception:
            pass
    st.session_state["create_df_cfg"]["combos"] = combos_preview

    if st.button("Fusionner", key="fusionner_groups"):
        groupes = st.session_state["create_df_cfg"]["combos"]
        if not groupes:
            st.warning("Aucune combinaison valide.")
        else:
            regroupements = extraire_et_fusionner_par_multi_gabarits_interactif(
                dfs_clean, groupes_predefinis=groupes
            )
            non_vides = {k: v for k, v in regroupements.items() if isinstance(v, pd.DataFrame)}
            if not non_vides:
                st.error("Aucun regroupement utile.")
            else:
                st.session_state["df"]["groups"] = non_vides
                st.success(f"{len(non_vides)} regroupement(s) créé(s).")

    # À partir d'ici : seulement si des regroupements existent
    groups = st.session_state.get("df", {}).get("groups")
    if not groups:
        st.info("Aucun regroupement pour le moment. Clique d’abord sur **Fusionner** ci-dessus.")
        return

    # Aperçu + Export Excel
    with st.expander("Aperçu des regroupements (résultats)"):
        for nom, dfm in groups.items():
            st.write(f"**{nom}** — {dfm.shape[0]}x{dfm.shape[1]}")
            st.dataframe(dfm.head(200))
    st.download_button(
        "📥 Exporter les regroupements (Excel)",
        _export_xlsx(groups),
        "regroupements.xlsx",
        key="dl_groups_xlsx",
    )

    # ------------------------- 2) RENOMMER DF -------------------------
    st.subheader("2) Renommer les DataFrames (résultats)")

    mapping_df_upload = st.file_uploader(
        "Importer mapping DF (JSON)", type=["json"], key="map_df_upl"
    )
    mapping_df_patterns = {}
    if mapping_df_upload:
        try:
            mapping_df_patterns = _load_mapping_any(mapping_df_upload)
            st.success("Mapping DF chargé.")
        except Exception as e:
            st.error(f"JSON invalide (DF): {e}")

    actual_names = list(groups.keys())

    # Defaults affichés = mapping importé (si présent), sinon identité
    mapping_df_default = {name: mapping_df_patterns.get(name, name) for name in actual_names}

    # Pré-remplissage depuis l’état si présent
    if st.session_state["create_df_cfg"].get("rename_df_map"):
        for k, v in st.session_state["create_df_cfg"]["rename_df_map"].items():
            if k in mapping_df_default:
                mapping_df_default[k] = v

    mapping_df_user = {}
    with st.form("rename_df_form"):
        for old_name in actual_names:
            new_name = st.text_input(
                f"Nom pour {old_name}",
                value=mapping_df_default[old_name],
                key=f"rename_df_{old_name}",
            )
            mapping_df_user[old_name] = new_name

        # met à jour l’état AVANT l’application
        st.session_state["create_df_cfg"]["rename_df_map"] = mapping_df_user or mapping_df_default

        apply_df = st.form_submit_button("Appliquer renommage DF")

    if apply_df:
        mapping_to_apply = st.session_state["create_df_cfg"]["rename_df_map"]
        new_dict, _ = renommer_dataframes(
            groups, mapping_renommage=mapping_to_apply, return_mapping=True
        )
        st.session_state["df"]["groups"] = new_dict
        groups = new_dict  # refresh
        st.success("Renommage DF appliqué ✅")
        st.download_button(
            "💾 Exporter mapping DF (JSON)",
            data=json.dumps(mapping_to_apply, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="mapping_renommage_df.json",
            mime="application/json",
            key="dl_map_df",
        )

    # ------------------------- 3) RENOMMER COLONNES -------------------------
    st.subheader("3) Renommer les colonnes (par DF)")

    mapping_cols_upload = st.file_uploader(
        "Importer mapping colonnes (JSON)", type=["json"], key="map_cols_upl"
    )
    mapping_cols_patterns = {}
    if mapping_cols_upload:
        try:
            mapping_cols_patterns = _load_mapping_any(mapping_cols_upload)
            st.success("Mapping colonnes chargé.")
        except Exception as e:
            st.error(f"JSON invalide (colonnes): {e}")

    # Préremplissage exact : si une entrée existe pour ce DF, on la propose
    mapping_total_default = {}
    for df_name, df in groups.items():
        if df_name in mapping_cols_patterns:
            mapping_total_default[df_name] = {
                col: mapping_cols_patterns[df_name].get(col, col) for col in df.columns
            }
        else:
            mapping_total_default[df_name] = {col: col for col in df.columns}

    # fusion avec l’état si présent
    if st.session_state["create_df_cfg"].get("rename_cols_map"):
        for df_name, mp in st.session_state["create_df_cfg"]["rename_cols_map"].items():
            if df_name in mapping_total_default:
                for col, newc in mp.items():
                    if col in mapping_total_default[df_name]:
                        mapping_total_default[df_name][col] = newc

    with st.form("rename_cols_form"):
        mapping_total_user = {}
        for df_name, df in groups.items():
            st.markdown(f"**{df_name}**")
            per_df_map = {}
            for col in df.columns:
                default = mapping_total_default[df_name].get(col, col)
                new_col = st.text_input(
                    f"{df_name} → {col}",
                    value=default,
                    key=f"rename_col_{df_name}_{col}",
                )
                if new_col != col:
                    per_df_map[col] = new_col
            if per_df_map:
                mapping_total_user[df_name] = per_df_map

        # mets à jour l’état AVANT l’application
        st.session_state["create_df_cfg"]["rename_cols_map"] = mapping_total_user or mapping_total_default

        apply_cols = st.form_submit_button("Appliquer renommage colonnes")

    if apply_cols:
        mapping_to_apply = st.session_state["create_df_cfg"]["rename_cols_map"]
        new_dict, _ = renommer_colonnes_df_dict(
            groups, mapping_renommage_colonnes=mapping_to_apply, return_mapping=True
        )
        st.session_state["df"]["groups"] = new_dict
        groups = new_dict  # refresh
        st.success("Renommage colonnes appliqué ✅")
        st.download_button(
            "💾 Exporter mapping colonnes (JSON)",
            data=json.dumps(mapping_to_apply, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="mapping_renommage_colonnes.json",
            mime="application/json",
            key="dl_map_cols",
        )

    # ------------------------- 4) SUPPRIMER COLONNES -------------------------
    st.subheader("4) Supprimer des colonnes")

    flag_key = "show_delete_cols_groups"
    st.session_state.setdefault(flag_key, False)
    if st.button("🗑️  Supprimer des colonnes ?", key="toggle_delete_groups"):
        st.session_state[flag_key] = not st.session_state[flag_key]

    if st.session_state[flag_key]:
        st.info("Sélectionne ou saisis des indices à supprimer (ex: 1,3-5,8).")
        selections = {}

        # préremplissage depuis l’état si existant
        previous = st.session_state["create_df_cfg"].get("delete_cols", {})

        for nom_df, df in groups.items():
            st.subheader(nom_df)

            # aide visuelle: index : nom
            st.code("\n".join([f"{i}: {c}" for i, c in enumerate(df.columns)]), language="text")

            # options multiselect (affichage)
            options = [f"{i}: {col}" for i, col in enumerate(df.columns)]
            preselect = []
            if nom_df in previous:
                preselect = [f"{i}: {df.columns[i]}" for i in previous.get(nom_df, []) if 0 <= i < len(df.columns)]

            choix = st.multiselect(
                f"Colonnes à supprimer dans {nom_df} (sélection)",
                options=options,
                default=preselect,
                key=f"drop_groups_{nom_df}",
            )

            saisie = st.text_input(
                f"… ou saisis des indices/plages pour {nom_df} (ex: 1,3-5,8)",
                value=",".join(str(i) for i in previous.get(nom_df, [])) if nom_df in previous else "",
                key=f"drop_ranges_{nom_df}",
            )

            idx_from_select = [int(x.split(":")[0]) for x in choix]
            idx_from_ranges = _parse_index_ranges(saisie)
            indices = sorted(set(idx_from_select + idx_from_ranges))
            if indices:
                selections[nom_df] = indices

        # MAJ état en live
        st.session_state["create_df_cfg"]["delete_cols"] = selections

        if st.button("Appliquer la suppression", key="apply_delete_groups"):
            if not selections:
                st.warning("Aucune colonne sélectionnée.")
            else:
                filtrer_colonnes_par_numero(groups, selections_predefinies=selections)  # inplace
                st.session_state["df"]["groups"] = groups
                st.session_state[flag_key] = False  # referme le panneau
                st.success("Colonnes supprimées ✅")
                st.experimental_rerun()

    # ------------------------- Export des regroupements à jour -------------------------
    st.markdown("---")
    st.download_button(
        "📦 Exporter les regroupements (Excel, à jour)",
        _export_xlsx(st.session_state.get("df", {}).get("groups", {})),
        "regroupements_apres_ops.xlsx",
        key="dl_groups_post_xlsx",
    )

    # Bouton mémorisation locale (facultatif)
    if st.button("💾 Mémoriser ces réglages (cette page)"):
        st.success("Réglages Create DF mémorisés ✅ (ils seront inclus si tu fais un export global ailleurs)")
