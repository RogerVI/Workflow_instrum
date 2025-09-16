# services/D00_streamlit_doc_export.py
from __future__ import annotations
import streamlit as st
from core.D00_docx_utils import insert_image_after_marker_docx



def render_docx_export_block():
    st.subheader("Exporter vers Word (insertion des graphiques)")

    # On attend que la page graph ait stocké les PNG en mémoire:
    # st.session_state["last_fig_pngs"] = { "NomDuGraph": png_bytes }
    imgs = st.session_state.get("last_fig_pngs", {})
    if not imgs:
        st.info("Aucune image en mémoire. Génére (ou régénère) les graphiques d’abord.")
        return

    with st.expander("Images détectées", expanded=False):
        st.write(list(imgs.keys()))

    tpl = st.file_uploader("Modèle Word (.docx) avec repères ### NomDuGraph", type=["docx"])
    colW, colBtn = st.columns([1, 1])
    with colW:
        width_in = st.number_input("Largeur des images (en pouces)", min_value=2.0, max_value=8.0, value=6.0, step=0.25)

    if colBtn.button("Générer le Word", key="gen_word_from_imgs"):
        if not tpl:
            st.warning("Charge d’abord un document Word (.docx) modèle.")
            return
        try:
            docx_out, ok, miss = insert_image_after_marker_docx(
                tpl.read(), imgs, width_in_inches=width_in, marker_prefix="### "
            )
            st.success(f"Terminé : {ok} image(s) insérée(s), {miss} repère(s) introuvable(s).")
            st.download_button(
                "📥 Télécharger le Word généré",
                data=docx_out,
                file_name="rapport_graphs.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_word_graphs",
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération: {e}")
