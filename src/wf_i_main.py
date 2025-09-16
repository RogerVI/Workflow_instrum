# wf_i_main.py (ou src/01_Import.py)

# --- bootstrap pour imports locaux si besoin ---
import sys, os
SRC = os.path.dirname(__file__)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# ----------------------------------------------

import streamlit as st

# Pages
from services.A00_streamlit_import_data import render_import_page
from services.A10_streamlit_create_df import (render_create_df_page)
from services.A11_streamlit_preprocessing_data import render_preprocessing_page
from services.C01_streamlit_graph import render_graph_page
from services.C10_streamlit_stats import render_stats_page
from services.C11_streamlit_correlation import render_correlation_page
from services.Z00_global_toolbar import render_global_toolbar

# Initialise les clés une seule fois
defaults = {
    "df": {},              # contiendra sources / groups / preprocessed
    "viz": {},             # contiendra thresholds
    "viz_units": {},       # unités par DF
    "graph_style": {},     # options Matplotlib
    "plotly_style": {},    # options Plotly
    "preproc_cfg": {},     # moyennes/vitesses/lissage etc.
    "nan_cfg": {},         # traitement NaN
    "excluded_dates_cfg": {},
    "corr_cfg": {},        # options corrélation
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


st.set_page_config(page_title="Workflow Instrum", layout="wide")

st.image(str("../SIXENSE_logo.png"), width=120)


# État global
st.session_state.setdefault("dfs_clean", None)
st.session_state.setdefault("regroupements", None)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Importer", "Créer DF",
    "Pré-traitement", "Graphiques",
    "Statistiques", "Corrélation"],
    index=0,
)

if page == "Importer":
    render_import_page()
elif page == "Créer DF":
    render_create_df_page()
elif page == "Pré-traitement":
    render_preprocessing_page()
elif page == "Graphiques":
    render_graph_page()
elif page == "Statistiques":
    render_stats_page()
elif page == "Corrélation":
    render_correlation_page()
