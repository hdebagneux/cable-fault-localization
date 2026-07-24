"""
app.py
------
Main entry point and lightweight router for the Cable Fault Simulation and
Localization Tool.

Navigation is driven by `st.session_state.page`. This keeps the app on a
single Streamlit script for now while remaining trivially extensible: each
new page is just another branch in `main()` (or a file under pages/ if you
later switch to Streamlit's native multipage mode — see README).

Run from inside the "streamlit interface" folder with:
    streamlit run app.py
"""

import streamlit as st

from styles import apply_custom_css
from components import (
    render_header,
    render_domain_selection,
    render_placeholder_page,
    render_footer,
)
from fdr_page import render_fdr_page
from time_domain import render_time_domain_page


def init_session_state() -> None:
    """Initialize navigation state once, safely (no overwrite on rerun)."""
    if "page" not in st.session_state:
        st.session_state.page = "home"       # "home" | "frequency" | "time"
    if "domain" not in st.session_state:
        st.session_state.domain = None       # "frequency" | "time"


def render_welcome_page() -> None:
    """The landing page: header + domain selection."""
    render_header()
    render_domain_selection()


def main() -> None:
    # Page-level configuration must be the first Streamlit call.
    st.set_page_config(
        page_title="Cable Fault Simulation & Localization Tool",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_custom_css()
    init_session_state()

    # --- Simple session-state router --------------------------------------
    page = st.session_state.page
    if page == "frequency":
        render_fdr_page()                    # full FDR interface (Hana)
    elif page == "time":
        render_time_domain_page()            # full TDR interface (coworker)
    else:
        render_welcome_page()

    render_footer()


if __name__ == "__main__":
    main()
