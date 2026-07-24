"""
components.py
-------------
Reusable UI building blocks (header, domain selection, placeholder pages).

Layout logic lives here; styling lives in styles.py. Each render function is
self-contained so app.py reads as a simple router.
"""

import streamlit as st

from theme import COLORS, PROJECT_LABEL, APP_TITLE, APP_SUBTITLE

# --- Domain metadata -------------------------------------------------------
# Declarative description of each top-level choice. Adding/adjusting a domain
# (or wiring it to a real page later) is just an edit to this dict.
DOMAINS = {
    "frequency": {
        "title": "Frequency Domain",
        "description": (
            "Analyze the cable's frequency response and reflection coefficient "
            "to simulate faults and estimate their positions through "
            "frequency-domain reflectometry."
        ),
        "button_label": "Explore Frequency Domain",
        "placeholder_title": "FDR — Frequency Domain Reflectometry",
        "placeholder_message": "Frequency-domain tools will be implemented here.",
    },
    "time": {
        "title": "Time Domain",
        "description": (
            "Observe reflected signals directly in the time domain to analyse "
            "the propagations of signals in a wire and the impact of faults on these."
        ),
        "button_label": "Explore Time Domain",
        "placeholder_title": "TDR — Time Domain Reflectometry",
        "placeholder_message": "Time-domain tools will be implemented here.",
    },
}


def go_to(page: str, domain: str | None = None) -> None:
    """Update navigation state and rerun. Central place for all navigation."""
    st.session_state.page = page
    if domain is not None:
        st.session_state.domain = domain
    st.rerun()


def render_header() -> None:
    """Render the EDF project label, app title, and subtitle."""
    st.markdown(
        f'<span class="project-label">{PROJECT_LABEL}</span>',
        unsafe_allow_html=True,
    )
    # The word "Fault" is accented in orange for a subtle scientific highlight.
    title_html = APP_TITLE.replace(
        "Fault", '<span class="accent">Fault</span>', 1
    )
    st.markdown(f'<h1 class="app-title">{title_html}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)
    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)


def _render_domain_card(domain_key: str) -> None:
    """Render a single domain card plus its action button."""
    meta = DOMAINS[domain_key]
    st.markdown(
        f"""
        <div class="domain-card">
            <div class="card-title">{meta['title']}</div>
            <div class="card-desc">{meta['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Button sits below the card; Streamlit buttons can't be embedded in HTML.
    # The domain key ("frequency" / "time") doubles as the destination page.
    if st.button(meta["button_label"], key=f"btn_{domain_key}"):
        go_to(domain_key, domain=domain_key)


def render_domain_selection() -> None:
    """Render the two-card domain selection (the user's first choice)."""
    st.markdown(
        '<div class="section-heading">Choose an analysis domain to begin</div>',
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        _render_domain_card("frequency")
    with col_right:
        _render_domain_card("time")


def render_placeholder_page(domain_key: str) -> None:
    """Render a temporary page for the selected domain, with a back button."""
    meta = DOMAINS[domain_key]

    # Back navigation (secondary style) at the top of the page.
    if st.button("← Back to Home", key="back_home", type="secondary"):
        go_to("home")

    st.markdown(
        f'<span class="project-label">{PROJECT_LABEL}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h1 class="app-title">{meta["title"]}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="placeholder-panel">
            <h3>{meta['placeholder_title']}</h3>
            <p>{meta['placeholder_message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Small footer note shown on every page."""
    st.markdown(
        '<div class="app-footer">EDF R&D — Cable Fault Diagnostics · '
        "Research prototype</div>",
        unsafe_allow_html=True,
    )
