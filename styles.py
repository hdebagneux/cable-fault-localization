"""
styles.py
---------
All custom CSS lives here so styling stays separated from layout logic.

`apply_custom_css()` injects a single <style> block built from the design
tokens in theme.py. This is the only place raw CSS should appear.
"""

import streamlit as st

from theme import COLORS


def apply_custom_css() -> None:
    """Inject the global stylesheet. Call once at the top of the app."""
    st.markdown(
        f"""
        <style>
            /* ---- Global layout ---- */
            .stApp {{
                background-color: {COLORS['background']};
            }}
            .block-container {{
                max-width: 1100px;
                padding-top: 2.5rem;
                padding-bottom: 3rem;
            }}

            /* ---- Typography ---- */
            html, body, [class*="css"] {{
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                color: {COLORS['text']};
            }}

            /* ---- Project label / header ---- */
            .project-label {{
                display: inline-block;
                background-color: {COLORS['edf_blue']};
                color: {COLORS['white']};
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                padding: 0.35rem 0.9rem;
                border-radius: 4px;
                margin-bottom: 1.2rem;
            }}
            .app-title {{
                font-size: 2.5rem;
                font-weight: 700;
                line-height: 1.15;
                margin: 0 0 0.6rem 0;
                color: {COLORS['text']};
            }}
            .app-title .accent {{
                color: {COLORS['edf_orange']};
            }}
            .app-subtitle {{
                font-size: 1.08rem;
                color: {COLORS['text_muted']};
                max-width: 720px;
                line-height: 1.55;
                margin-bottom: 0.5rem;
            }}

            /* ---- Section divider with thin accent rule ---- */
            .accent-rule {{
                height: 4px;
                width: 64px;
                background: {COLORS['edf_orange']};
                border-radius: 2px;
                margin: 1.4rem 0 2.2rem 0;
            }}
            .section-heading {{
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 1.4rem;
                color: {COLORS['text']};
            }}

            /* ---- Domain cards ---- */
            .domain-card {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-top: 4px solid {COLORS['edf_blue']};
                border-radius: 12px;
                padding: 1.8rem 1.6rem 1.4rem 1.6rem;
                height: 100%;
                box-shadow: 0 2px 6px rgba(10, 61, 145, 0.06);
                transition: box-shadow 0.2s ease, transform 0.2s ease;
            }}
            .domain-card:hover {{
                box-shadow: 0 6px 18px rgba(10, 61, 145, 0.12);
                transform: translateY(-2px);
            }}
            .domain-card .icon {{
                font-size: 2.4rem;
                line-height: 1;
            }}
            .domain-card .card-title {{
                font-size: 1.35rem;
                font-weight: 700;
                margin: 0.8rem 0 0.4rem 0;
                color: {COLORS['edf_blue']};
            }}
            .domain-card .card-desc {{
                font-size: 0.96rem;
                color: {COLORS['text_muted']};
                line-height: 1.55;
                min-height: 5.5em;
            }}

            /* ---- Buttons ---- */
            .stButton > button {{
                background-color: {COLORS['edf_orange']};
                color: {COLORS['white']};
                border: none;
                border-radius: 8px;
                padding: 0.6rem 1.2rem;
                font-size: 0.98rem;
                font-weight: 600;
                width: 100%;
                transition: background-color 0.2s ease;
            }}
            .stButton > button:hover {{
                background-color: {COLORS['edf_orange_dark']};
                color: {COLORS['white']};
            }}
            .stButton > button:focus {{
                box-shadow: 0 0 0 3px rgba(255, 107, 0, 0.35);
                color: {COLORS['white']};
            }}

            /* Secondary (back) buttons use a quiet blue outline style */
            .stButton > button[kind="secondary"] {{
                background-color: transparent;
                color: {COLORS['edf_blue']};
                border: 1px solid {COLORS['edf_blue']};
            }}
            .stButton > button[kind="secondary"]:hover {{
                background-color: {COLORS['edf_blue']};
                color: {COLORS['white']};
            }}

            /* ---- Placeholder / info panel ---- */
            .placeholder-panel {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-left: 4px solid {COLORS['edf_orange']};
                border-radius: 10px;
                padding: 1.6rem 1.8rem;
                margin-top: 0.5rem;
            }}
            .placeholder-panel h3 {{
                margin-top: 0;
                color: {COLORS['edf_blue']};
            }}
            .placeholder-panel p {{
                color: {COLORS['text_muted']};
                line-height: 1.6;
                margin-bottom: 0;
            }}

            /* ---- FDR page description under the title ---- */
            .fdr-description {{
                font-size: 1.0rem;
                color: {COLORS['text_muted']};
                max-width: 880px;
                line-height: 1.6;
                margin-bottom: 1.6rem;
            }}

            /* ---- Ground-truth callout (Panel 3 disclaimer) ---- */
            .ground-truth-note {{
                background: #FFF4E8;
                border-left: 4px solid {COLORS['edf_orange']};
                border-radius: 8px;
                padding: 0.85rem 1.1rem;
                font-size: 0.9rem;
                color: {COLORS['text']};
                line-height: 1.5;
                margin: 0.3rem 0 1.1rem 0;
            }}

            /* ---- Metric cards (Streamlit st.metric) ---- */
            div[data-testid="stMetric"] {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 0.9rem 1rem;
            }}
            div[data-testid="stMetricValue"] {{
                color: {COLORS['edf_blue']};
            }}

            /* ---- Footer ---- */
            .app-footer {{
                margin-top: 3rem;
                padding-top: 1.2rem;
                border-top: 1px solid {COLORS['border']};
                font-size: 0.8rem;
                color: {COLORS['text_muted']};
                text-align: center;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
