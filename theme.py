"""
theme.py
--------
Centralized design tokens (colors, fonts, spacing) for the application.

Keeping these values in one place means the whole visual identity can be
adjusted from a single file, and future pages (FDR / TDR) stay consistent
with the welcome page.
"""

# --- EDF-inspired color palette -------------------------------------------
# Blue + orange are EDF's signature colors; the rest are neutral support tones
# chosen to keep the interface scientific and clean rather than "marketing".
COLORS = {
    "edf_blue": "#0A3D91",       # primary brand blue
    "edf_blue_dark": "#062A66",  # darker blue for hover / depth
    "edf_orange": "#FF6B00",     # accent / call-to-action
    "edf_orange_dark": "#E25E00",
    "white": "#FFFFFF",
    "background": "#F5F7FA",     # light neutral page background
    "surface": "#FFFFFF",        # cards / panels
    "border": "#E2E8F0",         # subtle separators
    "text": "#1A2233",           # primary text
    "text_muted": "#5A6678",     # secondary text
}

# --- Project identity ------------------------------------------------------
PROJECT_LABEL = "EDF R&D · Cable Diagnostics"
APP_TITLE = "Cable Fault Simulation and Localization Tool"
APP_SUBTITLE = (
    "Simulate, visualize, and locate faults in electrical cables using "
    "time-domain and frequency-domain reflectometry methods."
)
