"""
time_domain.py
--------------
TDR (Time Domain Reflectometry) simulation page for the Streamlit app.

Rendered by app.py when st.session_state.page == "time".
"""

import io
import contextlib

import numpy as np
import pandas as pd

# Headless Matplotlib backend so figures render inside Streamlit (matches
# fdr_page.py). Must be set before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

from theme import COLORS, PROJECT_LABEL
from components import go_to
from second_order_streamlit import (
    DEFAULT_R, DEFAULT_L, DEFAULT_G, DEFAULT_C,
    build_sections, generate_signal, run_simulation,
    find_reflections_method_2, find_reflections_method_3, find_fault_size,
)

_PARAM_LABELS = {
    "R": f"R  (default {DEFAULT_R:.2e} Ω/m)",
    "L": f"L  (default {DEFAULT_L:.2e} H/m)",
    "G": f"G  (default {DEFAULT_G:.2e} S/m)",
    "C": f"C  (default {DEFAULT_C:.2e} F/m)",
}


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _signal_preview_fig(signal_type, f_max, V_amplitude):
    """Return a matplotlib Figure showing the chosen input signal."""
    sigma_t = 1.0 / (2.0 * np.pi * f_max)

    if signal_type == "Gaussian":
        t0     = 6.0 * sigma_t
        t_plot = np.linspace(0, t0 + 4.0 * sigma_t, 800)
        vs     = V_amplitude * np.exp(-((t_plot - t0) ** 2) / (2.0 * sigma_t ** 2))
    else:
        T      = 1.0 / f_max
        t_plot = np.linspace(0, 3.0 * T, 800)
        vs     = V_amplitude * np.sin(2.0 * np.pi * f_max * t_plot)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["surface"])
    ax.plot(t_plot * 1e9, vs, color=COLORS["edf_blue"], lw=1.8)
    ax.axhline(0, color=COLORS["border"], lw=0.8)
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.set_title("Input signal preview", fontsize=11, color=COLORS["text"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def _animate_propagation(sim, n_frames=180):
    """
    Stream a live animation of V(x) propagating and V(t) building up at x=0.

    Renders into a single st.empty() placeholder by encoding each frame as a
    PNG in memory — faster than st.pyplot() for rapid updates.
    """
    V, x, t_arr = sim["V"], sim["x"], sim["t_arr"]
    Nt, sections = sim["Nt"], sim["sections"]

    fault_idx      = 1 if len(sections) >= 2 else 0
    f_start, f_end = sections[fault_idx][0], sections[fault_idx][1]

    # Pre-compute fixed axis limits so axes don't jump between frames
    v_ymin  = np.min(V) - 0.3
    v_ymax  = np.max(V) + 0.3
    v0_ymin = np.min(V[:, 0]) - 0.3
    v0_ymax = np.max(V[:, 0]) + 0.3

    frame_indices = np.linspace(0, Nt - 1, min(n_frames, Nt), dtype=int)

    placeholder = st.empty()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 5))
    fig.patch.set_facecolor(COLORS["background"])

    for frame in frame_indices:
        ax_top.clear()
        ax_bot.clear()

        # Top panel: spatial voltage snapshot
        ax_top.set_facecolor(COLORS["surface"])
        ax_top.plot(x, V[frame, :], color=COLORS["edf_blue"], lw=1.6)
        ax_top.axvspan(f_start, f_end, alpha=0.12,
                       color=COLORS["edf_orange"], label="Fault region")
        ax_top.set_xlim(x[0], x[-1])
        ax_top.set_ylim(v_ymin, v_ymax)
        ax_top.set_xlabel("Position (m)", fontsize=10)
        ax_top.set_ylabel("Voltage (V)", fontsize=10)
        ax_top.set_title(
            f"Signal propagation  —  t = {t_arr[frame] * 1e9:.3f} ns",
            fontsize=11, color=COLORS["text"],
        )
        ax_top.legend(fontsize=9, loc="upper right")
        for sp in ("top", "right"):
            ax_top.spines[sp].set_visible(False)
        ax_top.tick_params(labelsize=9)

        # Bottom panel: V(t) at x=0 building up over time
        ax_bot.set_facecolor(COLORS["surface"])
        ax_bot.plot(t_arr[:frame + 1] * 1e9, V[:frame + 1, 0],
                    color=COLORS["edf_orange"], lw=1.4)
        ax_bot.set_xlim(0, t_arr[-1] * 1e9)
        ax_bot.set_ylim(v0_ymin, v0_ymax)
        ax_bot.set_xlabel("Time (ns)", fontsize=10)
        ax_bot.set_ylabel("Voltage at x = 0  (V)", fontsize=10)
        ax_bot.set_title(
            "Signal seen at source node (x = 0)",
            fontsize=11, color=COLORS["text"],
        )
        for sp in ("top", "right"):
            ax_bot.spines[sp].set_visible(False)
        ax_bot.tick_params(labelsize=9)

        fig.tight_layout(pad=2.0)

        # Encode to PNG in memory — faster than placeholder.pyplot()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=85, bbox_inches="tight",
                    facecolor=COLORS["background"])
        buf.seek(0)
        placeholder.image(buf, use_container_width=True)

    plt.close(fig)


def _tdr_fig(sim):
    """Static TDR trace: V(t) at x=0 for the full simulation."""
    V, t_arr = sim["V"], sim["t_arr"]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["surface"])
    ax.plot(t_arr * 1e9, V[:, 0], color=COLORS["edf_blue"], lw=1.4)
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Voltage at x = 0  (V)", fontsize=10)
    ax.set_title("TDR signal at source node (x = 0)", fontsize=11, color=COLORS["text"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ── Page renderer ─────────────────────────────────────────────────────────────

def render_time_domain_page():
    # Back navigation (uses the same "home" state as the rest of the app).
    if st.button("← Back to Home", key="back_home_td", type="secondary"):
        st.session_state.pop("tdr_sim", None)
        go_to("home")

    st.markdown(f'<span class="project-label">{PROJECT_LABEL}</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Time Domain Reflectometry</h1>', unsafe_allow_html=True)
    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)

    # ── Section 1: Input signal ───────────────────────────────────────────────
    st.markdown('<div class="section-heading">1 · Input signal</div>', unsafe_allow_html=True)

    signal_type = st.radio(
        "Signal type",
        ["Gaussian", "Sinusoidal"],
        horizontal=True,
        label_visibility="collapsed",
        key="td_signal_type",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        f_max_ghz = st.number_input(
            "f_max (GHz)", min_value=0.1, max_value=100.0, value=5.0, step=0.5,
            key="td_fmax",
        )
        f_max = f_max_ghz * 1e9
    with c2:
        V_amplitude = st.number_input(
            "Amplitude (V)", min_value=0.1, max_value=1000.0, value=10.0, step=1.0,
            key="td_vamp",
        )
    with c3:
        PPW = st.number_input(
            "PPW",
            min_value=10, max_value=50, value=10, step=5,
            help="Points per wavelength — controls spatial resolution (Gaussian only)",
            disabled=(signal_type == "Sinusoidal"),
            key="td_ppw",
        )

    st.pyplot(_signal_preview_fig(signal_type, f_max, V_amplitude))

    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)

    # ── Section 2: Cable & fault ──────────────────────────────────────────────
    st.markdown('<div class="section-heading">2 · Cable & fault</div>', unsafe_allow_html=True)

    col_len, col_fault = st.columns([1, 2])
    with col_len:
        st.markdown("**Cable length**")
        cable_length = st.number_input(
            "In m", min_value=0.1, max_value=100.0, value=0.5, step=0.1,
            key="td_length",
        )

    with col_fault:
        st.markdown("**Fault region**")
        fa, fb = st.columns(2)
        default_start = round(min(0.2, cable_length * 0.4), 2)
        default_end   = round(min(0.3, cable_length * 0.6), 2)
        with fa:
            fault_start = st.number_input(
                "From (m)", min_value=0.0, max_value=float(cable_length) - 0.01,
                value=default_start, step=0.05, key="td_fault_start",
            )
        with fb:
            fault_end = st.number_input(
                "To (m)",
                min_value=float(fault_start) + 0.01, max_value=float(cable_length),
                value=max(float(fault_start) + 0.01, default_end),
                step=0.05, key="td_fault_end",
            )

    col_param, col_mult = st.columns(2)
    with col_param:
        fault_param = st.selectbox(
            "Faulted RLGC parameter",
            list(_PARAM_LABELS.keys()),
            format_func=lambda k: _PARAM_LABELS[k],
            key="td_fault_param",
        )
    with col_mult:
        fault_multiplier = st.number_input(
            "Fault multiplier",
            min_value=0.1, max_value=10000.0, value=10.0, step=0.1,
            help=f"The fault section uses {fault_param} × this value",
            key="td_fault_mult",
        )

    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)

    # ── Section 3: Run ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">3 · Simulation</div>', unsafe_allow_html=True)

    if fault_start >= fault_end:
        st.warning("Fault start must be less than fault end.")
        return

    if st.button("▶  Run simulation", type="primary", key="td_run"):
        secs = build_sections(
            cable_length, fault_start, fault_end, fault_param, fault_multiplier
        )
        with st.spinner("Running FDTD simulation…"):
            sim = run_simulation(
                secs, cable_length, f_max, int(PPW), V_amplitude,
                signal_type=signal_type.lower(),
                verbose=False,
            )
        st.session_state["tdr_sim"] = sim
        st.session_state["tdr_animated"] = False  # trigger animation on this render

    sim = st.session_state.get("tdr_sim")
    if sim is not None:
        # Play the animation once (on the render immediately after clicking Run).
        # On subsequent reruns (e.g. expanding the analysis expander) skip straight
        # to the static plot so the animation doesn't replay.
        if not st.session_state.get("tdr_animated", True):
            _animate_propagation(sim)
            st.session_state["tdr_animated"] = True

        st.pyplot(_tdr_fig(sim))

        st.markdown("#### Simulated fault summary")
        text_buf = io.StringIO()
        with contextlib.redirect_stdout(text_buf):
            lc_uniform = all(
                L == sim["L_ref"] and C == sim["C_ref"]
                for (_, _, _, L, _, C) in sim["sections"]
            )
            if lc_uniform:
                try:
                    rows = find_reflections_method_2(sim)
                except TypeError:
                    rows = find_reflections_method_3(sim)
            else:
                rows = find_fault_size(sim)

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No reflections detected above threshold.")
