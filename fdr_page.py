"""
fdr_page.py
-----------
Frequency Domain Reflectometry (FDR) interface.

This page only collects user inputs, validates them, converts them into the
dataclasses/config expected by the existing simulation module, and renders the
results. The physical model is NOT reimplemented here — it is imported from
`fdr_to_distance.py` and reused unchanged.

Layout is split into small render functions so the page stays modular:
    render_fdr_page
        render_cable_configuration
        render_frequency_configuration
        render_fault_configuration
        validate_fdr_inputs
        render_fdr_results
"""

# Use a headless Matplotlib backend so figures render inside Streamlit without
# trying to open a GUI window. Must be set before pyplot is imported anywhere.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

# Reuse the existing simulation code (no duplication of the physical model).
# fdr_to_distance.py sits in this same folder, so a plain import is enough.
from fdr_to_distance import (
    CableParams,
    FaultParams,
    run_case,
    CONFIG,
    window_broadening,
    _local_rlgc,
)

from theme import PROJECT_LABEL

# Readable labels for the finite-length RLGC fault types.
FAULT_LABELS = {
    "R_increase": "R — Resistance",
    "L_increase": "L — Inductance",
    "G_increase": "G — Conductance",
    "C_increase": "C — Capacitance",
}

# Per-parameter display settings: (symbol, unit, scale to that unit, decimals).
_PARAM_DISPLAY = {
    "R_increase": ("R", "Ω/m", 1.0, 3),
    "L_increase": ("L", "µH/m", 1e6, 4),
    "G_increase": ("G", "µS/m", 1e6, 3),
    "C_increase": ("C", "pF/m", 1e12, 1),
}

# --- Mappings between UI labels and simulation identifiers -----------------
# Each selectable RLGC parameter maps to a finite-length "*_increase" fault
# type understood by _local_rlgc()/build_element_list() in the simulation.
FAULT_TYPE_MAP = {
    "R — Resistance": "R_increase",
    "L — Inductance": "L_increase",
    "G — Conductance": "G_increase",
    "C — Capacitance": "C_increase",
}
PARAM_SHORT = {
    "R — Resistance": "R",
    "L — Inductance": "L",
    "G — Conductance": "G",
    "C — Capacitance": "C",
}
# "Rectangular" must be mapped to the simulation's "rect" window key.
WINDOW_MAP = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Rectangular": "rect",
}

# --- Fixed healthy-cable model (NOT user-editable) -------------------------
# The healthy cable uses these distributed RLGC defaults from the simulation
# model. User faults locally multiply ONE of these along a chosen region.
DEFAULT_R = 20e-3       # Ω/m
DEFAULT_L = 0.25e-6     # H/m
DEFAULT_G = 0.7e-6      # S/m
DEFAULT_C = 100e-12     # F/m

PAGE_DESCRIPTION = (
    "The frequency-domain response shows how cable faults modify the reflection "
    "coefficient across the selected frequency range. The complex frequency "
    "response is then transformed into a distance-domain reflectogram to estimate "
    "the positions of discontinuities along the cable."
)


def render_cable_configuration() -> dict:
    """Collect only the user-relevant cable parameters (length + terminations).

    The healthy-cable RLGC values are fixed simulation defaults and are NOT
    user-editable here; they are attached to the returned dict automatically.
    """
    st.markdown(
        '<div class="section-heading">Section A — Cable configuration</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    total_length = c1.number_input(
        "Cable length [m]", min_value=0.0, value=1.0, step=0.1,
        format="%.3f", key="cable_length",
        help="Total physical length of the cable under test.",
    )
    Zs = c2.number_input(
        "Source impedance Zs [Ω]", min_value=0.0, value=50.0, step=1.0,
        key="cable_zs",
        help="Internal impedance of the measurement source / reference impedance "
             "used to define S11.",
    )
    ZL = c3.number_input(
        "Load impedance ZL [Ω]", min_value=0.0, value=50.0, step=1.0,
        key="cable_zl",
        help="Impedance terminating the far end of the cable.",
    )

    # Compact, read-only display of the fixed assumptions.
    with st.expander("Healthy cable model assumptions"):
        st.markdown(
            "These are **fixed simulation defaults**, not user inputs:\n\n"
            f"- R = {DEFAULT_R * 1e3:.3g} mΩ/m  (0.020 Ω/m)\n"
            f"- L = {DEFAULT_L * 1e6:.2f} µH/m\n"
            f"- G = {DEFAULT_G * 1e6:.2f} µS/m\n"
            f"- C = {DEFAULT_C * 1e12:.0f} pF/m\n\n"
            "Faults are applied locally by multiplying one of these parameters "
            "inside the selected fault region."
        )

    # Fixed RLGC defaults are merged with the user-editable terminations.
    return {
        "total_length": float(total_length),
        "Zs": float(Zs),
        "ZL": float(ZL),
        "R": DEFAULT_R,   # Ω/m  (fixed)
        "L": DEFAULT_L,   # H/m  (fixed)
        "G": DEFAULT_G,   # S/m  (fixed)
        "C": DEFAULT_C,   # F/m  (fixed)
    }


# =========================================================================
# Section B — Frequency sweep configuration
# =========================================================================
def render_frequency_configuration() -> dict:
    """Collect the three user-facing sweep parameters.

    The processing parameters WINDOW_TYPE, ZERO_PAD_FACTOR and REFERENCE_FREQUENCY
    are chosen automatically by the application (not exposed to the user). The
    detection settings keep their existing application defaults from CONFIG.
    """
    st.markdown(
        '<div class="section-heading">Section B — Frequency sweep configuration</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "FDR measures the complex reflection coefficient S11(f) over a range of "
        "frequencies. The selected bandwidth influences the ability to distinguish "
        "nearby reflections, while the number of frequency samples controls the "
        "frequency spacing."
    )

    # The three remaining inputs sit on one row when the screen is wide enough.
    c1, c2, c3 = st.columns(3)
    f_start_mhz = c1.number_input(
        "Start frequency [MHz]", min_value=0.0, value=1.0, step=1.0,
        format="%.3f", key="freq_start",
        help="Lowest frequency included in the stepped-frequency sweep.",
    )
    f_stop_mhz = c2.number_input(
        "Stop frequency [MHz]", min_value=0.0, value=1000.0, step=10.0,
        format="%.3f", key="freq_stop",
        help="Highest frequency included in the sweep. The difference between the stop "
             "and start frequencies defines the measurement bandwidth.",
    )
    n_freq = c3.number_input(
        "Number of frequency samples", min_value=2, value=5000, step=100,
        key="freq_n",
        help="Number of equally spaced frequencies used in the sweep. More samples "
             "provide finer frequency spacing but increase computation time.",
    )

    # Automatic processing defaults selected by the application (not user inputs):
    #   * Hann window — robust default for sidelobe reduction,
    #   * zero-padding ×4 — smoother display interpolation,
    #   * reference frequency = centre of the band — Panel 3 ground truth only.
    return {
        "F_START": float(f_start_mhz) * 1e6,
        "F_STOP": float(f_stop_mhz) * 1e6,
        "N_FREQ": int(n_freq),
        "WINDOW_TYPE": "hann",
        "ZERO_PAD_FACTOR": 4,
        "REFERENCE_FREQUENCY": (float(f_start_mhz) + float(f_stop_mhz)) * 0.5e6,
    }


def render_fault_configuration(number_of_faults: int, cable_length: float) -> list:
    """Render one card per fault and return the raw per-fault input values."""
    faults = []
    for i in range(number_of_faults):
        with st.container(border=True):
            st.markdown(f"**Fault {i + 1}**")
            c1, c2, c3, c4 = st.columns(4)
            # Sensible spread of default positions across the cable.
            default_pos = round(min(0.25 * (i + 1), max(cable_length - 0.05, 0.05)), 3)
            position = c1.number_input(
                "Position [m]", min_value=0.0, value=default_pos, step=0.01,
                format="%.3f", key=f"fault_position_{i}",
            )
            length = c2.number_input(
                "Length [m]", min_value=0.0, value=0.02, step=0.01,
                format="%.3f", key=f"fault_length_{i}",
            )
            parameter = c3.selectbox(
                "Modified parameter", list(FAULT_TYPE_MAP.keys()),
                key=f"fault_parameter_{i}",
            )
            factor = c4.number_input(
                "Factor (×)", min_value=0.0, value=2.0, step=0.5,
                format="%.2f", key=f"fault_factor_{i}",
                help="A factor of 5 multiplies the selected local RLGC "
                     "parameter by 5 (not a percentage).",
            )
            faults.append({
                "index": i,
                "position": float(position),
                "length": float(length),
                "param": parameter,
                "factor": float(factor),
            })
    return faults


def validate_fdr_inputs(cable: dict, faults: list, config: dict):
    """Return (errors, warnings). Errors block the run; warnings do not."""
    errors, warnings = [], []
    Ltot = cable["total_length"]

    # --- Cable ---
    if Ltot <= 0:
        errors.append("Cable length must be strictly positive.")
    for name in ("R", "L", "G", "C"):
        if cable[name] < 0:
            errors.append(f"Cable parameter {name} cannot be negative.")

    # --- Frequency sweep ---
    f0, f1 = config["F_START"], config["F_STOP"]
    if not (f0 < f1):
        errors.append("Start frequency must be strictly below the stop frequency.")
    if config["N_FREQ"] < 16:
        errors.append("Number of frequency samples is too small (use at least 16).")
    fref = config["REFERENCE_FREQUENCY"]
    if not (f0 <= fref <= f1):
        errors.append("Reference frequency must lie within the selected frequency range.")

    # --- Faults ---
    for f in faults:
        n = f["index"] + 1
        if f["position"] < 0:
            errors.append(f"Fault {n}: position cannot be negative.")
        if f["position"] > Ltot:
            errors.append(f"Fault {n}: position ({f['position']:.3f} m) exceeds the "
                          f"cable length ({Ltot:.3f} m).")
        if f["length"] <= 0:
            errors.append(f"Fault {n}: spatial length must be strictly positive.")
        if f["position"] + f["length"] > Ltot:
            errors.append(f"Fault {n}: position + length ({f['position'] + f['length']:.3f} m) "
                          f"exceeds the cable length ({Ltot:.3f} m).")
        if f["factor"] <= 0:
            errors.append(f"Fault {n}: multiplication factor must be strictly positive.")

    # --- Resolution / unambiguous range (needs a valid vp) ---
    if cable["L"] > 0 and cable["C"] > 0 and f1 > f0 and config["N_FREQ"] >= 2:
        vp = 1.0 / np.sqrt(cable["L"] * cable["C"])      # lossless approximation
        df = (f1 - f0) / (config["N_FREQ"] - 1)
        x_max = vp / (2.0 * df)
        if x_max < Ltot:
            errors.append(
                f"The frequency step yields a maximum unambiguous range of "
                f"{x_max:.3f} m, smaller than the cable length ({Ltot:.3f} m). "
                f"Increase the number of samples or reduce the bandwidth."
            )

        B = f1 - f0
        dx = vp / (2.0 * B)
        dx_eff = dx * window_broadening(config["WINDOW_TYPE"])

        ordered = sorted(faults, key=lambda d: d["position"])
        for a, b in zip(ordered, ordered[1:]):
            if a["position"] + a["length"] > b["position"]:
                warnings.append(
                    f"Faults {a['index'] + 1} and {b['index'] + 1} overlap spatially."
                )
            sep = b["position"] - a["position"]
            if 0 <= sep < dx_eff:
                warnings.append(
                    f"Faults {a['index'] + 1} and {b['index'] + 1} are separated by "
                    f"{sep * 100:.1f} cm, below the effective resolution "
                    f"{dx_eff * 100:.1f} cm; their peaks may merge."
                )

    if config["N_FREQ"] > 20000:
        warnings.append("A very high number of frequency samples may slow the simulation.")

    return errors, warnings


# =========================================================================
# Run handler
# =========================================================================
def _run_simulation(cable: dict, config: dict, fault_inputs: list) -> None:
    """Validate, build objects, and run the existing simulation."""
    # Build FaultParams from the UI rows.
    fault_objs = []
    for f in fault_inputs:
        fault_type = FAULT_TYPE_MAP[f["param"]]
        short = PARAM_SHORT[f["param"]]
        label = f"Fault {f['index'] + 1} — {short} ×{f['factor']:g}"
        fault_objs.append(FaultParams(
            position=f["position"],
            length=f["length"],
            fault_type=fault_type,
            factor=f["factor"],
            label=label,
        ))

    # Validate before touching the simulation.
    errors, warnings = validate_fdr_inputs(cable, fault_inputs, config)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    # Build the cable object.
    cable_params = CableParams(
        R=cable["R"], L=cable["L"], G=cable["G"], C=cable["C"],
        total_length=cable["total_length"], Zs=cable["Zs"], ZL=cable["ZL"],
    )

    # Copy the global defaults into a fresh dict, then override (never mutate CONFIG).
    # Only the user-facing sweep params and the automatic processing params are
    # overridden; the detection settings keep their existing CONFIG defaults.
    cfg = dict(CONFIG)
    for key in ("F_START", "F_STOP", "N_FREQ",
                "WINDOW_TYPE", "ZERO_PAD_FACTOR", "REFERENCE_FREQUENCY"):
        cfg[key] = config[key]

    # Free the previous figure to prevent Matplotlib memory accumulation.
    prev = st.session_state.get("fdr_results")
    if prev and prev.get("figure") is not None:
        plt.close(prev["figure"])

    with st.spinner("Running the frequency-domain simulation..."):
        try:
            results = run_case(cable_params, fault_objs, cfg, show_report=False)
            st.session_state.fdr_results = results
            # Keep the inputs needed to build the ground-truth fault summary.
            st.session_state.fdr_cable = cable_params
            st.session_state.fdr_faults = fault_objs
            st.session_state.fdr_ref_freq = cfg["REFERENCE_FREQUENCY"]
        except ValueError as exc:
            st.error(f"Invalid simulation configuration: {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the whole app
            st.error(f"The simulation could not be completed: {exc}")


# =========================================================================
# Results
# =========================================================================
def _healthy_param_value(cable: CableParams, fault_type: str) -> float:
    """Healthy distributed value of the parameter targeted by `fault_type`."""
    return {
        "R_increase": cable.R,
        "L_increase": cable.L,
        "G_increase": cable.G,
        "C_increase": cable.C,
    }[fault_type]


def build_fault_summary(cable: CableParams, faults: list, detected: list,
                        reference_frequency_hz: float) -> pd.DataFrame:
    """Build the per-fault ground-truth summary table (one row per fault).

    This is SIMULATION GROUND TRUTH: it reports the faults deliberately inserted
    into the model (true positions, the locally modified RLGC value, and the
    resulting ΔZc), plus the nearest detected peak for reference. ΔZc is computed
    from the same _local_rlgc() values the simulation uses.
    """
    omega = 2.0 * np.pi * reference_frequency_hz

    def zc(R, L, G, C):
        return abs(np.sqrt((R + 1j * omega * L) / (G + 1j * omega * C)))

    Zc_healthy = zc(cable.R, cable.L, cable.G, cable.C)

    rows = []
    for i, f in enumerate(faults):
        _, unit, scale, dec = _PARAM_DISPLAY[f.fault_type]
        healthy_value = _healthy_param_value(cable, f.fault_type)
        faulty_value = healthy_value * f.factor

        # Local RLGC inside the fault, taken from the exact simulation logic.
        R_l, L_l, G_l, C_l = _local_rlgc(cable, f)
        Zc_faulty = zc(R_l, L_l, G_l, C_l)
        delta_zc_percent = 100.0 * (Zc_faulty - Zc_healthy) / Zc_healthy

        start = f.position
        end = f.position + f.length

        # Nearest detected peak to the fault's START position (see tooltip).
        if detected:
            nearest = min(
                detected,
                key=lambda d: abs(d["estimated_position_m"] - f.position),
            )
            est = nearest["estimated_position_m"]
            nearest_str = f"{est:.4f}"
            error_str = f"{est - f.position:+.4f}"
        else:
            nearest_str = "Not detected"
            error_str = "Not detected"

        rows.append({
            "Fault": i + 1,
            "True region [m]": f"{start:.3f}–{end:.3f}",
            "Length [m]": f"{f.length:.3f}",
            "Modified parameter": FAULT_LABELS[f.fault_type],
            "Factor": f"×{f.factor:g}",
            "Healthy value": f"{healthy_value * scale:.{dec}f} {unit}",
            "Local faulty value": f"{faulty_value * scale:.{dec}f} {unit}",
            "Ground-truth ΔZc [%]": f"{delta_zc_percent:+.2f}",
            "Nearest peak [m]": nearest_str,
            "Error [m]": error_str,
        })

    return pd.DataFrame(rows)


def render_fault_summary(cable: CableParams, faults: list, results: dict,
                         reference_frequency_hz: float) -> None:
    """Render the simulated-fault ground-truth summary table (+ optional cards)."""
    if not faults:
        st.info("No faults were configured for this simulation.")
        return

    df = build_fault_summary(cable, faults, results["detected"],
                             reference_frequency_hz)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "Modified parameter": st.column_config.TextColumn(
                "Modified parameter",
                help="The RLGC parameter multiplied locally inside the fault region.",
            ),
            "Ground-truth ΔZc [%]": st.column_config.TextColumn(
                "Ground-truth ΔZc [%]",
                help="Simulated local change in characteristic-impedance magnitude at "
                     "the reference frequency — ground truth from the known RLGC, NOT "
                     "reconstructed from S11. Note: multiplying an RLGC parameter by a "
                     "factor does not produce the same percentage change in Zc.",
            ),
            "Nearest peak [m]": st.column_config.TextColumn(
                "Nearest peak [m]",
                help="Detected peak closest to the fault's START position. It is not "
                     "guaranteed to correspond to this fault.",
            ),
            "Error [m]": st.column_config.TextColumn(
                "Error [m]",
                help="Nearest detected position minus the fault's true start position "
                     "(signed: + means detected beyond the fault).",
            ),
        },
    )
    st.caption(
        "**ΔZc is the simulated local change in characteristic-impedance magnitude at the "
        "automatically selected reference frequency. It is ground-truth information derived "
        "from the known RLGC parameters, not a direct measurement reconstructed from "
        "S11.**  ΔZc evaluated at "
        f"{reference_frequency_hz / 1e6:.1f} MHz."
    )

    # Optional compact cards when exactly one fault is simulated.
    if len(faults) == 1:
        f = faults[0]
        symbol, _, _, _ = _PARAM_DISPLAY[f.fault_type]
        row = df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Fault location", f"{row['True region [m]']} m")
        c2.metric("Modified parameter", f"{symbol} ×{f.factor:g}")
        c3.metric("Ground-truth ΔZc", f"{row['Ground-truth ΔZc [%]']} %")


def render_fdr_results(results: dict, cable: CableParams, faults: list,
                       reference_frequency_hz: float) -> None:
    """Render results: figure first (primary output), then summary, then table, then notes."""
    metrics = results["metrics"]
    detected = results["detected"]
    vp = results["vp"]

    # --- 1. Three-panel figure (the primary result) ---
    st.subheader("FDR simulation results")
    st.pyplot(results["figure"], use_container_width=True)
    st.caption(
        "Panel 1: Healthy, faulty, and baseline-subtracted frequency-domain reflection "
        "responses. Panel 2: Distance-domain reflection strength and detected peak "
        "locations. Panel 3: Simulation ground truth for the local characteristic-impedance "
        "change. Panel 3 is not reconstructed from S11."
    )

    # --- 2. Simulation summary (three cards only) ---
    st.markdown('<div class="section-heading">Simulation summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Propagation velocity", f"{vp:.3e} m/s",
        help="Calculated from the fixed healthy-cable parameters: vp ≈ 1 / √(L·C).",
    )
    c2.metric(
        "Frequency bandwidth", f"{metrics['B'] / 1e6:,.1f} MHz",
        help="B = f_stop − f_start.",
    )
    c3.metric("Detected reflections", f"{len(detected)}")

    # --- 3. Simulated fault summary (ground truth) ---
    render_fault_summary(cable, faults, results, reference_frequency_hz)

def render_fdr_page() -> None:
    """Top-level FDR page: header, input sections, run button, and results."""
    # Back navigation (same mechanism as the rest of the app).
    if st.button("← Back to Home", key="fdr_back", type="secondary"):
        st.session_state.page = "home"
        st.rerun()

    # Header + description, matching the welcome page's visual identity.
    st.markdown(f'<span class="project-label">{PROJECT_LABEL}</span>',
                unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">FDR — Frequency Domain Reflectometry</h1>',
                unsafe_allow_html=True)
    st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="fdr-description">{PAGE_DESCRIPTION}</p>',
                unsafe_allow_html=True)

    # Input sections.
    cable = render_cable_configuration()
    config = render_frequency_configuration()

    st.markdown('<div class="section-heading">Section C — Fault configuration</div>',
                unsafe_allow_html=True)
    number_of_faults = st.number_input(
        "Number of faults", min_value=1, max_value=6, value=2, step=1,
        key="n_faults",
    )
    fault_inputs = render_fault_configuration(int(number_of_faults),
                                              cable["total_length"])

    # Run button (styled EDF-orange via the global CSS).
    st.markdown("")
    if st.button("Run FDR Simulation", key="run_fdr"):
        _run_simulation(cable, config, fault_inputs)

    # Results persist in session_state so they survive unrelated reruns.
    if st.session_state.get("fdr_results") is not None:
        st.markdown('<div class="accent-rule"></div>', unsafe_allow_html=True)
        render_fdr_results(
            st.session_state.fdr_results,
            st.session_state.fdr_cable,
            st.session_state.fdr_faults,
            st.session_state.fdr_ref_freq,
        )
