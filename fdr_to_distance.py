
"""
  A. S11(f)              -- complex source-port FREQUENCY-DOMAIN MEASUREMENT.
                           Measurable with a VNA / stepped-frequency reflectometer.
  B. r(x) = F^-1{S11(f)} -- DISTANCE-DOMAIN REFLECTOGRAM (reflection / discontinuity
                           STRENGTH vs distance). NOT the impedance value at x.
  C. dZc(x) [%]          -- GROUND-TRUTH local characteristic-impedance change.
                           Known ONLY because the simulator knows R,L,G,C(x).
                           NOT reconstructed from S11. Labelled "Simulation ground truth".


FREQUENCY -> DISTANCE: ASSUMPTIONS (see `reflectogram_from_S11`)

  * The sampled band {S11(f_k)} is treated as a COMPLEX baseband frequency
    response. Its inverse DFT yields the baseband-equivalent impulse response;
    a discontinuity at delay tau = 2 d / vp appears as a peak at that delay.
    The nonzero start frequency only multiplies each peak by a constant phase
    exp(-j 4 pi F_START d / vp); it does NOT move the peak, and we display the
    magnitude/envelope, so it is harmless for LOCALISATION.
  * vp ~ 1 / sqrt(L*C)  (lossless approximation, thesis eq. 1.24).
  * x = vp * t / 2.
  * Delta_x  ~ vp / (2 B)     spatial resolution     (B = f_max - f_min)
  * x_max    ~ vp / (2 Delta_f) maximum unambiguous one-way distance.

  Processing rules enforced below:
    - the COMPLEX phase of S11 is preserved (using only |S11| would destroy
      all distance information);
    - a selectable window (Hann by default) reduces sidelobes but BROADENS peaks;
    - zero-padding INTERPOLATES/smooths the displayed curve but does NOT improve
      the true resolution Delta_x (B is unchanged);
    - the main fault curve uses the COMPLEX baseline difference
      dS11 = S11_faulty - S11_healthy, which coherently cancels reflections shared
      by both systems (NEVER subtract magnitudes).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

import matplotlib.pyplot as plt

# scipy is optional; we provide a NumPy fallback for peak finding.
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class CableParams:
    """Per-unit-length (distributed) RLGC parameters and terminations."""
    R: float            # Ohm/m
    L: float            # H/m
    G: float            # S/m
    C: float            # F/m
    total_length: float  # m
    Zs: float = 50.0     # source impedance (Ohm)
    ZL: float = 50.0     # load impedance (Ohm)


@dataclass
class FaultParams:
   
    position: float                       # start of fault region (m)
    length: float = 0.02                  # spatial extent (m); ignored for hard faults
    fault_type: str = "impedance_change"
    delta_zc_percent: float = 50.0        # used by "impedance_change"
    factor: float = 5.0                   # multiplier for R/C/G_increase
    zc_mechanism: str = "L"               # "L" (default) or "C" for impedance_change
    label: str = ""

    def __post_init__(self):
        if self.fault_type == "impedance_change":
            ratio = 1.0 + self.delta_zc_percent / 100.0
            if ratio <= 0.0:
                raise ValueError(
                    f"Invalid delta_zc_percent={self.delta_zc_percent}: requires "
                    f"(1 + dZc/100) > 0, got {ratio}."
                )



@dataclass
class Excitation:
    """
    Stepped-frequency excitation: a uniform grid of CW tones measured one after
    another. Each tone is an independent steady-state sinusoidal experiment whose
    complex reflected/incident ratio is S11(f_k).

    (An OFDM/OMTDR excitation would inject all tones simultaneously and recover
    the same S11(f_k) by channel estimation; it would replace THIS class only.)
    """
    f_start: float
    f_stop: float
    n_freq: int
    name: str = "Stepped-frequency FDR"

    @property
    def frequencies(self) -> np.ndarray:
        return np.linspace(self.f_start, self.f_stop, self.n_freq)



# fdr modelling
def secondary_params(R, L, G, C, freqs) -> Tuple[np.ndarray, np.ndarray]:
    w = 2.0 * np.pi * freqs
    series = R + 1j * w * L          # R + jwL
    shunt = G + 1j * w * C          # G + jwC
    gamma = np.sqrt(series * shunt)  # (1.16)
    Zc = np.sqrt(series / shunt)     # (1.19)
    return gamma, Zc


def abcd_line(R, L, G, C, length, freqs) -> np.ndarray:
    #ABCD of a uniform line section. Thesis eq. (1.39). Returns (Nf,2,2)
    gamma, Zc = secondary_params(R, L, G, C, freqs)
    gl = gamma * length
    ch, sh = np.cosh(gl), np.sinh(gl)
    M = np.empty((freqs.size, 2, 2), dtype=complex)
    M[:, 0, 0] = ch
    M[:, 0, 1] = Zc * sh
    M[:, 1, 0] = sh / Zc
    M[:, 1, 1] = ch
    return M


def abcd_series(Z: complex, nf: int) -> np.ndarray:
    #Lumped series impedance Z: [[1, Z],[0,1]]. Used for an open circuit (Z -> large)
    M = np.broadcast_to(np.eye(2, dtype=complex), (nf, 2, 2)).copy()
    M[:, 0, 1] = Z
    return M


def abcd_shunt(Y: complex, nf: int) -> np.ndarray:
    #Lumped shunt admittance Y: [[1,0],[Y,1]]. Used for a SHORT (Y -> large)
    M = np.broadcast_to(np.eye(2, dtype=complex), (nf, 2, 2)).copy()
    M[:, 1, 0] = Y
    return M


# magnitudes used to approximate ideal hard faults as strong lumped discontinuities
_OPEN_SERIES_Z = 1e9    # Ohm  (series break ~ open circuit, Gamma ~ +1)
_SHORT_SHUNT_Y = 1e9    # S    (shunt short  ~ short circuit, Gamma ~ -1)


def _local_rlgc(cable: CableParams, fault: FaultParams) -> Tuple[float, float, float, float]:
    """Return the (R,L,G,C) inside a finite-length (non-hard) fault section."""
    R, L, G, C = cable.R, cable.L, cable.G, cable.C
    ft = fault.fault_type
    if ft == "impedance_change":
        ratio = 1.0 + fault.delta_zc_percent / 100.0
        if fault.zc_mechanism == "L":
            # keep C, change L so that Zc ~ sqrt(L/C) scales by `ratio`
            L = L * ratio ** 2
        elif fault.zc_mechanism == "C":
            # keep L, change C so that Zc ~ sqrt(L/C) scales by `ratio`
            C = C / ratio ** 2
        else:
            raise ValueError("zc_mechanism must be 'L' or 'C'")
    elif ft == "R_increase":
        R = R * fault.factor
    elif ft == "C_increase":
        C = C * fault.factor
    elif ft == "G_increase":
        G = G * fault.factor
    elif ft == "L_increase":
        L = L * fault.factor
    else:
        raise ValueError(f"_local_rlgc called for non-finite fault '{ft}'")
    return R, L, G, C


#modelisation de la ligne

def build_element_list(cable: CableParams, faults: List[FaultParams]):
    """
    Partition the cable [0, total_length] into an ordered list of two-port
    elements: uniform line segments (healthy or faulty RLGC) plus lumped
    series/shunt elements for hard (open/short) faults.

    Returns a list of tuples describing each element, in propagation order.
    """
    Ltot = cable.total_length
    tol = 1e-12

    interval_faults = [f for f in faults
                       if f.fault_type in ("impedance_change", "R_increase",
                                           "L_increase", "C_increase",
                                           "G_increase")]
    point_faults = [f for f in faults
                    if f.fault_type in ("open_circuit", "short_circuit")]

    # boundary positions: cable ends, all interval edges, all point positions
    bounds = {0.0, Ltot}
    for f in interval_faults:
        bounds.add(max(0.0, f.position))
        bounds.add(min(Ltot, f.position + f.length))
    for f in point_faults:
        bounds.add(min(Ltot, max(0.0, f.position)))
    bounds = sorted(bounds)

    def fault_at_midpoint(mid: float) -> Optional[FaultParams]:
        for f in interval_faults:
            if (f.position - tol) <= mid < (f.position + f.length - tol):
                return f
        return None

    elements = []  # each: dict with keys for later ABCD construction
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        # insert any point (hard) fault sitting exactly at boundary `a`
        for f in point_faults:
            if abs(f.position - a) <= 1e-9:
                elements.append({"kind": f.fault_type, "pos": a, "fault": f})
        seg_len = b - a
        if seg_len <= tol:
            continue
        f_here = fault_at_midpoint(0.5 * (a + b))
        if f_here is None:
            R, L, G, C = cable.R, cable.L, cable.G, cable.C
            faulty = False
        else:
            R, L, G, C = _local_rlgc(cable, f_here)
            faulty = True
        elements.append({"kind": "line", "a": a, "b": b, "len": seg_len,
                         "R": R, "L": L, "G": G, "C": C, "faulty": faulty})
    # point faults located exactly at the far end (L)
    for f in point_faults:
        if abs(f.position - Ltot) <= 1e-9:
            elements.append({"kind": f.fault_type, "pos": Ltot, "fault": f})

    return elements


def compute_S11(cable: CableParams, faults: List[FaultParams],
                freqs: np.ndarray) -> np.ndarray:
    """
    Full Stage-1 chain -> complex S11(f) at the source port.
        ABCD cascade -> Zin = (A ZL + B)/(C ZL + D) -> S11 = (Zin - Zs)/(Zin + Zs)
    """
    nf = freqs.size
    elements = build_element_list(cable, faults)

    Mtot = np.broadcast_to(np.eye(2, dtype=complex), (nf, 2, 2)).copy()
    for el in elements:
        if el["kind"] == "line":
            M = abcd_line(el["R"], el["L"], el["G"], el["C"], el["len"], freqs)
        elif el["kind"] == "open_circuit":
            M = abcd_series(_OPEN_SERIES_Z, nf)
        elif el["kind"] == "short_circuit":
            M = abcd_shunt(_SHORT_SHUNT_Y, nf)
        else:
            raise ValueError(f"unknown element kind {el['kind']}")
        Mtot = np.matmul(Mtot, M)

    A = Mtot[:, 0, 0]; B = Mtot[:, 0, 1]
    C = Mtot[:, 1, 0]; D = Mtot[:, 1, 1]
    Zin = (A * cable.ZL + B) / (C * cable.ZL + D)
    S11 = (Zin - cable.Zs) / (Zin + cable.Zs)
    return S11


# S11(f) -> window -> IFFT -> r(x)

def make_window(name: str, n: int) -> np.ndarray:
    name = (name or "rect").lower()
    if name in ("rect", "none", "boxcar"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window '{name}'")


# The rectangular first-null resolution is dx = vp/(2B); tapering trades resolution for lower sidelobes, so the EFFECTIVE resolution
# is roughly (factor * dx). These are standard, approximate rule-of-thumb values.
_WINDOW_BROADENING = {
    "rect": 1.0, "none": 1.0, "boxcar": 1.0,
    "hamming": 1.5,
    "hann": 2.0,
    "blackman": 2.7,
}


def window_broadening(name: str) -> float:
    return _WINDOW_BROADENING.get((name or "rect").lower(), 1.0)


def velocity_lossless(cable: CableParams) -> float:
    """vp ~ 1/sqrt(L*C)  (thesis eq. 1.24, lossless approximation)."""
    return 1.0 / np.sqrt(cable.L * cable.C)


def reflectogram_from_S11(S11: np.ndarray, freqs: np.ndarray, vp: float,
                          window: str = "hann", zero_pad_factor: int = 4
                          ) -> Tuple[np.ndarray, np.ndarray]:
    #Convert a complex frequency response to a distance-domain response.
    """
    Steps (see module docstring for the assumptions):
        1. preserve complex S11 (phase carries the delay/distance);
        2. apply window (reduces sidelobes, broadens peaks);
        3. zero-pad in frequency (interpolates time axis);
        4. inverse DFT -> impulse response;
        5. build delay axis dt = 1/(Npad * df);  t = arange(Npad)*dt;
        6. x = vp * t / 2  (factor 2 = round trip: out to fault and back).
    """
    n = S11.size
    df = float(np.mean(np.diff(freqs)))
    w = make_window(window, n)
    Sw = S11 * w
    n_pad = int(n * max(1, zero_pad_factor))
    Spad = np.zeros(n_pad, dtype=complex)
    Spad[:n] = Sw
    r = np.fft.ifft(Spad)                 # baseband-equivalent impulse response
    dt = 1.0 / (n_pad * df)               # time step; span dt*n_pad = 1/df (unchanged by padding)
    t = np.arange(n_pad) * dt
    x = vp * t / 2.0
    return x, r


def restrict_range(x: np.ndarray, r: np.ndarray, x_limit: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    mask = x <= x_limit
    return x[mask], r[mask]



# GROUND-TRUTH local Zc(x) and dZc(x)[%] 

def ground_truth_delta_zc(cable: CableParams, faults: List[FaultParams],
                          x_grid: np.ndarray, f_ref: float
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local |Zc(x, f_ref)| and percentage change vs the healthy cable.

        Zc(x,f) = sqrt((R(x)+jwL(x)) / (G(x)+jwC(x)))                 
        dZc(x)[%] = 100 * (|Zc(x,f_ref)| - |Zc_h(f_ref)|)/|Zc_h(f_ref)|
    """
    w = 2.0 * np.pi * f_ref
    _, Zc_h = secondary_params(cable.R, cable.L, cable.G, cable.C,
                               np.array([f_ref]))
    Zc_h = abs(Zc_h[0])

    interval_faults = [f for f in faults
                       if f.fault_type in ("impedance_change", "R_increase",
                                           "L_increase", "C_increase",
                                           "G_increase")]
    point_faults = [f for f in faults
                    if f.fault_type in ("open_circuit", "short_circuit")]

    Zc_local = np.full_like(x_grid, Zc_h, dtype=float)
    for i, x in enumerate(x_grid):
        for f in interval_faults:
            if f.position <= x < f.position + f.length:
                R, L, G, C = _local_rlgc(cable, f)
                series = R + 1j * w * L
                shunt = G + 1j * w * C
                Zc_local[i] = abs(np.sqrt(series / shunt))
                break

    dZc_pct = 100.0 * (Zc_local - Zc_h) / Zc_h

    # mark hard-fault positions (impedance step is effectively infinite/zero)
    for f in point_faults:
        idx = int(np.argmin(np.abs(x_grid - f.position)))
        dZc_pct[idx] = np.nan
    return Zc_local, dZc_pct


# ISOLATED-STEP REFLECTION (theory only; NOT what the IFFT peak equals exactly)

def gamma_isolated(delta_zc_percent: float) -> float:
#        Gamma = (Z2 - Z1)/(Z2 + Z1) = (dZc/100) / (2 + dZc/100)    
    d = delta_zc_percent / 100.0
    return d / (2.0 + d)



# FAULT LOCALISATION -- peak detection on |dr(x)|
"""
    Δr(x) is the distance-domain version of the baseline-subtracted reflection response, and |Δr(x)| is its envelope (magnitude).
=>distance is encoded in the phase slope of S₁₁(f); the IFFT decodes that slope into position
"""

def _numpy_find_peaks(y, min_prominence, min_distance_samples):
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            # crude prominence: height above the lower of the neighbouring minima
            left = y[:i]
            right = y[i + 1:]
            base = max(left.min() if left.size else 0.0,
                       right.min() if right.size else 0.0)
            if (y[i] - base) >= min_prominence:
                peaks.append((i, y[i] - base))
    # enforce minimum spacing, keep strongest
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for idx, prom in peaks:
        if all(abs(idx - k) >= min_distance_samples for k, _ in kept):
            kept.append((idx, prom))
    kept.sort(key=lambda p: p[0])
    return np.array([k for k, _ in kept], dtype=int), {
        "prominences": np.array([p for _, p in kept])}


def detect_faults(x: np.ndarray, r_mag_norm: np.ndarray,
                  min_prominence: float, min_distance_m: float,
                  max_peaks: int) -> List[Dict]:
    if x.size < 3:
        return []
    dx = float(np.mean(np.diff(x)))
    min_dist_samples = max(1, int(round(min_distance_m / dx)))

    if _HAVE_SCIPY:
        idx, props = _scipy_find_peaks(r_mag_norm, prominence=min_prominence,
                                       distance=min_dist_samples)
        proms = props.get("prominences", np.zeros(idx.size))
    else:
        idx, props = _numpy_find_peaks(r_mag_norm, min_prominence,
                                       min_dist_samples)
        proms = props["prominences"]

    order = np.argsort(-r_mag_norm[idx])[:max_peaks]
    idx = idx[order]; proms = proms[order]
    order2 = np.argsort(x[idx])
    idx = idx[order2]; proms = proms[order2]

    return [{"estimated_position_m": float(x[i]),
             "normalized_reflection_strength": float(r_mag_norm[i]),
             "prominence": float(p)}
            for i, p in zip(idx, proms)]



# FREQUENCY-SWEEP / DISTANCE-PERFORMANCE METRICS

def sweep_metrics(freqs: np.ndarray, vp: float, cable_len: float,
                  faults: List[FaultParams], window: str = "rect") -> Dict:
    df_vec = np.diff(freqs)
    df = float(np.mean(df_vec))
    uniform = bool(np.allclose(df_vec, df, rtol=1e-9, atol=0.0))
    B = float(freqs[-1] - freqs[0])
    dx = vp / (2.0 * B)                       # theoretical (rect window) resolution
    dx_eff = dx * window_broadening(window)   # broadened by the chosen window
    x_max = vp / (2.0 * df)
    fault_extents = []
    for f in faults:
        if f.fault_type in ("open_circuit", "short_circuit"):
            fault_extents.append((f.label or f.fault_type, 0.0, True))
        else:
            fault_extents.append((f.label or f.fault_type, f.length,
                                  f.length < dx))
    return {"df": df, "uniform": uniform, "B": B, "dx": dx, "dx_eff": dx_eff,
            "window": window, "x_max": x_max, "cable_len": cable_len,
            "cable_fits": cable_len <= x_max, "fault_extents": fault_extents}



def print_report(exc, metrics, vp, faults, detected):
    print("SIMULATED FAULTS")
    for i, f in enumerate(faults, 1):
        if f.fault_type in ("open_circuit", "short_circuit"):
            print(f"    Fault {i} [{f.label or f.fault_type}]:")
            print(f"        Position : {f.position:.3f} m (hard fault, point)")
            print(f"        Type     : {f.fault_type}")
        elif f.fault_type == "impedance_change":
            g = gamma_isolated(f.delta_zc_percent)
            print(f"    Fault {i} [{f.label or 'impedance_change'}]:")
            print(f"        Region   : {f.position:.3f} - {f.position+f.length:.3f} m")
            print(f"        dZc      : {f.delta_zc_percent:+.1f} %")
            print(f"        Theoretical isolated-step Gamma : {g:+.4f}")
        else:
            print(f"    Fault {i} [{f.label or f.fault_type}]:")
            print(f"        Region   : {f.position:.3f} - {f.position+f.length:.3f} m")
            print(f"        Type     : {f.fault_type} (factor {f.factor})")
    print()
    print("DETECTED REFLECTIONS  (peaks on baseline-subtracted |dr(x)|)")
    if not detected:
        print("    (none)")
    for i, d in enumerate(detected, 1):
        print(f"    Peak {i}:")
        print(f"        Estimated distance : {d['estimated_position_m']:.3f} m")
        print(f"        Normalized strength: {d['normalized_reflection_strength']:.3f}")
        print(f"        Prominence         : {d['prominence']:.3f}")


def run_validation(metrics, faults, detected, dr_norm, dZc_pct):
    """Lightweight self-checks; prints PASS/INFO lines (non-fatal)."""
    print("VALIDATION")
    # 1. non-trivial reflectogram when a fault is present
    has_fault = len(faults) > 0
    nontrivial = float(np.nanmax(dr_norm)) > 0.05
    if has_fault:
        print(f"    [{'PASS' if nontrivial else 'WARN'}] baseline-subtracted "
              f"reflectogram is non-trivial (max={np.nanmax(dr_norm):.3f})")
    # 2. detected positions vs simulated positions (interval faults)
    sim_pos = [f.position for f in faults
               if f.fault_type not in ("open_circuit", "short_circuit")]
    sim_pos += [f.position for f in faults
                if f.fault_type in ("open_circuit", "short_circuit")]
    for sp in sim_pos:
        if detected:
            nearest = min(detected, key=lambda d: abs(d["estimated_position_m"] - sp))
            err = abs(nearest["estimated_position_m"] - sp)
            ok = err <= max(metrics["dx_eff"], 0.02)
            print(f"    [{'PASS' if ok else 'INFO'}] fault @ {sp:.3f} m -> nearest "
                  f"peak @ {nearest['estimated_position_m']:.3f} m (err {err*100:.1f} cm)")
    # resolvability of fault pairs vs effective resolution
    pf = sorted(sim_pos)
    for a, b in zip(pf, pf[1:]):
        sep = b - a
        verdict = "resolvable" if sep >= metrics["dx_eff"] else \
                  "NOT separable (sep < effective resolution -> peaks merge)"
        print(f"    [INFO] fault pair {a:.3f}/{b:.3f} m: separation {sep*100:.1f} cm "
              f"vs Δx_eff {metrics['dx_eff']*100:.1f} cm -> {verdict}")
    # 3. ground-truth dZc matches definitions (interval faults only)
    print(f"    [INFO] ground-truth dZc(x) spans "
          f"[{np.nanmin(dZc_pct):+.1f}, {np.nanmax(dZc_pct):+.1f}] % "
          f"(exact, simulation-defined)")
    print("-" * 78)

#plot
def make_figure(freqs, S11_h, S11_f, dS11,
                x_d, dr_norm, detected,
                x_gt, dZc_pct, faults, metrics, exc, out_path):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 11),
        gridspec_kw={"height_ratios": [1.0, 1.1, 0.9]})

    fMHz = freqs / 1e6
    # frequency-domain S11 sweep
    ax1.plot(fMHz, np.abs(S11_h), lw=1.0, color="tab:green", label="|S11| healthy")
    ax1.plot(fMHz, np.abs(S11_f), lw=1.0, color="tab:red", label="|S11| faulty")
    ax1.plot(fMHz, np.abs(dS11), lw=1.0, color="tab:blue", alpha=0.8,
             label="|ΔS11| = |S11_faulty − S11_healthy|")
    ax1.set_title(f"Panel 1 — Frequency-domain response  ({exc.name})")
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("|S11|  (linear)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    # distance-domain reflection strength 
    ax2.plot(x_d, dr_norm, color="tab:blue", lw=1.2,
             label="|Δr(x)| normalized (reflection strength)")
    for d in detected:
        ax2.axvline(d["estimated_position_m"], color="k", ls=":", lw=0.8, alpha=0.6)
        ax2.plot(d["estimated_position_m"], d["normalized_reflection_strength"],
                 "v", color="black", ms=8)
    # true fault boundaries
    first = True
    for f in faults:
        if f.fault_type in ("open_circuit", "short_circuit"):
            ax2.axvline(f.position, color="tab:red", ls="--", lw=1.2,
                        label="true hard-fault position" if first else None)
        else:
            ax2.axvspan(f.position, f.position + f.length, color="tab:orange",
                        alpha=0.15,
                        label="true fault extent" if first else None)
            ax2.axvline(f.position, color="tab:orange", ls="--", lw=1.0)
            ax2.axvline(f.position + f.length, color="tab:purple", ls="--", lw=1.0)
        first = False
    # resolution bars: theoretical (rect) and effective (current window)
    ax2.annotate("", xy=(metrics["dx"], 0.78), xytext=(0.0, 0.78),
                 arrowprops=dict(arrowstyle="<->", color="gray"))
    ax2.text(metrics["dx"] / 2, 0.80, f"Δx≈{metrics['dx']*100:.1f} cm (rect)",
             ha="center", va="bottom", fontsize=7.5, color="gray")
    ax2.annotate("", xy=(metrics["dx_eff"], 0.90), xytext=(0.0, 0.90),
                 arrowprops=dict(arrowstyle="<->", color="dimgray"))
    ax2.text(metrics["dx_eff"] / 2, 0.92,
             f"Δx_eff≈{metrics['dx_eff']*100:.1f} cm ({metrics['window']})",
             ha="center", va="bottom", fontsize=7.5, color="dimgray")
    ax2.set_title("Panel 2 — Distance-domain reflectogram  "
                  "(reflection strength, NOT impedance)")
    ax2.set_xlabel("Distance from source [m]")
    ax2.set_ylabel("normalized |Δr(x)|")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    # ground-truth dZc% 
    ax3.plot(x_gt, dZc_pct, color="tab:green", lw=1.4,
             label="ΔZc(x) [%]  (SIMULATION GROUND TRUTH)")
    ax3.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    for f in faults:
        if f.fault_type not in ("open_circuit", "short_circuit"):
            ax3.axvline(f.position, color="tab:orange", ls="--", lw=1.0)
            ax3.axvline(f.position + f.length, color="tab:purple", ls="--", lw=1.0)
    ax3.set_title("Panel 3 — Ground-truth characteristic-impedance change "
                  "(NOT reconstructed from S11)")
    ax3.set_xlabel("Distance from source [m]")
    ax3.set_ylabel("ΔZc [%]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc="upper right")

    # align distance axes of panels 2 & 3
    xmax = max(x_d.max(), x_gt.max())
    ax2.set_xlim(0, xmax)
    ax3.set_xlim(0, xmax)

    fig.suptitle("Stepped-frequency FDR: frequency-domain modelling → "
                 "distance-domain interpretation", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    # NOTE: no plt.show() here so the figure can be embedded in Streamlit
    # (st.pyplot) or any non-interactive backend. The caller owns the figure.
    return fig


def run_case(cable: CableParams, faults: List[FaultParams], cfg: Dict,
             out_path: str = "fdr_result.png", show_report: bool = True):
    exc = Excitation(cfg["F_START"], cfg["F_STOP"], cfg["N_FREQ"])
    freqs = exc.frequencies

    # sanity: uniform grid
    df_vec = np.diff(freqs)
    assert np.allclose(df_vec, df_vec[0]), "frequency grid must be uniform"

    vp = velocity_lossless(cable)
    metrics = sweep_metrics(freqs, vp, cable.total_length, faults,
                            window=cfg["WINDOW_TYPE"])

    # Stage 1: complex S11 (A) 
    S11_h = compute_S11(cable, [], freqs)        # healthy baseline
    S11_f = compute_S11(cable, faults, freqs)    # faulty
    dS11 = S11_f - S11_h                          # COMPLEX baseline difference

    # Stage 2: distance (B)
    win = cfg["WINDOW_TYPE"]; zpf = cfg["ZERO_PAD_FACTOR"]
    x_full, r_h = reflectogram_from_S11(S11_h, freqs, vp, win, zpf)
    _,      r_f = reflectogram_from_S11(S11_f, freqs, vp, win, zpf)
    _,      r_d = reflectogram_from_S11(dS11,  freqs, vp, win, zpf)

    x_limit = cable.total_length * (1.0 + cfg.get("RANGE_MARGIN", 0.15))
    x_d, r_d_disp = restrict_range(x_full, r_d, x_limit)
    # unnormalised magnitude kept internally; normalise ONLY for display
    r_d_mag = np.abs(r_d_disp)
    dr_norm = r_d_mag / (r_d_mag.max() + 1e-300)

    # ground-truth dZc(x) 
    x_gt = np.linspace(0, cable.total_length, 2000)
    _, dZc_pct = ground_truth_delta_zc(cable, faults, x_gt, cfg["REFERENCE_FREQUENCY"])

    # localisation 
    detected = detect_faults(x_d, dr_norm,
                             cfg["MIN_PEAK_PROMINENCE"],
                             cfg["MIN_PEAK_DISTANCE_M"],
                             cfg["MAX_NUMBER_OF_PEAKS"])

    if show_report:
        print_report(exc, metrics, vp, faults, detected)

    fig = make_figure(freqs, S11_h, S11_f, dS11,
                      x_d, dr_norm, detected,
                      x_gt, dZc_pct, faults, metrics, exc, out_path)

    return {"freqs": freqs, "S11_h": S11_h, "S11_f": S11_f, "dS11": dS11,
            "x": x_d, "r_delta": r_d_disp, "dr_norm": dr_norm,
            "dZc_pct": dZc_pct, "detected": detected, "metrics": metrics,
            "vp": vp, "figure": fig}


#USERRRRR PART
CONFIG = {
    "F_START": 1e6,            # Hz
    "F_STOP": 1e9,             # Hz
    "N_FREQ": 5000,
    "REFERENCE_FREQUENCY": 500e6,  # Hz, for ground-truth Zc(x, f_ref)
    "WINDOW_TYPE": "hann",     # "rect" | "hann" | "hamming" | "blackman"
    "ZERO_PAD_FACTOR": 4,      # interpolation only (does NOT improve resolution)
    "RANGE_MARGIN": 0.15,      # display up to (1+margin)*cable_length
    "MIN_PEAK_PROMINENCE": 0.1,
    "MIN_PEAK_DISTANCE_M": 0.02,
    "MAX_NUMBER_OF_PEAKS": 6,
}

CABLE = CableParams(
    R=20e-3,       # Ohm/m
    L=0.25e-6,     # H/m
    G=0.7e-6,      # S/m
    C=100e-12,     # F/m
    total_length=1,  # m
    Zs=50.0,
    ZL=50.0,
)
def main():
    faults = [
        FaultParams(position=0.25, length=0.02, fault_type="impedance_change",
                    delta_zc_percent=50.0),
        FaultParams(position=0.45, length=0.02, fault_type="impedance_change",
                    delta_zc_percent=100.0),
    ]
    run_case(CABLE, faults, CONFIG, out_path="fdr_result.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# to chnage RLGC parameters instead specifically
#FaultParams(0.25, 0.02, "R_increase", factor=5.0)   # or C_increase, G_increase