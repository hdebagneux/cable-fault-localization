"""
second_order_streamlit.py
-------------------------
Importable version of second_order.py for the Streamlit interface.

The original second_order.py remains fully runnable in VSCode/terminal.
This file exposes the same physics as callable functions so time_domain.py
can invoke the simulation with user-supplied parameters.
"""

import numpy as np


# Nominal (healthy) distributed parameters — used as defaults and reference values
DEFAULT_R = 20e-3   # Ω/m
DEFAULT_L = 2.5e-7  # H/m
DEFAULT_G = 0.7e-6  # S/m
DEFAULT_C = 1e-10   # F/m


def build_sections(length, fault_start, fault_end, fault_param, fault_multiplier):
    """
    Build the sections list for a cable with a single fault region.

    Each section is a tuple (x_start, x_end, R, L, G, C).
    fault_param must be one of 'R', 'L', 'G', 'C'.
    The fault section's chosen parameter is multiplied by fault_multiplier.
    """
    def rlgc(is_fault):
        R, L, G, C = DEFAULT_R, DEFAULT_L, DEFAULT_G, DEFAULT_C
        if is_fault:
            if fault_param == 'R':
                R *= fault_multiplier
            elif fault_param == 'L':
                L *= fault_multiplier
            elif fault_param == 'G':
                G *= fault_multiplier
            elif fault_param == 'C':
                C *= fault_multiplier
        return R, L, G, C

    sections = []
    if fault_start > 0:
        R, L, G, C = rlgc(False)
        sections.append((0.0, fault_start, R, L, G, C))
    R, L, G, C = rlgc(True)
    sections.append((fault_start, fault_end, R, L, G, C))
    if fault_end < length:
        R, L, G, C = rlgc(False)
        sections.append((fault_end, length, R, L, G, C))
    return sections


def generate_signal(Nt, dt, f_max, V_amplitude, signal_type='gaussian'):
    """
    Return (Vs, t_arr) for the chosen source signal.
    signal_type: 'gaussian' or 'sinusoidal'
    """
    sigma_t = 1.0 / (2.0 * np.pi * f_max)
    t_arr   = np.arange(Nt) * dt
    t0      = 6.0 * sigma_t

    if signal_type == 'gaussian':
        Vs = V_amplitude * np.exp(-((t_arr - t0) ** 2) / (2.0 * sigma_t ** 2))
    else:
        Vs = V_amplitude * np.sin(2.0 * np.pi * f_max * t_arr)

    return Vs, t_arr


def run_simulation(sections, length, f_max, PPW, V_amplitude,
                   signal_type='gaussian', Zs=50, verbose=False):
    """
    Run the FDTD TDR simulation.

    Parameters
    ----------
    sections      : list of (x_start, x_end, R, L, G, C) — use build_sections()
    length        : total cable length (m)
    f_max         : maximum frequency / Gaussian bandwidth (Hz)
    PPW           : points per wavelength (spatial resolution, min 10)
    V_amplitude   : source pulse amplitude (V)
    signal_type   : 'gaussian' or 'sinusoidal'
    Zs            : source impedance (Ω), default 50
    verbose       : print grid diagnostics to stdout

    Returns
    -------
    dict with keys:
        V, I        – voltage (Nt×Nx) and current (Nt×(Nx-1)) grids
        x, t_arr    – spatial (m) and temporal (s) coordinate arrays
        dt, Nt, Nx  – grid sizes
        v, v_avg, Z0, length, sections
        R_ref, L_ref, G_ref, C_ref  – first-section (nominal) parameters
    """
    # ── Propagation speeds ───────────────────────────────────────────────────
    v_sections = [1.0 / np.sqrt(L * C) for (_, _, _, L, _, C) in sections]
    v_max = max(v_sections)
    v_min = min(v_sections)

    R_ref, L_ref, G_ref, C_ref = (
        sections[0][2], sections[0][3], sections[0][4], sections[0][5]
    )
    v  = 1.0 / np.sqrt(L_ref * C_ref)
    Z0 = np.sqrt(L_ref / C_ref)

    # Length-weighted harmonic mean: correct average for one-way transit time
    T_one_way = sum(
        (x_end - x_start) / (1.0 / np.sqrt(L * C))
        for (x_start, x_end, _, L, _, C) in sections
    )
    v_avg = length / T_one_way

    # ── Grid sizing ──────────────────────────────────────────────────────────
    sigma_t     = 1.0 / (2.0 * np.pi * f_max)
    sigma_x_min = v_min * sigma_t
    dx          = sigma_x_min / PPW
    Nx          = max(2, int(np.ceil(length / dx)))
    dx          = length / Nx

    dt_cfl = 0.9 * dx / v_max
    dt_nyq = 1.0 / (2.0 * f_max)
    dt     = min(dt_cfl, dt_nyq)

    T_total = 2.0 * length / v_avg
    Nt      = int(np.ceil(T_total / dt))

    if verbose:
        lambda_min = v / f_max
        print(f"Wave speed          v       = {v:.4e} m/s")
        print(f"Pulse width         σ_t     = {sigma_t:.4e} s")
        print(f"Max frequency       f_max   = {f_max:.4e} Hz")
        print(f"Min wavelength      λ_min   = {lambda_min:.4e} m")
        print(f"Space step          dx      = {dx:.4e} m  ({PPW} pts/wavelength)")
        print(f"Time step           dt      = {dt:.4e} s")
        print(f"CFL limit                   = {dt_cfl:.4e} s  {'✓' if dt <= dt_cfl else '✗'}")
        print(f"Nyquist limit               = {dt_nyq:.4e} s  {'✓' if dt <= dt_nyq else '✗'}")
        print(f"Nodes Nx            = {Nx},   Steps Nt = {Nt}")
        print(f"Total sim time      = {T_total:.4e} s")

    # ── Per-node RLGC arrays ─────────────────────────────────────────────────
    x     = np.linspace(0, length, Nx)
    R_arr = np.zeros(Nx)
    L_arr = np.zeros(Nx)
    G_arr = np.zeros(Nx)
    C_arr = np.zeros(Nx)

    for (x_start, x_end, R_sec, L_sec, G_sec, C_sec) in sections:
        mask = (x >= x_start) & (x < x_end)
        if x_end == length:
            mask |= (x == length)
        R_arr[mask] = R_sec
        L_arr[mask] = L_sec
        G_arr[mask] = G_sec
        C_arr[mask] = C_sec

    # ── FDTD coefficients ────────────────────────────────────────────────────
    Zl = Z0

    A_V = (2 * C_arr - G_arr * dt) / (2 * C_arr + G_arr * dt)
    B_V = (2 * dt) / ((2 * C_arr + G_arr * dt) * dx)

    # Current lives at half-nodes k+1/2; indexed by the LEFT node k.
    A_I = (L_arr[:-1] - R_arr[:-1] * dt) / L_arr[:-1]
    B_I = dt / (L_arr[:-1] * dx)

    denom_S = (2 * C_arr[0]  + G_arr[0]  * dt) * dx * Zs + 2 * dt
    denom_L = (2 * C_arr[-1] + G_arr[-1] * dt) * dx * Zl + 2 * dt

    coef_S_first  = ((2 * C_arr[0]  - G_arr[0]  * dt) * dx * Zs - 2 * dt) / denom_S
    coef_S_second = (2 * dt) / denom_S
    coef_L_first  = ((2 * C_arr[-1] - G_arr[-1] * dt) * dx * Zl - 2 * dt) / denom_L
    coef_L_second = (2 * dt * 2 * Zl) / denom_L

    # ── Source signal ────────────────────────────────────────────────────────
    Vs, t_arr = generate_signal(Nt, dt, f_max, V_amplitude, signal_type)

    # ── Time stepping ────────────────────────────────────────────────────────
    V = np.zeros((Nt, Nx))
    I = np.zeros((Nt, Nx - 1))

    for n in range(Nt - 1):
        I[n + 1, :] = A_I * I[n, :] - B_I * (V[n, 1:] - V[n, :-1])

        V[n + 1, 0] = (coef_S_first  * V[n, 0]
                       + coef_S_second * (Vs[n + 1] + Vs[n] - 2 * Zs * I[n + 1, 0]))
        V[n + 1, -1] = coef_L_first * V[n, -1] + coef_L_second * I[n + 1, -1]

        # U_k^{n+1} = A_V[k]*U_k^n - B_V[k]*(I_{k+1/2}^{n+1/2} - I_{k-1/2}^{n+1/2})
        V[n + 1, 1:-1] = (A_V[1:-1] * V[n, 1:-1]
                          - B_V[1:-1] * (I[n + 1, 1:] - I[n + 1, :-1]))

    return dict(
        V=V, I=I, x=x, t_arr=t_arr, dt=dt, Nt=Nt, Nx=Nx,
        v=v, v_avg=v_avg, Z0=Z0, length=length, sections=sections,
        R_ref=R_ref, L_ref=L_ref, G_ref=G_ref, C_ref=C_ref,
    )


# ── Reflection / fault detection ──────────────────────────────────────────────

def _detect_events(sim, threshold_factor):
    """Segment V(t) at x=0 into above-threshold pulses."""
    V, Nt, dt = sim['V'], sim['Nt'], sim['dt']
    v0        = V[:, 0]
    t_vals    = np.arange(Nt) * dt
    threshold = threshold_factor * np.max(np.abs(v0))

    above   = np.abs(v0) > threshold
    edges   = np.diff(above.astype(int))
    rising  = np.where(edges ==  1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    if above[0]:
        rising  = np.concatenate(([0], rising))
    if above[-1]:
        falling = np.concatenate((falling, [Nt]))

    return [{'t': t_vals[s:e], 'v': v0[s:e]} for s, e in zip(rising, falling)]


def find_reflections_method_2(sim, threshold_factor=0.001):
    """Time-ratio method — anchors fault distances to the end-of-cable reflection."""
    events     = _detect_events(sim, threshold_factor)
    v, length  = sim['v'], sim['length']

    if not events:
        print("No events detected.")
        return []

    t_emit = events[0]['t'][np.argmax(np.abs(events[0]['v']))]
    reflections = events[1:]

    estimated_time    = (length * 2) / v
    hundredth_of_time = estimated_time / 100
    end_peak_time     = None

    print(f"Found {len(reflections)} reflection(s) at x=0:")
    for ev in reflections:
        peak_idx  = np.argmax(np.abs(ev['v']))
        rel_time  = ev['t'][peak_idx] - t_emit
        if estimated_time - hundredth_of_time < rel_time < estimated_time + hundredth_of_time:
            end_peak_time = rel_time

    rows = []
    for k, ev in enumerate(reflections):
        peak_idx           = np.argmax(np.abs(ev['v']))
        time               = ev['t'][peak_idx]
        relative_peak_time = time - t_emit
        if end_peak_time is not None:
            start_time     = ev['t'][0]  - t_emit
            end_time       = ev['t'][-1] - t_emit
            start_distance = length * (start_time / end_peak_time)
            end_distance   = length * (end_time   / end_peak_time)
            peak_distance  = length * (relative_peak_time / end_peak_time)
            print(f"  #{k+1}  peak = {ev['v'][peak_idx]:.4f} V  "
                  f"at t = {ev['t'][peak_idx]:.3e} s  "
                  f"peak distance = {peak_distance:.3e} m  "
                  f"start distance = {start_distance:.4f} m  "
                  f"end distance = {end_distance:.4f} m")
            rows.append({
                "Fault #": k + 1,
                "Peak (V)": ev['v'][peak_idx],
                "Time (s)": ev['t'][peak_idx],
                "Distance (m)": peak_distance,
                "Extent (m)": f"{start_distance:.4f} – {end_distance:.4f}",
            })
        else:
            print("Peak from end reflection was not detected, so resorting to method based on default velocity of propagation...")
            return find_reflections_method_3(sim)
    return rows


def find_reflections_method_3(sim, threshold_factor=0.005):
    """Use wave speed v directly as the spatial ruler."""
    events = _detect_events(sim, threshold_factor)
    v      = sim['v']

    if not events:
        print("No events detected.")
        return []

    t_emit      = events[0]['t'][np.argmax(np.abs(events[0]['v']))]
    reflections = events[1:]

    print(f"\nFound {len(reflections)} reflection(s) at x=0 (using v = {v:.4e} m/s):")
    rows = []
    for k, ev in enumerate(reflections):
        peak_idx     = np.argmax(np.abs(ev['v']))
        t_peak       = ev['t'][peak_idx]
        t_round_trip = t_peak - t_emit
        d_peak       = v * t_round_trip / 2
        d_start      = v * (ev['t'][0]  - t_emit) / 2
        d_end        = v * (ev['t'][-1] - t_emit) / 2
        print(f"  #{k+1}  peak = {ev['v'][peak_idx]:+.4f} V  |  "
              f"round-trip = {t_round_trip:.3e} s  |  "
              f"fault at ≈ {d_peak:.4f} m  "
              f"(extent: {d_start:.4f} – {d_end:.4f} m)")
        rows.append({
            "Fault #": k + 1,
            "Peak (V)": ev['v'][peak_idx],
            "Time (s)": t_round_trip,
            "Distance (m)": d_peak,
            "Extent (m)": f"{d_start:.4f} – {d_end:.4f}",
        })
    return rows


def find_fault_size(sim, threshold_factor=0.005, tolerance=0.15):
    """
    Pair entry/exit reflections to estimate fault impedance and length.

    For a fault of impedance Z_fault in a cable of Z0:
      Γ = (Z_fault - Z0) / (Z_fault + Z0)
      Predicted exit amplitude: A_exit = A_entry × -(1 - Γ²)

    Wave speed inside the fault:
      L fault (C fixed): v_fault = v × Z0/Z_fault
      C fault (L fixed): v_fault = v × Z_fault/Z0
    """
    events       = _detect_events(sim, threshold_factor)
    v, Z0        = sim['v'], sim['Z0']
    Nt, dt       = sim['Nt'], sim['dt']
    sections     = sim['sections']
    L_ref, C_ref = sim['L_ref'], sim['C_ref']

    c_changed = any(C != C_ref for (_, _, _, _, _, C) in sections)
    l_changed = any(L != L_ref for (_, _, _, L, _, _) in sections)

    if not events:
        print("No events detected.")
        return []

    emitted  = events[0]
    emit_idx = np.argmax(np.abs(emitted['v']))
    t_emit   = emitted['t'][emit_idx]
    V_emit   = emitted['v'][emit_idx]
    reflections = events[1:]

    print(f"\nFound {len(reflections)} reflection(s) at x=0 (using v = {v:.4e} m/s):")
    matched = set()
    rows = []

    for k, ev in enumerate(reflections):
        if k in matched:
            continue

        peak_idx     = np.argmax(np.abs(ev['v']))
        t_peak       = ev['t'][peak_idx]
        V_entry      = ev['v'][peak_idx]
        t_round_trip = t_peak - t_emit
        Gamma        = V_entry / V_emit
        d_entry      = v * t_round_trip / 2

        Z_fault = Z0 * (1 + Gamma) / (1 - Gamma)
        v_fault = v * (Z_fault / Z0) if (c_changed and not l_changed) else v * (Z0 / Z_fault)
        V_exit_predicted = V_entry * (-(1 - Gamma ** 2))

        print(f"\n  #{k+1} [ENTRY]  peak = {V_entry:+.4f} V  |  Γ = {Gamma:+.4f}  |  "
              f"Z_fault ≈ {Z_fault:.2f} Ω  |  v_fault ≈ {v_fault:.3e} m/s  |  "
              f"d_entry ≈ {d_entry:.4f} m")
        print(f"          Predicted exit amplitude: {V_exit_predicted:+.4f} V  "
              f"(tolerance ±{tolerance * 100:.0f}%)")

        row = {
            "Fault #": k + 1,
            "Peak (V)": V_entry,
            "Time (s)": t_round_trip,
            "Distance (m)": d_entry,
            "Extent (m)": f"{d_entry:.4f}",
            "Z_fault (Ω)": Z_fault,
        }

        for j, ev2 in enumerate(reflections[k + 1:], start=k + 1):
            peak_idx2 = np.argmax(np.abs(ev2['v']))
            V_peak2   = ev2['v'][peak_idx2]
            t_peak2   = ev2['t'][peak_idx2]

            if abs(V_peak2 - V_exit_predicted) <= tolerance * abs(V_exit_predicted):
                t_round_trip2 = t_peak2 - t_emit
                delta_t       = t_round_trip2 - t_round_trip
                fault_length  = v_fault * delta_t / 2
                d_exit        = d_entry + fault_length
                matched.add(j)

                print(f"          EXIT (#{j+1})   peak = {V_peak2:+.4f} V  |  "
                      f"Δt = {delta_t:.3e} s  |  "
                      f"fault length ≈ {fault_length:.4f} m  |  "
                      f"d_exit ≈ {d_exit:.4f} m")

                row["Extent (m)"] = f"{d_entry:.4f} – {d_exit:.4f}"

                t_max = (Nt - 1) * dt
                reverberation_tol = 0.15 * delta_t
                for n_rev in range(1, 20):
                    t_reverb = t_round_trip2 + n_rev * delta_t
                    if t_reverb > t_max:
                        break
                    for m, ev3 in enumerate(reflections[j + 1:], start=j + 1):
                        if m in matched:
                            continue
                        peak_idx3 = np.argmax(np.abs(ev3['v']))
                        t_rt3     = ev3['t'][peak_idx3] - t_emit
                        if abs(t_rt3 - t_reverb) < reverberation_tol:
                            matched.add(m)
                            print(f"          REVERBERATION (#{m+1})  "
                                  f"t = {t_rt3:.3e} s  → skipped")
                            break
                break
        else:
            print("          No matching exit reflection found.")

        rows.append(row)

    return rows
