import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


length = 1 # meters

sections = [
    # x_start  x_end   R [Ω/m]   L [H/m]    G [S/m]    C [F/m]
    (0.00,     0.69,   20e-3,    2.5e-7,    0.7e-6,    1e-10),   # section 1 – nominal
    (0.69,     0.71,   20e-3,   5e-7,    0.7e-6,    0.5e-10),   # section 2 – high R (fault)
    (0.71,   length,   20e-3,    2.5e-7,    0.7e-6,    1e-10),   # section 3 – back to nominal
]

R_ref = sections[0][2]
L_ref = sections[0][3]
G_ref = sections[0][4]
C_ref = sections[0][5]

Zs = 50
Z0 = np.sqrt(L_ref / C_ref)
v  = 1.0 / np.sqrt(L_ref * C_ref) #2e8, which can be found from 66% velocity factor

v_sections = [1.0/np.sqrt(L*C) for (_,_,_,L,_,C) in sections]
v_max = max(v_sections)   # fastest region
v_min = min(v_sections)   # slowest region

# Length-weighted harmonic mean: correct average for a one-way transit time
T_one_way = sum((x_end - x_start) / (1.0 / np.sqrt(L * C))
                for (x_start, x_end, _, L, _, C) in sections)
v_avg = length / T_one_way   # average propagation speed across all sections

# FROM new_variable_RLGC.py (replaces fixed Nx=500 / Nt=10000)
f_max = 5e9 #chosen by the user
sigma_t = 1/(2*np.pi*f_max)  # pulse width in seconds
PPW     = 10     # points per wavelength (accuracy knob, min = 10)
n_trips = 2   # number of round trips (back AND forth) to simulate

sigma_x_min = v_min * sigma_t
dx = sigma_x_min / PPW
Nx = max(2, int(np.ceil(length / dx)))
dx = length / Nx


lambda_min = v / f_max 
dt_cfl = 0.9 * dx / v_max
dt_nyq = 1.0 / (2.0 * f_max)
dt = min(dt_cfl, dt_nyq)

T_total = n_trips * 2 * length / v_avg  # n_trips round trips at average speed
Nt = int(np.ceil(T_total / dt))

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


V_amplitude = 10
t_arr = np.arange(Nt) * dt
t0    = 6.0 * sigma_t
Vs    = V_amplitude * np.exp(-((t_arr - t0) ** 2) / (2.0 * sigma_t ** 2))


V = np.zeros((Nt, Nx))
I = np.zeros((Nt, Nx-1))

Zl = 100
R_arr = np.zeros(Nx)
L_arr = np.zeros(Nx)
G_arr = np.zeros(Nx)
C_arr = np.zeros(Nx)

x = np.linspace(0, length, Nx)

for (x_start, x_end, R_sec, L_sec, G_sec, C_sec) in sections:
    mask = (x >= x_start) & (x < x_end)
    if x_end == length:
        mask |= (x == length)
    R_arr[mask] = R_sec
    L_arr[mask] = L_sec
    G_arr[mask] = G_sec
    C_arr[mask] = C_sec

A_V = (2*C_arr - G_arr*dt) / (2*C_arr + G_arr*dt)          # shape (Nx,)
B_V = (2*dt)   / ((2*C_arr + G_arr*dt) * dx)               # shape (Nx,)

# Current lives at half-nodes k+1/2; we index by the LEFT node k.
A_I = (L_arr[:-1] - R_arr[:-1]*dt) / L_arr[:-1]            # shape (Nx-1,)
B_I = dt / (L_arr[:-1] * dx)                               # shape (Nx-1,)

denom_S = (2*C_arr[0]  + G_arr[0]  * dt) * dx * Zs + 2*dt
denom_L = (2*C_arr[-1] + G_arr[-1] * dt) * dx * Zl + 2*dt

coef_S_first  = ((2*C_arr[0]  - G_arr[0] *dt)*dx*Zs - 2*dt) / denom_S
coef_S_second = (2*dt) / denom_S          # multiplies (Vs^{n+1}+Vs^n - 2Zs*I[n,0])

coef_L_first  = ((2*C_arr[-1] - G_arr[-1]*dt)*dx*Zl - 2*dt) / denom_L
coef_L_second = (2*dt * 2*Zl) / denom_L  # multiplies I[n, Nx-2]  (= I_{K-1/2})

def propagate_impulse():
    for n in range(Nt - 1):

        I[n+1, :] = A_I * I[n, :] - B_I * (V[n, 1:] - V[n, :-1])

        # update boundary voltages using fresh currents

        V[n+1, 0] = (coef_S_first  * V[n, 0]
                     + coef_S_second * (Vs[n+1] + Vs[n] - 2*Zs * I[n+1, 0]))


        V[n+1, -1] = (coef_L_first  * V[n, -1]
                      + coef_L_second * I[n+1, -1])

        # update interior voltages
        # U_k^{n+1} = A_V[k]*U_k^n  -  B_V[k]*(I_{k+1/2}^{n+1/2} - I_{k-1/2}^{n+1/2})
        V[n+1, 1:-1] = (A_V[1:-1] * V[n, 1:-1]
                        - B_V[1:-1] * (I[n+1, 1:] - I[n+1, :-1]))


propagate_impulse()


def find_reflections(threshold_factor=0.005):
    """
    Detect pulses that return to x=0 after the emitted signal.

    threshold_factor : fraction of max |V[:,0]| used as detection threshold.

    Returns a list of dicts, one per reflection, each with:
        't' : 1-D array of time values (s) spanning the pulse
        'v' : 1-D array of voltage values (V) spanning the pulse
    """
    v0 = V[:, 0]
    t_vals = np.arange(Nt) * dt
    threshold = threshold_factor * np.max(np.abs(v0))

    above = np.abs(v0) > threshold
    edges = np.diff(above.astype(int))
    rising  = np.where(edges ==  1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    if above[0]:
        rising = np.concatenate(([0], rising))
    if above[-1]:
        falling = np.concatenate((falling, [Nt]))

    events = [{'t': t_vals[s:e], 'v': v0[s:e]} for s, e in zip(rising, falling)]

    reflections = events[1:]   # first event is the emitted signal
    estimated_time = (length*2)/v
    tenth_of_time = estimated_time/10
    print(f"Found {len(reflections)} reflection(s) at x=0:")
    for k, ev in enumerate(reflections):
        peak_idx = np.argmax(np.abs(ev['v']))
        time = ev['t'][peak_idx]
        origin = (v*time)/2
        print(f"  #{k+1}  peak = {ev['v'][peak_idx]:.4f} V  "
              f"at t = {ev['t'][peak_idx]:.3e} s  "
              f"duration = {ev['t'][-1] - ev['t'][0]:.3e} s  "
              f"origin = {origin:4f} m")

    return reflections


fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7))
plt.tight_layout(pad=3.0)

# Top panel – propagation snapshot
line_top, = ax_top.plot(x, V[0])
ax_top.set_xlim(x[0], x[-1])
ax_top.set_ylim(np.min(V) - 0.5, np.max(V) + 0.5)
ax_top.set_xlabel("Position along transmission line (m)")
ax_top.set_ylabel("Voltage (V)")
ax_top.set_title("2nd-order approximation – signal propagation")

# Bottom panel – voltage at x=0 over time
t_axis = np.arange(Nt) * dt
line_bot, = ax_bot.plot([], [])
ax_bot.set_xlim(0, (Nt - 1) * dt)
ax_bot.set_ylim(np.min(V[:, 0]) - 0.5, np.max(V[:, 0]) + 0.5)
ax_bot.set_xlabel("Time (s)")
ax_bot.set_ylabel("Voltage at x=0 (V)")
ax_bot.set_title("Signal seen at source node (x = 0)")

def update(frame):
    line_top.set_ydata(V[frame, :])
    ax_top.set_title(f"2nd-order approx, signal propagating")
    line_bot.set_xdata(t_axis[:frame + 1])
    line_bot.set_ydata(V[:frame + 1, 0])
    return line_top, line_bot

ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)
plt.show()

# reflections = find_reflections()

def find_reflections_method_2(threshold_factor=0.001):
    "Detects faults based on finding the end point reflection "
    "We can estimate this position using a calculation of the speed of propagation in the wire"
    "We then find the exact point of the end reflection, and measure other reflections due to faults"
    "As a ration of the time they took to come back"
    v0 = V[:, 0]
    t_vals = np.arange(Nt) * dt
    threshold = threshold_factor * np.max(np.abs(v0))

    above = np.abs(v0) > threshold
    edges = np.diff(above.astype(int))
    rising  = np.where(edges ==  1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    if above[0]:
        rising = np.concatenate(([0], rising))
    if above[-1]:
        falling = np.concatenate((falling, [Nt]))

    events = [{'t': t_vals[s:e], 'v': v0[s:e]} for s, e in zip(rising, falling)]
    emitted = events[0]
    t_emit = emitted['t'][np.argmax(np.abs(emitted['v']))]

    reflections = events[1:]   # first event is the emitted signal
    estimated_time = (length*2)/v
    hundredth_of_time = estimated_time/100
    end_peak_time = None
    print(f"Found {len(reflections)} reflection(s) at x=0:")
    for k, ev in enumerate(reflections):
        peak_idx = np.argmax(np.abs(ev['v']))
        peak_time = ev['t'][peak_idx]
        time = peak_time-t_emit
        if time < estimated_time+hundredth_of_time and time > estimated_time-hundredth_of_time:
            end_peak_time = time
    for k, ev in enumerate(reflections):
        peak_idx = np.argmax(np.abs(ev['v']))
        time = ev['t'][peak_idx]
        relative_peak_time = time - t_emit
        if end_peak_time is not None:
            origin = length* (time/end_peak_time)
        else:
            origin = "Could not find end peak"
        start_time = ev['t'][0] - t_emit
        end_time = ev['t'][-1] - t_emit
        start_distance = length* (start_time/end_peak_time)
        end_distance = length* (end_time/end_peak_time)
        peak_distance =  length* (relative_peak_time/end_peak_time)
        print(f"  #{k+1}  peak = {ev['v'][peak_idx]:.4f} V  "
              f"at t = {ev['t'][peak_idx]:.3e} s  "
              f"peak distance = {peak_distance:.3e} m  "
              f"start distance = {start_distance:4f} m  "
              f"end distance = {end_distance:4f} m  "
              )

    return reflections




def find_reflections_method_3(threshold_factor=0.005):
    """""
    ratio of the time taken to come back over the time 
    taken for the end signal to come back
    """""
    v0 = V[:, 0]
    t_vals = np.arange(Nt) * dt
    threshold = threshold_factor * np.max(np.abs(v0))

    above = np.abs(v0) > threshold
    edges = np.diff(above.astype(int))
    rising  = np.where(edges ==  1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    if above[0]:
        rising = np.concatenate(([0], rising))
    if above[-1]:
        falling = np.concatenate((falling, [Nt]))

    events = [{'t': t_vals[s:e], 'v': v0[s:e]} for s, e in zip(rising, falling)]

    if not events:
        print("No events detected.")
        return []

    # Time of the emitted pulse peak
    emitted = events[0]
    t_emit = emitted['t'][np.argmax(np.abs(emitted['v']))]

    reflections = events[1:]
    print(f"\nFound {len(reflections)} reflection(s) at x=0 (using v = {v:.4e} m/s):")

    for k, ev in enumerate(reflections):
        peak_idx         = np.argmax(np.abs(ev['v']))
        t_peak           = ev['t'][peak_idx]
        t_round_trip     = t_peak - t_emit          # time for signal to go there and back

        d_peak           = v * t_round_trip / 2

        # Spread of the detected pulse → spatial extent of the fault region
        t_start          = ev['t'][0]  - t_emit
        t_end            = ev['t'][-1] - t_emit
        d_start          = v * t_start / 2
        d_end            = v * t_end   / 2

        print(f"  #{k+1}  peak = {ev['v'][peak_idx]:+.4f} V  |  "
              f"round-trip = {t_round_trip:.3e} s  |  "
              f"fault at ≈ {d_peak:.4f} m  "
              f"(extent: {d_start:.4f} – {d_end:.4f} m)")
# uses known wave speed v directly as the ruler
        d_peak = v * t_round_trip / 2
    return reflections

def find_fault_size(threshold_factor=0.005, tolerance=0.15):
    """
    Detect prolonged reactive faults (L or C change) by pairing entry and exit reflections.

    For a fault of impedance Z_fault in a cable of Z0:
      - Entry reflection coefficient:  Γ = (Z_fault - Z0) / (Z_fault + Z0)
      - Predicted exit amplitude:       A_exit = A_entry × -(1 - Γ²)
        (derived from T_in × Γ_out × T_back = (1+Γ)(−Γ)(1−Γ))

    The time gap between entry and exit peaks gives the fault length via
    the wave speed inside the fault: L_fault = v_fault × Δt / 2.

    Wave speed inside the fault:
      - L fault (C fixed): Z_fault/Z0 = sqrt(L_f/L)  →  v_fault = v × Z0/Z_fault
      - C fault (L fixed): Z_fault/Z0 = sqrt(C/C_f)  →  v_fault = v × Z_fault/Z0
    Fault type is inferred from which parameter differs across sections.
    """
    c_changed = any(C != C_ref for (_, _, _, _, _, C) in sections)
    l_changed = any(L != L_ref for (_, _, _, L, _, _) in sections)
    v0 = V[:, 0]
    t_vals = np.arange(Nt) * dt
    threshold = threshold_factor * np.max(np.abs(v0))

    above = np.abs(v0) > threshold
    edges = np.diff(above.astype(int))
    rising  = np.where(edges ==  1)[0] + 1
    falling = np.where(edges == -1)[0] + 1

    if above[0]:
        rising = np.concatenate(([0], rising))
    if above[-1]:
        falling = np.concatenate((falling, [Nt]))

    events = [{'t': t_vals[s:e], 'v': v0[s:e]} for s, e in zip(rising, falling)]

    if not events:
        print("No events detected.")
        return []

    emitted  = events[0]
    emit_idx = np.argmax(np.abs(emitted['v']))
    t_emit   = emitted['t'][emit_idx]
    V_emit   = emitted['v'][emit_idx]

    reflections = events[1:]
    print(f"\nFound {len(reflections)} reflection(s) at x=0 (using v = {v:.4e} m/s):")

    matched = set()  # indices already claimed as exit reflections

    for k, ev in enumerate(reflections):
        if k in matched:
            continue

        peak_idx     = np.argmax(np.abs(ev['v']))
        t_peak       = ev['t'][peak_idx]
        V_entry      = ev['v'][peak_idx]
        t_round_trip = t_peak - t_emit
        Gamma        = V_entry / V_emit          # entry reflection coefficient
        d_entry      = v * t_round_trip / 2

        Z_fault = Z0 * (1 + Gamma) / (1 - Gamma)
        if c_changed and not l_changed:
            v_fault = v * (Z_fault / Z0)        # C fault: v_fault = v · Z_fault/Z0
        else:
            v_fault = v * (Z0 / Z_fault)        # L fault: v_fault = v · Z0/Z_fault

        # Exit reflection must satisfy: A_exit = A_entry × -(1 - Γ²)
        V_exit_predicted = V_entry * (-(1 - Gamma**2))

        print(f"\n  #{k+1} [ENTRY]  peak = {V_entry:+.4f} V  |  Γ = {Gamma:+.4f}  |  "
              f"Z_fault ≈ {Z_fault:.2f} Ω  |  v_fault ≈ {v_fault:.3e} m/s  |  "
              f"d_entry ≈ {d_entry:.4f} m")
        print(f"          Predicted exit amplitude: {V_exit_predicted:+.4f} V  "
              f"(tolerance ±{tolerance*100:.0f}%)")

        for j, ev2 in enumerate(reflections[k+1:], start=k+1):
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

                # Mark internal reverberations: echoes at t_exit + n*delta_t
                t_max = (Nt - 1) * dt
                reverberation_tol = 0.15 * delta_t
                for n in range(1, 20):
                    t_reverb = t_round_trip2 + n * delta_t
                    if t_reverb > t_max:
                        break
                    for m, ev3 in enumerate(reflections[j+1:], start=j+1):
                        if m in matched:
                            continue
                        peak_idx3 = np.argmax(np.abs(ev3['v']))
                        t_rt3 = ev3['t'][peak_idx3] - t_emit
                        if abs(t_rt3 - t_reverb) < reverberation_tol:
                            matched.add(m)
                            print(f"          REVERBERATION (#{m+1})  "
                                  f"t = {t_rt3:.3e} s  → skipped")
                            break
                break
        else:
            print(f"          No matching exit reflection found.")

    return reflections

lc_uniform = all(
    L == L_ref and C == C_ref
    for (_, _, _, L, _, C) in sections
)

if lc_uniform:
    print("\nL and C uniform across sections — using time-ratio method.")
    try:
        reflections2 = find_reflections_method_2()
    except TypeError:
        print("Method 2 unsuccessful - signal never reflected from end. Using method 3.")
        reflections3 = find_reflections_method_3()
else:
    print("\nL or C varies across sections — using impedance-pairing method.")
    find_fault_size()