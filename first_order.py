import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

length  = 1    # cable length in meters

# proportions hold for any cable length
_sections_norm = [
    (0.00, 0.69, 20e-3, 2.5e-7, 0.7e-6, 1e-10),  # 0–40 %
    (0.69, 0.71, 20e-3, 5e-7,  0.7e-6, 0.5e-10),   # 40–70 %
    (0.71, 1.00, 20e-3, 2.5e-7, 0.7e-6, 1e-10),   # 70–100 %
]

# scale to actual length
sections = [
    (s * length, e * length, R, L, G, C)
    for (s, e, R, L, G, C) in _sections_norm
]

R_ref = sections[0][2]
L_ref = sections[0][3]
G_ref = sections[0][4]
C_ref = sections[0][5]

Zs = 50
Z0 = np.sqrt(L_ref / C_ref)
v  = 1.0 / np.sqrt(L_ref * C_ref)

# MAIN INPUT: physical pulse width in seconds
sigma_t = 5e-11        # pulse width in seconds => change ONLY this
PPW     = 20             # points per wavelength (accuracy knob, min = 10)
n_trips = 6             # numb. of one-way trips to simulate
V_amplitude = 10

# derived frequency / wavelength limits
f_max  = 1.0 / (2.0 * np.pi * sigma_t)       # bandwidth of the Gaussian pulse
lambda_min = v / f_max                        # shortest wavelength to resolve

#  Spatial discretisation (driven by pulse width in space)
sigma_x = v * sigma_t                        # spatial pulse width (m)
dx      = sigma_x / PPW                      # PPW points across the pulse
Nx      = max(2, int(np.ceil(length / dx)))
dx      = length / Nx                        # recompute so grid fits exactly

#  Time discretisation (CFL + Nyquist)
dt_cfl    = 0.9 * dx / v                      # CFL stability limit
dt_nyq    = 1.0 / (2.0 * f_max)              # Nyquist accuracy limit
dt        = min(dt_cfl, dt_nyq)              

T_total   = n_trips * length / v             # total simulation time
Nt        = int(np.ceil(T_total / dt))        # number of time steps

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

V = np.zeros((Nt, Nx))
I = np.zeros((Nt, Nx - 1))

#  Source signal
t_arr  = np.arange(Nt) * dt                  
t0     = 6.0 * sigma_t                       # pulse centre (starts near 0)
Vs     = V_amplitude * np.exp(-((t_arr - t0) ** 2) / (2.0 * sigma_t ** 2))

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

#  FDTD propagation
def propagate_impulse():
    for time in range(Nt - 1):

        # Update interior currents
        for k in range(Nx - 1):
            I[time+1, k] = (
                (1 - R_arr[k] * dt / L_arr[k]) * I[time, k]
                - (dt / (L_arr[k] * dx)) * (V[time, k+1] - V[time, k])
            )

        # Source boundary (left)
        V[time+1, 0] = (
            V[time, 0] * ((2*Zs*C_arr[0]*dx - dt) / (2*Zs*C_arr[0]*dx + dt))
            + dt * (Vs[time+1] + Vs[time] - 2*Zs*I[time+1, 0])
            / (2*Zs*C_arr[0]*dx + dt)
        )

        # Load boundary 
        V[time+1, -1] = (
            V[time, -1] * ((2*Zl*C_arr[-1]*dx - dt) / (2*Zl*C_arr[-1]*dx + dt))
            + dt * (2*Zl*I[time+1, -1])
            / (2*Zl*C_arr[-1]*dx + dt)
        )

        # Interior voltages
        for k in range(1, Nx - 1):
            V[time+1, k] = (
                (1 - G_arr[k] * dt / C_arr[k]) * V[time, k]
                - (dt / (C_arr[k] * dx)) * (I[time+1, k] - I[time+1, k-1])
            )

def animate():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7))
    fig.tight_layout(pad=3.0)

    # Propagation snapshot
    line_top, = ax_top.plot(x, V[0, :], color="orange")
    ax_top.set_xlim(x[0], x[-1])
    ax_top.set_ylim(np.min(V) - 0.5, np.max(V) + 0.5)
    ax_top.set_xlabel("Position along transmission line (m)")
    ax_top.set_ylabel("Voltage (V)")
    ax_top.set_title("1st-order approximation - signal propagation")

    # Voltage at x=0 over time
    t_axis = np.arange(Nt) * dt
    line_bot, = ax_bot.plot([], [])
    ax_bot.set_xlim(0, (Nt - 1) * dt)
    ax_bot.set_ylim(np.min(V[:, 0]) - 0.5, np.max(V[:, 0]) + 0.5)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Voltage at x=0 (V)")
    ax_bot.set_title("Signal seen at source node (x = 0)")
    ax_bot.grid(True)

    skip = max(1, Nt // 200)           # controls animation speed
    frames = range(0, Nt, skip)

    def update(frame):
        line_top.set_ydata(V[frame, :])
        ax_top.set_title(f"1st-order approx, signal propagating")
        line_bot.set_xdata(t_axis[:frame + 1])
        line_bot.set_ydata(V[:frame + 1, 0])
        return line_top, line_bot

    ani = FuncAnimation(fig, update, frames=frames, interval=75, blit=True)
    plt.show()
    
    
propagate_impulse()
animate()
