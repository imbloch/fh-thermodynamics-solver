import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from physics import solve, compute_profiles, warmup

# ==========================================
# INTERACTIVE PLOT (MATPLOTLIB)
# ==========================================

# Initial Parameters
init_N = 300
init_k = 0.005
init_S = 1.5
init_U = 0.5
last_guess = [None]
last_params = [None]

warmup()

# Setup Figure
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

ax_plot = fig.add_subplot(111)
ax_plot.set_xlabel('Radius $r$ (lattice sites)')
ax_plot.set_ylabel('Density / Occupation per site')
ax_plot.set_title('Thermodynamics of 2D Lattice Fermions (Atomic Limit)')
ax_plot.grid(True, linestyle='--', alpha=0.6)
ax_plot.set_ylim(-0.05, 2.05)

# Initial Plot Lines (Empty)
line_tot, = ax_plot.plot([], [], 'k-', lw=2, label='Total Density')
line_sin, = ax_plot.plot([], [], 'b--', lw=1.5, label='Singlons')
line_dbl, = ax_plot.plot([], [], 'r-', lw=2, label='Doubles (Pairs)')
line_hol, = ax_plot.plot([], [], 'g-.', lw=1.5, label='Holes')
line_exc, = ax_plot.plot([], [], color='orange', ls=':', lw=2, label='Excited Band')
ax_plot.legend(loc='upper right')

# Text Output
text_res = plt.figtext(0.1, 0.92, "Adjust sliders to solve...",
                       fontsize=12, fontweight='bold', color='blue')

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_N = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_k = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_S = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_U = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

s_N = Slider(ax_N, 'Particle N',    100,  5000, valinit=init_N, valstep=10)
s_k = Slider(ax_k, 'Trap k',        5e-5, 0.02, valinit=init_k, valfmt='%.4f')
s_S = Slider(ax_S, 'Entropy S/N',   0.0,  2.0,  valinit=init_S)
s_U = Slider(ax_U, 'Interaction U (Δ)', -1.0, 1.0, valinit=init_U, valstep=0.05)

def update(event):
    val_N = s_N.val
    val_k = s_k.val
    val_S = s_S.val
    val_U = s_U.val
    params = (val_N, val_k, val_S, val_U)
    if last_params[0] == params:
        return
    last_params[0] = params

    text_res.set_text("Solving... Please wait.")
    plt.draw()

    mu_sol, T_sol, success, reason = solve(
        val_N,
        val_S,
        val_k,
        val_U,
        init_guess=last_guess[0],
        return_reason=True
    )

    if success:
        last_guess[0] = (mu_sol, T_sol)
        prof = compute_profiles(mu_sol, T_sol, val_k, val_U)
        n_check = np.trapezoid(
            prof['n_total'] * 2.0 * np.pi * prof['r'],
            prof['r']
        )

        status = (f"SOLVED:\n"
                  f"T = {T_sol:.4f} (Δ/kB)\n"
                  f"μ = {mu_sol:.4f} (Δ)\n"
                  f"Calc N = {n_check:.1f} (Target {val_N})")
        if reason:
            status += f"\nWarning: {reason}"
        text_res.set_text(status)
        text_res.set_color('darkgreen')

        r = prof['r']
        line_tot.set_data(r, prof['n_total'])
        line_sin.set_data(r, prof['p_singlon'])
        line_dbl.set_data(r, prof['p_doublon'])
        line_hol.set_data(r, prof['p_hole'])
        line_exc.set_data(r, prof['n_excited'])
        ax_plot.set_xlim(0, r[-1])
        ax_plot.set_ylim(-0.05, max(2.05, float(np.max(prof['n_total'])) * 1.1))
    else:
        text_res.set_text(reason or "Solver Failed! Try different parameters.")
        text_res.set_color('red')

    plt.draw()


s_N.on_changed(update)
s_k.on_changed(update)
s_S.on_changed(update)
s_U.on_changed(update)
update(None)

plt.show()
