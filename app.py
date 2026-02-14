import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from physics import solve, compute_profiles, warmup

# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Hubbard Solver", layout="wide")
st.title("Thermodynamics of 2D Lattice Fermions (Atomic Limit)")

@st.cache_resource(show_spinner=False)
def _warmup_once():
    warmup()


_warmup_once()

if "last_guess" not in st.session_state:
    st.session_state.last_guess = None
if "result" not in st.session_state:
    st.session_state.result = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None

st.sidebar.header("Parameters")
val_N = st.sidebar.slider("Particle N", 100, 5000, 300, step=10)
val_k = st.sidebar.slider("Trap k", 5e-5, 0.02, 0.005, step=1e-4, format="%.4f")
val_S = st.sidebar.slider("Entropy S/N", 0.0, 2.0, 1.5, step=0.1)
val_U = st.sidebar.slider("Interaction U (Δ units)", -1.0, 1.0, 0.5, step=0.05)

params = (float(val_N), float(val_k), float(val_S), float(val_U))
if st.session_state.last_params != params:
    with st.spinner("Solving..."):
        mu_sol, T_sol, success, reason = solve(
            float(val_N),
            val_S,
            val_k,
            val_U,
            init_guess=st.session_state.last_guess,
            return_reason=True,
        )
    st.session_state.result = {
        "mu": mu_sol,
        "T": T_sol,
        "success": success,
        "reason": reason,
        "N": val_N,
        "k": val_k,
        "S": val_S,
        "U": val_U,
    }
    if success:
        st.session_state.last_guess = (mu_sol, T_sol)
        st.session_state.profile = compute_profiles(mu_sol, T_sol, val_k, val_U)
    else:
        st.session_state.profile = None
    st.session_state.last_params = params

res = st.session_state.result

if res and res["success"]:
    prof = st.session_state.profile
    st.caption(
        f"Solved for N={res['N']}, k={res['k']:.4g}, S/N={res['S']:.2f}, U={res['U']:.2f}"
    )
    if res["reason"]:
        st.warning(res["reason"])

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature T", f"{res['T']:.4f} (Δ/kB)")
    col2.metric("Chemical Potential μ", f"{res['mu']:.4f} (Δ)")
    n_check = np.trapezoid(prof['n_total'] * 2.0 * np.pi * prof['r'], prof['r'])
    col3.metric("Calc N", f"{n_check:.1f}  (target {res['N']})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    r = prof['r']
    ax.plot(r, prof['n_total'],   'k-',  lw=2,   label='Total Density')
    ax.plot(r, prof['p_singlon'], 'b--', lw=1.5, label='Singlons')
    ax.plot(r, prof['p_doublon'], 'r-',  lw=2,   label='Doubles (Pairs)')
    ax.plot(r, prof['p_hole'],    'g-.', lw=1.5, label='Holes')
    ax.plot(r, prof['n_excited'], color='orange', ls=':', lw=2, label='Excited Band')
    ax.set_xlabel('Radius $r$ (lattice sites)')
    ax.set_ylabel('Density / Occupation per site')
    ax.set_ylim(-0.05, max(2.05, float(np.max(prof['n_total'])) * 1.1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    st.pyplot(fig)
    plt.close(fig)

    # Local entropy profile
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.plot(r, prof['s_local'], 'm-', lw=2, label='Total Local Entropy')
    ax_s.plot(r, prof['s_ground'], 'c--', lw=1.5, label='Ground Band Entropy')
    ax_s.plot(r, prof['s_excited'], color='brown', ls='-.', lw=1.5, label='Excited Band Entropy')
    ax_s.set_xlabel('Radius $r$ (lattice sites)')
    ax_s.set_ylabel('Local Entropy per Site')
    ax_s.set_ylim(bottom=-0.02)
    ax_s.grid(True, linestyle='--', alpha=0.6)
    ax_s.legend(loc='upper right')
    st.pyplot(fig_s)
    plt.close(fig_s)
elif res:
    st.error(res["reason"] or "Solver failed! Try different parameters.")
