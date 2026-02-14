# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Matplotlib GUI
.venv/bin/python3 Hubbard-Solver.py

# Streamlit web app
.venv/bin/streamlit run app.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `numba`, `streamlit`. Installed in `.venv/`.

## What This Project Does

Solves thermodynamic properties of 2D trapped lattice fermions in the **atomic limit of the Hubbard model** using the grand canonical ensemble. Interactive GUIs (matplotlib or Streamlit) let users adjust physical parameters (particle number N, trap strength k, entropy per particle S/N, interaction strength U, band gap Δ) and solve for temperature T and chemical potential μ.

## Architecture

- **`physics.py`** — All computation. Shared by both UIs.
  - JIT-compiled kernels (`@jit(nopython=True, cache=True)`): `_orbital_probs`, `_orbital_entropy`, `_integrate_ns`. Must remain Numba-compatible (no Python objects, no scipy inside JIT).
  - Gauss-Legendre quadrature (200 nodes, pre-computed at import) replaces `scipy.integrate.quad` for performance — both N and S computed in a single JIT pass.
  - `solve()` → (mu, T, success). Initial guess via `brentq` for μ at trial T, then `scipy.optimize.root` (hybr method).
  - `compute_profiles()` → dict of vectorised radial arrays for plotting.

- **`Hubbard-Solver.py`** — Matplotlib GUI. Sliders + "Calculate" button.
- **`app.py`** — Streamlit GUI. Auto-solves on slider change, cached via `@st.cache_data`.

## Multi-band Physics

The partition function includes **ground band** (1 s-orbital) and **first excited band** (2 degenerate p-orbitals at energy Δ). In the atomic limit with no inter-orbital interactions, the site partition function factorizes: Z_site = Z_ground × (Z_excited)². Each orbital's Z uses the log-sum-exp trick for numerical stability at low T.

## Key Conventions

- Natural units: kB = 1. Energies and temperatures in units of the tunneling energy.
- Local density approximation: μ_eff(r) = μ_global − k·r² (trap applies to all bands).
- Integration cutoff radius: `sqrt(max(15, |μ|×5) / k)`.
- Entropy per orbital uses spin-degeneracy correction for singlons: `p_s · ln(p_s / 2)`.
- Plot shows radial density profile: total density, ground-band singlons/doublons/holes, and excited-band density.
