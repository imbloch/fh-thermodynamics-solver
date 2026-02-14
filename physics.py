"""Physics engine for the Hubbard model atomic limit solver.

Computes thermodynamic properties of 2D trapped lattice fermions in the
atomic limit, including ground band (s-orbital) and first excited band
(2 degenerate p-orbitals at energy gap Delta).

All energies (U, k, T) are in units of the band gap Delta.

All intensive kernels are JIT-compiled with Numba. Integration uses
Gauss-Legendre quadrature (fully in JIT, no Python callbacks).
"""

import numpy as np
from scipy.optimize import root, brentq
from numba import jit

kB = 1.0
DELTA = 1.0  # Band gap — the energy unit. Excited band sits at this energy.
_MIN_T = 1e-4
_WARMED_UP = False

# Pre-compute Gauss-Legendre quadrature nodes and weights on [-1, 1]
_N_QUAD = 100
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_N_QUAD)

# =====================================================================
# JIT-compiled kernels
# =====================================================================

@jit(nopython=True, cache=True)
def _orbital_probs(mu_eff, T, u):
    """Occupation probabilities for a single orbital (numerically stable).

    mu_eff = mu - epsilon - V(r)  (effective chemical potential for this orbital).
    Returns (p_hole, p_singlon, p_double).
    p_singlon includes both spin species.
    """
    beta = 1.0 / T
    a0 = 0.0
    a1 = beta * mu_eff
    a2 = beta * (2.0 * mu_eff - u)
    a_max = max(a0, max(a1, a2))
    e0 = np.exp(a0 - a_max)
    e1 = np.exp(a1 - a_max)
    e2 = np.exp(a2 - a_max)
    z = e0 + 2.0 * e1 + e2
    return e0 / z, 2.0 * e1 / z, e2 / z


@jit(nopython=True, cache=True)
def _orbital_entropy(p_h, p_s, p_d):
    """Entropy from one orbital given its occupation probabilities.

    Uses spin-degeneracy correction for singlons: -p_s * ln(p_s / 2).
    """
    s = 0.0
    if p_h > 1e-15:
        s += p_h * np.log(p_h)
    if p_s > 1e-15:
        s += p_s * np.log(p_s / 2.0)
    if p_d > 1e-15:
        s += p_d * np.log(p_d)
    return -s


@jit(nopython=True, cache=True)
def _integrate_ns(mu, T, k, u, delta, max_r, gl_nodes, gl_weights):
    """Gauss-Legendre integration of total N and S over the 2D trap.

    Computes both N and S in a single pass over radial quadrature points,
    summing ground band (1 s-orbital) and excited band (2 p-orbitals at
    energy delta) contributions at each point.
    """
    half = max_r * 0.5
    N_tot = 0.0
    S_tot = 0.0
    for i in range(gl_nodes.shape[0]):
        r = half * (gl_nodes[i] + 1.0)   # map [-1,1] -> [0, max_r]
        w = half * gl_weights[i]
        mu_loc = mu - k * r * r

        # Ground band (1 s-orbital, epsilon = 0)
        p0h, p0s, p0d = _orbital_probs(mu_loc, T, u)
        n_site = p0s + 2.0 * p0d
        s_site = _orbital_entropy(p0h, p0s, p0d)

        # First excited band (2 degenerate p-orbitals, epsilon = delta)
        p1h, p1s, p1d = _orbital_probs(mu_loc - delta, T, u)
        n_site += 2.0 * (p1s + 2.0 * p1d)
        s_site += 2.0 * _orbital_entropy(p1h, p1s, p1d)

        jac = 2.0 * np.pi * r  # polar integration jacobian
        N_tot += w * n_site * jac
        S_tot += w * s_site * jac
    return N_tot, S_tot


# =====================================================================
# Public API
# =====================================================================

def _max_r(mu, k):
    """Integration cutoff radius."""
    return np.sqrt(max(15.0, abs(mu) * 5.0) / k)


def calculate_ns(mu, T, k, u):
    """Calculate total particle number N and entropy S."""
    mr = _max_r(mu, k)
    return _integrate_ns(mu, T, k, u, DELTA, mr, _GL_NODES, _GL_WEIGHTS)


def system_equations(params, target_n, target_s_per_n, k, u):
    """Residual equations F(mu, T) = 0 for the root finder."""
    mu, T = params
    if T <= _MIN_T:
        T = _MIN_T
    n_calc, s_calc = calculate_ns(mu, T, k, u)
    eq1 = (n_calc - target_n) / target_n
    eq2 = (s_calc / n_calc - target_s_per_n) if n_calc > 1e-5 else 0.0
    return [eq1, eq2]


def _mu_for_target_n(target_n, T, k, u, mu_center=0.0, mu_span=20.0):
    """Solve N(mu, T) = target_n for mu using adaptive bracketing."""

    def n_res(mu):
        n, _ = calculate_ns(mu, T, k, u)
        return n - target_n

    lo = mu_center - mu_span
    hi = mu_center + mu_span
    f_lo = n_res(lo)
    f_hi = n_res(hi)
    for _ in range(6):
        if f_lo * f_hi <= 0.0:
            return brentq(n_res, lo, hi, xtol=1e-3)
        mu_span *= 2.0
        lo = mu_center - mu_span
        hi = mu_center + mu_span
        f_lo = n_res(lo)
        f_hi = n_res(hi)
    raise ValueError("Could not bracket mu for target N.")


def _estimate_low_t_entropy_floor(target_n, k, u):
    """Conservative low-temperature S/N floor estimate for infeasibility checks."""
    floor = np.inf
    for t_probe in (0.02, 0.01):
        try:
            mu_probe = _mu_for_target_n(target_n, t_probe, k, u)
        except ValueError:
            continue
        n_probe, s_probe = calculate_ns(mu_probe, t_probe, k, u)
        if n_probe > 1e-8:
            floor = min(floor, s_probe / n_probe)
    return floor


def warmup():
    """Compile hot JIT kernels once to reduce first-interaction latency."""
    global _WARMED_UP
    if _WARMED_UP:
        return
    calculate_ns(0.0, 0.5, 0.005, 0.0)
    _WARMED_UP = True


def solve(target_n, target_s_per_n, k, u, init_guess=None, return_reason=False):
    """Solve for (mu, T) given target N and S/N.

    If return_reason=True, returns (mu, T, success, reason).
    On success, reason may contain a warning message.
    """
    def _pack(mu, t, success, reason=""):
        if return_reason:
            return mu, t, success, reason
        return mu, t, success

    if target_n <= 0.0 or k <= 0.0:
        return _pack(0.0, 0.0, False, "Invalid target N or trap k.")

    warning_reason = ""

    # For very low entropy requests, estimate a floor and warn if likely infeasible.
    if target_s_per_n < 1.0:
        s_floor = _estimate_low_t_entropy_floor(target_n, k, u)
        if np.isfinite(s_floor) and target_s_per_n < s_floor - 0.05:
            warning_reason = (
                f"Requested S/N may be infeasible (estimated floor ~ {s_floor:.3f}). "
                "Attempting solve anyway."
            )

    try:
        cache = {}

        def residuals(params):
            mu = float(params[0])
            T = float(params[1])
            if T <= _MIN_T:
                T = _MIN_T
            key = (mu, T)
            if key in cache:
                n_calc, s_calc = cache[key]
            else:
                n_calc, s_calc = calculate_ns(mu, T, k, u)
                cache[key] = (n_calc, s_calc)
            eq1 = (n_calc - target_n) / target_n
            eq2 = (s_calc / n_calc - target_s_per_n) if n_calc > 1e-8 else 0.0
            return [eq1, eq2]

        if init_guess is not None:
            mu_init, t_init = init_guess
            x0 = [float(mu_init), max(float(t_init), _MIN_T)]
            sol = root(residuals, x0, method='hybr')
            if sol.success and sol.x[1] > _MIN_T:
                return _pack(sol.x[0], sol.x[1], True, warning_reason)

        # Fallback initial guess: find mu that gives target_n at trial temperature.
        T_trial = 0.5
        mu0 = _mu_for_target_n(target_n, T_trial, k, u)
        sol = root(residuals, [mu0, T_trial], method='hybr')
    except ValueError:
        if warning_reason:
            return _pack(
                0.0,
                0.0,
                False,
                f"{warning_reason} Unable to bracket the particle-number equation.",
            )
        return _pack(0.0, 0.0, False, "Unable to bracket the particle-number equation.")

    if sol.success and sol.x[1] > _MIN_T:
        return _pack(sol.x[0], sol.x[1], True, warning_reason)
    if warning_reason:
        return _pack(0.0, 0.0, False, f"{warning_reason} Root solve failed to converge.")
    return _pack(0.0, 0.0, False, "Root solve failed to converge.")


def compute_profiles(mu, T, k, u, n_points=300):
    """Vectorised radial profiles for plotting.

    Returns dict with keys: r, n_total, n_ground, n_excited,
    p_hole, p_singlon, p_doublon.
    """
    max_r = np.sqrt((abs(mu) + 12.0 * T) / k)
    r = np.linspace(0, max_r, n_points)
    mu_loc = mu - k * r ** 2
    beta = 1.0 / T

    def _band_probs(mu_eff):
        a0 = np.zeros_like(mu_eff)
        a1 = beta * mu_eff
        a2 = beta * (2.0 * mu_eff - u)
        a_max = np.maximum(a0, np.maximum(a1, a2))
        e0 = np.exp(a0 - a_max)
        e1 = np.exp(a1 - a_max)
        e2 = np.exp(a2 - a_max)
        z = e0 + 2.0 * e1 + e2
        return e0 / z, 2.0 * e1 / z, e2 / z

    # Ground band
    p0_h, p0_s, p0_d = _band_probs(mu_loc)
    n_ground = p0_s + 2.0 * p0_d

    # Excited band (per orbital, then x2 for degeneracy)
    p1_h, p1_s, p1_d = _band_probs(mu_loc - DELTA)
    n_excited = 2.0 * (p1_s + 2.0 * p1_d)

    return {
        'r': r,
        'n_total': n_ground + n_excited,
        'n_ground': n_ground,
        'n_excited': n_excited,
        'p_hole': p0_h,
        'p_singlon': p0_s,
        'p_doublon': p0_d,
    }
