"""Synthetic surface Cp-field generator for DeepONet training.

Generates physically realistic Cp(x/c) distributions without CFD data, using:
  - Thin-airfoil theory (vortex sheet) for circulation-based Cp
  - NACA thickness correction for velocity distribution
  - Prandtl-Glauert compressibility factor
  - Transonic shock model (wave drag onset)
  - Separation/stall softening at high angle of attack
  - 5 flow regimes matching the integral synthetic generator

Output: (params_df, fields) where fields.shape == (n_samples, n_points).
The 200-point vector represents the airfoil surface in wrap-around order:
  points 0..99   → upper surface, LE to TE  (x: 0→1)
  points 100..199→ lower surface, TE to LE  (x: 1→0)

Physical model references:
  Abbott & von Doenhoff, "Theory of Wing Sections" (1959)
  Katz & Plotkin, "Low-Speed Aerodynamics" (2001)
  Drela, "Flight Vehicle Aerodynamics" (2014) — BL coupling
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Cp physics model
# ═══════════════════════════════════════════════════════════════

def _naca_thickness_velocity(x: np.ndarray, t: float) -> np.ndarray:
    """Local velocity ratio V/V_inf due to thickness using NACA 4-digit formula.

    From Abbott & von Doenhoff: y_t = 5t(0.2969√x - 0.1260x - 0.3516x² + 0.2843x³ - 0.1015x⁴)
    Velocity perturbation ≈ dy_t/dx
    Note: clipped at x≥0.002 to avoid the LE sqrt singularity.
    """
    x = np.clip(x, 0.002, 1.0)  # avoid 1/sqrt(0) singularity at LE
    dydt = 5 * t * (
        0.2969 / (2 * np.sqrt(x))
        - 0.1260
        - 2 * 0.3516 * x
        + 3 * 0.2843 * x ** 2
        - 4 * 0.1015 * x ** 3
    )
    return np.clip(dydt, -2.0, 2.0)  # physical bound on velocity perturbation


def _camber_cp_contribution(x: np.ndarray, camber: float, xc: float) -> np.ndarray:
    """Cp contribution from camber line via thin-airfoil vortex sheet.

    Uses NACA 4-digit camber: parabolic arcs front/rear of max-camber point.
    Returns dz_c/dx (camber slope) which enters Cp through circulation.
    """
    x = np.clip(x, 0.0, 1.0)
    dzc_dx = np.where(
        x <= xc,
        2 * camber / xc ** 2 * (xc - x),
        2 * camber / (1 - xc) ** 2 * (xc - x),
    )
    return dzc_dx


def _suction_peak_shape(x: np.ndarray, x_peak: float = 0.02, decay: float = 3.5) -> np.ndarray:
    """Shape function for leading-edge suction peak.

    Rises sharply from x=0, peaks at x_peak, then decays exponentially.
    Normalised so max = 1.0.
    """
    x = np.clip(x, 0.0, 1.0)
    shape = (x / x_peak) ** 0.5 * np.exp(-decay * (x - x_peak))
    shape = np.where(x < x_peak, (x / x_peak) ** 0.5, shape)
    peak = np.max(shape)
    return shape / (peak + 1e-9)


def _pressure_recovery(x: np.ndarray, x_trans: float = 0.6) -> np.ndarray:
    """Turbulent boundary-layer pressure recovery function (Cp → 0 at TE).

    Approximates adverse pressure gradient recovery downstream of suction peak.
    """
    x = np.clip(x, 0.0, 1.0)
    x_trans = np.clip(x_trans, 0.1, 0.99)  # guard against degenerate value
    denom = max(1.0 - x_trans, 1e-4)
    recovery = np.where(
        x < x_trans,
        np.ones_like(x),
        1.0 - np.clip((x - x_trans) / denom, 0.0, 1.0) ** 1.2,
    )
    return np.clip(recovery, 0.0, 1.0)


def _shock_cp(x: np.ndarray, m: float, t: float, cl: float) -> np.ndarray:
    """Transonic shock model: adds a Cp step at estimated shock location.

    Korn equation for critical Mach; shock position estimated from
    supersonic pocket extent.
    """
    if m < 0.7:
        return np.zeros_like(x)

    m_crit = 0.87 - t / 3.0 - cl / 10.0
    m_crit = np.clip(m_crit, 0.5, 0.88)
    m_excess = max(0.0, m - m_crit)

    if m_excess < 0.01:
        return np.zeros_like(x)

    # Shock location moves forward as M increases beyond M_crit
    x_shock = np.clip(0.6 - 1.5 * m_excess, 0.15, 0.65)
    shock_width = 0.04 + 0.02 * m_excess

    # Cp jump across shock: ΔCp ~ 4/3 * M_excess (strong shock limit)
    delta_cp = 1.2 * m_excess ** 0.7

    # Gaussian approximation of shock compression
    shock_profile = delta_cp * np.exp(-((x - x_shock) / shock_width) ** 2)
    return shock_profile


def generate_cp_distribution(
    alpha: float,
    mach: float,
    thickness: float,
    camber: float,
    camber_pos: float,
    reynolds: float,
    n_points: int = 200,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a physically realistic Cp(x/c) distribution.

    Returns a 1D array of shape (n_points,):
      indices [0, n_points//2)   → upper surface, x: 0 → 1
      indices [n_points//2, end] → lower surface, x: 1 → 0

    Args:
        alpha:       Angle of attack [deg]
        mach:        Freestream Mach number
        thickness:   Max thickness ratio (t/c)
        camber:      Max camber ratio (z_max/c)
        camber_pos:  Chordwise position of max camber (x/c)
        reynolds:    Reynolds number
        n_points:    Total number of surface points
        rng:         Random state for noise injection (optional)
    """
    if rng is None:
        rng = np.random.RandomState(0)

    n_half = n_points // 2
    # x arrays for upper and lower surfaces
    x_upper = np.linspace(0.0, 1.0, n_half)
    x_lower = np.linspace(1.0, 0.0, n_half)

    alpha_rad = np.radians(alpha)

    # Prandtl-Glauert compressibility factor
    beta = np.sqrt(max(1 - mach ** 2, 0.01))

    # ── Effective angle of attack accounting for camber ──────────────
    # In thin-airfoil theory: Cl = 2π(α + 2*camber) → α_eff includes camber
    alpha_eff = alpha + np.degrees(2 * camber)  # keep in degrees for scaling

    # ── Stall model ──────────────────────────────────────────────────
    if reynolds < 5e5:
        stall_angle = 8.0 + 2.0 * thickness / 0.12
    elif reynolds < 3e6:
        stall_angle = 12.0 + 4.0 * thickness / 0.12
    else:
        stall_angle = 15.0 + 5.0 * thickness / 0.12

    stall_ratio = max(0.0, (abs(alpha) - stall_angle) / stall_angle)
    stall_factor = max(0.0, 1.0 - 1.5 * stall_ratio ** 1.5)

    # ── Upper surface Cp ─────────────────────────────────────────────
    x_pk = max(0.005, 0.005 + 0.015 * (1 - min(abs(alpha) / 15, 1)))  # peak moves LE at high AoA

    # Suction peak magnitude (scales with alpha + camber)
    cp_min_base = -(0.3 + 3.5 * np.radians(abs(alpha_eff)) + 8 * camber + 3 * thickness)
    cp_min = cp_min_base / beta  # compressibility correction
    cp_min = np.clip(cp_min, -8.0, -0.1) * stall_factor

    # Shape: suction peak + boundary layer recovery
    peak_shape = _suction_peak_shape(x_upper, x_peak=x_pk, decay=2.8 + mach)
    recovery = _pressure_recovery(x_upper, x_trans=0.4 + 0.2 * (1 - abs(alpha) / 20))
    thickness_vel = _naca_thickness_velocity(x_upper, thickness)
    camber_slope = _camber_cp_contribution(x_upper, camber, camber_pos)

    cp_upper = (
        cp_min * peak_shape * recovery  # circulation-driven suction
        - 0.8 * thickness_vel / beta  # thickness velocity perturbation
        + 0.3 * camber_slope / beta  # camber contribution (rear loading)
        + _shock_cp(x_upper, mach, thickness, -cp_min * 0.15)  # shock (transonic)
    )

    # Stagnation point at x~0: Cp=+1 at the very leading edge
    cp_upper[0] = 1.0

    # ── Lower surface Cp ─────────────────────────────────────────────
    # Lower surface: mostly positive pressure (slight suction at high alpha)
    cp_min_lower = (0.05 + 0.5 * camber - 0.8 * np.radians(abs(alpha))) / beta
    cp_min_lower = np.clip(cp_min_lower, -0.8, 0.5) * stall_factor

    # Gentle shape: peaks near LE, flat recovery
    lower_shape = np.exp(-4.0 * x_lower)  # x_lower goes 1→0, so peak at x/c≈0 (mapped)
    # Map x_lower (1→0) to (0→1) for physical ordering
    x_l_phys = 1.0 - x_lower
    thickness_lower = _naca_thickness_velocity(x_l_phys, thickness)
    camber_lower = _camber_cp_contribution(x_l_phys, camber, camber_pos)

    cp_lower = (
        cp_min_lower * lower_shape * 0.6
        + 0.5 * thickness_lower / beta
        - 0.3 * camber_lower / beta
    )
    cp_lower[-1] = 1.0  # stagnation at LE (end of lower = physical LE)

    # ── Add realistic noise (CFD numerical + model uncertainty) ─────
    noise_scale = 0.015 + 0.025 * abs(alpha) / 15.0 + 0.03 * stall_ratio
    cp_upper += rng.normal(0, noise_scale, n_half)
    cp_lower += rng.normal(0, noise_scale * 0.6, n_half)

    # Smooth with 3-point running average (mimics surface mesh smoothness)
    cp_upper[1:-1] = 0.25 * cp_upper[:-2] + 0.5 * cp_upper[1:-1] + 0.25 * cp_upper[2:]
    cp_lower[1:-1] = 0.25 * cp_lower[:-2] + 0.5 * cp_lower[1:-1] + 0.25 * cp_lower[2:]

    # Physical bounds: Cp ∈ [-(1+M)²/β², 1.0]  (isentropic limit is ~+1 stagnation)
    cp_max_suction = -((1 + mach) ** 2) / (beta ** 2 + 1e-4)
    cp_upper = np.clip(cp_upper, max(cp_max_suction, -12.0), 1.05)
    cp_lower = np.clip(cp_lower, max(cp_max_suction, -12.0), 1.05)

    return np.concatenate([cp_upper, cp_lower]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Adapter class (compatible with VTKAdapter interface)
# ═══════════════════════════════════════════════════════════════

class SyntheticFieldAdapter:
    """Generates synthetic Cp(x/c) field data for DeepONet training.

    Produces (params_df, fields) matching the exact interface of VTKAdapter.load_fields().
    Uses the same 5-regime parameter distribution as the integral synthetic generator
    so the field model covers the same input domain as the ensemble.

    Config keys (all optional):
        n_samples    int   Number of Cp field samples to generate (default 3000)
        n_points     int   Points per surface wrap (default 200)
        random_state int   RNG seed (default 42)
    """

    def __init__(self, config: dict):
        self.n_samples = int(config.get("n_samples", 3000))
        self.n_points = int(config.get("n_points", 200))
        self.random_state = int(config.get("random_state", 42))

    # ── Public interface ──────────────────────────────────────

    def load_fields(self, n_points: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate and return (params_df, fields array).

        Returns:
            params_df: DataFrame with same columns as integral synthetic data.
            fields:    ndarray of shape (n_samples, n_points).
        """
        rng = np.random.RandomState(self.random_state)
        n = self.n_samples
        n_pts = n_points or self.n_points

        params = self._generate_params(n, rng)
        logger.info(
            f"SyntheticFieldAdapter: generating {n} Cp fields "
            f"({n_pts} points each)…"
        )

        fields = np.zeros((n, n_pts), dtype=np.float32)
        for i in range(n):
            fields[i] = generate_cp_distribution(
                alpha=params["alpha"][i],
                mach=params["mach"][i],
                thickness=params["thickness_ratio"][i],
                camber=params["camber"][i],
                camber_pos=params["camber_position"][i],
                reynolds=params["reynolds"][i],
                n_points=n_pts,
                rng=rng,
            )

        params_df = pd.DataFrame(params)
        logger.info(
            f"SyntheticFieldAdapter: done. Fields shape={fields.shape}, "
            f"Cp range=[{fields.min():.3f}, {fields.max():.3f}]"
        )
        return params_df, fields

    # ── Parameter generation (5 regimes, same as loader.py) ──

    def _generate_params(self, n: int, rng: np.random.RandomState) -> dict:
        """Sample aerodynamic parameters across 5 flow regimes."""
        per = n // 5
        chunks = []

        # Regime 1 — low-Re (kanakaero domain)
        chunks.append(dict(
            mach=rng.uniform(0.01, 0.10, per),
            reynolds=rng.uniform(5e4, 3e5, per),
            alpha=rng.uniform(-5, 18, per),
            thickness_ratio=rng.uniform(0.06, 0.24, per),
            camber=rng.uniform(0.0, 0.08, per),
            camber_position=rng.uniform(0.20, 0.60, per),
        ))

        # Regime 2 — medium-Re subsonic
        chunks.append(dict(
            mach=rng.uniform(0.10, 0.50, per),
            reynolds=rng.uniform(3e5, 3e6, per),
            alpha=rng.uniform(-5, 15, per),
            thickness_ratio=rng.uniform(0.06, 0.21, per),
            camber=rng.uniform(0.0, 0.06, per),
            camber_position=rng.uniform(0.25, 0.50, per),
        ))

        # Regime 3 — high-Re subsonic (transport)
        chunks.append(dict(
            mach=rng.uniform(0.30, 0.70, per),
            reynolds=rng.uniform(3e6, 2e7, per),
            alpha=rng.uniform(-3, 12, per),
            thickness_ratio=rng.uniform(0.08, 0.16, per),
            camber=rng.uniform(0.01, 0.05, per),
            camber_position=rng.uniform(0.30, 0.45, per),
        ))

        # Regime 4 — transonic (NASA CRM)
        chunks.append(dict(
            mach=rng.uniform(0.70, 0.95, per),
            reynolds=rng.uniform(5e6, 2e7, per),
            alpha=rng.uniform(-2, 8, per),
            thickness_ratio=rng.uniform(0.09, 0.14, per),
            camber=rng.uniform(0.01, 0.04, per),
            camber_position=rng.uniform(0.30, 0.40, per),
        ))

        # Regime 5 — remaining, broad sweep
        n5 = n - 4 * per
        chunks.append(dict(
            mach=rng.uniform(0.05, 0.90, n5),
            reynolds=rng.uniform(5e4, 2e7, n5),
            alpha=rng.uniform(-6, 20, n5),
            thickness_ratio=rng.uniform(0.05, 0.25, n5),
            camber=rng.uniform(0.0, 0.10, n5),
            camber_position=rng.uniform(0.20, 0.60, n5),
        ))

        merged = {}
        for key in chunks[0]:
            merged[key] = np.concatenate([c[key] for c in chunks])

        # Add extra columns to match integral dataset schema
        merged["leading_edge_radius"] = rng.uniform(0.005, 0.04, n)
        merged["trailing_edge_angle"] = rng.uniform(5, 25, n)
        merged["beta"] = np.zeros(n)
        merged["altitude"] = rng.uniform(0, 12000, n)

        return merged
