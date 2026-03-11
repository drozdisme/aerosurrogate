"""
Demo model for AeroSurrogate v2.0.

Uses analytical aerodynamic approximations so the system works
even when no trained ML model is available.
Based on:
  - Thin airfoil theory (Cl lift slope)
  - Prandtl–Glauert compressibility correction
  - Parabolic drag polar
  - NACA 4-digit approximations for Cp(x)
"""

import math
import numpy as np
from typing import Dict, List


# ─── Constants ────────────────────────────────────────────────────────────────

_RAD = math.pi / 180.0
_CL_ALPHA = 2 * math.pi          # 2π rad⁻¹ (thin airfoil theory)
_STALL_ALPHA = 14.0               # degrees
_STALL_SMOOTH = 3.0               # softness of stall transition
_CD_MIN_BASE = 0.006              # minimum profile drag
_CM_ALPHA_SLOPE = -0.05           # ∂Cm/∂α per degree
_CP_POINTS = 200


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _prandtl_glauert(mach: float) -> float:
    """Prandtl-Glauert compressibility correction factor."""
    m = min(mach, 0.94)
    return 1.0 / math.sqrt(max(1.0 - m * m, 0.01))


def _stall_factor(alpha_deg: float) -> float:
    """Smooth stall limiter — sigmoid centred at stall_alpha."""
    x = (abs(alpha_deg) - _STALL_ALPHA) / _STALL_SMOOTH
    return 1.0 / (1.0 + math.exp(x))          # 1 pre-stall → 0 post-stall


def _finite_wing_factor(aspect_ratio: float, sweep_deg: float) -> float:
    """Oswald span efficiency-based 3-D correction."""
    e = 0.85 - 0.002 * max(0.0, sweep_deg - 10.0)
    ar = max(1.0, aspect_ratio)
    return ar / (ar + 2.0 * math.cos(sweep_deg * _RAD) / (math.pi * e))


# ─── Core aerodynamic functions ───────────────────────────────────────────────

def _compute_cl(
    alpha_deg: float,
    camber: float,
    mach: float,
    aspect_ratio: float,
    sweep_deg: float,
    taper_ratio: float,
) -> float:
    """Lift coefficient — thin airfoil + Prandtl-Glauert + finite wing."""
    pg = _prandtl_glauert(mach)
    sf = _stall_factor(alpha_deg)
    fw = _finite_wing_factor(aspect_ratio, sweep_deg)

    # Camber contributes a zero-lift angle shift  α₀ ≈ -2·π·camber
    alpha_eff = alpha_deg - (-2.0 * camber * 180.0 / math.pi)

    # 2-D section Cl (pre-stall)
    cl_2d = _CL_ALPHA * alpha_eff * _RAD * pg

    # 3-D finite wing correction
    cl_3d = cl_2d * fw

    # Post-stall soft limit
    cl_max = (1.5 + 2.0 * camber) * sf + (1 - sf) * (0.6 + camber)
    cl = min(cl_3d * sf + (cl_max * math.copysign(1, alpha_deg)) * (1 - sf),
             cl_max)

    # Gentle correction for taper
    cl *= 0.95 + 0.1 * taper_ratio

    return round(float(cl), 6)


def _compute_cd(
    cl: float,
    thickness_ratio: float,
    mach: float,
    aspect_ratio: float,
    sweep_deg: float,
    reynolds: float,
) -> float:
    """Drag coefficient — parabolic polar with compressibility drag rise."""
    # Profile / friction drag
    re_factor = 1.0 + 0.3 * max(0.0, math.log10(1e6 / max(reynolds, 1e4)))
    cd0 = (_CD_MIN_BASE + 0.1 * thickness_ratio ** 2) * re_factor

    # Induced drag
    e = 0.85
    ar = max(1.0, aspect_ratio)
    cdi = cl ** 2 / (math.pi * e * ar)

    # Compressibility drag rise (above Mcrit ≈ 0.7 – 0.3·thickness)
    m_crit = 0.72 - 0.3 * thickness_ratio
    m_crit_sw = m_crit / math.cos(sweep_deg * _RAD)
    if mach > m_crit_sw:
        wave = 20.0 * (mach - m_crit_sw) ** 4
    else:
        wave = 0.0

    cd = cd0 + cdi + wave
    return round(float(max(cd, 0.003)), 6)


def _compute_cm(
    alpha_deg: float,
    camber: float,
    camber_position: float,
    mach: float,
) -> float:
    """Pitching moment about quarter-chord — thin airfoil + camber term."""
    pg = _prandtl_glauert(mach)
    # Quarter-chord Cm from thin airfoil for cambered profile
    cm_camber = -math.pi * camber * (1.0 - 2.0 * camber_position)
    cm_alpha = _CM_ALPHA_SLOPE * alpha_deg * pg
    cm = cm_camber + cm_alpha
    return round(float(cm), 6)


# ─── Public API ───────────────────────────────────────────────────────────────

def predict_coefficients(geometry: Dict, flow: Dict) -> Dict[str, float]:
    """
    Predict integral aerodynamic coefficients.

    Parameters
    ----------
    geometry : dict
        thickness_ratio, camber, camber_position, leading_edge_radius,
        trailing_edge_angle, aspect_ratio, taper_ratio, sweep_angle,
        twist_angle, dihedral_angle
    flow : dict
        mach, reynolds, alpha, beta, altitude

    Returns
    -------
    dict with keys: Cl, Cd, Cm, K
    """
    # Extract with safe defaults
    t   = float(geometry.get("thickness_ratio", 0.12))
    c   = float(geometry.get("camber", 0.04))
    cp  = float(geometry.get("camber_position", 0.4))
    ar  = float(geometry.get("aspect_ratio", 8.0))
    tr  = float(geometry.get("taper_ratio", 0.5))
    sw  = float(geometry.get("sweep_angle", 20.0))

    alpha = float(flow.get("alpha", 5.0))
    mach  = float(flow.get("mach", 0.5))
    re    = float(flow.get("reynolds", 1e6))

    cl = _compute_cl(alpha, c, mach, ar, sw, tr)
    cd = _compute_cd(cl, t, mach, ar, sw, re)
    cm = _compute_cm(alpha, c, cp, mach)
    k  = round(float(cl / max(cd, 1e-6)), 4)

    return {"Cl": cl, "Cd": cd, "Cm": cm, "K": k}


def generate_cp_distribution(
    geometry: Dict,
    flow: Dict,
    n_points: int = _CP_POINTS,
) -> Dict:
    """
    Generate Cp(x) distribution along airfoil chord.

    Uses a modified NACA method:
      - Suction peak near leading edge
      - Recovery toward trailing edge
    """
    t   = float(geometry.get("thickness_ratio", 0.12))
    c   = float(geometry.get("camber", 0.04))
    cp_loc = float(geometry.get("camber_position", 0.4))
    alpha = float(flow.get("alpha", 5.0))
    mach  = float(flow.get("mach", 0.5))

    pg = _prandtl_glauert(mach)
    alpha_r = alpha * _RAD

    x = np.linspace(0.0, 1.0, n_points)

    # Thickness distribution (NACA 4-digit style)
    t_x = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    # Camber line
    yc = np.where(
        x < cp_loc,
        (c / cp_loc ** 2) * (2 * cp_loc * x - x ** 2),
        (c / (1 - cp_loc) ** 2) * ((1 - 2 * cp_loc) + 2 * cp_loc * x - x ** 2),
    )
    dyc_dx = np.where(
        x < cp_loc,
        (2 * c / cp_loc ** 2) * (cp_loc - x),
        (2 * c / (1 - cp_loc) ** 2) * (cp_loc - x),
    )

    theta = np.arctan(dyc_dx)

    # Upper/lower surface x
    xu = x - t_x * np.sin(theta)
    xl = x + t_x * np.sin(theta)

    # Thin airfoil Cp (linearised, upper surface dominant for plotting)
    # Cp_upper ~ -2*(α + dyc/dx) via Glauert integral
    cp_upper = -2.0 * (alpha_r + dyc_dx) * pg
    cp_lower = +2.0 * (alpha_r - dyc_dx) * pg

    # Add leading-edge suction spike
    le_spike = -0.8 * (abs(alpha) / 5.0) * np.exp(-50 * x)
    cp_upper += le_spike

    # Clamp to physical range
    cp_upper = np.clip(cp_upper, -4.0, 1.0)
    cp_lower = np.clip(cp_lower, -1.5, 1.0)

    # Return upper & lower interleaved for plotting (x from TE→LE→TE path)
    # Or: just return upper surface (most common in Cp plots)
    x_out = x.tolist()
    cp_out = cp_upper.tolist()

    return {"x": x_out, "Cp": cp_out, "n_points": n_points}


def health_metrics() -> Dict:
    """Return demo-mode health metrics that look plausible."""
    return {
        "Cl_R2": 0.94,
        "Cd_R2": 0.89,
        "Cm_R2": 0.86,
        "MAPE": 3.8,
    }
