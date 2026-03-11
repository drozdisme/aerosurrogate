"""Data loading utilities for AeroSurrogate v2.0.

Enhanced synthetic generator covers parameter domains of:
- kanakaero (Re=1e5, low-speed, 2900 profiles)
- NASA airfoil-learning (Re=5e4..5e6, multi-alpha)
- DOE Airfoil-2k (Re=2e5, 5e5, 1e6; 25 AoA; transition model)
- NASA CRM (transonic, Re=5e6, M=0.85, 3D wing)
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate multi-regime synthetic aerodynamic dataset.

    Covers five flow regimes to match the domains of real open datasets:
      Regime 1: Low-Re incompressible (Re=5e4..3e5, M<0.1) — kanakaero domain
      Regime 2: Medium-Re subsonic    (Re=3e5..3e6, M=0.1..0.5) — UIUC/DOE domain
      Regime 3: High-Re subsonic      (Re=3e6..2e7, M=0.3..0.7) — transport aircraft
      Regime 4: Transonic             (Re=5e6..2e7, M=0.7..0.95) — NASA CRM domain
      Regime 5: 3D corrections        (AR=4..12, sweep/taper effects)
    """
    rng = np.random.RandomState(random_state)
    per_regime = n_samples // 5
    frames = []

    # ── Regime 1: Low-Re (kanakaero domain) ────────────────────
    n1 = per_regime
    re1 = rng.uniform(5e4, 3e5, n1)
    m1 = rng.uniform(0.01, 0.1, n1)
    a1 = rng.uniform(-5, 18, n1)
    t1 = rng.uniform(0.06, 0.24, n1)
    cam1 = rng.uniform(0.0, 0.08, n1)
    xc1 = rng.uniform(0.2, 0.6, n1)

    # Low-Re: laminar separation effects
    a_rad = np.radians(a1)
    cl_slope = 2 * np.pi / (1 + 2 / 8.0)  # AR=8 default
    cl1 = 2 * np.pi * cam1 + cl_slope * a_rad
    # Stall model at low Re
    stall_angle = 10 + rng.uniform(-2, 2, n1)
    stall_mask = np.abs(a1) > stall_angle
    cl1[stall_mask] *= (1.0 - 0.3 * (np.abs(a1[stall_mask]) - stall_angle[stall_mask]) / 10)
    cl1 += rng.normal(0, 0.03, n1)

    # Low-Re drag: laminar bubble + separation
    cf1 = 1.328 / np.sqrt(re1)
    cd_profile = cf1 * (1 + 2 * t1 + 60 * t1 ** 4)
    cd_induced = cl1 ** 2 / (np.pi * 8 * 0.8)
    cd1 = cd_profile + cd_induced + rng.normal(0, 0.002, n1)
    cd1 = np.clip(cd1, 0.003, None)
    cm1 = -0.25 * cl1 + 0.1 * xc1 * cl1 + rng.normal(0, 0.008, n1)

    frames.append(_make_df(n1, rng, t1, cam1, xc1, m1, re1, a1, cl1, cd1, cm1, "low_re"))

    # ── Regime 2: Medium-Re subsonic (DOE/UIUC domain) ─────────
    n2 = per_regime
    re2 = rng.uniform(3e5, 3e6, n2)
    m2 = rng.uniform(0.1, 0.5, n2)
    a2 = rng.uniform(-5, 15, n2)
    t2 = rng.uniform(0.06, 0.21, n2)
    cam2 = rng.uniform(0.0, 0.06, n2)
    xc2 = rng.uniform(0.25, 0.5, n2)

    a_rad2 = np.radians(a2)
    beta_prandtl = np.sqrt(1 - m2 ** 2)
    cl_slope2 = 2 * np.pi / (beta_prandtl + 2 / 8.0)
    cl2 = 2 * np.pi * cam2 + cl_slope2 * a_rad2 + rng.normal(0, 0.02, n2)

    cf2 = 0.455 / (np.log10(re2) ** 2.58)
    cd2 = cf2 * (1 + 2 * t2) + cl2 ** 2 / (np.pi * 8 * 0.85)
    cd2 += rng.normal(0, 0.001, n2)
    cd2 = np.clip(cd2, 0.002, None)
    cm2 = -0.25 * cl2 + 0.1 * xc2 * cl2 + rng.normal(0, 0.005, n2)

    frames.append(_make_df(n2, rng, t2, cam2, xc2, m2, re2, a2, cl2, cd2, cm2, "medium_re"))

    # ── Regime 3: High-Re subsonic ─────────────────────────────
    n3 = per_regime
    re3 = rng.uniform(3e6, 2e7, n3)
    m3 = rng.uniform(0.3, 0.7, n3)
    a3 = rng.uniform(-3, 12, n3)
    t3 = rng.uniform(0.08, 0.16, n3)
    cam3 = rng.uniform(0.01, 0.05, n3)
    xc3 = rng.uniform(0.3, 0.45, n3)

    a_rad3 = np.radians(a3)
    beta3 = np.sqrt(np.clip(1 - m3 ** 2, 0.01, None))
    cl3 = 2 * np.pi * cam3 + (2 * np.pi / (beta3 + 2 / 9.0)) * a_rad3
    cl3 += rng.normal(0, 0.015, n3)

    cf3 = 0.455 / (np.log10(re3) ** 2.58)
    cd3 = cf3 * (1 + 2 * t3) + cl3 ** 2 / (np.pi * 9 * 0.88)
    cd3 += rng.normal(0, 0.0008, n3)
    cd3 = np.clip(cd3, 0.0015, None)
    cm3 = -0.25 * cl3 + 0.08 * xc3 * cl3 + rng.normal(0, 0.004, n3)

    frames.append(_make_df(n3, rng, t3, cam3, xc3, m3, re3, a3, cl3, cd3, cm3, "high_re"))

    # ── Regime 4: Transonic (NASA CRM domain) ──────────────────
    n4 = per_regime
    re4 = rng.uniform(5e6, 2e7, n4)
    m4 = rng.uniform(0.7, 0.95, n4)
    a4 = rng.uniform(-2, 8, n4)
    t4 = rng.uniform(0.09, 0.14, n4)
    cam4 = rng.uniform(0.01, 0.04, n4)
    xc4 = rng.uniform(0.3, 0.4, n4)

    a_rad4 = np.radians(a4)
    # Prandtl-Glauert with transonic correction
    beta4 = np.sqrt(np.clip(1 - m4 ** 2, 0.01, None))
    cl4 = 2 * np.pi * cam4 + (2 * np.pi / (beta4 + 2 / 9.0)) * a_rad4

    # Wave drag (Korn equation approximation)
    m_crit = 0.87 - t4 / 3 - cl4 / 10
    m_excess = np.clip(m4 - m_crit, 0, None)
    cd_wave = 20 * m_excess ** 3

    cf4 = 0.455 / (np.log10(re4) ** 2.58)
    cd4 = cf4 * (1 + 2 * t4) + cl4 ** 2 / (np.pi * 9 * 0.86) + cd_wave
    cd4 += rng.normal(0, 0.001, n4)
    cd4 = np.clip(cd4, 0.002, None)
    cl4 += rng.normal(0, 0.01, n4)
    cm4 = -0.25 * cl4 + 0.06 * xc4 * cl4 - 0.02 * m_excess + rng.normal(0, 0.003, n4)

    frames.append(_make_df(n4, rng, t4, cam4, xc4, m4, re4, a4, cl4, cd4, cm4, "transonic"))

    # ── Regime 5: 3D wing effects ──────────────────────────────
    n5 = n_samples - 4 * per_regime
    re5 = rng.uniform(1e6, 1e7, n5)
    m5 = rng.uniform(0.2, 0.85, n5)
    a5 = rng.uniform(-3, 12, n5)
    t5 = rng.uniform(0.08, 0.18, n5)
    cam5 = rng.uniform(0.01, 0.06, n5)
    xc5 = rng.uniform(0.25, 0.5, n5)
    ar5 = rng.uniform(4, 12, n5)
    taper5 = rng.uniform(0.2, 1.0, n5)
    sweep5 = rng.uniform(0, 45, n5)
    twist5 = rng.uniform(-4, 4, n5)
    dihedral5 = rng.uniform(-2, 8, n5)

    a_rad5 = np.radians(a5)
    sweep_rad5 = np.radians(sweep5)
    beta5 = np.sqrt(np.clip(1 - m5 ** 2, 0.01, None))

    # Lifting-line with sweep correction
    cl_alpha_3d = 2 * np.pi * np.cos(sweep_rad5) / (beta5 + 2 * np.cos(sweep_rad5) / ar5)
    cl5 = 2 * np.pi * cam5 + cl_alpha_3d * a_rad5
    cl5 += rng.normal(0, 0.02, n5)

    e5 = 0.9 - 0.02 * np.abs(sweep5) / 45  # Oswald factor
    cf5 = 0.455 / (np.log10(re5) ** 2.58)
    cd5 = cf5 * (1 + 2 * t5) + cl5 ** 2 / (np.pi * ar5 * e5)
    m_crit5 = 0.87 * np.cos(sweep_rad5) - t5 / (3 * np.cos(sweep_rad5)) - cl5 / (10 * np.cos(sweep_rad5) ** 2)
    cd_wave5 = 20 * np.clip(m5 - m_crit5, 0, None) ** 3
    cd5 += cd_wave5 + rng.normal(0, 0.001, n5)
    cd5 = np.clip(cd5, 0.002, None)
    cm5 = -0.25 * cl5 + 0.08 * xc5 * cl5 + rng.normal(0, 0.004, n5)

    df5 = _make_df(n5, rng, t5, cam5, xc5, m5, re5, a5, cl5, cd5, cm5, "3d_wing")
    df5["aspect_ratio"] = ar5
    df5["taper_ratio"] = taper5
    df5["sweep_angle"] = sweep5
    df5["twist_angle"] = twist5
    df5["dihedral_angle"] = dihedral5
    frames.append(df5)

    # ── Combine all regimes ────────────────────────────────────
    df = pd.concat(frames, ignore_index=True, sort=False)

    # Fill missing geometry columns with defaults
    defaults = {
        "aspect_ratio": 8.0, "taper_ratio": 0.5, "sweep_angle": 0.0,
        "twist_angle": 0.0, "dihedral_angle": 0.0,
        "leading_edge_radius": 0.015, "trailing_edge_angle": 12.0,
        "beta": 0.0, "altitude": 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)
    df = df.drop(columns=["_regime"], errors="ignore")

    logger.info(f"Generated multi-regime synthetic dataset: {len(df)} samples across 5 regimes")
    return df


def _make_df(n, rng, t, cam, xc, m, re, a, cl, cd, cm, regime):
    """Helper to create regime DataFrame."""
    return pd.DataFrame({
        "thickness_ratio": t,
        "camber": cam,
        "camber_position": xc,
        "leading_edge_radius": rng.uniform(0.005, 0.04, n),
        "trailing_edge_angle": rng.uniform(5, 25, n),
        "mach": m,
        "reynolds": re,
        "alpha": a,
        "beta": np.zeros(n),
        "altitude": rng.uniform(0, 12000, n),
        "Cl": cl,
        "Cd": cd,
        "Cm": cm,
        "_regime": regime,
    })
