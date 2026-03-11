"""
Synthetic XFOIL-equivalent dataset generator for AeroSurrogate v2.0.

Uses analytical aerodynamic theory (thin airfoil + corrections) to
produce a realistic training dataset that mimics XFOIL output for:
  - NACA 0012, 2412, 4412
  - Random CST-parameterised profiles
  
Parameters swept:
  Re:    5e5 – 5e6
  alpha: -5° → 15°
  Mach:  0.05 – 0.3
"""

import math
import random

import numpy as np
import pandas as pd

# ── reuse demo_model physics ───────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.demo.demo_model import (
    _prandtl_glauert, _stall_factor, _finite_wing_factor,
    _compute_cl, _compute_cd, _compute_cm,
)

RNG = np.random.default_rng(42)

# ── NACA profile definitions ────────────────────────────────────────────────────

PROFILES = {
    "NACA0012": dict(thickness_ratio=0.12, camber=0.00, camber_position=0.30),
    "NACA2412": dict(thickness_ratio=0.12, camber=0.02, camber_position=0.40),
    "NACA4412": dict(thickness_ratio=0.12, camber=0.04, camber_position=0.40),
}


def random_cst_profile() -> dict:
    """Random airfoil via crude CST-like parameter sampling."""
    t = RNG.uniform(0.06, 0.22)
    c = RNG.uniform(0.00, 0.08)
    cp = RNG.uniform(0.25, 0.60)
    return dict(thickness_ratio=t, camber=c, camber_position=cp)


def add_noise(val: float, frac: float = 0.01) -> float:
    """Add ±frac*|val| Gaussian noise to simulate XFOIL scatter."""
    sigma = abs(val) * frac + 1e-6
    return float(val + RNG.normal(0, sigma))


def generate_dataset(n_samples: int = 60_000) -> pd.DataFrame:
    rows = []

    # Fractions per profile type
    naca_samples = int(n_samples * 0.6)        # 60 % NACA
    cst_samples  = n_samples - naca_samples    # 40 % random

    # NACA sweep
    per_naca = naca_samples // len(PROFILES)
    for name, geom in PROFILES.items():
        for _ in range(per_naca):
            alpha = float(RNG.uniform(-5, 15))
            mach  = float(RNG.uniform(0.05, 0.30))
            re    = float(RNG.uniform(5e5, 5e6))
            ar    = float(RNG.uniform(5, 12))
            tr    = float(RNG.uniform(0.3, 0.8))
            sw    = float(RNG.uniform(0, 30))

            cl = _compute_cl(alpha, geom["camber"], mach, ar, sw, tr)
            cd = _compute_cd(cl, geom["thickness_ratio"], mach, ar, sw, re)
            cm = _compute_cm(alpha, geom["camber"], geom["camber_position"], mach)
            k  = cl / max(cd, 1e-6)

            rows.append({
                "profile": name,
                "thickness_ratio": geom["thickness_ratio"],
                "camber": geom["camber"],
                "camber_position": geom["camber_position"],
                "leading_edge_radius": round(0.5 * geom["thickness_ratio"] ** 2, 5),
                "trailing_edge_angle": round(12 + 20 * geom["thickness_ratio"], 2),
                "aspect_ratio": round(ar, 3),
                "taper_ratio": round(tr, 3),
                "sweep_angle": round(sw, 2),
                "mach": round(mach, 4),
                "reynolds": round(re, 0),
                "alpha": round(alpha, 3),
                "Cl": round(add_noise(cl, 0.015), 6),
                "Cd": round(max(add_noise(cd, 0.025), 0.003), 6),
                "Cm": round(add_noise(cm, 0.02), 6),
                "K": round(k, 4),
            })

    # CST / random profiles
    for _ in range(cst_samples):
        geom = random_cst_profile()
        alpha = float(RNG.uniform(-5, 15))
        mach  = float(RNG.uniform(0.05, 0.30))
        re    = float(RNG.uniform(5e5, 5e6))
        ar    = float(RNG.uniform(4, 14))
        tr    = float(RNG.uniform(0.2, 0.9))
        sw    = float(RNG.uniform(0, 40))

        cl = _compute_cl(alpha, geom["camber"], mach, ar, sw, tr)
        cd = _compute_cd(cl, geom["thickness_ratio"], mach, ar, sw, re)
        cm = _compute_cm(alpha, geom["camber"], geom["camber_position"], mach)
        k  = cl / max(cd, 1e-6)

        rows.append({
            "profile": "CST_random",
            "thickness_ratio": round(geom["thickness_ratio"], 4),
            "camber": round(geom["camber"], 4),
            "camber_position": round(geom["camber_position"], 3),
            "leading_edge_radius": round(0.5 * geom["thickness_ratio"] ** 2, 5),
            "trailing_edge_angle": round(12 + 20 * geom["thickness_ratio"], 2),
            "aspect_ratio": round(ar, 3),
            "taper_ratio": round(tr, 3),
            "sweep_angle": round(sw, 2),
            "mach": round(mach, 4),
            "reynolds": round(re, 0),
            "alpha": round(alpha, 3),
            "Cl": round(add_noise(cl, 0.015), 6),
            "Cd": round(max(add_noise(cd, 0.025), 0.003), 6),
            "Cm": round(add_noise(cm, 0.02), 6),
            "K": round(k, 4),
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(df):,} samples. Columns: {list(df.columns)}")
    print(df[["Cl", "Cd", "Cm"]].describe().round(4))
    return df


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    df = generate_dataset(60_000)
    out = "dataset/xfoil_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
