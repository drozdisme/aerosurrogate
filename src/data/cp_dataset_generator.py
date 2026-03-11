"""Generate Cp(x) training dataset for the DeepONet surrogate."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from src.demo.demo_model import _prandtl_glauert

RNG = np.random.default_rng(42)
N_PROFILES = 4_000
N_X_POINTS = 30
TOTAL = N_PROFILES * N_X_POINTS   # 120k rows — fast to train


def _cp_profile(alpha_deg, mach, thickness, camber, camber_pos, n=N_X_POINTS):
    pg = _prandtl_glauert(mach)
    a  = alpha_deg * np.pi / 180
    p  = max(0.1, min(camber_pos, 0.9))
    x  = np.linspace(0.0, 1.0, n)
    dyc = np.where(x < p,
        (2*camber/p**2)*(p-x),
        (2*camber/(1-p)**2)*(p-x))
    cp = -2.0*(a + dyc)*pg
    cp += -0.6*(abs(alpha_deg)/5.0)*np.exp(-40*x)
    cp += (1.0 - cp[-1]) * np.exp(-8*(1-x)) * 0.15
    cp = np.clip(cp + RNG.normal(0, 0.015, n), -5.0, 1.0)
    return x, cp


def generate_cp_dataset():
    rows = []
    for _ in range(N_PROFILES):
        alpha  = float(RNG.uniform(-5, 15))
        mach   = float(RNG.uniform(0.05, 0.50))
        re     = float(RNG.uniform(5e5, 5e6))
        t      = float(RNG.uniform(0.06, 0.22))
        c      = float(RNG.uniform(0.00, 0.08))
        cp_loc = float(RNG.uniform(0.25, 0.65))
        ar     = float(RNG.uniform(5, 12))
        tr     = float(RNG.uniform(0.3, 0.8))
        sw     = float(RNG.uniform(0, 35))
        x_arr, cp_arr = _cp_profile(alpha, mach, t, c, cp_loc)
        for x_val, cp_val in zip(x_arr, cp_arr):
            rows.append({"thickness_ratio":t,"camber":c,"camber_position":cp_loc,
                "aspect_ratio":ar,"taper_ratio":tr,"sweep_angle":sw,
                "mach":mach,"reynolds":re,"alpha":alpha,
                "x_pos":round(float(x_val),4),"Cp":round(float(cp_val),6)})
    df = pd.DataFrame(rows)
    print(f"Cp dataset: {len(df):,} rows  Cp range [{df.Cp.min():.3f}, {df.Cp.max():.3f}]")
    return df


if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    df = generate_cp_dataset()
    df.to_csv("dataset/cp_dataset.csv", index=False)
    print("Saved → dataset/cp_dataset.csv")
