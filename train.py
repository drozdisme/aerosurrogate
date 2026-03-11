"""
AeroSurrogate v2 — unified training pipeline.

Trains:
  1. Cl / Cd / Cm coefficient models  (XGBoost → LightGBM → sklearn GBM)
  2. Cp(x) field surrogate            (DeepONet-like; same backend priority)

Run:
  python train.py
  MODELS_DIR=models ARTIFACTS_DIR=artifacts python train.py
"""

import json, os, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:    import xgboost as xgb;     HAS_XGB = True
except: HAS_XGB = False
try:    import lightgbm as lgb;    HAS_LGB = True
except: HAS_LGB = False

MODELS_DIR    = Path(os.getenv("MODELS_DIR",    "models"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DATASET       = os.getenv("DATASET_PATH",    "dataset/xfoil_dataset.csv")
CP_DATASET    = os.getenv("CP_DATASET_PATH", "dataset/cp_dataset.csv")

COEF_BASE  = ["thickness_ratio","camber","camber_position","leading_edge_radius",
              "trailing_edge_angle","aspect_ratio","taper_ratio","sweep_angle",
              "mach","reynolds","alpha"]
COEF_EXTRA = ["mach_alpha","alpha2","alpha3","log_re","ar_tr","tc_ratio","pg"]
COEF_FEATS = COEF_BASE + COEF_EXTRA
TARGETS    = ["Cl","Cd","Cm"]

CP_BASE    = ["thickness_ratio","camber","camber_position","aspect_ratio",
              "taper_ratio","sweep_angle","mach","reynolds","alpha"]
CP_EXTRA   = ["mach_alpha","alpha2","alpha3","log_re",
              "x_pos","x_sqrt","one_minus_x","mach_x","alpha_x"]
CP_FEATS   = CP_BASE + CP_EXTRA


def _coef_feats(df):
    df = df.copy()
    df["mach_alpha"] = df["mach"] * df["alpha"]
    df["alpha2"]     = df["alpha"]**2
    df["alpha3"]     = df["alpha"]**3
    df["log_re"]     = np.log1p(df["reynolds"])
    df["ar_tr"]      = df["aspect_ratio"] * df["taper_ratio"]
    df["tc_ratio"]   = df["thickness_ratio"] / df["camber"].replace(0, 1e-8)
    df["pg"]         = 1.0 / np.sqrt(1.0 - df["mach"].clip(upper=0.999)**2)
    return df


def _cp_feats(df):
    df = df.copy()
    df["mach_alpha"]  = df["mach"]  * df["alpha"]
    df["alpha2"]      = df["alpha"]**2
    df["alpha3"]      = df["alpha"]**3
    df["log_re"]      = np.log1p(df["reynolds"])
    x = df["x_pos"].clip(lower=1e-4)
    df["x_sqrt"]      = np.sqrt(x)
    df["one_minus_x"] = 1.0 - df["x_pos"]
    df["mach_x"]      = df["mach"]  * df["x_pos"]
    df["alpha_x"]     = df["alpha"] * df["x_pos"]
    return df


def _best_model(tag="coef", n_est=300):
    if HAS_XGB:
        print(f"  [{tag}] backend: XGBoost")
        return xgb.XGBRegressor(n_estimators=n_est, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
    if HAS_LGB:
        print(f"  [{tag}] backend: LightGBM")
        return lgb.LGBMRegressor(n_estimators=n_est, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    print(f"  [{tag}] backend: sklearn GBM")
    return GradientBoostingRegressor(n_estimators=min(n_est, 150), max_depth=5,
        learning_rate=0.08, subsample=0.8, random_state=42)


# ── 1. Coefficient models ─────────────────────────────────────────────────────

def train_coef():
    print("\n" + "─"*55)
    print("  COEFFICIENT MODELS  (Cl / Cd / Cm)")
    print("─"*55)

    if not Path(DATASET).exists():
        print("  Generating xfoil dataset...")
        os.makedirs("dataset", exist_ok=True)
        from src.data.xfoil_generator import generate_dataset
        generate_dataset().to_csv(DATASET, index=False)

    df = pd.read_csv(DATASET)
    df = _coef_feats(df)
    print(f"  Rows: {len(df):,}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}

    for t in TARGETS:
        t0 = time.time()
        X  = df[COEF_FEATS].values.astype(np.float32)
        y  = df[t].values.astype(np.float32)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.15, random_state=42)

        m = _best_model(t, n_est=300)
        m.fit(Xtr, ytr)
        r2 = r2_score(yte, m.predict(Xte))
        print(f"  {t}: R²={r2:.4f}  ({time.time()-t0:.1f}s)")
        metrics[f"{t}_R2"] = round(float(r2), 4)

        with open(MODELS_DIR / f"{t.lower()}_model.pkl", "wb") as f:
            pickle.dump(m, f)

    return metrics


# ── 2. Cp surrogate ───────────────────────────────────────────────────────────

def train_cp():
    print("\n" + "─"*55)
    print("  Cp(x) SURROGATE  (DeepONet-like)")
    print("─"*55)

    if not Path(CP_DATASET).exists():
        print("  Generating Cp dataset...")
        os.makedirs("dataset", exist_ok=True)
        from src.data.cp_dataset_generator import generate_cp_dataset
        generate_cp_dataset().to_csv(CP_DATASET, index=False)

    df = pd.read_csv(CP_DATASET)
    df = _cp_feats(df)
    print(f"  Rows: {len(df):,}")

    X  = df[CP_FEATS].values.astype(np.float32)
    y  = df["Cp"].values.astype(np.float32)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.10, random_state=42)

    t0 = time.time()
    m  = _best_model("Cp", n_est=400)
    m.fit(Xtr, ytr)
    r2  = r2_score(yte, m.predict(Xte))
    mae = float(np.mean(np.abs(m.predict(Xte) - yte)))
    print(f"  Cp: R²={r2:.4f}  MAE={mae:.4f}  ({time.time()-t0:.1f}s)")

    with open(MODELS_DIR / "cp_surrogate.pkl", "wb") as f:
        pickle.dump(m, f)
    with open(MODELS_DIR / "cp_features.json", "w") as f:
        json.dump({"features": CP_FEATS}, f)

    return {"Cp_R2": round(float(r2), 4), "Cp_MAE": round(mae, 5)}


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    t_start = time.time()
    coef = train_coef()
    cp   = train_cp()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    backend = "xgboost" if HAS_XGB else ("lightgbm" if HAS_LGB else "sklearn")
    meta = {**coef, **cp, "model_version":"v2", "backend":backend,
            "n_samples":60000, "features":COEF_FEATS}
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  Done in {time.time()-t_start:.0f}s")
    print(json.dumps({k:v for k,v in meta.items() if k!="features"}, indent=2))
    return meta


if __name__ == "__main__":
    train()
