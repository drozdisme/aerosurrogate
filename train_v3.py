"""
AeroSurrogate v3.0 — Unified training entry point.

Produces exactly what SklearnPredictor expects:

    models/
        cl_model.pkl
        cd_model.pkl
        cm_model.pkl
        cp_surrogate.pkl
        fno.pt              (if torch available and SKIP_FNO != 1)
    artifacts/
        metrics.json

Environment variables (all optional):
    MODELS_DIR      output dir for model files   (default: models)
    ARTIFACTS_DIR   output dir for metrics.json  (default: artifacts)
    DATASET_PATH    path to coefficient CSV       (default: dataset/xfoil_dataset.csv)
    CP_DATASET_PATH path to Cp CSV               (default: dataset/cp_dataset.csv)
    SKIP_FNO        "1" to skip FNO training     (default: 0)
    N_SAMPLES       samples for synthetic data    (default: 60000)
    FNO_PROFILES    FNO training profiles         (default: 3000)
    FNO_EPOCHS      FNO training epochs           (default: 150)
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path(os.getenv("MODELS_DIR",    "models"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DATASET_PATH  = Path(os.getenv("DATASET_PATH",    "dataset/xfoil_dataset.csv"))
CP_DATASET    = Path(os.getenv("CP_DATASET_PATH", "dataset/cp_dataset.csv"))
SKIP_FNO      = os.getenv("SKIP_FNO", "0") == "1"
N_SAMPLES     = int(os.getenv("N_SAMPLES",    "60000"))
FNO_PROFILES  = int(os.getenv("FNO_PROFILES", "3000"))
FNO_EPOCHS    = int(os.getenv("FNO_EPOCHS",   "150"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train_v3")

# ── Optional ML backends ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Feature columns — must match SklearnPredictor exactly ──────────────────────
COEF_BASE  = [
    "thickness_ratio", "camber", "camber_position",
    "leading_edge_radius", "trailing_edge_angle",
    "aspect_ratio", "taper_ratio", "sweep_angle",
    "mach", "reynolds", "alpha",
]
COEF_EXTRA = ["mach_alpha", "alpha2", "alpha3", "log_re", "ar_tr", "tc_ratio", "pg"]
COEF_FEATS = COEF_BASE + COEF_EXTRA

CP_BASE    = [
    "thickness_ratio", "camber", "camber_position",
    "aspect_ratio", "taper_ratio", "sweep_angle",
    "mach", "reynolds", "alpha",
]
CP_EXTRA   = ["mach_alpha", "alpha2", "alpha3", "log_re",
               "x_pos", "x_sqrt", "one_minus_x", "mach_x", "alpha_x"]
CP_FEATS   = CP_BASE + CP_EXTRA

TARGETS = ["Cl", "Cd", "Cm"]


# ── Feature engineering ─────────────────────────────────────────────────────────

def _add_coef_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mach_alpha"] = df["mach"] * df["alpha"]
    df["alpha2"]     = df["alpha"] ** 2
    df["alpha3"]     = df["alpha"] ** 3
    df["log_re"]     = np.log1p(df["reynolds"])
    df["ar_tr"]      = df["aspect_ratio"] * df["taper_ratio"]
    df["tc_ratio"]   = df["thickness_ratio"] / df["camber"].replace(0, 1e-8)
    m = df["mach"].clip(upper=0.999)
    df["pg"]         = 1.0 / np.sqrt(1.0 - m ** 2)
    return df


def _add_cp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mach_alpha"]  = df["mach"] * df["alpha"]
    df["alpha2"]      = df["alpha"] ** 2
    df["alpha3"]      = df["alpha"] ** 3
    df["log_re"]      = np.log1p(df["reynolds"])
    x = df["x_pos"].clip(lower=1e-4)
    df["x_sqrt"]      = np.sqrt(x)
    df["one_minus_x"] = 1.0 - df["x_pos"]
    df["mach_x"]      = df["mach"] * df["x_pos"]
    df["alpha_x"]     = df["alpha"] * df["x_pos"]
    return df


# ── Model factory ───────────────────────────────────────────────────────────────

def _make_model(name: str, n_est: int = 400):
    """Return best available regression model for this environment."""
    if HAS_XGB:
        log.info(f"    [{name}] XGBoost (n_estimators={n_est})")
        return xgb.XGBRegressor(
            n_estimators=n_est, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    if HAS_LGB:
        log.info(f"    [{name}] LightGBM (n_estimators={n_est})")
        return lgb.LGBMRegressor(
            n_estimators=n_est, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            num_leaves=63, random_state=42, n_jobs=-1, verbose=-1,
        )
    log.info(f"    [{name}] sklearn GradientBoosting (n_estimators={min(n_est, 200)})")
    return GradientBoostingRegressor(
        n_estimators=min(n_est, 200), max_depth=5,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )


# ── Step 0: Dataset generation ──────────────────────────────────────────────────

def ensure_datasets() -> None:
    Path("dataset").mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        log.info("Generating aerodynamic coefficient dataset...")
        from src.data.xfoil_generator import generate_dataset
        df = generate_dataset(N_SAMPLES)
        df.to_csv(DATASET_PATH, index=False)
        log.info(f"  Saved {len(df):,} rows -> {DATASET_PATH}")

    if not CP_DATASET.exists():
        log.info("Generating Cp field dataset...")
        from src.data.cp_dataset_generator import generate_cp_dataset
        df = generate_cp_dataset()
        df.to_csv(CP_DATASET, index=False)
        log.info(f"  Saved {len(df):,} rows -> {CP_DATASET}")


# ── Step 1: Coefficient models ──────────────────────────────────────────────────

def train_coef_models() -> dict:
    log.info("")
    log.info("─" * 55)
    log.info("  Step 1 / 3 — Coefficient models (Cl, Cd, Cm)")
    log.info("─" * 55)

    df = pd.read_csv(DATASET_PATH)
    df = _add_coef_features(df)
    log.info(f"  Loaded {len(df):,} rows from {DATASET_PATH}")

    metrics = {}
    backend = "xgboost" if HAS_XGB else ("lightgbm" if HAS_LGB else "sklearn_gbm")

    for target in TARGETS:
        t0 = time.time()
        X  = df[COEF_FEATS].values.astype(np.float32)
        y  = df[target].values.astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        model = _make_model(target, n_est=400)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        r2  = float(r2_score(y_te, y_pred))
        mae = float(np.mean(np.abs(y_te - y_pred)))

        log.info(f"  {target}: R²={r2:.4f}  MAE={mae:.5f}  ({time.time()-t0:.1f}s)")
        metrics[f"{target}_R2"]  = round(r2, 4)
        metrics[f"{target}_MAE"] = round(mae, 6)

        out = MODELS_DIR / f"{target.lower()}_model.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved -> {out}")

    metrics["backend"] = backend
    return metrics


# ── Step 2: Cp point-wise surrogate ────────────────────────────────────────────

def train_cp_surrogate() -> dict:
    log.info("")
    log.info("─" * 55)
    log.info("  Step 2 / 3 — Cp(x) surrogate (point-wise GBM)")
    log.info("─" * 55)

    df = pd.read_csv(CP_DATASET)
    df = _add_cp_features(df)
    log.info(f"  Loaded {len(df):,} rows from {CP_DATASET}")

    X  = df[CP_FEATS].values.astype(np.float32)
    y  = df["Cp"].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    t0    = time.time()
    model = _make_model("Cp", n_est=500)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    r2  = float(r2_score(y_te, y_pred))
    mae = float(np.mean(np.abs(y_te - y_pred)))

    log.info(f"  Cp: R²={r2:.4f}  MAE={mae:.5f}  ({time.time()-t0:.1f}s)")

    out = MODELS_DIR / "cp_surrogate.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    log.info(f"  Saved -> {out}")

    return {"Cp_R2": round(r2, 4), "Cp_MAE": round(mae, 6)}


# ── Step 3: FNO field model ─────────────────────────────────────────────────────

def _build_fno_dataset(n_profiles: int = 3000, n_points: int = 200):
    """Build synthetic (params, Cp-field) pairs for FNO training."""
    from src.demo.demo_model import _prandtl_glauert

    rng = np.random.default_rng(42)
    X_list, Y_list = [], []

    for _ in range(n_profiles):
        alpha  = float(rng.uniform(-5, 15))
        mach   = float(rng.uniform(0.05, 0.40))
        re     = float(rng.uniform(5e5, 5e6))
        t      = float(rng.uniform(0.06, 0.22))
        c      = float(rng.uniform(0.00, 0.08))
        cp_loc = float(rng.uniform(0.25, 0.65))
        ar     = float(rng.uniform(5, 12))
        tr     = float(rng.uniform(0.3, 0.8))
        sw     = float(rng.uniform(0, 30))

        pg  = _prandtl_glauert(mach)
        a   = alpha * np.pi / 180
        p   = max(0.1, min(cp_loc, 0.9))
        x   = np.linspace(0.0, 1.0, n_points)
        dyc = np.where(x < p,
                       (2*c/p**2)*(p-x),
                       (2*c/(1-p)**2)*(p-x))
        cp_field = -2.0 * (a + dyc) * pg
        cp_field += -0.6 * (abs(alpha) / 5.0) * np.exp(-40 * x)
        cp_field += (1.0 - cp_field[-1]) * np.exp(-8 * (1 - x)) * 0.15
        cp_field  = np.clip(
            cp_field + rng.normal(0, 0.015, n_points), -5.0, 1.0
        ).astype(np.float32)

        params = np.array([
            t, c, cp_loc,
            0.5 * t**2,           # leading_edge_radius
            12 + 20 * t,          # trailing_edge_angle
            ar, tr, sw,
            mach, re, alpha,
            # derived
            mach * alpha,
            alpha ** 2,
            alpha ** 3,
            np.log1p(re),
            ar * tr,
            t / max(c, 1e-8),
            1.0 / np.sqrt(max(1.0 - mach**2, 0.001)),
        ], dtype=np.float32)

        X_list.append(params)
        Y_list.append(cp_field)

    return np.array(X_list), np.array(Y_list)


def train_fno_model() -> dict:
    log.info("")
    log.info("─" * 55)
    log.info("  Step 3 / 3 — FNO field model (resolution-independent Cp)")
    log.info("─" * 55)

    if not HAS_TORCH:
        log.warning("  torch not installed — skipping FNO (Cp surrogate used instead)")
        return {}

    try:
        from models.fno.fno2d import FNOModel
    except ImportError as e:
        log.warning(f"  models/fno not available ({e}) — skipping FNO")
        return {}

    n_points = 200
    log.info(f"  Building dataset: {FNO_PROFILES} profiles x {n_points} points")
    X, Y = _build_fno_dataset(n_profiles=FNO_PROFILES, n_points=n_points)
    param_dim = X.shape[1]

    # Normalise parameters
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_s  = ((X - mean) / std).astype(np.float32)

    n_val = max(1, int(len(X_s) * 0.15))
    idx   = np.random.default_rng(0).permutation(len(X_s))
    X_tr, Y_tr = X_s[idx[n_val:]], Y[idx[n_val:]]
    X_vl, Y_vl = X_s[idx[:n_val]], Y[idx[:n_val]]
    log.info(f"  Split: {len(X_tr)} train / {len(X_vl)} val | param_dim={param_dim}")

    hparams = {
        "n_modes":       int(os.getenv("FNO_MODES",  "16")),
        "width":         int(os.getenv("FNO_WIDTH",  "32")),
        "n_layers":      int(os.getenv("FNO_LAYERS", "3")),
        "epochs":        FNO_EPOCHS,
        "batch_size":    64,
        "patience":      30,
        "learning_rate": 5e-4,
        "weight_decay":  1e-4,
        "grad_clip":     1.0,
    }

    t0    = time.time()
    model = FNOModel(param_dim=param_dim, params=hparams)
    model.fit(X_tr, Y_tr, X_vl, Y_vl)

    Y_pred = model.predict(X_vl, n_points=n_points)
    norms  = np.linalg.norm(Y_vl, axis=1) + 1e-8
    rel_l2 = float(np.mean(np.linalg.norm(Y_pred - Y_vl, axis=1) / norms))
    elapsed = time.time() - t0
    log.info(f"  FNO done: rel-L2={rel_l2:.5f}  time={elapsed:.0f}s")

    fno_path = MODELS_DIR / "fno.pt"
    torch.save({
        "state_dict":    model.network.state_dict(),
        "param_dim":     param_dim,
        "hparams":       model.hparams,
        "train_history": model.train_history,
        "model_class":   "FNO1d",
        "param_mean":    mean.tolist(),
        "param_std":     std.tolist(),
    }, fno_path)
    log.info(f"  Saved -> {fno_path}")

    return {
        "fno_val_rel_l2": round(rel_l2, 6),
        "fno_trained":    True,
        "fno_param_dim":  param_dim,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    log.info("")
    log.info("=" * 60)
    log.info("  AeroSurrogate v3.0 — Training Pipeline")
    log.info("=" * 60)
    log.info(f"  MODELS_DIR    : {MODELS_DIR}")
    log.info(f"  ARTIFACTS_DIR : {ARTIFACTS_DIR}")
    log.info(f"  N_SAMPLES     : {N_SAMPLES:,}")
    log.info(f"  SKIP_FNO      : {SKIP_FNO}")
    log.info(f"  XGBoost       : {HAS_XGB}")
    log.info(f"  LightGBM      : {HAS_LGB}")
    log.info(f"  PyTorch       : {HAS_TORCH}")
    log.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate datasets (no-op if already present)
    ensure_datasets()

    # Train models
    coef_metrics = train_coef_models()
    cp_metrics   = train_cp_surrogate()
    fno_metrics  = {} if SKIP_FNO else train_fno_model()

    # Save metrics
    all_metrics = {
        **coef_metrics,
        **cp_metrics,
        **fno_metrics,
        "model_version": "v3",
        "n_samples":     N_SAMPLES,
        "features":      COEF_FEATS,
        "MAPE":          round(
            float(np.mean([coef_metrics.get(f"{t}_MAE", 0) for t in TARGETS])) * 100,
            4
        ),
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - t_start

    log.info("")
    log.info("=" * 60)
    log.info(f"  Training complete in {elapsed:.0f}s")
    log.info("")
    for k, v in all_metrics.items():
        if k not in ("features",):
            log.info(f"    {k}: {v}")
    log.info("")
    log.info(f"  -> Models  : {MODELS_DIR}/")
    log.info(f"  -> Metrics : {metrics_path}")
    log.info("=" * 60)
    log.info("")


if __name__ == "__main__":
    main()