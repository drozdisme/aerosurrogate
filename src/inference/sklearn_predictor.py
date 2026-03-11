"""
SklearnPredictor v2 — final.

Loads:
  - cl_model.pkl / cd_model.pkl / cm_model.pkl   (XGBoost / sklearn GBM)
  - cp_surrogate.pkl                              (DeepONet-like Cp(x) surrogate)

Returns uppercase confidence levels: HIGH / MEDIUM / LOW
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGETS    = ["Cl", "Cd", "Cm"]

COEF_BASE  = ["thickness_ratio","camber","camber_position","leading_edge_radius",
              "trailing_edge_angle","aspect_ratio","taper_ratio","sweep_angle",
              "mach","reynolds","alpha"]
COEF_EXTRA = ["mach_alpha","alpha2","alpha3","log_re","ar_tr","tc_ratio","pg"]
COEF_FEATS = COEF_BASE + COEF_EXTRA

CP_BASE    = ["thickness_ratio","camber","camber_position","aspect_ratio",
              "taper_ratio","sweep_angle","mach","reynolds","alpha"]
CP_EXTRA   = ["mach_alpha","alpha2","alpha3","log_re",
              "x_pos","x_sqrt","one_minus_x","mach_x","alpha_x"]
CP_FEATS   = CP_BASE + CP_EXTRA


# ── Feature helpers ───────────────────────────────────────────────────────────

def _make_coef_row(d: dict) -> pd.DataFrame:
    for col in COEF_BASE:
        d.setdefault(col, 0.0)
    df = pd.DataFrame([d])
    df["mach_alpha"] = df["mach"] * df["alpha"]
    df["alpha2"]     = df["alpha"] ** 2
    df["alpha3"]     = df["alpha"] ** 3
    df["log_re"]     = np.log1p(df["reynolds"])
    df["ar_tr"]      = df["aspect_ratio"] * df["taper_ratio"]
    df["tc_ratio"]   = df["thickness_ratio"] / df["camber"].replace(0, 1e-8)
    m = df["mach"].clip(upper=0.999)
    df["pg"]         = 1.0 / np.sqrt(1.0 - m ** 2)
    return df


def _make_cp_batch(d: dict, x_arr: np.ndarray) -> pd.DataFrame:
    n = len(x_arr)
    rows = {col: np.full(n, float(d.get(col, 0.0))) for col in CP_BASE}
    rows["x_pos"] = x_arr
    df = pd.DataFrame(rows)
    df["mach_alpha"]  = df["mach"]  * df["alpha"]
    df["alpha2"]      = df["alpha"] ** 2
    df["alpha3"]      = df["alpha"] ** 3
    df["log_re"]      = np.log1p(df["reynolds"])
    x = df["x_pos"].clip(lower=1e-4)
    df["x_sqrt"]      = np.sqrt(x)
    df["one_minus_x"] = 1.0 - df["x_pos"]
    df["mach_x"]      = df["mach"]  * df["x_pos"]
    df["alpha_x"]     = df["alpha"] * df["x_pos"]
    return df


# ── Uncertainty estimation ────────────────────────────────────────────────────

def _estimate_std(model, X: np.ndarray, val: float) -> float:
    """
    Estimate prediction std by comparing early-stopping trees vs full ensemble.
    Uses actual boosted rounds (not n_estimators param) and a 1/5 split
    to get meaningful variation even for well-converged models.
    """
    # XGBoost: actual rounds via get_booster(), compare first 20% vs full
    try:
        actual_n = model.get_booster().num_boosted_rounds()
        early_n  = max(1, actual_n // 5)   # first 20% of trees
        p_early  = float(model.predict(X, iteration_range=(0, early_n))[0])
        p_mid    = float(model.predict(X, iteration_range=(0, actual_n // 2))[0])
        # dispersion across tree checkpoints = proxy for epistemic uncertainty
        spread   = abs(val - p_early) * 0.6 + abs(val - p_mid) * 0.4
        return spread + abs(val) * 0.002 + 1e-6
    except (AttributeError, TypeError):
        pass
    # sklearn GBM: variance across staged predictions (use full range)
    try:
        stages = list(model.staged_predict(X))
        n      = len(stages)
        # sample checkpoints at 20%, 40%, 60%, 80%, 100%
        checkpoints = [stages[max(0, int(n * frac) - 1)][0]
                       for frac in (0.2, 0.4, 0.6, 0.8, 1.0)]
        spread = float(np.std(checkpoints))
        return spread + abs(val) * 0.002 + 1e-6
    except AttributeError:
        pass
    # LightGBM: compare 20% vs full
    try:
        actual_n = model.booster_.num_trees()
        early_n  = max(1, actual_n // 5)
        p_early  = float(model.predict(X, num_iteration=early_n)[0])
        spread   = abs(val - p_early) * 0.6
        return spread + abs(val) * 0.002 + 1e-6
    except (AttributeError, TypeError):
        pass
    return abs(val) * 0.025 + 1e-6


def _confidence(alpha: float, mach: float, reynolds: float,
                stds: dict, preds: dict) -> dict:
    """Physics-aware confidence — always returns UPPERCASE level."""
    # ── Physics regime penalties (these drive most of the variation) ──────────
    stall  = max(0.0, (abs(alpha) - 10.0) / 8.0)   # non-linear above 10°
    comp   = max(0.0, (mach - 0.65) / 0.30)         # transonic above M=0.65
    low_re = max(0.0, (3e5 - reynolds) / 2.5e5)     # laminar separation risk

    # ── Model uncertainty from spread between tree checkpoints ───────────────
    # Normalise by a meaningful scale, not just the predicted value
    # (avoids near-zero division when Cl ≈ 0 near zero-lift alpha)
    cl_scale = max(abs(preds["Cl"]), 0.15)
    cd_scale = max(abs(preds["Cd"]), 0.008)
    cl_rel   = min(stds["Cl"] / cl_scale, 1.0)
    cd_rel   = min(stds["Cd"] / cd_scale, 1.0)
    model_unc = (cl_rel * 0.6 + cd_rel * 0.4)      # weighted model uncertainty

    # ── Combine ───────────────────────────────────────────────────────────────
    penalty = (stall   * 0.35 +
               comp    * 0.25 +
               low_re  * 0.15 +
               model_unc * 0.25)
    score   = float(np.clip(0.97 - penalty * 0.60, 0.20, 0.97))

    if score >= 0.70:
        return {"score": round(score, 4), "level": "HIGH",   "color": "#16a34a"}
    elif score >= 0.42:
        return {"score": round(score, 4), "level": "MEDIUM", "color": "#b45309"}
    else:
        return {"score": round(score, 4), "level": "LOW",    "color": "#dc2626"}


# ── Predictor class ───────────────────────────────────────────────────────────

class SklearnPredictor:

    def __init__(self, models_dir: str = "models",
                 metrics_path: str = "artifacts/metrics.json"):
        self.models_dir   = Path(models_dir)
        self.metrics_path = Path(metrics_path)
        self.models: Dict       = {}
        self.cp_model           = None
        self.metrics: Dict      = {}
        self.is_loaded          = False
        self.has_deeponet       = False   # True when any field model (FNO or DeepONet) is loaded
        self._field_model_type  = None    # "fno" | "deeponet" | None

    def load(self) -> None:
        missing = [t for t in TARGETS
                   if not (self.models_dir / f"{t.lower()}_model.pkl").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing pkl files for: {missing} — run train.py first")

        for t in TARGETS:
            path = self.models_dir / f"{t.lower()}_model.pkl"
            with open(path, "rb") as f:
                self.models[t] = pickle.load(f)
            logger.info(f"Loaded {path.name}")

        # Cp surrogate: try FNO (v3) first, then legacy pkl (v2)
        fno_path = self.models_dir / "fno.pt"
        cp_path  = self.models_dir / "cp_surrogate.pkl"

        if fno_path.exists():
            try:
                from models.fno.fno2d import FNOModel
                # param_dim discovered lazily from checkpoint
                fno = FNOModel.__new__(FNOModel)
                fno.device = __import__("torch").device("cpu")
                fno.is_fitted = False
                fno.load(str(fno_path))
                self.cp_model = fno
                self.has_deeponet = True
                self._field_model_type = "fno"
                logger.info("Loaded FNO field model (v3) ✓")
            except Exception as e:
                logger.warning(f"FNO load failed ({e}), trying legacy Cp surrogate")

        if not self.has_deeponet and cp_path.exists():
            with open(cp_path, "rb") as f:
                self.cp_model = pickle.load(f)
            self.has_deeponet = True
            self._field_model_type = "deeponet"
            logger.info("Loaded cp_surrogate.pkl (v2 legacy) ✓")

        if not self.has_deeponet:
            logger.warning("No field model found — Cp predictions use analytical model")

        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                self.metrics = json.load(f)

        self.is_loaded = True
        logger.info(
            f"SklearnPredictor ready | backend={self.metrics.get('backend','?')} | "
            f"Cl R²={self.metrics.get('Cl_R2')} | "
            f"Cp surrogate={'yes' if self.has_deeponet else 'no'}"
        )

    # ── Integral coefficients ─────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Not loaded — call load() first")

        df = _make_coef_row(dict(data))
        for col in COEF_FEATS:
            if col not in df.columns:
                df[col] = 0.0
        X = df[COEF_FEATS].values.astype(np.float64)

        vals, stds = {}, {}
        for t in TARGETS:
            v = float(self.models[t].predict(X)[0])
            vals[t] = v
            stds[t] = _estimate_std(self.models[t], X, v)

        vals["Cd"] = max(vals["Cd"], 0.003)
        k           = vals["Cl"] / max(vals["Cd"], 1e-6)
        vals["K"]   = round(k, 4)
        stds["K"]   = abs(k) * 0.03 + 1e-6

        alpha    = float(data.get("alpha", 0))
        mach     = float(data.get("mach",  0))
        reynolds = float(data.get("reynolds", 1e6))
        conf     = _confidence(alpha, mach, reynolds, stds, vals)

        return {
            "predictions": {t: {"value": round(float(vals[t]), 6),
                                 "std":   round(float(stds[t]), 6)}
                             for t in ["Cl","Cd","Cm","K"]},
            "confidence": conf,
        }

    # ── Cp(x) field — DeepONet surrogate ─────────────────────────────────────

    def predict_field(self, data: dict, n_points: int = 200) -> Optional[dict]:
        if not self.has_deeponet or self.cp_model is None:
            return None

        x_arr = np.linspace(0.0, 1.0, n_points)
        df    = _make_cp_batch(data, x_arr)
        for col in CP_FEATS:
            if col not in df.columns:
                df[col] = 0.0
        X  = df[CP_FEATS].values.astype(np.float64)
        cp = np.clip(self.cp_model.predict(X), -5.0, 1.0)

        return {"x": x_arr.tolist(), "Cp": cp.tolist(), "n_points": n_points}