"""Optuna-based hyperparameter optimization for AeroSurrogate v2.0."""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_idx: int = 0,
    n_trials: int = 100,
    timeout: int = 3600,
    metric: str = "mape",
) -> Dict:
    """Run Optuna hyperparameter search.

    Tunes XGBoost, LightGBM, and MLP hyperparameters jointly.
    Returns the best parameter set found.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. Returning default parameters.")
        return _default_params()

    y_t = y_train[:, target_idx]
    y_v = y_val[:, target_idx]

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", ["xgboost", "lightgbm", "mlp"])

        if model_type == "xgboost":
            from src.models.xgb_model import XGBModel
            params = {
                "n_estimators": trial.suggest_int("xgb_n_est", 100, 1000),
                "max_depth": trial.suggest_int("xgb_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_sub", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("xgb_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
            }
            model = XGBModel("tune", params=params)
            model.fit(X_train, y_t)
            pred = model.predict(X_val)

        elif model_type == "lightgbm":
            from src.models.lgbm_model import LGBMModel
            params = {
                "n_estimators": trial.suggest_int("lgbm_n_est", 100, 1000),
                "max_depth": trial.suggest_int("lgbm_depth", 3, 10),
                "learning_rate": trial.suggest_float("lgbm_lr", 0.005, 0.3, log=True),
                "num_leaves": trial.suggest_int("lgbm_leaves", 15, 127),
                "subsample": trial.suggest_float("lgbm_sub", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgbm_col", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("lgbm_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgbm_lambda", 1e-3, 10.0, log=True),
            }
            model = LGBMModel("tune", params=params)
            model.fit(X_train, y_t)
            pred = model.predict(X_val)

        else:
            from src.models.mlp_model import MLPModel
            n_layers = trial.suggest_int("mlp_layers", 2, 5)
            layers = []
            dim = trial.suggest_categorical("mlp_first", [64, 128, 256, 512])
            for l in range(n_layers):
                layers.append(dim)
                dim = max(16, dim // 2)
            params = {
                "hidden_layers": layers,
                "dropout": trial.suggest_float("mlp_drop", 0.0, 0.3),
                "learning_rate": trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
                "epochs": 150,
                "batch_size": trial.suggest_categorical("mlp_bs", [32, 64, 128, 256]),
                "patience": 20,
                "random_state": 42,
            }
            model = MLPModel("tune", input_dim=X_train.shape[1], params=params)
            model.fit(X_train, y_t, X_val, y_v)
            pred = model.predict(X_val)

        if metric == "mape":
            mask = np.abs(y_v) > 1e-8
            if mask.sum() > 0:
                return np.mean(np.abs((y_v[mask] - pred[mask]) / y_v[mask])) * 100
            return np.mean(np.abs(y_v - pred))
        elif metric == "rmse":
            return np.sqrt(np.mean((y_v - pred) ** 2))
        else:
            return np.mean(np.abs(y_v - pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Tuning complete: best {metric}={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return _extract_best_params(study.best_params)


def _extract_best_params(best: Dict) -> Dict:
    """Convert flat Optuna params to structured config."""
    result = {"xgboost": {}, "lightgbm": {}, "mlp": {}}

    for k, v in best.items():
        if k.startswith("xgb_"):
            name = k.replace("xgb_", "").replace("n_est", "n_estimators").replace("depth", "max_depth").replace("lr", "learning_rate").replace("sub", "subsample").replace("col", "colsample_bytree").replace("alpha", "reg_alpha").replace("lambda", "reg_lambda")
            result["xgboost"][name] = v
        elif k.startswith("lgbm_"):
            name = k.replace("lgbm_", "").replace("n_est", "n_estimators").replace("depth", "max_depth").replace("lr", "learning_rate").replace("leaves", "num_leaves").replace("sub", "subsample").replace("col", "colsample_bytree").replace("alpha", "reg_alpha").replace("lambda", "reg_lambda")
            result["lightgbm"][name] = v
        elif k.startswith("mlp_"):
            name = k.replace("mlp_", "").replace("drop", "dropout").replace("lr", "learning_rate").replace("bs", "batch_size")
            result["mlp"][name] = v

    return result


def _default_params() -> Dict:
    return {
        "xgboost": {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03},
        "lightgbm": {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.03},
        "mlp": {"hidden_layers": [256, 128, 64, 32], "learning_rate": 0.001},
    }
