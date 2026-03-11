"""Ensemble model for AeroSurrogate v2.0.

v2 changes:
  - N MLP instances with different seeds (configurable, default 5)
  - Physics-informed loss option for MLP training
  - Better variance estimation from larger ensemble
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.base_model import BaseModel
from src.models.xgb_model import XGBModel
from src.models.lgbm_model import LGBMModel
from src.models.mlp_model import MLPModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted ensemble: XGBoost + LightGBM + N×MLP per target.

    Total models = n_targets × (2 + n_seeds).
    Default: 4 targets × 7 models = 28 estimators.
    """

    def __init__(
        self,
        target_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        xgb_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
        mlp_params: Optional[Dict] = None,
        input_dim: int = 20,
        n_seeds: int = 5,
        base_seed: int = 42,
    ):
        self.target_names = target_names
        self.weights = weights or {"xgboost": 0.30, "lightgbm": 0.30, "mlp": 0.40}
        self.input_dim = input_dim
        self.n_seeds = n_seeds
        self.base_seed = base_seed

        self.models: Dict[str, Dict[str, BaseModel]] = {}
        for target in target_names:
            group = {
                "xgboost": XGBModel(target, params=xgb_params),
                "lightgbm": LGBMModel(target, params=lgbm_params),
            }
            for s in range(n_seeds):
                seed = base_seed + s * 7
                seed_params = dict(mlp_params or {})
                seed_params["random_state"] = seed
                group[f"mlp_{s}"] = MLPModel(target, input_dim=input_dim, params=seed_params)
            self.models[target] = group

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train all models for all targets."""
        total = len(self.target_names) * (2 + self.n_seeds)
        count = 0

        for i, target in enumerate(self.target_names):
            y_t = y_train[:, i]
            y_v = y_val[:, i] if y_val is not None else None

            logger.info(f"Training models for '{target}'...")

            self.models[target]["xgboost"].fit(X_train, y_t)
            count += 1

            self.models[target]["lightgbm"].fit(X_train, y_t)
            count += 1

            for s in range(self.n_seeds):
                key = f"mlp_{s}"
                self.models[target][key].fit(X_train, y_t, X_val, y_v)
                count += 1
                logger.info(f"  [{count}/{total}] {target}/{key} done")

        logger.info(f"Ensemble training complete: {total} models")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble predictions with variance.

        Returns:
            predictions: (n_samples, n_targets)
            std_devs: (n_samples, n_targets)
        """
        n = X.shape[0]
        n_t = len(self.target_names)
        predictions = np.zeros((n, n_t))
        std_devs = np.zeros((n, n_t))

        w_xgb = self.weights["xgboost"]
        w_lgbm = self.weights["lightgbm"]
        w_mlp = self.weights["mlp"]
        w_total = w_xgb + w_lgbm + w_mlp

        for i, target in enumerate(self.target_names):
            all_preds = []

            p_xgb = self.models[target]["xgboost"].predict(X)
            p_lgbm = self.models[target]["lightgbm"].predict(X)
            all_preds.extend([p_xgb, p_lgbm])

            mlp_preds = []
            for s in range(self.n_seeds):
                p = self.models[target][f"mlp_{s}"].predict(X)
                mlp_preds.append(p)
                all_preds.append(p)

            # Weighted mean: boosting models + average of MLP seeds
            mlp_mean = np.mean(mlp_preds, axis=0)
            predictions[:, i] = (w_xgb * p_xgb + w_lgbm * p_lgbm + w_mlp * mlp_mean) / w_total

            # Std across ALL models (2 + n_seeds)
            stacked = np.stack(all_preds, axis=0)
            std_devs[:, i] = np.std(stacked, axis=0)

        return predictions, std_devs

    def save(self, directory: str) -> None:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        for target in self.target_names:
            td = base / target
            td.mkdir(exist_ok=True)
            for name, model in self.models[target].items():
                if name.startswith("mlp"):
                    model.save(str(td / f"{name}.pt"))
                else:
                    model.save(str(td / f"{name}.joblib"))

        logger.info(f"Ensemble saved to {base}")

    def load(self, directory: str) -> None:
        base = Path(directory)
        for target in self.target_names:
            td = base / target
            for name, model in self.models[target].items():
                if name.startswith("mlp"):
                    model.load(str(td / f"{name}.pt"))
                else:
                    model.load(str(td / f"{name}.joblib"))

        logger.info(f"Ensemble loaded from {base}")
