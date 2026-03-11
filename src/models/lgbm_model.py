"""LightGBM model for AeroSurrogate."""

import logging
from typing import Dict, Optional

import joblib
import numpy as np
import lightgbm as lgb

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """LightGBM regression model for single-target prediction."""

    def __init__(self, target_name: str, params: Optional[Dict] = None):
        super().__init__(name=f"lgbm_{target_name}", params=params)
        self.target_name = target_name

        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        self.model = lgb.LGBMRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LightGBM on features X and single target y."""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"LightGBM model '{self.name}' trained on {X.shape[0]} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained LightGBM model."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' is not fitted.")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to joblib file."""
        joblib.dump(self.model, path)
        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from joblib file."""
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"LightGBM model loaded from {path}")
