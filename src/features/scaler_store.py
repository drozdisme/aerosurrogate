"""Scaler persistence for AeroSurrogate."""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ScalerStore:
    """Manages feature and target scalers with persistence."""

    def __init__(self):
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None

    def fit_feature_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform feature scaler."""
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        logger.info(f"Feature scaler fitted on shape {X.shape}")
        return X_scaled

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.feature_scaler is None:
            raise RuntimeError("Feature scaler not fitted. Call fit_feature_scaler first.")
        return self.feature_scaler.transform(X)

    def fit_target_scaler(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform target scaler."""
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y)
        logger.info(f"Target scaler fitted on shape {y.shape}")
        return y_scaled

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if self.target_scaler is None:
            raise RuntimeError("Target scaler not fitted.")
        return self.target_scaler.transform(y)

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original space."""
        if self.target_scaler is None:
            raise RuntimeError("Target scaler not fitted.")
        return self.target_scaler.inverse_transform(y_scaled)

    def save(self, directory: str) -> None:
        """Save scalers to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, path / "feature_scaler.joblib")
        if self.target_scaler is not None:
            joblib.dump(self.target_scaler, path / "target_scaler.joblib")

        logger.info(f"Scalers saved to {path}")

    def load(self, directory: str) -> None:
        """Load scalers from disk."""
        path = Path(directory)

        feature_path = path / "feature_scaler.joblib"
        target_path = path / "target_scaler.joblib"

        if feature_path.exists():
            self.feature_scaler = joblib.load(feature_path)
            logger.info("Feature scaler loaded")
        if target_path.exists():
            self.target_scaler = joblib.load(target_path)
            logger.info("Target scaler loaded")
