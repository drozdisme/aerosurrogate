"""Abstract base class for all AeroSurrogate models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class defining the model interface.

    All models in AeroSurrogate implement this interface to ensure
    consistent training, prediction, saving, and loading.
    """

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix X and target vector y.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,) — single target.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        ...
