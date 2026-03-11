"""Ensemble variance-based uncertainty estimation."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class VarianceEstimator:
    """Computes uncertainty from ensemble prediction variance.

    Normalizes raw standard deviations into [0, 1] using training-set statistics.
    """

    def __init__(self):
        self.mean_std: float = 0.0
        self.max_std: float = 1.0

    def fit(self, std_devs: np.ndarray) -> None:
        """Compute normalization statistics from training-set ensemble std.

        Args:
            std_devs: Array of shape (n_samples, n_targets).
        """
        mean_across_targets = std_devs.mean(axis=1)
        self.mean_std = float(np.mean(mean_across_targets))
        self.max_std = float(np.percentile(mean_across_targets, 99))
        if self.max_std < 1e-8:
            self.max_std = 1.0
        logger.info(f"VarianceEstimator fitted: mean_std={self.mean_std:.6f}, max_std={self.max_std:.6f}")

    def score(self, std_devs: np.ndarray) -> np.ndarray:
        """Compute normalized variance score in [0, 1].

        Higher score → higher uncertainty.

        Args:
            std_devs: Array of shape (n_samples, n_targets).

        Returns:
            Scores of shape (n_samples,) in [0, 1].
        """
        mean_across_targets = std_devs.mean(axis=1)
        scores = mean_across_targets / self.max_std
        return np.clip(scores, 0.0, 1.0)

    def get_params(self) -> dict:
        return {"mean_std": self.mean_std, "max_std": self.max_std}

    def set_params(self, params: dict) -> None:
        self.mean_std = params["mean_std"]
        self.max_std = params["max_std"]
