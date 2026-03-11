"""Distance-based uncertainty estimation using Mahalanobis distance."""

import logging

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class DistanceEstimator:
    """Estimates uncertainty via Mahalanobis distance to training data centroid.

    Points far from the training distribution receive higher uncertainty scores.
    """

    def __init__(self):
        self.mean: np.ndarray = np.array([])
        self.cov_inv: np.ndarray = np.array([])
        self.max_distance: float = 1.0

    def fit(self, X_train: np.ndarray) -> None:
        """Compute training data statistics for Mahalanobis distance.

        Args:
            X_train: Feature matrix of shape (n_samples, n_features).
        """
        self.mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)

        # Regularize covariance to avoid singularity
        reg = 1e-6 * np.eye(cov.shape[0])
        cov_reg = cov + reg

        self.cov_inv = np.linalg.inv(cov_reg)

        # Compute max distance from training data for normalization
        distances = self._compute_distances(X_train)
        self.max_distance = float(np.percentile(distances, 99))
        if self.max_distance < 1e-8:
            self.max_distance = 1.0

        logger.info(f"DistanceEstimator fitted: max_distance={self.max_distance:.4f}")

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each sample."""
        diff = X - self.mean
        left = diff @ self.cov_inv
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized distance score in [0, 1].

        Higher score → further from training distribution → higher uncertainty.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Scores of shape (n_samples,) in [0, 1].
        """
        distances = self._compute_distances(X)
        scores = distances / self.max_distance
        return np.clip(scores, 0.0, 1.0)

    def get_params(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "cov_inv": self.cov_inv.tolist(),
            "max_distance": self.max_distance,
        }

    def set_params(self, params: dict) -> None:
        self.mean = np.array(params["mean"])
        self.cov_inv = np.array(params["cov_inv"])
        self.max_distance = params["max_distance"]
