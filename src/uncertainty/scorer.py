"""Combined confidence scorer for AeroSurrogate predictions."""

import logging
from typing import Dict, List, Tuple

import numpy as np

from src.uncertainty.variance import VarianceEstimator
from src.uncertainty.distance import DistanceEstimator

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Combines variance and distance scores into a single confidence level.

    Final confidence = 1 - weighted_uncertainty, where:
        weighted_uncertainty = w_var * variance_score + w_dist * distance_score

    Confidence levels:
        HIGH   (green):  confidence >= 0.7
        MEDIUM (yellow): 0.4 <= confidence < 0.7
        LOW    (red):    confidence < 0.4
    """

    LEVELS = {
        "HIGH": "green",
        "MEDIUM": "yellow",
        "LOW": "red",
    }

    def __init__(
        self,
        variance_weight: float = 0.5,
        distance_weight: float = 0.5,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
    ):
        self.variance_weight = variance_weight
        self.distance_weight = distance_weight
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        self.variance_estimator = VarianceEstimator()
        self.distance_estimator = DistanceEstimator()

    def fit(self, X_train: np.ndarray, ensemble_std: np.ndarray) -> None:
        """Fit both estimators on training data.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            ensemble_std: Ensemble std on training data (n_samples, n_targets).
        """
        self.variance_estimator.fit(ensemble_std)
        self.distance_estimator.fit(X_train)
        logger.info("ConfidenceScorer fitted")

    def score(
        self, X: np.ndarray, ensemble_std: np.ndarray
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Compute confidence scores and levels.

        Args:
            X: Feature matrix (n_samples, n_features).
            ensemble_std: Ensemble std (n_samples, n_targets).

        Returns:
            confidence_scores: Array of shape (n_samples,) in [0, 1].
            levels: List of level names (HIGH/MEDIUM/LOW).
            colors: List of color names (green/yellow/red).
        """
        var_score = self.variance_estimator.score(ensemble_std)
        dist_score = self.distance_estimator.score(X)

        total_w = self.variance_weight + self.distance_weight
        uncertainty = (
            self.variance_weight * var_score + self.distance_weight * dist_score
        ) / total_w

        confidence = 1.0 - np.clip(uncertainty, 0.0, 1.0)

        levels = []
        colors = []
        for c in confidence:
            if c >= self.high_threshold:
                levels.append("HIGH")
                colors.append("green")
            elif c >= self.medium_threshold:
                levels.append("MEDIUM")
                colors.append("yellow")
            else:
                levels.append("LOW")
                colors.append("red")

        return confidence, levels, colors

    def get_params(self) -> Dict:
        return {
            "variance_weight": self.variance_weight,
            "distance_weight": self.distance_weight,
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
            "variance_estimator": self.variance_estimator.get_params(),
            "distance_estimator": self.distance_estimator.get_params(),
        }

    def set_params(self, params: Dict) -> None:
        self.variance_weight = params["variance_weight"]
        self.distance_weight = params["distance_weight"]
        self.high_threshold = params["high_threshold"]
        self.medium_threshold = params["medium_threshold"]
        self.variance_estimator.set_params(params["variance_estimator"])
        self.distance_estimator.set_params(params["distance_estimator"])
