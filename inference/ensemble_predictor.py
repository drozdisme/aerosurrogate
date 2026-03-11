"""
EnsemblePredictor — backed by SklearnPredictor.
Used by RealTimePredictor so that digital_twin.load() succeeds.
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(self, artifacts_dir="artifacts", config_dir="configs",
                 n_workers=None, use_fno=True):
        self.artifacts_dir = Path(artifacts_dir)
        models_dir = Path(os.getenv("MODELS_DIR", "models"))
        metrics_path = self.artifacts_dir / "metrics.json"

        from src.inference.sklearn_predictor import SklearnPredictor
        self._sk = SklearnPredictor(
            models_dir=str(models_dir),
            metrics_path=str(metrics_path),
        )
        self.is_loaded = False
        self.field_model_type: Optional[str] = None

    def load(self):
        self._sk.load()
        self.is_loaded = self._sk.is_loaded
        self.field_model_type = self._sk._field_model_type or "none"
        logger.info(f"EnsemblePredictor ready | backend={self._sk.metrics.get('backend','?')} "
                    f"| Cl R²={self._sk.metrics.get('Cl_R2')} | field={self.field_model_type}")

    def predict(self, data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Not loaded — call load() first")
        return self._sk.predict(data)

    def predict_field(self, data: dict, n_points: int = 200):
        if not self.is_loaded:
            raise RuntimeError("Not loaded — call load() first")
        return self._sk.predict_field(data, n_points=n_points)

    @property
    def metrics(self):
        return self._sk.metrics if self._sk else {}
