"""Inference predictor for AeroSurrogate v2.0."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.features.engineer import FeatureEngineer
from src.features.scaler_store import ScalerStore
from src.models.ensemble import EnsembleModel
from src.models.deeponet import DeepONetModel
from src.uncertainty.scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class Predictor:
    """Loads trained artifacts and runs inference."""

    def __init__(self, artifacts_dir: str = "artifacts", config_dir: str = "configs"):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_dir = Path(config_dir)
        self.is_loaded = False
        self.has_deeponet = False

        self.engineer = FeatureEngineer()
        self.scaler_store = ScalerStore()
        self.ensemble: Optional[EnsembleModel] = None
        self.deeponet: Optional[DeepONetModel] = None
        self.scorer: Optional[ConfidenceScorer] = None

        self.feature_names: List[str] = []
        self.target_names: List[str] = []

    def load(self) -> None:
        meta_path = self.artifacts_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        self.target_names = meta["target_names"]

        with open(self.config_dir / "model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)

        self.scaler_store.load(str(self.artifacts_dir))

        mlp_cfg = model_cfg.get("models", {}).get("mlp", {})
        n_seeds = mlp_cfg.get("n_seeds", 5)
        base_seed = mlp_cfg.get("base_seed", 42)

        self.ensemble = EnsembleModel(
            target_names=self.target_names,
            weights=model_cfg["ensemble"]["weights"],
            input_dim=meta["n_features"],
            n_seeds=n_seeds,
            base_seed=base_seed,
        )
        self.ensemble.load(str(self.artifacts_dir / "models"))

        with open(self.artifacts_dir / "uncertainty_params.json") as f:
            scorer_params = json.load(f)
        self.scorer = ConfidenceScorer()
        self.scorer.set_params(scorer_params)

        # DeepONet (optional)
        deeponet_path = self.artifacts_dir / "models" / "deeponet.pt"
        if deeponet_path.exists():
            deeponet_cfg = model_cfg.get("models", {}).get("deeponet", {})
            self.deeponet = DeepONetModel(param_dim=meta["n_features"], params=deeponet_cfg)
            self.deeponet.load(str(deeponet_path))
            self.has_deeponet = True
            logger.info("DeepONet field model loaded")

        self.is_loaded = True
        logger.info("Predictor v2.0 loaded")

    def predict(self, input_data: Dict[str, float]) -> Dict:
        if not self.is_loaded:
            raise RuntimeError("Predictor not loaded")

        import pandas as pd

        df = pd.DataFrame([input_data])
        df = self.engineer.transform(df)

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_names].values.astype(np.float32)
        X_s = self.scaler_store.transform_features(X)

        preds, std_devs = self.ensemble.predict(X_s)
        confidence, levels, colors = self.scorer.score(X_s, std_devs)

        result = {
            "predictions": {},
            "confidence": {
                "score": round(float(confidence[0]), 4),
                "level": levels[0],
                "color": colors[0],
            },
        }

        for i, target in enumerate(self.target_names):
            result["predictions"][target] = {
                "value": round(float(preds[0, i]), 6),
                "std": round(float(std_devs[0, i]), 6),
            }

        return result

    def predict_field(self, input_data: Dict[str, float],
                      n_points: int = 200) -> Optional[Dict]:
        """Predict surface Cp distribution via DeepONet."""
        if not self.has_deeponet or self.deeponet is None:
            return None

        import pandas as pd

        df = pd.DataFrame([input_data])
        df = self.engineer.transform(df)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_names].values.astype(np.float32)
        X_s = self.scaler_store.transform_features(X)

        cp = self.deeponet.predict(X_s, n_points=n_points)
        x_coords = np.linspace(0, 1, n_points).tolist()
        cp_values = cp[0].tolist()

        return {"x": x_coords, "Cp": cp_values, "n_points": n_points}

    def predict_batch(self, inputs: List[Dict[str, float]]) -> List[Dict]:
        return [self.predict(inp) for inp in inputs]
