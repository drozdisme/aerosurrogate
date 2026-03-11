"""Feature engineering for AeroSurrogate v2.0."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates derived features from geometry and flow parameters.

    v2.0 additions: compressibility factor, sweep-Mach interaction,
    alpha^3, reduced frequency, CST coefficient support.
    """

    def __init__(self, feature_config: Optional[Dict] = None):
        self.feature_config = feature_config or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        df = df.copy()

        # Drop internal columns
        if "_source" in df.columns:
            df = df.drop(columns=["_source"])

        # ── v1.0 derived features ──────────────────────────────
        if "mach" in df.columns and "alpha" in df.columns:
            df["mach_alpha_interaction"] = df["mach"] * df["alpha"]

        if "reynolds" in df.columns:
            df["reynolds_log"] = np.log1p(df["reynolds"])

        if "alpha" in df.columns:
            df["alpha_squared"] = df["alpha"] ** 2
            df["alpha_cubed"] = df["alpha"] ** 3

        if "aspect_ratio" in df.columns and "taper_ratio" in df.columns:
            df["aspect_ratio_taper_interaction"] = df["aspect_ratio"] * df["taper_ratio"]

        if "thickness_ratio" in df.columns and "camber" in df.columns:
            safe_camber = df["camber"].replace(0, 1e-8)
            df["thickness_camber_ratio"] = df["thickness_ratio"] / safe_camber

        # ── v2.0 derived features ──────────────────────────────
        if "sweep_angle" in df.columns and "mach" in df.columns:
            sweep_rad = np.radians(df["sweep_angle"])
            df["sweep_mach_interaction"] = df["mach"] * np.cos(sweep_rad)

        if "mach" in df.columns:
            m = df["mach"].clip(upper=0.999)
            df["compressibility_factor"] = 1.0 / np.sqrt(1.0 - m ** 2)

        if "mach" in df.columns and "reynolds" in df.columns:
            df["reduced_frequency"] = df["mach"] / np.log1p(df["reynolds"])

        logger.info(f"Feature engineering complete: {len(df.columns)} columns")
        return df

    def get_feature_columns(self, df: pd.DataFrame, target_cols: List[str]) -> List[str]:
        """Return feature column names (everything except targets and metadata)."""
        exclude = set(target_cols) | {"_source", "case_id", "index"}
        return [c for c in df.columns if c not in exclude]
