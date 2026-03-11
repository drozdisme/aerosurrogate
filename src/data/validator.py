"""Data validation utilities for AeroSurrogate."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates aerodynamic dataset for quality and consistency."""

    PHYSICAL_BOUNDS = {
        "mach": (0.0, 5.0),
        "reynolds": (1e3, 1e9),
        "alpha": (-30.0, 30.0),
        "beta": (-30.0, 30.0),
        "altitude": (-500, 100000),
        "thickness_ratio": (0.01, 0.5),
        "camber": (0.0, 0.2),
        "Cl": (-3.0, 4.0),
        "Cd": (0.0, 2.0),
        "Cm": (-2.0, 2.0),
    }

    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns or ["mach", "reynolds", "alpha", "Cl"]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all validation checks and return cleaned DataFrame."""
        df = df.copy()

        self._check_required_columns(df)
        df = self._remove_duplicates(df)
        df = self._remove_nan_rows(df)
        df = self._enforce_physical_bounds(df)

        logger.info(f"Validation complete: {len(df)} rows retained")
        return df

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure all required columns are present."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate rows")
        return df

    def _remove_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN values in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_before = len(df)
        df = df.dropna(subset=numeric_cols)
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} rows with NaN values")
        return df

    def _enforce_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that violate physical bounds."""
        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in self.PHYSICAL_BOUNDS.items():
            if col in df.columns:
                col_mask = (df[col] >= lo) & (df[col] <= hi)
                n_violations = (~col_mask).sum()
                if n_violations > 0:
                    logger.warning(f"Column '{col}': {n_violations} out-of-bounds values removed")
                mask &= col_mask
        return df[mask].reset_index(drop=True)
