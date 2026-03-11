"""Abstract base adapter for data source ingestion."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseAdapter(ABC):
    """All data adapters implement this interface.

    An adapter reads data from a specific format (CSV, SU2, OpenFOAM, VLM, VTK)
    and returns a unified pandas DataFrame with standardized column names.
    """

    STANDARD_COLUMNS = {
        "geometry": [
            "thickness_ratio", "camber", "camber_position",
            "leading_edge_radius", "trailing_edge_angle",
            "aspect_ratio", "taper_ratio", "sweep_angle",
            "twist_angle", "dihedral_angle",
        ],
        "flow": ["mach", "reynolds", "alpha", "beta", "altitude"],
        "targets": ["Cl", "Cd", "Cm", "K"],
    }

    def __init__(self, config: Dict):
        self.config = config
        self.path = config.get("path", "")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return standardized DataFrame."""
        ...

    def apply_column_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to the column_map in config."""
        col_map = self.config.get("column_map", {})
        if col_map:
            reverse_map = {v: k for k, v in col_map.items()}
            df = df.rename(columns=reverse_map)
        return df

    def validate_minimum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure at least mach, reynolds, alpha, Cl are present."""
        required = {"mach", "reynolds", "alpha", "Cl"}
        present = required & set(df.columns)
        if len(present) < len(required):
            missing = required - present
            raise ValueError(
                f"Adapter {self.__class__.__name__}: missing columns {missing}"
            )
        return df
