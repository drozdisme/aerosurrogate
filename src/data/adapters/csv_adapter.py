"""CSV data adapter with flexible column mapping."""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.data.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class CSVAdapter(BaseAdapter):
    """Loads aerodynamic data from CSV/TSV files.

    Supports arbitrary column names via column_map in config.
    This is the primary adapter for purchased datasets, XFOIL output,
    and any tabular data.
    """

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        logger.info(f"CSVAdapter: loaded {len(df)} rows from {path.name}")

        df = self.apply_column_map(df)

        # Compute K if missing but Cl and Cd present
        if "K" not in df.columns and "Cl" in df.columns and "Cd" in df.columns:
            df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)

        # Compute Cm placeholder if missing
        if "Cm" not in df.columns and "Cl" in df.columns:
            df["Cm"] = -0.25 * df["Cl"]
            logger.warning("CSVAdapter: Cm not found, using approximation Cm = -0.25*Cl")

        df = self.validate_minimum(df)
        return df
