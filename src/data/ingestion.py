"""Unified data ingestion pipeline for AeroSurrogate v2.0.

Reads data_sources.yaml, instantiates the correct adapter for each
enabled source, and merges all results into a single training DataFrame.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.data.adapters.base_adapter import BaseAdapter
from src.data.adapters.csv_adapter import CSVAdapter
from src.data.adapters.su2_adapter import SU2Adapter
from src.data.adapters.openfoam_adapter import OpenFOAMAdapter
from src.data.adapters.vlm_adapter import VLMAdapter
from src.data.adapters.vtk_adapter import VTKAdapter
from src.data.adapters.synthetic_field_adapter import SyntheticFieldAdapter
from src.data.loader import generate_synthetic_data

logger = logging.getLogger(__name__)

ADAPTER_REGISTRY: Dict[str, type] = {
    "csv": CSVAdapter,
    "su2": SU2Adapter,
    "openfoam": OpenFOAMAdapter,
    "vlm": VLMAdapter,
    "vtk": VTKAdapter,
}


class DataIngestion:
    """Orchestrates data loading from multiple heterogeneous sources.

    Usage:
        ingestion = DataIngestion("configs/data_sources.yaml")
        df = ingestion.load_all()
    """

    def __init__(self, config_path: str = "configs/data_sources.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.source_stats: Dict[str, int] = {}

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning(f"Data sources config not found: {self.config_path}")
            return {"sources": {"synthetic_fallback": {"adapter": "synthetic", "enabled": True, "n_samples": 5000}}}
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def load_all(self) -> pd.DataFrame:
        """Load and merge all enabled data sources.

        Returns a single DataFrame with standardized column names,
        deduplicated according to merge settings.
        """
        sources = self.config.get("sources", {})
        merge_cfg = self.config.get("merge", {})
        quality_cfg = self.config.get("quality", {})

        frames: List[pd.DataFrame] = []

        for name, src_cfg in sources.items():
            if not src_cfg.get("enabled", False):
                logger.info(f"Source '{name}' disabled, skipping")
                continue

            adapter_type = src_cfg.get("adapter", "")
            logger.info(f"Loading source '{name}' (adapter={adapter_type})...")

            try:
                df = self._load_source(name, src_cfg)
                df["_source"] = name

                min_rows = quality_cfg.get("min_rows_per_source", 1)
                if len(df) < min_rows:
                    logger.warning(f"Source '{name}' has only {len(df)} rows (min={min_rows}), skipping")
                    continue

                max_nan = quality_cfg.get("max_nan_fraction", 1.0)
                nan_frac = df.isnull().mean().mean()
                if nan_frac > max_nan:
                    logger.warning(f"Source '{name}' has {nan_frac:.1%} NaN (max={max_nan:.0%}), skipping")
                    continue

                self.source_stats[name] = len(df)
                frames.append(df)
                logger.info(f"Source '{name}': {len(df)} rows loaded")

            except Exception as e:
                logger.error(f"Failed to load source '{name}': {e}")
                continue

        if not frames:
            logger.warning("No data sources loaded. Generating synthetic fallback.")
            df = generate_synthetic_data(n_samples=5000, random_state=42)
            df["_source"] = "synthetic_fallback"
            frames.append(df)
            self.source_stats["synthetic_fallback"] = len(df)

        # Merge
        combined = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"Combined dataset: {len(combined)} rows from {len(frames)} sources")

        # Deduplicate
        if merge_cfg.get("deduplicate", False):
            dedup_keys = merge_cfg.get("deduplicate_keys", [])
            available_keys = [k for k in dedup_keys if k in combined.columns]
            if available_keys:
                before = len(combined)
                combined = combined.drop_duplicates(subset=available_keys, keep="last")
                logger.info(f"Deduplication: {before} → {len(combined)} rows")

        return combined

    def _load_source(self, name: str, cfg: Dict) -> pd.DataFrame:
        """Load a single data source via its adapter."""
        adapter_type = cfg.get("adapter", "")

        if adapter_type == "synthetic":
            n = cfg.get("n_samples", 5000)
            seed = cfg.get("random_state", 42)
            return generate_synthetic_data(n_samples=n, random_state=seed)

        if adapter_type not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown adapter type: '{adapter_type}'. "
                             f"Available: {list(ADAPTER_REGISTRY.keys())}")

        adapter_cls = ADAPTER_REGISTRY[adapter_type]
        adapter = adapter_cls(cfg)
        return adapter.load()

    def load_field_data(self, source_name: Optional[str] = None,
                        n_points: int = 200) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """Load field data (Cp distributions) for DeepONet training.

        Priority:
          1. VTK sources (real CFD field data) — highest quality
          2. synthetic_field sources — physics-based Cp generator (no CFD needed)

        Returns (params_df, fields_array) or None if no field sources are enabled.
        """
        sources = self.config.get("sources", {})

        # ── Pass 1: prefer real VTK data ──────────────────────
        for name, cfg in sources.items():
            if not cfg.get("enabled", False):
                continue
            if source_name and name != source_name:
                continue
            if cfg.get("adapter") == "vtk":
                logger.info(f"Loading VTK field data from source '{name}'")
                adapter = VTKAdapter(cfg)
                return adapter.load_fields(n_points=n_points)

        # ── Pass 2: synthetic_field fallback ──────────────────
        for name, cfg in sources.items():
            if not cfg.get("enabled", False):
                continue
            if source_name and name != source_name:
                continue
            if cfg.get("adapter") == "synthetic_field":
                logger.info(
                    f"No VTK source found. Using synthetic Cp-field generator "
                    f"(source: '{name}') for DeepONet training."
                )
                adapter = SyntheticFieldAdapter(cfg)
                return adapter.load_fields(n_points=n_points)

        logger.info("No field data sources found (VTK or synthetic_field). DeepONet will be skipped.")
        return None

    def print_summary(self) -> None:
        """Print a summary of loaded data sources."""
        total = sum(self.source_stats.values())
        logger.info("=" * 50)
        logger.info("Data Ingestion Summary")
        logger.info("-" * 50)
        for name, count in self.source_stats.items():
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {name:30s} {count:>7d} rows ({pct:5.1f}%)")
        logger.info("-" * 50)
        logger.info(f"  {'TOTAL':30s} {total:>7d} rows")
        logger.info("=" * 50)
