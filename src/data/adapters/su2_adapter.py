"""SU2 CFD solver data adapter."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class SU2Adapter(BaseAdapter):
    """Parses SU2 solver output directories into standardized DataFrames.

    Expected directory structure:
        su2_runs/
        ├── case_001/
        │   ├── config.cfg          (SU2 configuration)
        │   ├── solution_flow.csv   (surface solution)
        │   └── history.csv         (convergence history with Cl, Cd, Cm)
        ├── case_002/
        │   └── ...

    The adapter extracts flow conditions from config.cfg and
    aerodynamic coefficients from history.csv (last converged iteration).
    """

    def load(self) -> pd.DataFrame:
        base = Path(self.path)
        if not base.exists():
            raise FileNotFoundError(f"SU2 directory not found: {base}")

        file_pattern = self.config.get("file_pattern", "*/history.csv")
        config_pattern = self.config.get("config_pattern", "*/config.cfg")

        history_files = sorted(base.glob(file_pattern))
        if not history_files:
            raise FileNotFoundError(f"No SU2 history files matching '{file_pattern}' in {base}")

        rows = []
        for hf in history_files:
            case_dir = hf.parent
            cfg_candidates = list(case_dir.glob("*.cfg"))
            if not cfg_candidates:
                logger.warning(f"No config.cfg in {case_dir}, skipping")
                continue

            try:
                flow_params = self._parse_su2_config(cfg_candidates[0])
                coeffs = self._parse_history(hf)
                row = {**flow_params, **coeffs}
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to parse {case_dir.name}: {e}")
                continue

        if not rows:
            raise ValueError("No valid SU2 cases parsed")

        df = pd.DataFrame(rows)
        logger.info(f"SU2Adapter: parsed {len(df)} cases from {base}")

        if "K" not in df.columns and "Cl" in df.columns and "Cd" in df.columns:
            df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)

        df = self.apply_column_map(df)
        df = self.validate_minimum(df)
        return df

    def _parse_su2_config(self, cfg_path: Path) -> Dict[str, float]:
        """Extract flow conditions from SU2 .cfg file."""
        text = cfg_path.read_text(errors="replace")
        params = {}

        patterns = {
            "mach": r"MACH_NUMBER\s*=\s*([\d.eE+-]+)",
            "reynolds": r"REYNOLDS_NUMBER\s*=\s*([\d.eE+-]+)",
            "alpha": r"AOA\s*=\s*([\d.eE+-]+)",
            "beta": r"SIDESLIP_ANGLE\s*=\s*([\d.eE+-]+)",
        }

        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                params[key] = float(m.group(1))

        if "mach" not in params:
            raise ValueError(f"MACH_NUMBER not found in {cfg_path}")

        params.setdefault("reynolds", 1e6)
        params.setdefault("alpha", 0.0)
        params.setdefault("beta", 0.0)
        params.setdefault("altitude", 0.0)

        return params

    def _parse_history(self, history_path: Path) -> Dict[str, float]:
        """Extract converged Cl, Cd, Cm from SU2 history.csv."""
        df = pd.read_csv(history_path)
        # SU2 history columns often have quotes and spaces
        df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

        last = df.iloc[-1]

        coeffs = {}
        cl_cols = [c for c in df.columns if "cl" in c.lower() or "lift" in c.lower()]
        cd_cols = [c for c in df.columns if "cd" in c.lower() or "drag" in c.lower()]
        cm_cols = [c for c in df.columns if "cm" in c.lower() or "moment" in c.lower()]

        if cl_cols:
            coeffs["Cl"] = float(last[cl_cols[0]])
        if cd_cols:
            coeffs["Cd"] = float(last[cd_cols[0]])
        if cm_cols:
            coeffs["Cm"] = float(last[cm_cols[0]])

        if "Cl" not in coeffs:
            raise ValueError(f"Cl not found in {history_path}")

        return coeffs

    @staticmethod
    def parse_surface_cp(solution_path: Path, n_points: int = 200) -> Optional[np.ndarray]:
        """Parse surface Cp distribution from SU2 solution_flow.csv.

        Returns array of shape (n_points,) with Cp interpolated
        onto a uniform arc-length parameterization.
        """
        if not solution_path.exists():
            return None

        df = pd.read_csv(solution_path)
        df.columns = [c.strip().strip('"') for c in df.columns]

        x_col = next((c for c in df.columns if c.lower() in ("x", "x_coord", "points:0")), None)
        cp_col = next((c for c in df.columns if "pressure_coefficient" in c.lower() or c.lower() == "cp"), None)

        if x_col is None or cp_col is None:
            return None

        x = df[x_col].values
        cp = df[cp_col].values

        # Sort by x and interpolate to uniform grid
        idx = np.argsort(x)
        x_sorted, cp_sorted = x[idx], cp[idx]

        x_uniform = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        cp_interp = np.interp(x_uniform, x_sorted, cp_sorted)

        return cp_interp
