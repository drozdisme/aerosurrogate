"""VTK field data adapter for surface Cp distributions."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class VTKAdapter(BaseAdapter):
    """Loads surface field data (Cp distributions) from VTK or CSV field files.

    Used to prepare training data for DeepONet / field-prediction models.

    Expected structure:
        field_data/
        ├── case_001/
        │   ├── params.csv          (mach, Re, alpha, ...)
        │   └── surface.csv         (x, y, Cp) or surface.vtk
        ├── case_002/
        │   └── ...

    Or single-file format (e.g. AirfRANS):
        airfrans/
        ├── manifest.csv            (case_id, mach, Re, alpha, ...)
        ├── fields/
        │   ├── 0000.npy            (x, y, Cp arrays)
        │   ├── 0001.npy
    """

    def load(self) -> pd.DataFrame:
        """Load integral coefficients from field data directory.

        For the integral-model training pipeline, returns a DataFrame
        with flow params and Cl, Cd, Cm. Field arrays are NOT included
        in the DataFrame (too large); use load_fields() separately
        for DeepONet training.
        """
        base = Path(self.path)
        if not base.exists():
            raise FileNotFoundError(f"VTK/field directory not found: {base}")

        # Check for manifest-based layout
        manifest = base / "manifest.csv"
        if manifest.exists():
            df = pd.read_csv(manifest)
            df = self.apply_column_map(df)
            logger.info(f"VTKAdapter: loaded {len(df)} entries from manifest")
            df = self.validate_minimum(df)
            return df

        # Case-directory layout
        case_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        if not case_dirs:
            raise FileNotFoundError(f"No case directories in {base}")

        rows = []
        for cd in case_dirs:
            params_file = cd / "params.csv"
            if not params_file.exists():
                continue
            try:
                row = pd.read_csv(params_file).iloc[0].to_dict()
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to parse {cd.name}: {e}")

        if not rows:
            raise ValueError("No valid field cases parsed")

        df = pd.DataFrame(rows)
        df = self.apply_column_map(df)
        df = self.validate_minimum(df)
        logger.info(f"VTKAdapter: parsed {len(df)} cases")
        return df

    def load_fields(self, n_points: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load both parameters and Cp field arrays for DeepONet training.

        Returns:
            params_df: DataFrame with flow/geometry parameters.
            fields: Array of shape (n_cases, n_points) with Cp distributions.
        """
        base = Path(self.path)
        field_name = self.config.get("field_name", "Cp")

        # Manifest + numpy layout
        manifest = base / "manifest.csv"
        fields_dir = base / "fields"
        if manifest.exists() and fields_dir.exists():
            params_df = pd.read_csv(manifest)
            params_df = self.apply_column_map(params_df)

            field_arrays = []
            for idx in range(len(params_df)):
                npy_path = fields_dir / f"{idx:04d}.npy"
                if npy_path.exists():
                    arr = np.load(npy_path)
                    # Interpolate to standard grid if needed
                    if len(arr.shape) == 1:
                        cp = self._interpolate_1d(arr, n_points)
                    elif arr.shape[1] >= 2:
                        # Assume columns: x, [y,] Cp
                        cp = self._interpolate_from_xy(arr, n_points)
                    else:
                        cp = self._interpolate_1d(arr.flatten(), n_points)
                    field_arrays.append(cp)
                else:
                    field_arrays.append(np.zeros(n_points))

            fields = np.stack(field_arrays, axis=0)
            logger.info(f"VTKAdapter: loaded {len(fields)} field arrays")
            return params_df, fields

        # Case-directory layout with surface.csv
        case_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        params_list = []
        field_list = []

        for cd in case_dirs:
            params_file = cd / "params.csv"
            surface_file = cd / "surface.csv"
            if not params_file.exists() or not surface_file.exists():
                continue
            try:
                row = pd.read_csv(params_file).iloc[0].to_dict()
                params_list.append(row)

                surf = pd.read_csv(surface_file)
                surf.columns = [c.strip().lower() for c in surf.columns]
                x_col = next((c for c in surf.columns if c in ("x", "x_coord")), surf.columns[0])
                cp_col = next((c for c in surf.columns if c in ("cp", "pressure_coefficient")), surf.columns[-1])
                arr = np.column_stack([surf[x_col].values, surf[cp_col].values])
                cp = self._interpolate_from_xy(arr, n_points)
                field_list.append(cp)
            except Exception as e:
                logger.warning(f"Failed to load field from {cd.name}: {e}")

        if not params_list:
            raise ValueError("No valid field cases found")

        params_df = pd.DataFrame(params_list)
        params_df = self.apply_column_map(params_df)
        fields = np.stack(field_list, axis=0)

        logger.info(f"VTKAdapter: loaded {len(fields)} field arrays from case dirs")
        return params_df, fields

    def _interpolate_1d(self, cp: np.ndarray, n_points: int) -> np.ndarray:
        """Interpolate 1D Cp array to n_points."""
        if len(cp) == n_points:
            return cp
        x_old = np.linspace(0, 1, len(cp))
        x_new = np.linspace(0, 1, n_points)
        return np.interp(x_new, x_old, cp)

    def _interpolate_from_xy(self, arr: np.ndarray, n_points: int) -> np.ndarray:
        """Interpolate (x, Cp) or (x, y, Cp) to uniform n_points grid."""
        if arr.shape[1] == 2:
            x, cp = arr[:, 0], arr[:, 1]
        else:
            x, cp = arr[:, 0], arr[:, -1]

        idx = np.argsort(x)
        x_sorted, cp_sorted = x[idx], cp[idx]
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        return np.interp(x_new, x_sorted, cp_sorted)
