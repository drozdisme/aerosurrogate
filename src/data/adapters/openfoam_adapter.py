"""OpenFOAM case data adapter."""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class OpenFOAMAdapter(BaseAdapter):
    """Parses OpenFOAM case directories into standardized DataFrames.

    Expected structure:
        openfoam_cases/
        ├── case_001/
        │   ├── 0/                     (initial conditions)
        │   ├── constant/
        │   │   └── transportProperties
        │   ├── system/
        │   │   └── controlDict
        │   └── postProcessing/
        │       └── forceCoeffs/
        │           └── 0/
        │               └── coefficient.dat
        ├── case_002/
        │   └── ...

    Flow conditions are extracted from boundary conditions and
    transport properties. Aerodynamic coefficients come from
    forceCoeffs function object output.
    """

    def load(self) -> pd.DataFrame:
        base = Path(self.path)
        if not base.exists():
            raise FileNotFoundError(f"OpenFOAM directory not found: {base}")

        # Find case directories (must contain 'constant' and 'system')
        case_dirs = [
            d for d in sorted(base.iterdir())
            if d.is_dir() and (d / "constant").exists() and (d / "system").exists()
        ]

        if not case_dirs:
            # Maybe base itself is a single case
            if (base / "constant").exists():
                case_dirs = [base]
            else:
                raise FileNotFoundError(f"No OpenFOAM cases found in {base}")

        coeffs_file = self.config.get(
            "coefficients_file",
            "postProcessing/forceCoeffs/0/coefficient.dat",
        )

        rows = []
        for case_dir in case_dirs:
            try:
                flow = self._parse_flow_conditions(case_dir)
                coeffs = self._parse_force_coeffs(case_dir / coeffs_file)
                rows.append({**flow, **coeffs})
            except Exception as e:
                logger.warning(f"Failed to parse {case_dir.name}: {e}")
                continue

        if not rows:
            raise ValueError("No valid OpenFOAM cases parsed")

        df = pd.DataFrame(rows)
        logger.info(f"OpenFOAMAdapter: parsed {len(df)} cases from {base}")

        if "K" not in df.columns and "Cl" in df.columns and "Cd" in df.columns:
            df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)

        df = self.apply_column_map(df)
        df = self.validate_minimum(df)
        return df

    def _parse_flow_conditions(self, case_dir: Path) -> Dict[str, float]:
        """Extract Mach, Re, alpha from OpenFOAM case files."""
        params = {}

        # Try reading from transportProperties or controlDict comments
        for search_file in ["constant/transportProperties", "system/controlDict",
                            "0/U", "0/include/initialConditions"]:
            fpath = case_dir / search_file
            if fpath.exists():
                text = fpath.read_text(errors="replace")
                self._extract_params_from_text(text, params)

        # Try to get velocity magnitude and compute Mach from 0/U
        u_file = case_dir / "0/U"
        if u_file.exists() and "mach" not in params:
            text = u_file.read_text(errors="replace")
            m = re.search(r"internalField\s+uniform\s+\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)", text)
            if m:
                ux, uy, uz = float(m.group(1)), float(m.group(2)), float(m.group(3))
                u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
                alpha_rad = np.arctan2(uy, ux)
                params.setdefault("alpha", np.degrees(alpha_rad))
                # Assume speed of sound ~340 m/s if not provided
                a = params.get("speed_of_sound", 340.0)
                params.setdefault("mach", u_mag / a)

        params.setdefault("mach", 0.3)
        params.setdefault("reynolds", 1e6)
        params.setdefault("alpha", 0.0)
        params.setdefault("beta", 0.0)
        params.setdefault("altitude", 0.0)

        return params

    def _extract_params_from_text(self, text: str, params: Dict) -> None:
        """Search text for common parameter definitions."""
        patterns = {
            "mach": r"[Mm]ach\w*\s+([\d.eE+-]+)",
            "reynolds": r"[Rr]e(?:ynolds)?\w*\s+([\d.eE+-]+)",
            "alpha": r"(?:AoA|alpha|angleOfAttack)\s+([\d.eE+-]+)",
            "speed_of_sound": r"speedOfSound\s+([\d.eE+-]+)",
        }
        for key, pat in patterns.items():
            if key not in params:
                m = re.search(pat, text)
                if m:
                    params[key] = float(m.group(1))

    def _parse_force_coeffs(self, coeffs_path: Path) -> Dict[str, float]:
        """Parse OpenFOAM forceCoeffs output file."""
        if not coeffs_path.exists():
            raise FileNotFoundError(f"Force coefficients not found: {coeffs_path}")

        lines = []
        header = None
        for line in coeffs_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                # Last comment line is typically the header
                header = stripped.lstrip("# ").split()
                continue
            if stripped:
                lines.append(stripped.split())

        if not lines:
            raise ValueError(f"No data in {coeffs_path}")

        # Take last timestep (converged)
        last = lines[-1]
        values = [float(v) for v in last]

        coeffs = {}
        if header:
            mapping = {}
            for i, h in enumerate(header):
                hl = h.lower()
                if "cl" in hl or "lift" in hl:
                    mapping["Cl"] = i
                elif "cd" in hl or "drag" in hl:
                    mapping["Cd"] = i
                elif "cm" in hl or "moment" in hl:
                    mapping["Cm"] = i
            for name, idx in mapping.items():
                if idx < len(values):
                    coeffs[name] = values[idx]
        else:
            # Fallback: assume time, Cd, Cl, Cm order (common OpenFOAM layout)
            if len(values) >= 4:
                coeffs["Cd"] = values[1]
                coeffs["Cl"] = values[2]
                coeffs["Cm"] = values[3]
            elif len(values) >= 3:
                coeffs["Cd"] = values[1]
                coeffs["Cl"] = values[2]

        if "Cl" not in coeffs:
            raise ValueError(f"Could not extract Cl from {coeffs_path}")

        return coeffs
