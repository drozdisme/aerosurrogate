"""Vortex Lattice Method (AVL / OpenVSP) data adapter."""

import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class VLMAdapter(BaseAdapter):
    """Parses output from Vortex Lattice Method solvers.

    Supported formats:
      - AVL (Athena Vortex Lattice) — .run and .st files
      - OpenVSP DegenGeom + VSPAERO output

    AVL expected structure:
        avl_runs/
        ├── case_001.run    (run file with flow conditions)
        ├── case_001.st     (stability derivatives and coefficients)
        ├── case_002.run
        ├── case_002.st

    OpenVSP expected structure:
        vspaero_runs/
        ├── model_001/
        │   ├── model.history   (convergence)
        │   ├── model.lod       (load distribution)
        │   └── model.polar     (sweep results)
    """

    def load(self) -> pd.DataFrame:
        fmt = self.config.get("format", "avl")
        base = Path(self.path)
        if not base.exists():
            raise FileNotFoundError(f"VLM directory not found: {base}")

        if fmt == "avl":
            df = self._load_avl(base)
        elif fmt == "vspaero":
            df = self._load_vspaero(base)
        else:
            raise ValueError(f"Unknown VLM format: {fmt}")

        if "K" not in df.columns and "Cl" in df.columns and "Cd" in df.columns:
            df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)

        df = self.apply_column_map(df)
        df = self.validate_minimum(df)
        return df

    # ── AVL parsing ────────────────────────────────────────────
    def _load_avl(self, base: Path) -> pd.DataFrame:
        st_files = sorted(base.glob("*.st"))
        if not st_files:
            st_files = sorted(base.glob("**/*.st"))
        if not st_files:
            raise FileNotFoundError(f"No AVL .st files found in {base}")

        rows = []
        for st_file in st_files:
            try:
                row = self._parse_avl_st(st_file)
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to parse AVL {st_file.name}: {e}")

        df = pd.DataFrame(rows)
        logger.info(f"VLMAdapter(AVL): parsed {len(df)} cases")
        return df

    def _parse_avl_st(self, st_path: Path) -> Dict[str, float]:
        """Parse AVL stability-derivative .st file."""
        text = st_path.read_text(errors="replace")
        params = {}

        patterns = {
            "alpha": r"Alpha\s*=\s*([\d.eE+-]+)",
            "beta": r"Beta\s*=\s*([\d.eE+-]+)",
            "mach": r"Mach\s*=\s*([\d.eE+-]+)",
            "Cl": r"CLtot\s*=\s*([\d.eE+-]+)",
            "Cd": r"CDtot\s*=\s*([\d.eE+-]+)",
            "Cm": r"Cmtot\s*=\s*([\d.eE+-]+)",
        }

        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                params[key] = float(m.group(1))

        # AVL may report CDind only (no viscous drag)
        if "Cd" not in params:
            m = re.search(r"CDind\s*=\s*([\d.eE+-]+)", text)
            if m:
                params["Cd"] = float(m.group(1))

        params.setdefault("mach", 0.0)
        params.setdefault("reynolds", 1e6)
        params.setdefault("beta", 0.0)
        params.setdefault("altitude", 0.0)

        if "Cl" not in params:
            raise ValueError(f"CLtot not found in {st_path}")

        return params

    # ── VSPAERO parsing ────────────────────────────────────────
    def _load_vspaero(self, base: Path) -> pd.DataFrame:
        polar_files = sorted(base.glob("**/*.polar"))
        history_files = sorted(base.glob("**/*.history"))

        if polar_files:
            return self._parse_vspaero_polar(polar_files)
        elif history_files:
            return self._parse_vspaero_history(history_files)
        else:
            raise FileNotFoundError(f"No VSPAERO .polar or .history files in {base}")

    def _parse_vspaero_polar(self, files: List[Path]) -> pd.DataFrame:
        """Parse VSPAERO polar sweep files."""
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, comment="#", sep=r"\s+", engine="python")
                df.columns = [c.strip() for c in df.columns]
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to parse VSPAERO polar {f}: {e}")

        if not dfs:
            raise ValueError("No valid VSPAERO polar files")

        combined = pd.concat(dfs, ignore_index=True)

        # Standardize column names
        rename = {}
        for col in combined.columns:
            cl = col.lower()
            if cl in ("aoa", "alpha"):
                rename[col] = "alpha"
            elif cl in ("cl", "cltot"):
                rename[col] = "Cl"
            elif cl in ("cd", "cdtot", "cdtotal"):
                rename[col] = "Cd"
            elif cl in ("cmy", "cm", "cmtot"):
                rename[col] = "Cm"
            elif cl == "mach":
                rename[col] = "mach"

        combined = combined.rename(columns=rename)
        combined.setdefault("mach", 0.0)
        combined.setdefault("reynolds", 1e6)
        combined.setdefault("beta", 0.0)
        combined.setdefault("altitude", 0.0)

        logger.info(f"VLMAdapter(VSPAERO): parsed {len(combined)} points")
        return combined

    def _parse_vspaero_history(self, files: List[Path]) -> pd.DataFrame:
        """Parse VSPAERO convergence history — take last line per case."""
        rows = []
        for f in files:
            try:
                df = pd.read_csv(f, comment="#", sep=r"\s+", engine="python")
                df.columns = [c.strip() for c in df.columns]
                last = df.iloc[-1].to_dict()
                rows.append(last)
            except Exception as e:
                logger.warning(f"Failed to parse {f}: {e}")

        if not rows:
            raise ValueError("No valid VSPAERO history files")

        return pd.DataFrame(rows)
