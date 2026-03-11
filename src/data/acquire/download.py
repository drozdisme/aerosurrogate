"""Data acquisition module for AeroSurrogate v2.0.

Downloads and parses open-source aerodynamic datasets:

1. kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils
   — 2900 airfoils, Cl/Cd at Re=1e5, CST coefficients (OpenFOAM CFD)
   — GitHub: https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils

2. NASA airfoil-learning
   — ~1500 airfoils, Cl/Cd/Cm at multiple Re/alpha (XFoil)
   — S3: https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip

3. DOE Airfoil CFD 2k
   — 2000 shapes × 25 AoA × 3 Re (RANS solver HAM2D)
   — OEDI: https://data.openei.org/submissions/5548

Usage:
    python -m src.data.acquire          # download all
    python -m src.data.acquire kanakaero  # download specific source
"""

import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# Source 1: kanakaero — 2900 airfoils with CST + Cl/Cd from OpenFOAM
# ════════════════════════════════════════════════════════════════════

def acquire_kanakaero(output_dir: Path = DATA_DIR) -> Optional[Path]:
    """Download and parse the kanakaero airfoil dataset.

    Source: GitHub (kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils)
    Contains: ~2900 airfoils × multiple AoA, Re=1e5, CST coefficients, Cl, Cd.
    Format: CSV with columns AoA, Cl, Cd, CST_upper_0..5, CST_lower_0..5.
    """
    import requests

    url = "https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils/archive/refs/heads/main.zip"
    out_csv = output_dir / "kanakaero_airfoils.csv"

    if out_csv.exists():
        logger.info(f"kanakaero dataset already exists: {out_csv}")
        return out_csv

    logger.info("Downloading kanakaero dataset...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    rows = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv") and "dataset" in n.lower()]
        if not csv_files:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]

        for csv_name in csv_files:
            try:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
                    if len(df.columns) >= 3:
                        rows.append(df)
                        logger.info(f"  Parsed {csv_name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to parse {csv_name}: {e}")

    if not rows:
        logger.error("No CSV data found in kanakaero archive")
        return None

    combined = pd.concat(rows, ignore_index=True)

    # Standardize column names
    rename = {}
    for col in combined.columns:
        cl = col.strip().lower()
        if cl in ("aoa", "angle_of_attack", "alpha"):
            rename[col] = "alpha"
        elif cl in ("cl", "c_l", "lift_coefficient"):
            rename[col] = "Cl"
        elif cl in ("cd", "c_d", "drag_coefficient"):
            rename[col] = "Cd"
    combined = combined.rename(columns=rename)

    # Add fixed flow condition (Re=1e5)
    combined["reynolds"] = 1e5
    combined["mach"] = 0.0  # Incompressible (low Re)

    # Compute derived targets
    if "Cl" in combined.columns and "Cd" in combined.columns:
        combined["K"] = combined["Cl"] / combined["Cd"].clip(lower=1e-8)
        combined["Cm"] = -0.25 * combined["Cl"]  # Thin-airfoil approximation

    combined.to_csv(out_csv, index=False)
    logger.info(f"kanakaero dataset saved: {out_csv} ({len(combined)} rows)")
    return out_csv


# ════════════════════════════════════════════════════════════════════
# Source 2: NASA airfoil-learning — XFoil multi-Re dataset
# ════════════════════════════════════════════════════════════════════

def acquire_nasa_airfoil(output_dir: Path = DATA_DIR) -> Optional[Path]:
    """Download and parse NASA airfoil-learning dataset.

    Source: NASA public S3 bucket.
    Contains: ~1500 airfoils × multiple Re × multiple alpha,
    Cl, Cd, Cm from XFoil.
    """
    import requests

    url = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip"
    out_csv = output_dir / "nasa_airfoil_learning.csv"

    if out_csv.exists():
        logger.info(f"NASA dataset already exists: {out_csv}")
        return out_csv

    logger.info("Downloading NASA airfoil-learning dataset...")
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    content = b""
    for chunk in resp.iter_content(chunk_size=8192):
        content += chunk

    rows = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            # NASA dataset stores XFoil results as text files
            if name.endswith((".dat", ".csv", ".txt")) and "result" in name.lower():
                try:
                    with zf.open(name) as f:
                        text = f.read().decode("utf-8", errors="replace")
                        parsed = _parse_xfoil_polar(text, name)
                        if parsed is not None:
                            rows.extend(parsed)
                except Exception as e:
                    continue

            # Also look for any consolidated CSV
            if name.endswith(".csv"):
                try:
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                        if "Cl" in df.columns or "cl" in df.columns.str.lower():
                            rows_from_csv = df.to_dict("records")
                            rows.extend(rows_from_csv)
                            logger.info(f"  Parsed CSV {name}: {len(rows_from_csv)} rows")
                except Exception:
                    continue

    if not rows:
        logger.warning("No polar data found in NASA archive. Trying alternative parsing...")
        # Alternative: look for numpy/pickle files
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if name.endswith(".npy"):
                    try:
                        with zf.open(name) as f:
                            arr = np.load(io.BytesIO(f.read()), allow_pickle=True)
                            logger.info(f"  Found numpy array {name}: shape={arr.shape}")
                    except Exception:
                        pass

    if not rows:
        logger.error("Could not parse NASA dataset. Download manually and inspect.")
        return None

    combined = pd.DataFrame(rows)
    # Standardize
    rename = {}
    for col in combined.columns:
        cl = col.strip().lower()
        if cl in ("alpha", "aoa", "angle"):
            rename[col] = "alpha"
        elif cl in ("re", "reynolds"):
            rename[col] = "reynolds"
        elif cl in ("cl",):
            rename[col] = "Cl"
        elif cl in ("cd",):
            rename[col] = "Cd"
        elif cl in ("cm",):
            rename[col] = "Cm"
    combined = combined.rename(columns=rename)

    combined.setdefault("mach", 0.0)
    if "Cl" in combined.columns and "Cd" in combined.columns:
        combined["K"] = combined["Cl"] / combined["Cd"].clip(lower=1e-8)

    combined.to_csv(out_csv, index=False)
    logger.info(f"NASA dataset saved: {out_csv} ({len(combined)} rows)")
    return out_csv


def _parse_xfoil_polar(text: str, filename: str) -> Optional[List[dict]]:
    """Parse XFoil polar output text into list of row dicts."""
    lines = text.strip().split("\n")

    # Extract Re from header
    reynolds = 1e6
    for line in lines[:15]:
        m = re.search(r"Re\s*=?\s*([\d.]+)\s*[eE]\s*([\d]+)", line)
        if m:
            reynolds = float(m.group(1)) * 10 ** int(m.group(2))
            break

    # Find data start (after dashed separator line)
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*-+\s*$", line):
            data_start = i + 1
            break

    if data_start is None:
        return None

    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                alpha = float(parts[0])
                cl = float(parts[1])
                cd = float(parts[2])
                cm = float(parts[3]) if len(parts) > 3 else -0.25 * cl
                rows.append({
                    "alpha": alpha, "reynolds": reynolds, "mach": 0.0,
                    "Cl": cl, "Cd": cd, "Cm": cm,
                })
            except ValueError:
                continue

    return rows if rows else None


# ════════════════════════════════════════════════════════════════════
# Source 3: GitHub CFD datasets search
# ════════════════════════════════════════════════════════════════════

def acquire_github_cfd(output_dir: Path = DATA_DIR) -> Optional[Path]:
    """Search GitHub for public CFD airfoil datasets and download CSV files.

    Uses GitHub API to find repositories tagged with 'cfd-dataset',
    'airfoil-data', or containing aerodynamic coefficient CSVs.
    """
    import requests

    out_csv = output_dir / "github_cfd_combined.csv"
    if out_csv.exists():
        logger.info(f"GitHub CFD dataset already exists: {out_csv}")
        return out_csv

    # Known high-quality repos with downloadable CSV data
    known_repos = [
        {
            "name": "kanakaero",
            "url": "https://raw.githubusercontent.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils/main/Dataset/dataset.csv",
        },
        {
            "name": "rafaelstevenson_transonic",
            "url": "https://raw.githubusercontent.com/rafaelstevenson/Transonic-Airfoil-ANN-Prediction/master/data/RAE2822_Data.csv",
        },
    ]

    frames = []
    for repo in known_repos:
        try:
            logger.info(f"Fetching {repo['name']}...")
            resp = requests.get(repo["url"], timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df["_source"] = repo["name"]
            frames.append(df)
            logger.info(f"  {repo['name']}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed to fetch {repo['name']}: {e}")

    # Also search GitHub API for more repos
    try:
        api_url = "https://api.github.com/search/repositories?q=airfoil+aerodynamic+dataset+csv&sort=stars&per_page=10"
        resp = requests.get(api_url, timeout=30)
        if resp.status_code == 200:
            repos = resp.json().get("items", [])
            for r in repos[:5]:
                logger.info(f"  Found repo: {r['full_name']} ({r['stargazers_count']} stars)")
    except Exception:
        pass

    if not frames:
        logger.warning("No GitHub CFD datasets fetched")
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out_csv, index=False)
    logger.info(f"GitHub CFD combined: {out_csv} ({len(combined)} rows)")
    return out_csv


# ════════════════════════════════════════════════════════════════════
# Master acquisition function
# ════════════════════════════════════════════════════════════════════

def acquire_all(output_dir: Path = DATA_DIR) -> List[Path]:
    """Download all available open-source datasets.

    Returns list of paths to downloaded CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    logger.info("=" * 60)
    logger.info("AeroSurrogate v2.0 — Data Acquisition")
    logger.info("=" * 60)

    # 1. kanakaero (most reliable — direct CSV)
    path = acquire_kanakaero(output_dir)
    if path:
        results.append(path)

    # 2. NASA airfoil-learning
    path = acquire_nasa_airfoil(output_dir)
    if path:
        results.append(path)

    # 3. GitHub CFD datasets
    path = acquire_github_cfd(output_dir)
    if path:
        results.append(path)

    logger.info("-" * 60)
    logger.info(f"Acquired {len(results)} dataset(s)")
    for p in results:
        n = len(pd.read_csv(p))
        logger.info(f"  {p.name}: {n} rows")
    logger.info("=" * 60)

    # Update data_sources.yaml with acquired files
    _update_data_sources(results)

    return results


def _update_data_sources(paths: List[Path]) -> None:
    """Update configs/data_sources.yaml with acquired dataset entries."""
    import yaml

    config_path = Path("configs/data_sources.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {"sources": {}, "merge": {"strategy": "concatenate", "deduplicate": True}}

    sources = config.get("sources", {})

    for p in paths:
        key = p.stem.replace("-", "_").replace(" ", "_")
        if key not in sources:
            sources[key] = {
                "adapter": "csv",
                "path": str(p),
                "enabled": True,
            }
            logger.info(f"Added source '{key}' to data_sources.yaml")

    config["sources"] = sources
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "kanakaero":
        acquire_kanakaero()
    elif target == "nasa":
        acquire_nasa_airfoil()
    elif target == "github":
        acquire_github_cfd()
    elif target == "all":
        paths = acquire_all()
        if paths:
            print(f"\nDone. Run training: python -m src.training.trainer")
    else:
        print(f"Unknown target: {target}")
        print("Usage: python -m src.data.acquire [all|kanakaero|nasa|github]")
