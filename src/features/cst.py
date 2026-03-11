"""Class Shape Transformation (CST) airfoil parameterization.

CST coefficients provide a compact, smooth representation of airfoil
geometry that is far more informative for ML models than simple
thickness/camber descriptors.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.special import comb

logger = logging.getLogger(__name__)


def class_function(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """CST class function C(x) = x^n1 * (1-x)^n2."""
    return np.power(x, n1) * np.power(1.0 - x, n2)


def shape_function(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """CST shape function S(x) using Bernstein polynomials.

    S(x) = sum_{i=0}^{n} A_i * K_i * x^i * (1-x)^{n-i}
    where K_i = C(n, i) is the binomial coefficient.
    """
    n = len(coeffs) - 1
    s = np.zeros_like(x)
    for i, a in enumerate(coeffs):
        k = comb(n, i, exact=True)
        s += a * k * np.power(x, i) * np.power(1.0 - x, n - i)
    return s


def cst_curve(x: np.ndarray, coeffs: np.ndarray,
              n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """Compute CST curve y(x) = C(x) * S(x)."""
    return class_function(x, n1, n2) * shape_function(x, coeffs)


def fit_cst_coefficients(x: np.ndarray, y: np.ndarray,
                         n_coeffs: int = 6) -> np.ndarray:
    """Fit CST coefficients to an airfoil surface (upper or lower).

    Uses least-squares fitting of Bernstein polynomial basis.

    Args:
        x: Chordwise coordinates normalized to [0, 1].
        y: Surface ordinates.
        n_coeffs: Number of CST coefficients.

    Returns:
        Array of n_coeffs CST coefficients.
    """
    c = class_function(x)
    n = n_coeffs - 1

    # Build Bernstein basis matrix
    basis = np.zeros((len(x), n_coeffs))
    for i in range(n_coeffs):
        k = comb(n, i, exact=True)
        basis[:, i] = c * k * np.power(x, i) * np.power(1.0 - x, n - i)

    # Regularized least squares
    lam = 1e-8
    ATA = basis.T @ basis + lam * np.eye(n_coeffs)
    ATy = basis.T @ y
    coeffs = np.linalg.solve(ATA, ATy)
    return coeffs


def extract_cst_from_coordinates(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_upper: int = 6,
    n_lower: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CST coefficients from raw airfoil coordinates.

    Assumes coordinates go from trailing edge along the upper surface
    to the leading edge, then back along the lower surface.

    Args:
        x_coords: Chordwise coordinates.
        y_coords: Surface ordinates.
        n_upper: Number of upper-surface CST coefficients.
        n_lower: Number of lower-surface CST coefficients.

    Returns:
        (upper_coeffs, lower_coeffs)
    """
    # Normalize x to [0, 1]
    x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-12)

    # Split upper and lower surfaces at the leading edge (min x)
    le_idx = np.argmin(x_norm)

    if le_idx > 0 and le_idx < len(x_norm) - 1:
        x_upper = x_norm[:le_idx + 1][::-1]
        y_upper = y_coords[:le_idx + 1][::-1]
        x_lower = x_norm[le_idx:]
        y_lower = y_coords[le_idx:]
    else:
        # Fallback: split by y sign
        upper_mask = y_coords >= 0
        lower_mask = y_coords <= 0
        x_upper = x_norm[upper_mask]
        y_upper = y_coords[upper_mask]
        x_lower = x_norm[lower_mask]
        y_lower = y_coords[lower_mask]

    # Clip x to (0, 1) open interval for stability
    eps = 1e-6
    x_upper = np.clip(x_upper, eps, 1.0 - eps)
    x_lower = np.clip(x_lower, eps, 1.0 - eps)

    upper_coeffs = fit_cst_coefficients(x_upper, y_upper, n_upper) if len(x_upper) > n_upper else np.zeros(n_upper)
    lower_coeffs = fit_cst_coefficients(x_lower, y_lower, n_lower) if len(x_lower) > n_lower else np.zeros(n_lower)

    return upper_coeffs, lower_coeffs
