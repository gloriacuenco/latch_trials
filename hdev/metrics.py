"""Volume-level measures shared by the E2 evaluations (1 mm reference space, values in [0, 1])."""

from __future__ import annotations

import math

import numpy as np

FG_THR = 0.02
LEVELS = np.linspace(0.005, 0.995, 199)


def psnr_fg(x: np.ndarray, y: np.ndarray, thr: float = FG_THR) -> float:
    """PSNR on a fixed [0, 1] scale over the input's foreground."""
    m = x > thr
    return -10 * math.log10(max(float(((y - x)[m] ** 2).mean()), 1e-12))


def quantile_profile(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Intensity quantiles of the voxels in ``mask``: a compact stand-in for the distribution."""
    return np.quantile(x[mask], LEVELS)


def w1(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Wasserstein-1 distance between two distributions given by quantiles at the same levels."""
    return float(np.abs(q_a - q_b).mean())
