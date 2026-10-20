"""Latent site probes shared by the E2 and E3 evaluations.

``coarse``: E1's features on the central crop (4 x 4 x 4 block averages, channel means and spreads),
mostly regional brightness and contrast. ``fine``: the same crop's detail energy (spread of what is
left after removing a 4-voxel blur, per block and per channel), which a coarse-only change cannot hide.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def coarse(z: np.ndarray, crop: int = 6) -> np.ndarray:
    t = torch.from_numpy(z[:, :, crop:-crop, crop:-crop, crop:-crop].astype(np.float32))
    pooled = torch.nn.functional.adaptive_avg_pool3d(t, 4).flatten(1)
    return torch.cat([pooled, t.mean((2, 3, 4)), t.std((2, 3, 4))], 1).numpy()


def fine(z: np.ndarray, crop: int = 6, scale: int = 4) -> np.ndarray:
    t = torch.from_numpy(z[:, :, crop:-crop, crop:-crop, crop:-crop].astype(np.float32))
    small = torch.nn.functional.avg_pool3d(t, scale, ceil_mode=True)
    detail = t - torch.nn.functional.interpolate(small, size=t.shape[2:], mode="trilinear", align_corners=False)
    energy = torch.nn.functional.adaptive_avg_pool3d(detail ** 2, 4).flatten(1).sqrt()
    return torch.cat([energy, detail.std((2, 3, 4))], 1).numpy()


def probe(x_tr, y_tr, x_te, y_te):
    """Linear site probe; returns (balanced accuracy on the test rows, fitted model)."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
    m.fit(x_tr, y_tr)
    return float(balanced_accuracy_score(y_te, m.predict(x_te))), m
