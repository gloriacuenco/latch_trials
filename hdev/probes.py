"""The latent site probe shared by the E2 and E3 evaluations (E1's linear probe on the central crop)."""

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


def probe(x_tr, y_tr, x_te, y_te):
    """Linear site probe; returns (balanced accuracy on the test rows, fitted model)."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
    m.fit(x_tr, y_tr)
    return float(balanced_accuracy_score(y_te, m.predict(x_te))), m
