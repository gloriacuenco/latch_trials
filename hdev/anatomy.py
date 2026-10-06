"""Anatomy at latent resolution, used to keep the flow from changing anatomy.

FastSurfer labels are grouped into a few tissue classes and summarized per latent voxel (4 x 4 x 4 mm)
as the fraction of each class. ``AnatomyNet`` is a small network that predicts those fractions from a
latent; it is trained once, on all sites, and then frozen. During flow training, the flow's predicted
end point is passed through it and compared with the source scan's own fractions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["background", "cortex", "cerebral_wm", "lateral_ventricles", "deep_gray", "hippocampus_amygdala",
           "cerebellum_cortex", "cerebellum_wm", "brainstem", "other"]
_GROUPS = {
    "cortex": [3, 42],  # plus every cortical parcel (label >= 1000)
    "cerebral_wm": [2, 41, 77, 251, 252, 253, 254, 255],
    "lateral_ventricles": [4, 5, 43, 44],
    "deep_gray": [10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60],
    "hippocampus_amygdala": [17, 18, 53, 54],
    "cerebellum_cortex": [8, 47],
    "cerebellum_wm": [7, 46],
    "brainstem": [16],
}


def group_labels(seg: np.ndarray) -> np.ndarray:
    """FastSurfer labels -> class index (0 background, last class = any other brain label)."""
    out = np.where(seg > 0, len(CLASSES) - 1, 0).astype(np.uint8)
    for name, ids in _GROUPS.items():
        out[np.isin(seg, ids)] = CLASSES.index(name)
    out[seg >= 1000] = CLASSES.index("cortex")
    return out


def fraction_maps(classes: np.ndarray, factor: int = 4) -> np.ndarray:
    """Class map [X, Y, Z] -> per-block class fractions [K, X/factor, Y/factor, Z/factor] (float32)."""
    onehot = F.one_hot(torch.from_numpy(classes.astype(np.int64)), len(CLASSES)).permute(3, 0, 1, 2).float()
    return F.avg_pool3d(onehot[None], factor)[0].numpy()


class AnatomyNet(nn.Module):
    """Latent [B, 4, X, Y, Z] -> class logits [B, K, X, Y, Z], at the latent's own resolution."""

    def __init__(self, width: int = 64, blocks: int = 6):
        super().__init__()
        layers = [nn.Conv3d(4, width, 3, padding=1)]
        for i in range(blocks):
            d = 2 ** (i % 3)  # dilations 1, 2, 4 widen the view without losing resolution
            layers += [nn.GroupNorm(8, width), nn.SiLU(), nn.Conv3d(width, width, 3, padding=d, dilation=d)]
        layers += [nn.GroupNorm(8, width), nn.SiLU(), nn.Conv3d(width, len(CLASSES), 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def soft_cross_entropy(logits: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against soft targets (class fractions per voxel)."""
    return -(fractions * F.log_softmax(logits, 1)).sum(1).mean()


def soft_dice(probs: torch.Tensor, fractions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice per class [K] between predicted probabilities and target fractions, summed over the batch."""
    inter = (probs * fractions).sum((0, 2, 3, 4))
    return (2 * inter + eps) / (probs.sum((0, 2, 3, 4)) + fractions.sum((0, 2, 3, 4)) + eps)
