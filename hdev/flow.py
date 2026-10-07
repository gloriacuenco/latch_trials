"""The LATCH flow: a site-conditioned velocity field in the latent space of a frozen VAE.

Pieces, in the order the training script uses them:

1. ``LatentStats``: per-channel mean and standard deviation of the training latents, so the flow
   works on roughly unit-scale numbers.
2. Couplings: which NYU latent each source latent is paired with during training. A coupling is a
   matrix P (sources x NYU scans) whose rows sum to 1; training samples a partner from row i.
   ``random`` pairs uniformly; ``latent`` and ``anatomy`` solve an entropic optimal-transport
   problem (Sinkhorn) on a cost: squared latent distance, or distance between anatomy features
   (FastSurfer regional volumes, age, sex). ``anatomy`` is the novel element.
3. ``VelocityNet``: a 3D U-Net v(z, t, site) that predicts the direction to move a latent at time
   t in [0, 1]. The site label tells it where the scan comes from; an extra "unknown site" label,
   used for a fraction of training steps, lets it handle scans from sites it never saw.
4. ``transport``: integrates dz/dt = v(z, t, site) from t = 0 (the source scan) to t = 1 (its NYU
   version) with a fixed number of midpoint steps. No noise anywhere, so it is deterministic, and
   running it from t = 1 back to t = 0 undoes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- latent scaling


@dataclass
class LatentStats:
    mean: torch.Tensor  # [C]
    std: torch.Tensor  # [C]

    @classmethod
    def fit(cls, z: np.ndarray) -> "LatentStats":
        t = torch.from_numpy(z.astype(np.float32))
        return cls(t.mean((0, 2, 3, 4)), t.std((0, 2, 3, 4)))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean.to(z)[:, None, None, None]) / self.std.to(z)[:, None, None, None]

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z)[:, None, None, None] + self.mean.to(z)[:, None, None, None]


# ----------------------------------------------------------------------------- couplings

ANATOMY_REGIONS = ["cortex", "cerebral_wm", "lateral_ventricles", "cerebellum_cortex", "cerebellum_wm",
                   "thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "brainstem"]


def anatomy_features(brain: pd.DataFrame) -> np.ndarray:
    """Anatomy of each scan as a vector: log total brain volume, each region as a share of the brain,
    age (log) and sex. Standardized later, on the training scans."""
    total = brain["vol_brain"].to_numpy(dtype=float)
    cols = [np.log(total)] + [brain[f"vol_{r}"].to_numpy(dtype=float) / total for r in ANATOMY_REGIONS]
    cols += [np.log(brain["age"].to_numpy(dtype=float)), (brain["sex"].to_numpy() == 1).astype(float)]
    return np.stack(cols, 1)


def standardize(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (a - ref.mean(0)) / (ref.std(0) + 1e-8)


def sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between every row of a [N, D] and b [M, D] -> [N, M]."""
    return (torch.cdist(a.float(), b.float()) ** 2) / a.shape[1]


def sinkhorn(cost: torch.Tensor, eps: float, iters: int = 500) -> torch.Tensor:
    """Entropic optimal transport with uniform marginals; returns a plan whose rows sum to 1.

    ``eps`` is relative to the median cost: small = close to exact OT, large = close to random."""
    c = cost / cost.median()
    n, m = c.shape
    log_k = -c / eps
    log_u, log_v = torch.zeros(n, dtype=c.dtype), torch.zeros(m, dtype=c.dtype)
    log_a, log_b = torch.full((n,), -np.log(n), dtype=c.dtype), torch.full((m,), -np.log(m), dtype=c.dtype)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], 1)
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], 0)
    plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
    return plan / plan.sum(1, keepdim=True)


def make_coupling(kind: str, z_src: np.ndarray, z_tgt: np.ndarray, a_src: np.ndarray | None,
                  a_tgt: np.ndarray | None, eps: float) -> torch.Tensor:
    """Rows: training source scans; columns: training NYU scans; each row is a distribution over partners."""
    n, m = len(z_src), len(z_tgt)
    if kind == "random":
        return torch.full((n, m), 1.0 / m)
    if kind == "latent":
        cost = sq_dist(torch.from_numpy(z_src.reshape(n, -1)), torch.from_numpy(z_tgt.reshape(m, -1)))
    elif kind == "anatomy":
        cost = sq_dist(torch.from_numpy(a_src), torch.from_numpy(a_tgt))
    else:
        raise ValueError(kind)
    return sinkhorn(cost.double(), eps).float()


# ----------------------------------------------------------------------------- velocity network


class VelocityNet(nn.Module):
    """v(z, t, site): MONAI's 3D diffusion U-Net with a learned embedding per source site.

    Site labels 0..n_sites-1; label ``n_sites`` means "unknown site"."""

    def __init__(self, n_sites: int, channels=(64, 128, 256), attention=(False, False, True)):
        super().__init__()
        from monai.networks.nets import DiffusionModelUNet

        self.unknown = n_sites
        self.unet = DiffusionModelUNet(
            spatial_dims=3, in_channels=4, out_channels=4, channels=channels, attention_levels=attention,
            num_res_blocks=2, num_head_channels=32, num_class_embeds=n_sites + 1,
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        # The U-Net embeds time with sinusoids made for diffusion steps 0-1000, so t in [0, 1] is scaled.
        return self.unet(z, timesteps=t * 1000, class_labels=site)


@torch.no_grad()
def transport(net: nn.Module, z: torch.Tensor, site: torch.Tensor, steps: int = 20,
              reverse: bool = False) -> torch.Tensor:
    """Solve dz/dt = v(z, t, site) with the midpoint rule: t from 0 to 1 (harmonize), or 1 to 0 (undo)."""
    ts = torch.linspace(1.0, 0.0, steps + 1) if reverse else torch.linspace(0.0, 1.0, steps + 1)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = float(t1 - t0)
        tt = torch.full((z.shape[0],), float(t0), device=z.device)
        k1 = net(z, tt, site)
        k2 = net(z + 0.5 * h * k1, tt + 0.5 * h, site)
        z = z + h * k2
    return z
