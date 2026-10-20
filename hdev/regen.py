"""LATCH regeneration: re-draw a scan as NYU from its own anatomy.

Why not the E2 flow: an ODE transport is invertible, so its output still carries everything about
the source, site included; it can only move site information where a frozen probe does not look.
Here the scan is partly replaced by noise and the lost detail is re-drawn by a model that only
knows "NYU" and "this anatomy", so site detail destroyed by the noise cannot come back.

Pieces:

1. ``RegenNet``: a 3D U-Net v(z, t, site, anatomy). It is a generator: trained on all 17 sites to
   move pure noise (t = 0) to a real latent (t = 1), told the scan's site and given its FastSurfer
   tissue-fraction maps (hdev/anatomy.py) as extra input channels. An extra "unknown site" label,
   used for a fraction of training steps, lets it generate without a site and enables site guidance.
2. ``regenerate``: harmonization. Keep a fraction ``keep`` of the source latent and fill the rest
   with noise, z = keep * source + (1 - keep) * noise (the point the generator would pass through at
   t = keep), then run the generator from t = keep to 1 as NYU, with the source's anatomy maps.
   keep = 1 returns the input; keep = 0 draws a new NYU scan from the anatomy maps alone. In between,
   ``keep`` trades site removal against faithfulness to the input.
3. Coarse/fine split (``fine_scale`` > 0): the site lives mostly in the coarse part of a latent
   (regional brightness and contrast), which survives noise; the subject's own detail lives in the
   fine part. At every step the fine part of the drawing is replaced by the source's fine part,
   noised to that step, so the generator re-draws only the coarse part, as NYU, from the anatomy
   maps. ``fine_scale`` is the blur size in latent voxels (4 mm each) that separates the two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdev.anatomy import CLASSES


class RegenNet(nn.Module):
    """v(z, t, site, anatomy): MONAI's 3D diffusion U-Net; the anatomy maps are concatenated to z.

    Site labels 0..n_sites-1; label ``n_sites`` means "unknown site"."""

    def __init__(self, n_sites: int, channels=(64, 128, 256), attention=(False, False, True)):
        super().__init__()
        from monai.networks.nets import DiffusionModelUNet

        self.unknown = n_sites
        self.unet = DiffusionModelUNet(
            spatial_dims=3, in_channels=4 + len(CLASSES), out_channels=4, channels=channels,
            attention_levels=attention, num_res_blocks=2, num_head_channels=32, num_class_embeds=n_sites + 1,
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, site: torch.Tensor, anatomy: torch.Tensor) -> torch.Tensor:
        # The U-Net embeds time with sinusoids made for diffusion steps 0-1000, so t in [0, 1] is scaled.
        return self.unet(torch.cat([z, anatomy.to(z.dtype)], 1), timesteps=t * 1000, class_labels=site)


def guided(net: RegenNet, z: torch.Tensor, t: torch.Tensor, site: torch.Tensor, anatomy: torch.Tensor,
           guidance: float) -> torch.Tensor:
    """Velocity with site guidance: guidance 1 = plain; above 1 pushes further toward ``site``'s look
    (unknown-site velocity + guidance x the difference the site label makes)."""
    v = net(z, t, site, anatomy)
    if guidance == 1.0:
        return v
    v_unknown = net(z, t, torch.full_like(site, net.unknown), anatomy)
    return v_unknown + guidance * (v - v_unknown)


def coarse_part(z: torch.Tensor, scale: int) -> torch.Tensor:
    """Low-pass a latent [B, C, X, Y, Z]: average over blocks of ``scale`` voxels, then interpolate back."""
    small = F.adaptive_avg_pool3d(z, [max(1, n // scale) for n in z.shape[2:]])
    return F.interpolate(small, size=z.shape[2:], mode="trilinear", align_corners=False)


@torch.no_grad()
def regenerate(net: RegenNet, z_src: torch.Tensor, noise: torch.Tensor, site: torch.Tensor,
               anatomy: torch.Tensor, keep: float, steps: int = 20, guidance: float = 1.0,
               fine_scale: int = 0) -> torch.Tensor:
    """Noise the source to t = keep, then integrate dz/dt = v from t = keep to 1 (midpoint rule,
    ``steps`` steps over that interval) as ``site`` with the source's anatomy maps. With
    ``fine_scale`` > 0, the fine part is reset to the noised source's after every step."""
    z = keep * z_src + (1 - keep) * noise
    ts = torch.linspace(keep, 1.0, steps + 1)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        h = float(t1 - t0)
        tt = torch.full((z.shape[0],), float(t0), device=z.device)
        k1 = guided(net, z, tt, site, anatomy, guidance)
        k2 = guided(net, z + 0.5 * h * k1, tt + 0.5 * h, site, anatomy, guidance)
        z = z + h * k2
        if fine_scale > 0:
            ref = float(t1) * z_src + (1 - float(t1)) * noise  # the source on the same noise path
            z = coarse_part(z, fine_scale) + (ref - coarse_part(ref, fine_scale))
    return z
