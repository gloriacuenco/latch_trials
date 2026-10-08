"""Pretrained NVIDIA 3D VAEs (MAISI and NV-Generate-MR), used frozen.

Both share the MAISI architecture: 4 latent channels at a quarter of the resolution on each axis.
Encoding uses the posterior mean, so it is deterministic.
"""

from __future__ import annotations

import torch

VAES = {
    "maisi": {"repo": "MONAI/maisi_ct_generative", "file": "models/autoencoder.pt", "num_splits": 2},
    "nvmr": {"repo": "nvidia/NV-Generate-MR", "file": "models/autoencoder_v2.pt", "num_splits": 4},
}
DOWNSAMPLE = 4
LATENT_CHANNELS = 4


def _state_dict(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("autoencoder_state_dict", "unet_state_dict", "model_state_dict", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def load_vae(name: str, ckpt: str, device: torch.device) -> torch.nn.Module:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

    model = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=LATENT_CHANNELS,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2], norm_num_groups=32, norm_eps=1e-6,
        attention_levels=[False, False, False], with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False, use_checkpointing=False, use_convtranspose=False,
        norm_float16=True, num_splits=VAES[name]["num_splits"], dim_split=1,
    )
    model.load_state_dict(_state_dict(ckpt), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


@torch.no_grad()
def encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """[B, 1, X, Y, Z] in [0, 1] -> [B, 4, X/4, Y/4, Z/4] (posterior mean, float32)."""
    # MONAI 1.5.1 encode_stage_2_inputs samples; eval() does not disable it.
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        z, _ = model.encode(x)
    return z.float()


@torch.no_grad()
def decode(model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16):
        x = model.decode_stage_2_outputs(z)
    return x.float()
