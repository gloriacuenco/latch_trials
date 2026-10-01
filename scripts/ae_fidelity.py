"""Reconstruction fidelity of pretrained 3D autoencoders on ABIDE I validation volumes.

TRDP2 feasibility gate: a latent harmonizer cannot preserve more anatomy than its
autoencoder reconstructs. Every model is scored in one reference space: the raw T1
volume reoriented to RAS, robust-normalized to [0, 1] as in the ISBI 2027 audit and
resampled to 1 mm isotropic. Uses the validation split only.

Models (``--models``):
  maisi        MONAI MAISI VAE (MONAI/maisi_ct_generative, models/autoencoder.pt)
  nvmr         NV-Generate-MR VAE (nvidia/NV-Generate-MR, models/autoencoder_v2.pt)
  hcld         adapted HCLD AutoencoderKL (input resized to 192x192x64 as in training)
  hcld_resize  the HCLD resize round trip alone, without the autoencoder
  blur05/blur10  Gaussian blur with sigma 0.5/1.0 mm, to calibrate the metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Preprocessing shared with the ISBI 2027 audit (HarmonIt, tag isbi2027-submission).
from harmonit.data.abide_slices_dataset import robust_normalize

FG_THR = 0.02
HCLD_SHAPE_DHW = (192, 192, 64)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="Directory with abide_manifest.csv, splits.json and ABIDE/.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--maisi-ckpt", default=None)
    p.add_argument("--nvmr-ckpt", default=None)
    p.add_argument("--hcld-root", default=None)
    p.add_argument("--hcld-ckpt", default=None)
    p.add_argument("--save-nifti", action="store_true", help="Save 1 mm input/outputs for one subject per site.")
    return p


# ---------------------------------------------------------------- data

def resolve_t1(path_value: str, data_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    parts = path.parts
    return data_dir / Path(*parts[parts.index("ABIDE"):])


def load_reference(t1_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple, tuple, np.ndarray]:
    """Raw T1 -> RAS -> robust [0,1] -> 1 mm isotropic (trilinear). Returns [X, Y, Z] tensor and its affine."""
    img = nib.as_closest_canonical(nib.load(str(t1_path)))
    vol = robust_normalize(img.get_fdata(dtype=np.float32))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    shape_1mm = tuple(max(1, int(round(s * z))) for s, z in zip(vol.shape, zooms))
    t = torch.from_numpy(np.ascontiguousarray(vol)).to(device)[None, None]
    t = F.interpolate(t, size=shape_1mm, mode="trilinear", align_corners=False)
    # align_corners=False: new index i maps to old index (i + 0.5) * s - 0.5, s = old/new size.
    scale = np.array([o / n for o, n in zip(vol.shape, shape_1mm)])
    step = np.eye(4)
    step[:3, :3] = np.diag(scale)
    step[:3, 3] = 0.5 * scale - 0.5
    return t[0, 0].clamp(0.0, 1.0), zooms, tuple(vol.shape), img.affine @ step


# ---------------------------------------------------------------- filters and metrics

def gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a [X, Y, Z] tensor with replicate padding."""
    k = gaussian_kernel1d(sigma, vol.device)
    r = (k.numel() - 1) // 2
    x = vol[None, None]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x[0, 0]


def ssim_map(a: torch.Tensor, b: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = blur3d(a, sigma), blur3d(b, sigma)
    s_aa = blur3d(a * a, sigma) - mu_a ** 2
    s_bb = blur3d(b * b, sigma) - mu_b ** 2
    s_ab = blur3d(a * b, sigma) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))


def grad_mag(vol: torch.Tensor) -> torch.Tensor:
    """Central-difference gradient magnitude on the interior [1:-1]^3."""
    gx = (vol[2:, 1:-1, 1:-1] - vol[:-2, 1:-1, 1:-1]) / 2
    gy = (vol[1:-1, 2:, 1:-1] - vol[1:-1, :-2, 1:-1]) / 2
    gz = (vol[1:-1, 1:-1, 2:] - vol[1:-1, 1:-1, :-2]) / 2
    return torch.sqrt(gx * gx + gy * gy + gz * gz)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12))


def metrics(x: torch.Tensor, r: torch.Tensor) -> dict[str, float]:
    m = x > FG_THR
    d = (r - x)
    mse_fg = float((d[m] ** 2).mean())
    mse_all = float((d ** 2).mean())
    s = ssim_map(x, r)
    gm_x, gm_r = grad_mag(x), grad_mag(r)
    mi = m[1:-1, 1:-1, 1:-1]
    hp_x = (x - blur3d(x, 1.0))[m]
    hp_r = (r - blur3d(r, 1.0))[m]
    xs = torch.sort(x[m])[0]
    rs = torch.sort(r[m])[0]
    return {
        "psnr_fg": -10 * math.log10(max(mse_fg, 1e-12)),
        "psnr_vol": -10 * math.log10(max(mse_all, 1e-12)),
        "mae_fg": float(d[m].abs().mean()),
        "ssim_fg": float(s[m].mean()),
        "ssim_vol": float(s.mean()),
        "grad_ratio": float(gm_r[mi].mean() / gm_x[mi].mean().clamp_min(1e-12)),
        "grad_corr": corr(gm_x[mi], gm_r[mi]),
        "highpass_corr": corr(hp_x, hp_r),
        "w1_fg": float((xs - rs).abs().mean()),
        "psnr_fg_s1": -10 * math.log10(max(float(((blur3d(r, 1.0) - blur3d(x, 1.0))[m] ** 2).mean()), 1e-12)),
        "ssim_fg_s1": float(ssim_map(blur3d(x, 1.0), blur3d(r, 1.0))[m].mean()),
        "recon_min": float(r.min()),
        "recon_max": float(r.max()),
        "fg_voxels": int(m.sum()),
    }


def mid_slices(vol: torch.Tensor, center: tuple[int, int, int]) -> np.ndarray:
    """Axial, coronal and sagittal slices through ``center`` padded to 256x256, float16."""
    cx, cy, cz = center
    out = []
    for sl in (vol[:, :, cz], vol[:, cy, :], vol[cx, :, :]):
        a = sl.float().cpu().numpy()
        canvas = np.zeros((256, 256), dtype=np.float32)
        h, w = min(256, a.shape[0]), min(256, a.shape[1])
        oh, ow = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
        ch, cw = (256 - h) // 2, (256 - w) // 2
        canvas[ch:ch + h, cw:cw + w] = a[oh:oh + h, ow:ow + w]
        out.append(canvas)
    return np.stack(out).astype(np.float16)


# ---------------------------------------------------------------- models

def load_state(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("autoencoder_state_dict", "unet_state_dict", "model_state_dict", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def build_maisi(ckpt: str, num_splits: int, device: torch.device) -> torch.nn.Module:
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

    model = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2], norm_num_groups=32, norm_eps=1e-6,
        attention_levels=[False, False, False], with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False, use_checkpointing=False, use_convtranspose=False,
        norm_float16=True, num_splits=num_splits, dim_split=1,
    )
    model.load_state_dict(load_state(ckpt), strict=True)
    return model.to(device).eval()


def build_hcld(hcld_root: str, ckpt: str, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(hcld_root))
    from generative.networks.nets import AutoencoderKL

    model = AutoencoderKL(
        spatial_dims=3, in_channels=1, out_channels=1, num_channels=(32, 64, 64), latent_channels=6,
        num_res_blocks=2, norm_num_groups=8, attention_levels=(False, False, True),
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_flash_attention=False, use_checkpointing=False,
    )
    model.load_state_dict(load_state(ckpt), strict=True)
    return model.to(device).eval()


def pad_to_multiple(x: torch.Tensor, k: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
    shape = x.shape[-3:]
    pads = []
    for s in reversed(shape):
        pads.extend([0, (-s) % k])
    return F.pad(x, pads), tuple(shape)


@torch.no_grad()
def recon_maisi(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    inp, shape = pad_to_multiple(x[None, None], 16)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            z = model.encode_stage_2_inputs(inp)
            r = model.decode_stage_2_outputs(z)
        mode = "full"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        from monai.inferers import SlidingWindowInferer

        inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.4, mode="gaussian")

        def ae(t):
            with torch.autocast("cuda", dtype=torch.float16):
                return model.decode_stage_2_outputs(model.encode_stage_2_inputs(t))

        r = inferer(inp, ae)
        mode = "sliding_window"
    r = r.float()[0, 0, : shape[0], : shape[1], : shape[2]]
    return r, mode


@torch.no_grad()
def recon_hcld(model: torch.nn.Module | None, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    # HCLD training volumes: XYZ -> (Z, X, Y), trilinear resize to 192x192x64 (HarmonIt scripts/methods/hcld_prepare_abide.py).
    zxy = x.permute(2, 0, 1)[None, None]
    small = F.interpolate(zxy, size=HCLD_SHAPE_DHW, mode="trilinear", align_corners=False)
    if model is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_mu, _ = model.encode(small)
            small = model.decode(z_mu)
        small = small.float()
    back = F.interpolate(small, size=tuple(zxy.shape[-3:]), mode="trilinear", align_corners=False)
    return back[0, 0].permute(1, 2, 0).contiguous(), "full"


# ---------------------------------------------------------------- main

def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(data_dir / "abide_manifest.csv")
    site_ids = {s: i for i, s in enumerate(sorted(manifest["site"].unique()))}
    subjects = json.loads((data_dir / "splits.json").read_text())[args.split]
    rows = manifest.set_index("subject_id").loc[subjects].reset_index().sort_values(["site", "subject_id"])
    if args.limit:
        rows = rows.head(args.limit)
    figure_subjects = set(rows.groupby("site")["subject_id"].first())

    models = {}
    for name in args.models:
        if name == "maisi":
            models[name] = build_maisi(args.maisi_ckpt, num_splits=2, device=device)
        elif name == "nvmr":
            models[name] = build_maisi(args.nvmr_ckpt, num_splits=4, device=device)
        elif name == "hcld":
            models[name] = build_hcld(args.hcld_root, args.hcld_ckpt, device)
        elif name in ("hcld_resize", "blur05", "blur10"):
            models[name] = None
        else:
            raise ValueError(name)

    records = []
    slices: dict[str, dict[str, np.ndarray]] = {name: {} for name in args.models}
    slices["input"] = {}
    for i, row in enumerate(rows.itertuples(index=False)):
        x, zooms, native_shape, affine = load_reference(resolve_t1(row.t1_path, data_dir), device)
        save_nifti = args.save_nifti and row.subject_id in figure_subjects
        if save_nifti:
            (out_dir / "nifti" / "input").mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(x.cpu().numpy().astype(np.float32), affine),
                     out_dir / "nifti" / "input" / f"{row.subject_id}.nii.gz")
        fg = torch.nonzero(x > FG_THR)
        center = tuple(int(v) for v in ((fg.min(0).values + fg.max(0).values) // 2).tolist())
        if row.subject_id in figure_subjects:
            slices["input"][row.subject_id] = mid_slices(x, center)
        for name, model in models.items():
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if name in ("maisi", "nvmr"):
                r, mode = recon_maisi(model, x)
            elif name in ("hcld", "hcld_resize"):
                r, mode = recon_hcld(model, x)
            else:
                r, mode = blur3d(x, 0.5 if name == "blur05" else 1.0), "filter"
            torch.cuda.synchronize()
            seconds = time.time() - t0
            rec = {
                "subject_id": row.subject_id, "site": row.site, "site_id": site_ids[row.site], "model": name,
                "mode": mode, "seconds": round(seconds, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "native_shape": "x".join(map(str, native_shape)),
                "zooms": "x".join(f"{z:.3g}" for z in zooms),
                "shape_1mm": "x".join(map(str, x.shape)),
            }
            rec.update(metrics(x, r))
            records.append(rec)
            if row.subject_id in figure_subjects:
                slices[name][row.subject_id] = mid_slices(r, center)
            if save_nifti and name in ("maisi", "nvmr", "hcld", "hcld_resize"):
                (out_dir / "nifti" / name).mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(r.clamp(0, 1).cpu().numpy().astype(np.float32), affine),
                         out_dir / "nifti" / name / f"{row.subject_id}.nii.gz")
            del r
        print(f"[{i + 1}/{len(rows)}] {row.subject_id} {tuple(x.shape)} "
              + " ".join(f"{r['model']}={r['psnr_fg']:.2f}dB/{r['ssim_fg']:.3f}" for r in records[-len(models):]),
              flush=True)
        pd.DataFrame(records).to_csv(out_dir / "per_subject.csv", index=False)

    for name, d in slices.items():
        if d:
            np.savez_compressed(out_dir / f"slices_{name}.npz", **d)
    (out_dir / "run.json").write_text(json.dumps({
        "args": vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "n_subjects": int(len(rows)), "reference": "RAS, robust_normalize [p1,p99]->[0,1], trilinear to 1 mm",
        "foreground": f"reference > {FG_THR}",
    }, indent=2))


if __name__ == "__main__":
    main()
