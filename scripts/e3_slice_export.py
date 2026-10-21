"""E3 outputs on the ISBI 2027 audit's frozen slices, for image-space site probes (train and val only).

Every training and validation scan is regenerated as NYU at one setting (same noise as
scripts/e3_eval_regen.py, so the validation outputs are the ones already evaluated), decoded, put
back on the scan's 1 mm grid (outside the grid the input is kept) and resampled to its native grid.
Each native volume is then cut at the audit's frozen slice with the raw slice's head mask, crop and
256 x 256 resize, exactly as the audit exported 3D methods (HarmonIt scripts/methods/hcld_export_ldm.py,
fixed_slice_from_volume). Gate: the recomputed raw slice must equal the audit export's raw slice.

Two exports per split, in the audit's NPZ format (scripts/train_slice_probe.py reads them):
``whole/<split>/latch_slices.npz`` (head) and ``brain/<split>/latch_slices.npz`` (images and raw images
multiplied by the HD-BET brain mask, as the audit's brain-only probes, protocol amendment 7).
Scans LATCH cannot process (QC exclusions: no anatomy maps) pass through unchanged and are listed in
``passthrough``. The regenerated latents are saved too (``latents.npz``) for the age/sex check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from harmonit.data.abide_slices_dataset import (bbox_from_mask, crop_with_bbox, make_head_mask, resize_mask_to_hw,
                                                resize_to_hw, robust_normalize)

from hdev.flow import LatentStats
from hdev.latents import load_latents
from hdev.preprocess import Placement, from_grid, to_native
from hdev.regen import RegenNet, regenerate
from hdev.vae import decode, load_vae

TARGET = "NYU"
HW = (256, 256)


def raw_volume(manifest: pd.DataFrame, data_root: Path, sid: str) -> np.ndarray:
    """The audit's raw volume: canonical orientation, robust [0, 1] scaling (AbideSlicesDataset._load_volume)."""
    path = Path(manifest.at[sid, "t1_path"])
    if not path.is_absolute():
        path = data_root.parent / path
    return robust_normalize(nib.as_closest_canonical(nib.load(str(path))).get_fdata(dtype=np.float32))


def cut(raw: np.ndarray, volume: np.ndarray, k: int, brain: np.ndarray) -> tuple[np.ndarray, ...]:
    """Raw slice, output slice (head-masked, cropped, resized) and the brain mask on the same 256 x 256 grid."""
    head_full = make_head_mask(raw[:, :, k], thr=0.02, dilate_iters=3)
    bbox = bbox_from_mask(head_full, margin=20)
    head = crop_with_bbox(head_full.astype(np.uint8), bbox).astype(bool)
    out = []
    for v in (raw, volume):
        s = np.array(crop_with_bbox(v[:, :, k], bbox), dtype=np.float32)
        s[~head] = 0.0
        out.append(resize_to_hw(s, HW).astype(np.float32))
    mask = resize_mask_to_hw(crop_with_bbox(brain[:, :, k], bbox) & head, HW)
    return out[0], np.clip(out[1], 0, 1), mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--audit-exports", required=True, help="audit whole-head exports: <dir>/<split>/*_slices.npz")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mask-dir", required=True, help="HD-BET masks, <subject>.nii.gz")
    p.add_argument("--keep", type=float, default=0.0)
    p.add_argument("--fine-scale", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- regenerate every training and validation latent (same noise per row as e3_eval_regen.py)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    n_sites = int(meta["site_id"].max()) + 1
    net = RegenNet(n_sites, channels=tuple(ck["config"]["channels"])).to(device).eval()
    net.load_state_dict(ck["ema"])
    stats = LatentStats(ck["stats"]["mean"], ck["stats"]["std"])
    maps = np.stack([np.load(Path(args.labels_dir) / f"{s}.npy") for s in meta["subject_id"]])
    nyu = int(meta.loc[meta["site"] == TARGET, "site_id"].iloc[0])
    zh = np.empty(z.shape, dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(z), 8):
            idx = np.arange(s, min(s + 8, len(z)))
            z0 = stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))
            noise = torch.stack([torch.randn(z0.shape[1:], generator=torch.Generator().manual_seed(args.seed * 100_000 + int(i)))
                                 for i in idx]).to(device)
            a = torch.from_numpy(maps[idx].astype(np.float32) / 255).to(device)
            zr = regenerate(net, z0, noise, torch.full((len(idx),), nyu, device=device), a, args.keep,
                            args.ode_steps, args.guidance, args.fine_scale).float()
            zh[idx] = stats.denormalize(zr).cpu().numpy()
    np.savez_compressed(out / "latents.npz", latents=zh.astype(np.float16), subject_ids=meta["subject_id"].to_numpy(str),
                        site_ids=meta["site_id"].to_numpy(), splits=meta["split"].to_numpy(str))
    print(f"regenerated {len(zh)} latents", flush=True)

    # ---- decode, back to native, cut at the frozen slices
    row = {sid: i for i, sid in enumerate(meta["subject_id"])}
    lat_index = pd.read_csv(Path(args.latent_dir) / "index.csv").set_index("subject_id")
    manifest = pd.read_csv(args.manifest).set_index("subject_id")
    data_root = Path(args.manifest).resolve().parent
    vae = load_vae(args.vae, args.vae_ckpt, device)
    report = {"config": vars(args)}
    for split in ("val", "train"):
        src = next((Path(args.audit_exports) / split).glob("*_slices.npz"))
        with np.load(src, allow_pickle=False) as d:
            ref = {k: d[k] for k in ("raw_images", "subject_ids", "site_ids", "slice_indices")}
        rows = {"images": [], "raw_images": [], "masks": [], "passthrough": []}
        max_diff = 0.0
        for n, (sid, k) in enumerate(zip(ref["subject_ids"].astype(str), ref["slice_indices"].astype(int))):
            raw = raw_volume(manifest, data_root, sid)
            brain = np.asarray(nib.as_closest_canonical(nib.load(str(Path(args.mask_dir) / f"{sid}.nii.gz"))).dataobj) > 0.5
            if sid in row:
                img = nib.load(str(Path(args.ref_dir) / f"{sid}.nii.gz"))
                x = np.asarray(img.dataobj, dtype=np.float32)
                li = lat_index.loc[sid]
                pl = Placement((zh.shape[2] * 4, zh.shape[3] * 4, zh.shape[4] * 4), (li.offset_x, li.offset_y, li.offset_z))
                g = decode(vae, torch.from_numpy(zh[row[sid]]).to(device)[None])[0, 0].clamp(0, 1).cpu().numpy()
                volume = to_native(from_grid(g, pl, x.shape, fill=x), raw.shape)
            else:
                volume = raw
            r, o, m = cut(raw, volume, k, brain)
            max_diff = max(max_diff, float(np.abs(r - ref["raw_images"][n, 0]).max()))
            rows["images"].append(o[None])
            rows["raw_images"].append(r[None])
            rows["masks"].append(m[None])
            rows["passthrough"].append(sid not in row)
            if (n + 1) % 100 == 0:
                print(f"{split} {n + 1}/{len(ref['subject_ids'])} (max raw diff {max_diff:.1e})", flush=True)
        if max_diff > 1e-5:
            raise SystemExit(f"{split}: recomputed raw slices differ from the audit export (max {max_diff:.2e})")
        images, raws = np.stack(rows["images"]), np.stack(rows["raw_images"])
        masks = np.stack(rows["masks"]).astype(np.float32)
        common = dict(subject_ids=ref["subject_ids"], site_ids=ref["site_ids"], slice_indices=ref["slice_indices"],
                      split=np.asarray(split), method=np.asarray("latch"), passthrough=np.asarray(rows["passthrough"]))
        for view, im, rw in (("whole", images, raws), ("brain", images * masks, raws * masks)):
            (out / view / split).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / view / split / "latch_slices.npz", images=im, raw_images=rw, **common)
        report[split] = {"n": len(images), "passthrough": int(np.sum(rows["passthrough"])), "max_raw_abs_diff": max_diff}
        print(split, json.dumps(report[split]), flush=True)
    (out / "export_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
