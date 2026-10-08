"""Small validation-only comparison of posterior-mean and sampled VAE reconstructions.

Reuse the existing latent index for crop offsets and array shapes, but do not use
or modify its latent values. Select one QC-passing validation subject per site.
This is a diagnostic, not a new harmonization result or a representative benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from hdev.metrics import FG_THR, psnr_fg
from hdev.preprocess import Placement, from_grid, to_grid
from hdev.vae import VAES, decode, encode, load_vae


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_subjects(meta, limit):
    if limit < 1 or meta.subject_id.duplicated().any():
        raise ValueError("Need positive subject count and unique subject IDs")
    meta = meta[meta.split == "val"].copy()
    if "qc_exclude" in meta:
        flag = meta.qc_exclude.astype(str).str.lower()
        if not flag.isin(["true", "false", "0", "1"]).all():
            raise ValueError("Missing or invalid QC flags")
        meta = meta[flag.isin(["false", "0"])]
    if meta.empty:
        raise ValueError("No QC-passing validation subjects")
    return meta.sort_values(["site", "subject_id"]).drop_duplicates("site").head(limit)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True, help="Existing cache with index.csv and .npy shapes; read only")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--vae", choices=sorted(VAES), default="nvmr")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subjects", type=int, default=3)
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.draws < 2:
        p.error("Use at least two sampling draws")
    if not torch.cuda.is_available():
        raise SystemExit("This pretrained 3D VAE experiment requires a CUDA GPU")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("Use a new empty output directory")
    index = Path(args.latent_dir) / "index.csv"
    selected = select_subjects(pd.read_csv(index), args.subjects)
    selected.to_csv(out / "selected_subjects.csv", index=False)
    device = torch.device("cuda")
    model = load_vae(args.vae, args.ckpt, device)
    records = []
    with torch.no_grad():
        for row in selected.itertuples(index=False):
            shape = np.load(Path(args.latent_dir) / f"{row.subject_id}.npy", mmap_mode="r").shape
            if len(shape) != 4 or shape[0] != 4:
                raise ValueError(f"Unexpected latent shape: {shape}")
            grid = tuple(int(v * 4) for v in shape[1:])
            pl = Placement(grid, tuple(int(v) for v in (row.offset_x, row.offset_y, row.offset_z)))
            img = nib.load(str(Path(args.ref_dir) / f"{row.subject_id}.nii.gz"))
            x = np.asarray(img.dataobj, dtype=np.float32)
            g = torch.from_numpy(to_grid(x, pl)).to(device)[None, None]
            mask = (x > FG_THR) & (from_grid(np.ones(grid, dtype=np.uint8), pl, x.shape) > 0)
            if not mask.any():
                raise ValueError("Empty processed foreground")
            torch.manual_seed(args.seed)
            z_mean = encode(model, g)
            torch.manual_seed(args.seed + 1)
            z_again = encode(model, g)
            repeat_error = float((z_again - z_mean).abs().max())
            del z_again
            with torch.autocast("cuda", dtype=torch.float16):
                mu, sigma = model.encode(g)
            mean_grid = decode(model, z_mean)[0, 0].cpu().numpy()
            mean_vol = from_grid(mean_grid, pl, x.shape, fill=x)
            for draw in [-1] + list(range(args.draws)):
                seed = args.seed + 100 + draw
                if draw == -1:
                    z, v = z_mean, mean_vol
                else:
                    torch.manual_seed(seed)
                    z = model.sampling(mu, sigma).float()
                    v = from_grid(decode(model, z)[0, 0].cpu().numpy(), pl, x.shape, fill=x)
                if not np.isfinite(v).all():
                    raise ValueError("Nonfinite reconstruction")
                records.append({
                    "subject_id": row.subject_id, "site": row.site,
                    "mode": "mean" if draw == -1 else "sample", "draw": draw,
                    "seed": None if draw == -1 else seed,
                    "mean_repeat_max_abs": repeat_error,
                    "latent_rmse_to_mean": float(((z - z_mean) ** 2).mean().sqrt()),
                    "psnr_head_unclipped": psnr_fg(x, v),
                    "psnr_head_clipped": psnr_fg(x, np.clip(v, 0, 1)),
                    "mae_processed_fg": float(np.abs(v[mask] - x[mask]).mean()),
                    "mae_to_mean_processed_fg": float(np.abs(v[mask] - mean_vol[mask]).mean()),
                    "out_of_range_processed_fg": float(((v[mask] < 0) | (v[mask] > 1)).mean()),
                })
            pd.DataFrame(records).to_csv(out / "per_subject.csv", index=False)
            print(f"{row.subject_id}: mean encoding repeat max error {repeat_error:.3g}", flush=True)
            del g, z_mean, mu, sigma, z
    import monai
    (out / "run.json").write_text(json.dumps({
        "args": vars(args), "subjects": selected.subject_id.tolist(),
        "torch": torch.__version__, "monai": monai.__version__, "gpu": torch.cuda.get_device_name(),
        "checkpoint_sha256": file_hash(args.ckpt), "index_sha256": file_hash(index),
        "selection": "first QC-passing validation subject per site, sites alphabetically ordered",
        "note": "No latent cache values used; no train/test subjects; no cache overwritten.",
    }, indent=2))
    print(pd.DataFrame(records).groupby("mode")[["mean_repeat_max_abs", "latent_rmse_to_mean",
          "psnr_head_unclipped", "mae_processed_fg", "mae_to_mean_processed_fg"]].mean().to_string())


if __name__ == "__main__":
    main()
