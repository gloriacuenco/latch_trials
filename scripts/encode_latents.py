"""E0 step 3: encode every reference volume into a latent with a frozen pretrained VAE.

For each scan: place a fixed grid on the head (hdev.preprocess.place), crop/pad, encode with the
posterior mean, save the latent as float16 ``<out>/<subject>.npy`` [4, X/4, Y/4, Z/4], and record
the placement in ``<out>/index.csv`` so outputs can be put back on the scan's own grid.
``--check N`` also decodes N validation scans and reports PSNR on the head, as in the gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from hdev.preprocess import HEAD_THR, cut_fraction, from_grid, place, place_on_brain, to_grid
from hdev.vae import VAES, decode, encode, load_vae


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True, help="index.csv from make_reference_volumes.py")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--vae", required=True, choices=sorted(VAES))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--grid", nargs=3, type=int, required=True)
    p.add_argument("--top-margin", type=int, default=8, help="Head-box placement only.")
    p.add_argument("--brain-index", default=None, help="brain_index.csv: centre the grid on the FastSurfer brain box")
    p.add_argument("--qc", default=None, help="qc_exclusions.csv: these scans fall back to head-box placement")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--check", type=int, default=8, help="Decode this many validation scans as a check.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if any(g % 16 for g in args.grid):
        raise SystemExit("grid sizes must be multiples of 16")

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit("Use a new empty output directory; do not mix sampled and posterior-mean caches.")
    df = pd.read_csv(args.index)
    if args.limit:
        df = df.head(args.limit)
    brain = pd.read_csv(args.brain_index).set_index("subject_id") if args.brain_index else None
    excluded = set(pd.read_csv(args.qc)["subject_id"]) if args.qc else set()
    check_ids = set(df[(df["split"] == "val") & ~df["subject_id"].isin(excluded)]
                    .groupby("site")["subject_id"].first().head(args.check))
    model = load_vae(args.vae, args.ckpt, device)

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        t0 = time.time()
        x = np.asarray(nib.load(str(Path(args.ref_dir) / f"{r.subject_id}.nii.gz")).dataobj, dtype=np.float32)
        if brain is not None and r.subject_id not in excluded:
            b = brain.loc[r.subject_id]
            pl = place_on_brain(((b.brain_x0, b.brain_x1), (b.brain_y0, b.brain_y1), (b.brain_z0, b.brain_z1)),
                                tuple(args.grid))
            how = "brain"
        else:
            bbox = ((r.bbox_x0, r.bbox_x1), (r.bbox_y0, r.bbox_y1), (r.bbox_z0, r.bbox_z1))
            pl = place(bbox, tuple(args.grid), args.top_margin)
            how = "head"
        g = torch.from_numpy(to_grid(x, pl)).to(device)[None, None]
        z = encode(model, g)
        np.save(out_dir / f"{r.subject_id}.npy", z[0].cpu().numpy().astype(np.float16))
        rec = {"subject_id": r.subject_id, "site": r.site, "site_id": r.site_id, "split": r.split,
               "placement": how, "qc_exclude": r.subject_id in excluded,
               "offset_x": pl.offset[0], "offset_y": pl.offset[1], "offset_z": pl.offset[2],
               "head_cut_fraction": round(cut_fraction(x, pl), 5)}
        if r.subject_id in check_ids:
            y = from_grid(decode(model, z)[0, 0].cpu().numpy(), pl, x.shape, fill=x)
            m = x > HEAD_THR
            rec["check_psnr_head"] = -10 * math.log10(float(((y - x)[m] ** 2).mean()))
        rec["seconds"] = round(time.time() - t0, 2)
        rows.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(df) or "check_psnr_head" in rec:
            extra = f" check PSNR {rec['check_psnr_head']:.2f} dB" if "check_psnr_head" in rec else ""
            print(f"{i + 1}/{len(df)} {r.subject_id} {rec['seconds']} s{extra}", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)
    (out_dir / "run.json").write_text(json.dumps({
        "args": vars(args), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "encoding": "posterior_mean", "monai": __import__("monai").__version__,
        "latent_shape": [4] + [g // 4 for g in args.grid], "dtype": "float16",
    }, indent=2))


if __name__ == "__main__":
    main()
