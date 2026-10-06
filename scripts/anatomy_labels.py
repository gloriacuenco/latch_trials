"""Anatomy targets at latent resolution, one per scan passing QC.

FastSurfer labels (computed on the 1 mm reference volume) are mapped back onto the reference grid,
grouped into tissue classes (hdev/anatomy.py), cropped to the same brain-centred grid as the latents,
and summarized per 4 x 4 x 4 mm block as class fractions. Saved as uint8 (fraction x 255)
``<out>/<subject>.npy`` of shape [K, X/4, Y/4, Z/4].
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to

from hdev.anatomy import fraction_maps, group_labels
from hdev.preprocess import place_on_brain, to_grid

SEG = "aparc.DKTatlas+aseg.deep.mgz"


def _one_thread() -> None:
    torch.set_num_threads(1)


def process(job: tuple) -> str:
    sid, bbox, grid, ref_dir, seg_dir, out_dir = job
    out = Path(out_dir) / f"{sid}.npy"
    if out.exists():
        return "skip"
    ref = nib.load(str(Path(ref_dir) / f"{sid}.nii.gz"))
    seg = nib.load(str(Path(seg_dir) / sid / SEG))
    labels = np.asarray(resample_from_to(seg, (ref.shape, ref.affine), order=0).dataobj).astype(np.int32)
    classes = to_grid(group_labels(labels), place_on_brain(bbox, grid))
    np.save(out, np.round(fraction_maps(classes) * 255).astype(np.uint8))
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--brain-index", required=True)
    p.add_argument("--qc", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--grid", nargs=3, type=int, default=[192, 224, 192])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(args.brain_index)
    b = b[~b["subject_id"].isin(pd.read_csv(args.qc)["subject_id"])]
    jobs = [(r.subject_id, ((r.brain_x0, r.brain_x1), (r.brain_y0, r.brain_y1), (r.brain_z0, r.brain_z1)),
             tuple(args.grid), args.ref_dir, args.seg_dir, args.out_dir) for r in b.itertuples(index=False)]
    with Pool(args.workers, initializer=_one_thread) as pool:
        for i, status in enumerate(pool.imap(process, jobs, chunksize=4)):
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} {status}", flush=True)


if __name__ == "__main__":
    main()
