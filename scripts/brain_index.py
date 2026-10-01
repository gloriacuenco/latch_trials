"""E0 step 2: brain boxes (for placing the grid) and anatomy features (for anatomy-matched pairing).

For each scan, the FastSurfer segmentation (computed on the 1 mm reference volume, in FastSurfer's
conformed space) is mapped back onto the reference grid with nearest-neighbour resampling. The brain
box is the bounding box of all labels. Regional volumes (mm^3) are counted in the conformed space,
whose voxels are 1 mm^3. Age and sex come from the ABIDE I phenotype file when given.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

SEG = "aparc.DKTatlas+aseg.deep.mgz"
REGIONS = {
    "cortex": "cortex",
    "cerebral_wm": [2, 41],
    "lateral_ventricles": [4, 43],
    "cerebellum_cortex": [8, 47],
    "cerebellum_wm": [7, 46],
    "thalamus": [10, 49],
    "caudate": [11, 50],
    "putamen": [12, 51],
    "pallidum": [13, 52],
    "hippocampus": [17, 53],
    "amygdala": [18, 54],
    "brainstem": [16],
}


def process(job: tuple[str, str, str]) -> dict:
    sid, ref_dir, seg_dir = job
    ref = nib.load(str(Path(ref_dir) / f"{sid}.nii.gz"))
    seg = nib.load(str(Path(seg_dir) / sid / SEG))
    lab = np.asarray(seg.dataobj).astype(np.int32)
    on_ref = np.asarray(resample_from_to(seg, (ref.shape, ref.affine), order=0).dataobj) > 0
    idx = np.nonzero(on_ref)
    rec = {"subject_id": sid}
    for a, i in zip("xyz", idx):
        rec[f"brain_{a}0"], rec[f"brain_{a}1"] = int(i.min()), int(i.max()) + 1
        rec[f"brain_extent_{a}"] = rec[f"brain_{a}1"] - rec[f"brain_{a}0"]
    rec["vol_brain"] = int((lab > 0).sum())
    for name, ids in REGIONS.items():
        rec[f"vol_{name}"] = int((lab >= 1000).sum()) if ids == "cortex" else int(np.isin(lab, ids).sum())
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True, help="index.csv from make_reference_volumes.py")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--seg-dir", required=True, help="FastSurfer outputs, one folder per subject")
    p.add_argument("--phenotypes", default=None, help="ABIDE I Phenotypic_V1_0b.csv (AGE_AT_SCAN, SEX, DX_GROUP)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    index = pd.read_csv(args.index)
    jobs = [(sid, args.ref_dir, args.seg_dir) for sid in index["subject_id"]]
    with Pool(args.workers) as pool:
        recs = list(pool.imap(process, jobs, chunksize=4))
    df = index[["subject_id", "site", "site_id", "split"]].merge(pd.DataFrame(recs), on="subject_id")
    if args.phenotypes:
        # ABIDE I SUB_IDs are unique across sites; our subject IDs end with them (site names differ).
        ph = pd.read_csv(args.phenotypes).rename(columns={"SUB_ID": "sub_id", "AGE_AT_SCAN": "age", "SEX": "sex",
                                                          "DX_GROUP": "dx"})[["sub_id", "age", "sex", "dx"]]
        df["sub_id"] = df["subject_id"].str.rsplit("_", n=1).str[1].astype(int)
        df = df.merge(ph, on="sub_id", how="left").drop(columns="sub_id")
        print(f"phenotypes matched for {df['age'].notna().sum()}/{len(df)} subjects")
    df.to_csv(args.out, index=False)
    print(df[["brain_extent_x", "brain_extent_y", "brain_extent_z"]].describe(percentiles=[0.5, 0.99]).round(1).to_string())


if __name__ == "__main__":
    main()
