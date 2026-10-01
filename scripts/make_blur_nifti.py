"""Blurred copies of the saved 1 mm input volumes, to calibrate segmentation agreement."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run = Path(args.run_dir)
    for f in sorted((run / "hcld" / "nifti" / "input").glob("*.nii.gz")):
        img = nib.load(str(f))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        for name, sigma in (("blur05", 0.5), ("blur10", 1.0)):
            out = run / "calib" / "nifti" / name / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(gaussian_filter(vol, sigma, mode="nearest"), img.affine), out)
        print(f.name, flush=True)


if __name__ == "__main__":
    main()
