"""How much do two segmenters disagree on the same scan, compared with how much LATCH changes anatomy?

FastSurfer and AssemblyNet segment the same 92 validation input scans. Both are mapped to six shared
tissues (cortex, cerebral white matter, deep grey incl. hippocampus/amygdala, cerebellar cortex,
cerebellar white matter, brainstem; CSF is left out because AssemblyNet's CSF includes the CSF
outside the brain). Reported per tissue and on average:

- between segmenters: FastSurfer(input) vs AssemblyNet(input), the noise floor of "anatomy agreement";
- within AssemblyNet: AssemblyNet(output) vs AssemblyNet(input) for each harmonized set.

If LATCH's within-segmenter agreement is close to the between-segmenter agreement, its anatomy
changes are about as large as the disagreement between two good segmenters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

SEG = "aparc.DKTatlas+aseg.deep.mgz"
TISSUES = ["cortex", "cerebral_wm", "deep_gray", "cerebellum_cortex", "cerebellum_wm", "brainstem"]
FS_GROUPS = {  # FreeSurfer label ids -> tissue index (1-based, as in TISSUES)
    1: [3, 42], 2: [2, 41, 77, 251, 252, 253, 254, 255],
    3: [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60],
    4: [8, 47], 5: [7, 46], 6: [16],
}
AN_TISSUES = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}  # AssemblyNet tissue label -> tissue index; 1 (CSF) dropped


def fastsurfer_tissues(seg: np.ndarray) -> np.ndarray:
    out = np.zeros(seg.shape, np.uint8)
    for k, ids in FS_GROUPS.items():
        out[np.isin(seg, ids)] = k
    out[seg >= 1000] = 1
    return out


def assemblynet_tissues(t: np.ndarray) -> np.ndarray:
    out = np.zeros(t.shape, np.uint8)
    for a, k in AN_TISSUES.items():
        out[t == a] = k
    return out


def dice(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [2 * np.sum((a == k) & (b == k)) / max(np.sum(a == k) + np.sum(b == k), 1) for k in range(1, len(TISSUES) + 1)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref-dir", required=True, help="reference_1mm volumes (the AssemblyNet inputs)")
    p.add_argument("--fs-dir", required=True, help="FastSurfer outputs of the inputs: <fs-dir>/<subject>/" + SEG)
    p.add_argument("--an-dir", required=True, help="AssemblyNet outputs: <an-dir>/<set>/native_tissues_<subject>.nii.gz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    an_dir = Path(args.an_dir)
    sets = sorted(d.name for d in an_dir.iterdir() if d.is_dir() and d.name != "input")
    rows = []
    for f in sorted((an_dir / "input").glob("native_tissues_*.nii.gz")):
        sid = f.name[len("native_tissues_"):-len(".nii.gz")]
        ref = nib.load(str(Path(args.ref_dir) / f"{sid}.nii.gz"))
        an_in = assemblynet_tissues(np.asarray(nib.load(str(f)).dataobj).astype(int))
        fs = resample_from_to(nib.load(str(Path(args.fs_dir) / sid / SEG)), (ref.shape, ref.affine), order=0)
        fs_in = fastsurfer_tissues(np.asarray(fs.dataobj).astype(int))
        rows.append({"subject_id": sid, "comparison": "FastSurfer vs AssemblyNet (input)",
                     **dict(zip(TISSUES, dice(fs_in, an_in)))})
        for s in sets:
            g = an_dir / s / f.name
            if g.exists():
                an_out = assemblynet_tissues(np.asarray(nib.load(str(g)).dataobj).astype(int))
                rows.append({"subject_id": sid, "comparison": f"AssemblyNet: {s} vs input",
                             **dict(zip(TISSUES, dice(an_out, an_in)))})
    df = pd.DataFrame(rows)
    df["mean"] = df[TISSUES].mean(1)
    df.to_csv(args.out, index=False)
    print(f"{df['subject_id'].nunique()} subjects")
    print(df.groupby("comparison")[TISSUES + ["mean"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
