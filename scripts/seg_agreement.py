"""Segmentation agreement between FastSurfer labels of input volumes and of reconstructions.

All volumes share one 1 mm grid, so FastSurfer conforms them identically and labels
compare voxel by voxel. Agreement is consistency with the input's segmentation, not
accuracy against ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SEG = "aparc.DKTatlas+aseg.deep.mgz"
GROUPS = {
    "cerebral_wm": [2, 41],
    "cortex": "cortex",
    "lateral_ventricles": [4, 43],
    "thalamus": [10, 49],
    "caudate": [11, 50],
    "putamen": [12, 51],
    "pallidum": [13, 52],
    "hippocampus": [17, 53],
    "amygdala": [18, 54],
    "cerebellum_cortex": [8, 47],
}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def confusion(ref: np.ndarray, out: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Counts of (ref label i, out label j) over voxels where either is labelled; index 0 = other."""
    lut = np.zeros(int(max(ref.max(), out.max())) + 1, dtype=np.int64)
    lut[labels] = np.arange(1, len(labels) + 1)
    keep = (ref > 0) | (out > 0)
    k = len(labels) + 1
    return np.bincount(lut[ref[keep]] * k + lut[out[keep]], minlength=k * k).reshape(k, k)


def group_dice(cm: np.ndarray, idx: list[int]) -> float:
    inter = cm[np.ix_(idx, idx)].sum()
    s = cm[idx, :].sum() + cm[:, idx].sum()
    return float(2 * inter / s) if s else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    seg_dir = Path(args.seg_dir)
    rows = []
    for ref_path in sorted((seg_dir / "input").glob(f"*/{SEG}")):
        sid = ref_path.parent.name
        ref = np.asarray(nib.load(str(ref_path)).dataobj).astype(np.int64)
        labels = np.array([int(v) for v in np.unique(ref) if v != 0])
        for model_dir in sorted(d for d in seg_dir.iterdir() if d.is_dir() and d.name != "input"):
            path = model_dir / sid / SEG
            if not path.exists():
                continue
            out = np.asarray(nib.load(str(path)).dataobj).astype(np.int64)
            cm = confusion(ref, out, labels)
            diag = np.diag(cm)[1:]
            ref_n, out_n = cm.sum(1)[1:], cm.sum(0)[1:]
            per_label = 2 * diag / np.maximum(ref_n + out_n, 1)
            vol_change = np.abs(out_n - ref_n) / np.maximum(ref_n, 1)
            idx = {int(v): i + 1 for i, v in enumerate(labels)}
            rec = {"subject_id": sid, "site": sid.rsplit("_", 1)[0], "model": model_dir.name,
                   "dice_mean_all_labels": float(per_label.mean()),
                   "dice_cortical_parcels": float(per_label[labels >= 1000].mean()),
                   "abs_volume_change_median": float(np.median(vol_change)),
                   "brain_dice": dice(ref > 0, out > 0)}
            for name, ids in GROUPS.items():
                sel = [idx[v] for v in labels if v >= 1000] if ids == "cortex" else [idx[v] for v in ids if v in idx]
                rec[f"dice_{name}"] = group_dice(cm, sel)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    cols = [c for c in df.columns if c.startswith("dice") or c.startswith("abs") or c == "brain_dice"]
    print(df.groupby("model")[cols].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
