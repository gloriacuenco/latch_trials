"""Anatomy agreement by AssemblyNet (volBrain team; Coupé et al., NeuroImage 2020), independent of FastSurfer.

E3 conditions its generator on FastSurfer tissue maps, so FastSurfer agreement is partly circular.
AssemblyNet is a different segmenter (133 BrainColor structures, plus tissues), never used by LATCH.
Every volume of a subject is on that subject's 1 mm grid, and AssemblyNet returns native-space labels,
so input and output segmentations compare voxel by voxel.

Layout: ``<seg-dir>/<set>/native_structures_<subject>.nii.gz`` (and ``native_tissues_...``), with
``<set>`` = ``input`` for the original scans and any other name for a harmonized set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def find(set_dir: Path, kind: str, sid: str) -> Path | None:
    # The README spells the structures file "strctures"; accept either spelling.
    hits = sorted(set_dir.rglob(f"native_{kind}_{sid}.nii*")) or sorted(set_dir.rglob(f"native_strctures_{sid}.nii*")
                                                                         if kind == "structures" else [])
    return hits[0] if hits else None


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj).astype(np.int64)


def dice_per_label(ref: np.ndarray, out: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dice and relative volume change per label (labels present in the reference)."""
    k = int(max(ref.max(), out.max())) + 1
    keep = (ref > 0) | (out > 0)
    cm = np.bincount(ref[keep] * k + out[keep], minlength=k * k).reshape(k, k)
    inter, ref_n, out_n = np.diag(cm)[labels], cm.sum(1)[labels], cm.sum(0)[labels]
    return 2 * inter / np.maximum(ref_n + out_n, 1), np.abs(out_n - ref_n) / np.maximum(ref_n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    seg_dir = Path(args.seg_dir)
    sids = sorted({f.name.split("_", 2)[2].split(".nii")[0] for f in (seg_dir / "input").rglob("native_tissues_*.nii*")})
    rows = []
    for sid in sids:
        ref_s, ref_t = find(seg_dir / "input", "structures", sid), find(seg_dir / "input", "tissues", sid)
        if ref_s is None or ref_t is None:
            continue
        ref_s, ref_t = load(ref_s), load(ref_t)
        labels = np.array([v for v in np.unique(ref_s) if v != 0])
        for set_dir in sorted(d for d in seg_dir.iterdir() if d.is_dir() and d.name != "input"):
            out_s, out_t = find(set_dir, "structures", sid), find(set_dir, "tissues", sid)
            if out_s is None or out_t is None:
                continue
            d, vc = dice_per_label(ref_s, load(out_s), labels)
            td, _ = dice_per_label(ref_t, load(out_t), np.array([v for v in np.unique(ref_t) if v != 0]))
            rows.append({"subject_id": sid, "site": sid.rsplit("_", 1)[0], "set": set_dir.name,
                         "dice_structures_mean": float(d.mean()), "abs_volume_change_median": float(np.median(vc)),
                         "dice_tissues_mean": float(td.mean()), **{f"dice_tissue_{i + 1}": float(x) for i, x in enumerate(td)}})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{df['subject_id'].nunique()} subjects")
    print(df.groupby("set").median(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    main()
