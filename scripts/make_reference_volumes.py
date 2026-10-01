"""E0 step 1: write every ABIDE I scan in the 1 mm reference space and index it.

Output: one float32 NIfTI per subject (``<out>/<subject_id>.nii.gz``) and ``<out>/index.csv`` with
site, split, native geometry and the head bounding box, which E0 step 2 uses to choose the grid.
Existing NIfTIs are kept; their index rows are recomputed.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from hdev.preprocess import HEAD_THR, head_bbox, load_reference


def resolve_t1(path_value: str, data_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    parts = path.parts
    return data_dir / Path(*parts[parts.index("ABIDE"):])


def _one_thread() -> None:
    torch.set_num_threads(1)  # parallelism comes from the worker processes


def process(job: tuple[dict, str, str]) -> dict:
    row, data_dir, out_dir = job
    out = Path(out_dir) / f"{row['subject_id']}.nii.gz"
    t1 = resolve_t1(row["t1_path"], Path(data_dir))
    if out.exists():
        x = np.asarray(nib.load(str(out)).dataobj, dtype=np.float32)
        raw = nib.as_closest_canonical(nib.load(str(t1)))
        native_shape, zooms = tuple(raw.shape[:3]), tuple(float(z) for z in raw.header.get_zooms()[:3])
    else:
        ref = load_reference(t1)
        x, native_shape, zooms = ref.volume, ref.native_shape, ref.zooms
        nib.save(nib.Nifti1Image(x, ref.affine), out)
    bb = head_bbox(x)
    return {
        "subject_id": row["subject_id"], "site": row["site"], "site_id": row["site_id"], "split": row["split"],
        "native_shape": "x".join(map(str, native_shape)),
        "zooms": "x".join(f"{z:.4g}" for z in zooms),
        "shape_1mm": "x".join(map(str, x.shape)),
        **{f"bbox_{a}{e}": v for a, (s, t) in zip("xyz", bb) for e, v in (("0", s), ("1", t))},
        **{f"extent_{a}": t - s for a, (s, t) in zip("xyz", bb)},
        "head_voxels": int((x > HEAD_THR).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="Directory with abide_manifest.csv, splits.json and ABIDE/.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(data_dir / "abide_manifest.csv")
    site_ids = {s: i for i, s in enumerate(sorted(manifest["site"].unique()))}
    split_of = {sid: name for name, ids in json.loads((data_dir / "splits.json").read_text()).items()
                if isinstance(ids, list) for sid in ids}
    manifest["site_id"] = manifest["site"].map(site_ids)
    manifest["split"] = manifest["subject_id"].map(split_of)
    rows = manifest.sort_values(["site", "subject_id"]).to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    with Pool(args.workers, initializer=_one_thread) as pool:
        records = []
        for i, rec in enumerate(pool.imap(process, [(r, str(data_dir), str(out_dir)) for r in rows], chunksize=2)):
            records.append(rec)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"{i + 1}/{len(rows)} {rec['subject_id']} {rec['shape_1mm']}", flush=True)
    pd.DataFrame(records).to_csv(out_dir / "index.csv", index=False)


if __name__ == "__main__":
    main()
