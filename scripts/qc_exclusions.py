"""E0 QC: scans excluded from flow training and from segmentation-based metrics.

Rule, fixed on 2026-10-01 before any flow result: FastSurfer failed on the 1 mm reference volume,
meaning either the brain box is wider (left-right) than it is long (front-back), which flags a scan
whose header orientation is wrong, or the cortical volume is under 300,000 mm^3 (typical: about
600,000). These scans are still encoded and harmonized; they are only left out of training and of
FastSurfer-based evaluation.
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--brain-index", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = pd.read_csv(args.brain_index)
    reasons = []
    for r in d.itertuples(index=False):
        why = []
        if r.brain_extent_x > r.brain_extent_y:
            why.append("orientation")
        if r.vol_cortex < 300_000:
            why.append("segmentation")
        reasons.append("+".join(why))
    d["qc_exclude"] = reasons
    out = d.loc[d["qc_exclude"] != "", ["subject_id", "site", "split", "qc_exclude", "vol_cortex",
                                        "brain_extent_x", "brain_extent_y", "brain_extent_z"]]
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} excluded: {out['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
