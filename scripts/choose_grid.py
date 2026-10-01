"""E0 step 2: report head sizes in the 1 mm space and how candidate grids would fit them.

Placement (hdev.preprocess.place) centres the grid in x and y and keeps the top of the head,
so a grid shorter than the head in z cuts the neck first. Grid sizes must be multiples of 16.
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True, help="index.csv from make_reference_volumes.py")
    p.add_argument("--top-margin", type=int, default=8)
    p.add_argument("--brain-index", default=None, help="brain_index.csv: report brain boxes instead")
    p.add_argument("--qc", default=None, help="qc_exclusions.csv: leave these scans out of the brain report")
    args = p.parse_args()
    df = pd.read_csv(args.index)
    if args.brain_index:
        b = pd.read_csv(args.brain_index)
        if args.qc:
            b = b[~b["subject_id"].isin(pd.read_csv(args.qc)["subject_id"])]
        cols = ["brain_extent_x", "brain_extent_y", "brain_extent_z"]
        print(f"{len(b)} scans passing QC, brain extent in mm (FastSurfer)")
        print(b[cols].describe(percentiles=[0.5, 0.99]).round(1).to_string())
        print("\nsmallest margin around the brain on each side (mm), grid centred on the brain:")
        for grid in [(176, 208, 176), (192, 224, 192), (208, 240, 208)]:
            margins = [(g - b[c].max()) / 2 for g, c in zip(grid, cols)]
            lat = "x".join(str(g // 4) for g in grid)
            print(f"  grid {'x'.join(map(str, grid))} (latent 4x{lat}): " + ", ".join(f"{a} {m:.0f}" for a, m in zip("xyz", margins)))
        return

    print(f"{len(df)} scans, head extent in mm (RAS: x left-right, y posterior-anterior, z inferior-superior)")
    print(df[["extent_x", "extent_y", "extent_z"]].describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())
    print("\nlargest per site:")
    print(df.groupby("site")[["extent_x", "extent_y", "extent_z"]].max().to_string())

    print("\nscans whose head is wider than the grid (x, y) or taller than grid - top margin (z):")
    for gx, gy, gz in [(176, 208, 176), (192, 224, 176), (192, 224, 192), (192, 240, 208)]:
        over = {a: int((df[f"extent_{a}"] > g - (args.top_margin if a == "z" else 0)).sum())
                for a, g in zip("xyz", (gx, gy, gz))}
        lat = f"{gx // 4}x{gy // 4}x{gz // 4}"
        print(f"  grid {gx}x{gy}x{gz} (latent 4x{lat}): x {over['x']}, y {over['y']}, z {over['z']} (z: neck cut)")


if __name__ == "__main__":
    main()
