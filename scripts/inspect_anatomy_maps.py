"""Walk through how one scan's anatomy maps are made, stage by stage, and save every stage to look at.

The same calls as scripts/anatomy_labels.py, split into cells (# %%) so each stage can be run, stepped
through in the debugger, and inspected. Every stage is saved as NIfTI with a correct affine, so the
volumes line up when opened together in a viewer (e.g. the NiiVue extension in VS Code):

  1_t1_1mm.nii.gz           the reference T1 the latents come from (1 mm, RAS)
  2_fastsurfer_seg.nii.gz   FastSurfer's labels, resampled onto that 1 mm volume (~95 labels)
  3_tissue_1mm.nii.gz       the labels grouped into 10 tissue classes (0 = background)
  4_t1_grid.nii.gz          the T1 cropped to the 192 x 224 x 192 brain-centred grid
  5_tissue_grid.nii.gz      the tissue classes on the same grid
  6_fractions_4mm.nii.gz    the anatomy maps: 10 channels at 48 x 56 x 48, the share of each class
                            in every 4 x 4 x 4-voxel block (4D: one volume per class)
  overview.png              one axial slice through every stage

Usage (on vpulab, from the repo root):
  PYTHONPATH=. /home/mmi/envs/hdev/bin/python scripts/inspect_anatomy_maps.py --subject KKI_50782
"""

# %% setup
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

from hdev.anatomy import CLASSES, fraction_maps, group_labels
from hdev.preprocess import place_on_brain, to_grid

D = Path("/mnt/rhome/mmi/projects/harmonit-data")
SEG = "aparc.DKTatlas+aseg.deep.mgz"
GRID = (192, 224, 192)

p = argparse.ArgumentParser()
p.add_argument("--subject", default="KKI_50782")
p.add_argument("--out-dir", default=str(D / "inspect"))
args, _ = p.parse_known_args()
sid = args.subject
out = Path(args.out_dir) / sid
out.mkdir(parents=True, exist_ok=True)


def save(name: str, x: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(x, affine), out / name)
    print(f"saved {name:26s} shape {x.shape}  dtype {x.dtype}")


# %% stage 1: the reference T1 (1 mm, RAS, intensities scaled to [0, 1])
ref = nib.load(str(D / "reference_1mm" / f"{sid}.nii.gz"))
t1 = np.asarray(ref.dataobj, dtype=np.float32)
print("T1:", t1.shape, "voxel size", ref.header.get_zooms()[:3], "range", t1.min(), t1.max())
save("1_t1_1mm.nii.gz", t1, ref.affine)

# %% stage 2: FastSurfer's segmentation, one label per voxel
# FastSurfer works in its own 256^3 1 mm space; nearest-neighbour resampling (order=0) puts its labels
# onto the reference volume without mixing label numbers.
seg_img = nib.load(str(D / "fastsurfer" / "reference_1mm" / sid / SEG))
print("FastSurfer output:", seg_img.shape, "(its own space)")
seg = np.asarray(resample_from_to(seg_img, (ref.shape, ref.affine), order=0).dataobj).astype(np.int32)
labels, counts = np.unique(seg, return_counts=True)
print(f"on the T1 grid: {seg.shape}, {len(labels)} distinct labels, e.g. {labels[:12].tolist()} ...")
save("2_fastsurfer_seg.nii.gz", seg, ref.affine)

# %% stage 3: group ~95 labels into 10 tissue classes (hdev/anatomy.py, _GROUPS)
tissue = group_labels(seg)
for k, name in enumerate(CLASSES):
    print(f"  class {k} {name:22s} {int((tissue == k).sum()):>9d} voxels")
save("3_tissue_1mm.nii.gz", tissue, ref.affine)

# %% stage 4: crop to the fixed 192 x 224 x 192 grid, centred on the FastSurfer brain box
row = pd.read_csv("results/e0_data/brain_index.csv").set_index("subject_id").loc[sid]
bbox = ((row.brain_x0, row.brain_x1), (row.brain_y0, row.brain_y1), (row.brain_z0, row.brain_z1))
placement = place_on_brain(bbox, GRID)
print("brain box", bbox, "-> grid offset", placement.offset, "(grid voxel 0 = T1 voxel offset)")
grid_affine = ref.affine.copy()
grid_affine[:3, 3] = ref.affine[:3, :3] @ np.array(placement.offset) + ref.affine[:3, 3]
t1_grid, tissue_grid = to_grid(t1, placement), to_grid(tissue, placement)
save("4_t1_grid.nii.gz", t1_grid, grid_affine)
save("5_tissue_grid.nii.gz", tissue_grid, grid_affine)

# %% stage 5: the anatomy maps -- share of each class in every 4 x 4 x 4-voxel block
fractions = fraction_maps(tissue_grid)  # [10, 48, 56, 48]
print("anatomy maps:", fractions.shape, "| each block's shares sum to", np.unique(fractions.sum(0).round(4)))
# block (I, J, K) covers grid voxels 4I..4I+3, so its centre is grid voxel 4I + 1.5
block_affine = grid_affine.copy()
block_affine[:3, :3] = grid_affine[:3, :3] * 4
block_affine[:3, 3] = grid_affine[:3, :3] @ np.full(3, 1.5) + grid_affine[:3, 3]
save("6_fractions_4mm.nii.gz", np.moveaxis(fractions, 0, -1).astype(np.float32), block_affine)

# %% check: identical to what training used (stored as uint8, share x 255)
stored = np.load(D / "anatomy_labels" / f"{sid}.npy")
print("stored:", stored.shape, "| max difference:", np.abs(stored / 255 - fractions).max().round(4), "(rounding <= 0.002)")

# %% overview: one axial slice through every stage
z_grid = 96
z_t1 = z_grid + placement.offset[2]
panels = [("1 T1, 1 mm", np.rot90(t1[:, :, z_t1]), "gray"), ("2 FastSurfer labels", np.rot90(seg[:, :, z_t1] % 40), "tab20"),
          ("3 10 tissues", np.rot90(tissue[:, :, z_t1]), "tab10"), ("4 T1 on grid", np.rot90(t1_grid[:, :, z_grid]), "gray"),
          ("5 tissues on grid", np.rot90(tissue_grid[:, :, z_grid]), "tab10"),
          ("6 cortex share, 4 mm", np.rot90(fractions[CLASSES.index("cortex"), :, :, z_grid // 4]), "viridis"),
          ("6 white matter share", np.rot90(fractions[CLASSES.index("cerebral_wm"), :, :, z_grid // 4]), "viridis"),
          ("6 ventricle share", np.rot90(fractions[CLASSES.index("lateral_ventricles"), :, :, z_grid // 4]), "viridis")]
fig, ax = plt.subplots(2, 4, figsize=(16, 8.5))
for a, (title, img, cmap) in zip(ax.ravel(), panels):
    a.imshow(img, cmap=cmap, interpolation="nearest")
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
plt.savefig(out / "overview.png", dpi=110)
print("saved overview.png in", out)
