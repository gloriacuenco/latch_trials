"""Minimal, reversible preprocessing shared by every HarmonIt experiment.

Reference space, as in the ISBI 2027 audit and the autoencoder gate: the raw T1 reoriented to
RAS, robust-normalized to [0, 1] (clipped to its 1st-99th percentiles) and resampled to 1 mm
isotropic with trilinear interpolation. A fixed-size crop/pad around the head then gives every
scan the same grid for the autoencoder.

Both steps are undone before evaluation: the crop/pad exactly (inside the grid), the resampling
only up to interpolation smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# Preprocessing shared with the ISBI 2027 audit (HarmonIt, tag isbi2027-submission).
from harmonit.data.abide_slices_dataset import robust_normalize

HEAD_THR = 0.02


@dataclass
class Reference:
    """A scan in the 1 mm reference space, with what is needed to go back to its native grid."""

    volume: np.ndarray  # [X, Y, Z] float32 in [0, 1], RAS, 1 mm
    affine: np.ndarray  # 1 mm voxel -> world
    native_shape: tuple[int, int, int]
    native_affine: np.ndarray  # RAS-canonical native voxel -> world
    zooms: tuple[float, float, float]


def resample(vol: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None]
    return F.interpolate(t, size=tuple(shape), mode="trilinear", align_corners=False)[0, 0].numpy()


def load_reference(t1_path: str | Path) -> Reference:
    img = nib.as_closest_canonical(nib.load(str(t1_path)))
    vol = robust_normalize(img.get_fdata(dtype=np.float32))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    shape_1mm = tuple(max(1, int(round(s * z))) for s, z in zip(vol.shape, zooms))
    x = np.clip(resample(vol, shape_1mm), 0.0, 1.0).astype(np.float32)
    # align_corners=False: 1 mm index i maps to native index (i + 0.5) * s - 0.5, s = native/new size.
    scale = np.array([o / n for o, n in zip(vol.shape, shape_1mm)])
    step = np.eye(4)
    step[:3, :3] = np.diag(scale)
    step[:3, 3] = 0.5 * scale - 0.5
    return Reference(x, img.affine @ step, tuple(vol.shape), img.affine, zooms)


def to_native(x_1mm: np.ndarray, native_shape: tuple[int, int, int]) -> np.ndarray:
    """Undo the 1 mm resampling (approximate: interpolation smooths slightly)."""
    return resample(x_1mm, native_shape)


def head_bbox(x: np.ndarray, thr: float = HEAD_THR) -> tuple[tuple[int, int], ...]:
    """Bounding box [start, stop) per axis of the largest connected component above ``thr``."""
    mask = x > thr
    labels, n = ndimage.label(mask)
    if n == 0:
        return tuple((0, s) for s in x.shape)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = labels == int(np.argmax(sizes)) + 1
    sl = ndimage.find_objects(keep.astype(np.int8))[0]
    return tuple((s.start, s.stop) for s in sl)


@dataclass
class Placement:
    """Where a fixed grid sits in a 1 mm volume: grid voxel (0, 0, 0) is volume index ``offset``.

    Offsets may be negative or run past the volume (zero padding)."""

    grid: tuple[int, int, int]
    offset: tuple[int, int, int]


def place(bbox: tuple[tuple[int, int], ...], grid: tuple[int, int, int], top_margin: int = 8) -> Placement:
    """Centre the grid on the head in x (left-right) and y (posterior-anterior); in z (inferior-superior)
    put the top of the head ``top_margin`` voxels below the grid's top, so a crop cuts the neck first."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    ox = (x0 + x1) // 2 - grid[0] // 2
    oy = (y0 + y1) // 2 - grid[1] // 2
    oz = z1 + top_margin - grid[2]
    return Placement(tuple(grid), (int(ox), int(oy), int(oz)))


def _overlap(offset: int, size: int, length: int) -> tuple[slice, slice]:
    """Slices into (volume, grid) along one axis for a grid of ``size`` at ``offset`` in ``length``."""
    v0, v1 = max(offset, 0), min(offset + size, length)
    if v1 <= v0:
        return slice(0, 0), slice(0, 0)
    return slice(v0, v1), slice(v0 - offset, v1 - offset)


def to_grid(x: np.ndarray, p: Placement) -> np.ndarray:
    """Crop/pad a 1 mm volume to the grid; outside the volume is zero."""
    out = np.zeros(p.grid, dtype=x.dtype)
    sv, sg = zip(*(_overlap(o, g, n) for o, g, n in zip(p.offset, p.grid, x.shape)))
    out[sg] = x[sv]
    return out


def from_grid(g: np.ndarray, p: Placement, shape: tuple[int, int, int], fill: np.ndarray | None = None) -> np.ndarray:
    """Put a grid back into a 1 mm volume of ``shape``. Voxels the grid did not cover come from ``fill``
    (for harmonized outputs: the unchanged input) or are zero."""
    out = np.zeros(shape, dtype=g.dtype) if fill is None else fill.astype(g.dtype, copy=True)
    sv, sg = zip(*(_overlap(o, s, n) for o, s, n in zip(p.offset, p.grid, shape)))
    out[sv] = g[sg]
    return out


def cut_fraction(x: np.ndarray, p: Placement, thr: float = HEAD_THR) -> float:
    """Share of head voxels (above ``thr``) that fall outside the grid."""
    head = x > thr
    inside = np.zeros_like(head)
    sv, _ = zip(*(_overlap(o, g, n) for o, g, n in zip(p.offset, p.grid, x.shape)))
    inside[sv] = True
    total = int(head.sum())
    return float((head & ~inside).sum() / total) if total else 0.0
