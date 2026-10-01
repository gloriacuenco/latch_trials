"""Latents written by scripts/encode_latents.py: one float16 [4, X, Y, Z] array per scan plus index.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_latents(latent_dir: str | Path, splits: tuple[str, ...] = ("train", "val"),
                 exclude_qc: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack the latents of the given splits, in index order. Kept as float16 to save memory."""
    latent_dir = Path(latent_dir)
    meta = pd.read_csv(latent_dir / "index.csv")
    meta = meta[meta["split"].isin(splits)]
    if exclude_qc and "qc_exclude" in meta:
        meta = meta[~meta["qc_exclude"].astype(bool)]
    meta = meta.reset_index(drop=True)
    x = np.stack([np.load(latent_dir / f"{s}.npy") for s in meta["subject_id"]])
    return x, meta
