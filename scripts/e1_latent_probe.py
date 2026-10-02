"""E1: how much site information do the latents carry?

Two site probes, trained on the training latents and scored once on the validation latents (QC
exclusions left out), with balanced accuracy (BA) over the 17 sites (chance 1/17 = 0.06):

- linear: logistic regression on coarse features (each channel average-pooled to 4 x 4 x 4, plus
  channel means and standard deviations);
- cnn: a small 3D CNN on the full latent, trained for a fixed number of epochs (no checkpoint is
  picked on validation), with left-right flips and small shifts as augmentation.

Every probe runs twice: on the full latent, and on a central crop (``--crop`` latent voxels removed
on each side; the grid is centred on the brain, so the crop is roughly the brain). The full latent
also sees field of view, padding and neck cut-off, which no harmonizer should change; the central
crop is closer to what harmonization can act on, like the audit's brain-only probes.

A shuffled-label control (site labels permuted between training subjects) gives the BA expected by
chance. These numbers are the baseline the flow has to bring down; the same probes are rerun on
transported latents in E2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hdev.latents import load_latents


def coarse_features(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32))
    pooled = F.adaptive_avg_pool3d(t, 4).flatten(1)
    return torch.cat([pooled, t.mean((2, 3, 4)), t.std((2, 3, 4))], 1).numpy()


class Probe(nn.Module):
    """Small 3D CNN: 4 latent channels in, one logit per site out."""

    def __init__(self, n_sites: int, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int, stride: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv3d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())

        self.features = nn.Sequential(block(4, width, 1), block(width, 2 * width, 2),
                                      block(2 * width, 4 * width, 2), block(4 * width, 4 * width, 2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * width, n_sites))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(z).mean((2, 3, 4)))


def augment(z: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """One latent [C, X, Y, Z]: random left-right flip and a shift of up to 2 latent voxels."""
    if rng.random() < 0.5:
        z = z.flip(1)
    shifts = tuple(int(s) for s in rng.integers(-2, 3, size=3))
    return torch.roll(z, shifts=shifts, dims=(1, 2, 3))


def train_cnn(xtr, ytr, xva, n_sites, seed, epochs, device) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Probe(n_sites).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(xtr) / 16))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    counts = np.bincount(ytr, minlength=n_sites)
    weights = torch.tensor(len(ytr) / np.maximum(counts, 1) / n_sites, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            z = torch.from_numpy(xtr[idx].astype(np.float32)).to(device)
            z = torch.stack([augment(zz, rng) for zz in z])
            loss = F.cross_entropy(model(z), torch.from_numpy(ytr[idx]).to(device), weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xva), 32):
            preds.append(model(torch.from_numpy(xva[i:i + 32].astype(np.float32)).to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--out", required=True, help="JSON results file")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--crop", type=int, default=6, help="Latent voxels removed per side for the central crop (6 = 24 mm).")
    args = p.parse_args()
    device = torch.device("cuda")

    x, meta = load_latents(args.latent_dir)
    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    y = meta["site_id"].to_numpy()
    n_sites = int(y.max()) + 1
    sites = meta.drop_duplicates("site_id").sort_values("site_id")["site"].tolist()
    res = {"latent_dir": str(args.latent_dir), "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "chance": 1 / n_sites, "sites": sites}

    c = args.crop
    for view, xv in (("full", x), ("central", x[:, :, c:-c, c:-c, c:-c])):
        feats = coarse_features(xv)
        lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
        lin.fit(feats[tr], y[tr])
        res[f"{view}_linear_ba"] = balanced_accuracy_score(y[va], lin.predict(feats[va]))
        print(f"{view} linear probe BA {res[f'{view}_linear_ba']:.3f}", flush=True)
        for name, shuffle in (("cnn", False), ("cnn_shuffled", True)):
            bas, cm = [], np.zeros((n_sites, n_sites), dtype=int)
            for seed in args.seeds:
                ytr = np.random.default_rng(100 + seed).permutation(y[tr]) if shuffle else y[tr]
                pred = train_cnn(xv[tr], ytr, xv[va], n_sites, seed, args.epochs, device)
                bas.append(balanced_accuracy_score(y[va], pred))
                cm += confusion_matrix(y[va], pred, labels=list(range(n_sites)))
                print(f"{view} {name} seed {seed} BA {bas[-1]:.3f}", flush=True)
            res[f"{view}_{name}_ba"] = bas
            res[f"{view}_{name}_ba_mean"] = float(np.mean(bas))
            res[f"{view}_{name}_confusion"] = cm.tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
