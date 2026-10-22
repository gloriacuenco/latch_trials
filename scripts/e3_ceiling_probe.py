"""How much site information does LATCH's conditioning carry? The leakage ceiling, measured.

At keep 0 the harmonized latent depends on the source scan only through its anatomy maps c (plus
noise, the NYU label and weights that do not depend on the scan). So, in theory, no probe can find
more site information in the outputs than the best possible probe finds in c (data processing
inequality; Moyer et al. 2020). A trained probe only estimates that from below, so every number here
is a lower estimate; what matters is that the same probe families are run on every representation:

- brain mask: 1 - background fraction of the maps (shape only, no tissue information);
- anatomy maps: the 10 tissue-fraction channels the generator is given;
- keep-0 outputs: the regenerated latents (NYU scans regenerated too);
- original latents.

Probes: the linear probe on pooled features used throughout (hdev/probes.py ``coarse``), and a small
3D CNN on the full 48 x 56 x 48 grid (stronger; three seeds, fixed budget, no selection on
validation). Trained on the training scans, scored on the validation scans (17 sites), balanced
accuracy with a subject bootstrap 95% interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from hdev.latents import load_latents
from hdev.probes import coarse, probe


class CNNProbe(nn.Module):
    def __init__(self, channels: int, n_sites: int, width: int = 32):
        super().__init__()
        layers, c = [], channels
        for w in (width, 2 * width, 4 * width):
            layers += [nn.Conv3d(c, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
                       nn.Conv3d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(c, n_sites))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean((2, 3, 4)))


def cnn_probe(x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, n_sites: int, seed: int,
              epochs: int, device: torch.device) -> np.ndarray:
    """Train on tr, predict va. Inputs standardized per channel with training statistics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = torch.from_numpy(x).to(device, torch.float16)
    mean = xt[torch.from_numpy(tr).to(device)].float().mean((0, 2, 3, 4), keepdim=True)
    std = xt[torch.from_numpy(tr).to(device)].float().std((0, 2, 3, 4), keepdim=True) + 1e-6
    net = CNNProbe(x.shape[1], n_sites).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    idx_tr = np.where(tr)[0]
    by_site = {s: idx_tr[y[idx_tr] == s] for s in np.unique(y[idx_tr])}
    steps = epochs * len(idx_tr) // 16
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    yt = torch.from_numpy(y).to(device)
    net.train()
    for _ in range(steps):
        rows = np.array([rng.choice(by_site[s]) for s in rng.choice(list(by_site), 16)])  # site-balanced
        b = (xt[torch.from_numpy(rows).to(device)].float() - mean) / std
        if rng.random() < 0.5:
            b = b.flip(2)  # left-right
        b = torch.roll(b, shifts=tuple(int(s) for s in rng.integers(-2, 3, 3)), dims=(2, 3, 4))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(b).float(), yt[torch.from_numpy(rows).to(device)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in np.array_split(np.where(va)[0], 8):
            preds.append(net((xt[torch.from_numpy(i).to(device)].float() - mean) / std).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def bootstrap_ba(y: np.ndarray, preds: list[np.ndarray], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean BA over prediction sets, with a 95% interval from resampling subjects."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([balanced_accuracy_score(y, p) for p in preds]))
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        draws.append(np.mean([balanced_accuracy_score(y[i], p[i]) for p in preds]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--keep0-latents", required=True, help="latents.npz from e3_slice_export.py")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda")

    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    y = meta["site_id"].to_numpy()
    n_sites = int(y.max()) + 1
    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    maps = np.stack([np.load(Path(args.labels_dir) / f"{s}.npy") for s in meta["subject_id"]]).astype(np.float16) / 255
    with np.load(args.keep0_latents, allow_pickle=False) as d:
        order = {s: i for i, s in enumerate(d["subject_ids"].astype(str))}
        keep0 = d["latents"][[order[s] for s in meta["subject_id"]]]
    reps = {"brain_mask": (1 - maps[:, :1]).astype(np.float16), "anatomy_maps": maps,
            "keep0_outputs": keep0.astype(np.float16), "original_latents": z.astype(np.float16)}

    res = {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_sites": n_sites, "epochs": args.epochs,
           "seeds": args.seeds}
    for name, x in reps.items():
        f = coarse(x.astype(np.float32))
        _, lin = probe(f[tr], y[tr], f[va], y[va])
        linear = bootstrap_ba(y[va], [lin.predict(f[va])])
        cnn = bootstrap_ba(y[va], [cnn_probe(x, y, tr, va, n_sites, s, args.epochs, device) for s in args.seeds])
        res[name] = {"linear": linear, "cnn": cnn}
        print(f"{name:18s} linear {linear[0]:.3f} [{linear[1]:.3f}, {linear[2]:.3f}]   "
              f"cnn {cnn[0]:.3f} [{cnn[1]:.3f}, {cnn[2]:.3f}]", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
