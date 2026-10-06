"""Train the frozen anatomy reader used by the flow's anatomy-consistency term.

Input: latents of every site (training split, QC exclusions out); target: the FastSurfer class
fractions per latent voxel (scripts/anatomy_labels.py). Validation: soft Dice per class on the
validation latents, overall and per site, to check the reader works equally well on every site.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hdev.anatomy import CLASSES, AnatomyNet, soft_cross_entropy, soft_dice
from hdev.flow import LatentStats
from hdev.latents import load_latents


def load_labels(labels_dir: Path, ids) -> np.ndarray:
    return np.stack([np.load(labels_dir / f"{s}.npy") for s in ids])  # uint8, fraction x 255


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out", required=True, help="checkpoint path")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    tr, va = np.where(meta["split"] == "train")[0], np.where(meta["split"] == "val")[0]
    stats = LatentStats.fit(z[tr])
    labels = load_labels(Path(args.labels_dir), meta["subject_id"])

    def batch(idx: np.ndarray, flip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        zz = stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))
        yy = torch.from_numpy(labels[idx].astype(np.float32) / 255).to(device)
        if flip:  # classes merge left and right, so a left-right flip is a valid augmentation
            zz, yy = zz.flip(2), yy.flip(2)
        return zz, yy

    net = AnatomyNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(tr)
        losses = []
        for i in range(0, len(order), args.batch):
            zz, yy = batch(order[i:i + args.batch], flip=bool(rng.random() < 0.5))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(zz)
            loss = soft_cross_entropy(logits.float(), yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} loss {np.mean(losses):.4f}", flush=True)

    net.eval()
    dice_all, per_site = [], {}
    with torch.no_grad():
        for i in va:
            zz, yy = batch(np.array([i]))
            d = soft_dice(torch.softmax(net(zz).float(), 1), yy).cpu().numpy()
            dice_all.append(d)
            per_site.setdefault(meta.at[i, "site"], []).append(d[1:].mean())
    dice = np.mean(dice_all, 0)
    res = {"val_dice": {c: round(float(v), 4) for c, v in zip(CLASSES, dice)},
           "val_dice_mean_brain_classes": round(float(dice[1:].mean()), 4),
           "val_dice_by_site": {s: round(float(np.mean(v)), 4) for s, v in sorted(per_site.items())}}
    print(json.dumps(res, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "stats": {"mean": stats.mean, "std": stats.std},
                "classes": CLASSES, "val": res, "args": vars(args)}, args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
