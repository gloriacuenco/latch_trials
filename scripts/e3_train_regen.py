"""E3: train the LATCH regenerator (hdev/regen.py) on one VAE's latents.

Data: the training latents of all 17 sites (NYU included) minus QC exclusions, each with its own
FastSurfer tissue-fraction maps (scripts/anatomy_labels.py).

One training step (flow matching from noise to data):
1. pick a batch of scans: NYU with probability ``--target-share`` (the site we generate at
   harmonization), otherwise a site drawn uniformly from the other 16, then a scan of that site;
2. pair each scan with fresh Gaussian noise;
3. pick a time t in [0, 1] per scan and the point on the straight line between them,
   z_t = (1 - t) noise + t z_scan;
4. ask the network for the velocity at z_t, given the scan's site and anatomy maps, and penalize
   its squared difference from the line's direction z_scan - noise.
With probability ``--label-dropout`` the site label is replaced by "unknown" (needed for site
guidance and for unseen sites).

Anatomy consistency (``--anatomy-weight`` > 0), as in E2: the velocity implies an end point
z_end = z_t + (1 - t) v; the frozen anatomy reader maps it to tissue fractions, and their
cross-entropy against the scan's own fractions is added. The anatomy maps are an input here, so this
term makes sure the generator actually uses them.

Every ``--val-every`` steps, with the averaged (EMA) weights, every validation scan (NYU included)
is regenerated as NYU at ``--val-keep`` with fixed noise, and three cheap latent-space numbers are
logged: how far scans moved, how far per-channel statistics of the source scans are from NYU's
validation latents (before vs after), and the anatomy reader's agreement with the source maps
(before vs after). The real evaluation is scripts/e3_eval_regen.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hdev.anatomy import AnatomyNet, soft_cross_entropy, soft_dice
from hdev.flow import LatentStats
from hdev.latents import load_latents
from hdev.regen import RegenNet, regenerate

TARGET = "NYU"


def channel_stats(z: torch.Tensor) -> torch.Tensor:
    """Per-channel mean and std over scans and voxels -> [2C]."""
    return torch.cat([z.mean((0, 2, 3, 4)), z.std((0, 2, 3, 4))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--labels-dir", required=True, help="anatomy maps from anatomy_labels.py (model input)")
    p.add_argument("--anatomy-net", default=None, help="checkpoint from train_anatomy_net.py")
    p.add_argument("--anatomy-weight", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-share", type=float, default=0.25)
    p.add_argument("--label-dropout", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--channels", nargs=3, type=int, default=[64, 128, 256])
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-keep", type=float, default=0.5)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.anatomy_weight > 0 and not args.anatomy_net:
        raise SystemExit("--anatomy-weight needs --anatomy-net")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ---- data: latents and their anatomy maps
    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    is_tgt = (meta["site"] == TARGET).to_numpy()
    site_id = meta["site_id"].to_numpy()
    nyu = int(site_id[is_tgt][0])
    n_sites = int(site_id.max()) + 1
    stats = LatentStats.fit(z[tr])
    labels_dir = Path(args.labels_dir)
    missing = [s for s in meta["subject_id"] if not (labels_dir / f"{s}.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} scans have no anatomy maps, e.g. {missing[:3]}")
    maps = np.stack([np.load(labels_dir / f"{s}.npy") for s in meta["subject_id"]])  # uint8, fraction x 255
    idx_tr, idx_va = np.where(tr)[0], np.where(va)[0]
    by_site = {s: idx_tr[site_id[idx_tr] == s] for s in np.unique(site_id[idx_tr]) if s != nyu}
    print(f"train: {len(idx_tr)} scans ({int((tr & is_tgt).sum())} NYU), val: {len(idx_va)}; "
          f"{n_sites} sites", flush=True)

    def latents(idx: np.ndarray) -> torch.Tensor:
        return stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))

    def anatomy(idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(maps[idx].astype(np.float32) / 255).to(device)

    # ---- model, optimizer, averaged copy
    net = RegenNet(n_sites, channels=tuple(args.channels)).to(device)
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"regenerator: {sum(q.numel() for q in net.parameters()) / 1e6:.1f} M parameters", flush=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "n_sites": n_sites, "target_site_id": nyu,
                                                 "n_train": len(idx_tr), "n_val": len(idx_va)}, indent=2))

    # ---- frozen anatomy reader
    anat = None
    if args.anatomy_net:
        ack = torch.load(args.anatomy_net, map_location="cpu", weights_only=False)
        anat = AnatomyNet().to(device).eval()
        anat.load_state_dict(ack["state_dict"])
        for q in anat.parameters():
            q.requires_grad_(False)
        anat_stats = LatentStats(ack["stats"]["mean"], ack["stats"]["std"])

        def read_anatomy(z_norm: torch.Tensor) -> torch.Tensor:
            """Regenerator-normalized latents -> anatomy logits (the reader has its own scaling)."""
            return anat(anat_stats.normalize(stats.denormalize(z_norm)))

        def reader_dice(z_norm: torch.Tensor, idx: np.ndarray) -> float:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                probs = torch.cat([torch.softmax(read_anatomy(z_norm[i:i + 8]).float(), 1)
                                   for i in range(0, len(idx), 8)])
            return float(soft_dice(probs, anatomy(idx))[1:].mean())

    # ---- fixed validation noise, so checkpoints are compared on the same draws
    z_va = latents(idx_va)
    noise_va = torch.randn(z_va.shape, generator=torch.Generator().manual_seed(args.seed + 1)).to(device)
    src_va = torch.from_numpy(~is_tgt[idx_va]).to(device)
    gap_before = float((channel_stats(z_va[src_va]) - channel_stats(z_va[~src_va])).abs().mean())
    dice_before = reader_dice(z_va, idx_va) if anat is not None else None
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "loss", "loss_anatomy", "seconds_per_step", "peak_gb", "val_moved", "val_gap_before",
                     "val_gap_after", "val_anatomy_dice_before", "val_anatomy_dice_after"])

    losses, anat_losses, tic = [], [], time.time()
    for step in range(1, args.steps + 1):
        # 1. NYU with probability target_share, otherwise a uniformly chosen other site
        rows = np.array([rng.choice(idx_tr[site_id[idx_tr] == nyu]) if rng.random() < args.target_share
                         else rng.choice(by_site[rng.choice(list(by_site))]) for _ in range(args.batch)])
        z1, a = latents(rows), anatomy(rows)
        site = torch.from_numpy(site_id[rows]).to(device)
        site = torch.where(torch.rand(args.batch, device=device) < args.label_dropout, torch.full_like(site, n_sites), site)
        # 2-3. fresh noise and a point on the line from it to the scan
        z0 = torch.randn_like(z1)
        t = torch.rand(args.batch, device=device)
        tb = t[:, None, None, None, None]
        zt = (1 - tb) * z0 + tb * z1
        # 4. regress the velocity onto the line's direction
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(zt, t, site, a)
        loss = F.mse_loss(v.float(), z1 - z0)
        losses.append(loss.item())
        # 5. anatomy consistency: where this velocity leads should have the scan's anatomy
        if args.anatomy_weight > 0:
            z_end = zt + (1 - tb) * v.float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = read_anatomy(z_end)
            loss_anat = soft_cross_entropy(logits.float(), a)
            anat_losses.append(loss_anat.item())
            loss = loss + args.anatomy_weight * loss_anat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for qe, q in zip(ema.parameters(), net.parameters()):
                qe.lerp_(q, 1 - args.ema)

        if step % args.val_every == 0 or step == args.steps:
            sec = (time.time() - tic) / len(losses)
            nyu_label = torch.full((len(idx_va),), nyu, device=device)
            a_va = anatomy(idx_va)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zh = torch.cat([regenerate(ema, z_va[i:i + 8], noise_va[i:i + 8], nyu_label[i:i + 8], a_va[i:i + 8],
                                           args.val_keep, args.ode_steps).float() for i in range(0, len(idx_va), 8)])
            val_moved = float(((zh - z_va) ** 2).mean())
            gap_after = float((channel_stats(zh[src_va]) - channel_stats(z_va[~src_va])).abs().mean())
            dice_after = reader_dice(zh, idx_va) if anat is not None else None
            peak = torch.cuda.max_memory_allocated() / 1e9
            la = float(np.mean(anat_losses)) if anat_losses else None
            writer.writerow([step, np.mean(losses), la, round(sec, 3), round(peak, 2), round(val_moved, 4),
                             round(gap_before, 4), round(gap_after, 4), dice_before, dice_after])
            log.flush()
            anat_txt = f" anatomy dice {dice_before:.3f} -> {dice_after:.3f}" if anat is not None else ""
            print(f"step {step} loss {np.mean(losses):.4f} ({sec:.2f} s/step, {peak:.1f} GB) "
                  f"val moved {val_moved:.3f} gap {gap_before:.3f} -> {gap_after:.3f}{anat_txt}", flush=True)
            ckpt = {"step": step, "net": net.state_dict(), "ema": ema.state_dict(),
                    "stats": {"mean": stats.mean, "std": stats.std}, "config": vars(args),
                    "n_sites": n_sites, "target_site_id": nyu}
            torch.save(ckpt, out / "checkpoints" / "latest.pt")
            if step % 10000 == 0 or step == args.steps:  # keep a few for checkpoint selection on validation
                torch.save({k: v for k, v in ckpt.items() if k != "net"}, out / "checkpoints" / f"ema_{step:06d}.pt")
            losses, anat_losses, tic = [], [], time.time()
    log.close()


if __name__ == "__main__":
    main()
