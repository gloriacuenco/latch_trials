"""E2: train the LATCH flow on one VAE's latents with one coupling.

Data: the training latents minus QC exclusions; sources are the 16 non-NYU sites, the target is NYU.

One training step (conditional flow matching):
1. pick a batch of source scans, each site equally likely;
2. draw each scan's NYU partner from the coupling (see hdev/flow.py);
3. pick a time t in [0, 1] per pair and the point on the straight line between them,
   z_t = (1 - t) z_source + t z_partner;
4. ask the network for the velocity at z_t, and penalize its squared difference from the
   line's direction z_partner - z_source.
With probability ``--label-dropout`` the site label is replaced by "unknown", so the flow also
learns to harmonize scans from sites it has not seen.

Anatomy consistency (``--anatomy-weight`` > 0): the network's velocity also implies where the scan
would end up, z_end = z_t + (1 - t) v. A frozen anatomy reader (scripts/train_anatomy_net.py) maps
z_end to tissue-class fractions, and their cross-entropy against the source scan's own FastSurfer
fractions is added to the loss. The pairing pulls toward NYU's appearance; this term holds the source
scan's anatomy in place.

Every ``--val-every`` steps, with the averaged (EMA) weights, the validation source latents are
transported to NYU and two cheap numbers are logged in latent space: how far they moved, and how
far their per-channel statistics are from NYU's validation latents (before vs after). The real
evaluation (decode, FastSurfer, site probes) runs afterwards on saved checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hdev.anatomy import AnatomyNet, soft_cross_entropy, soft_dice
from hdev.flow import LatentStats, VelocityNet, anatomy_features, make_coupling, standardize, transport
from hdev.latents import load_latents

TARGET = "NYU"


def channel_stats(z: torch.Tensor) -> torch.Tensor:
    """Per-channel mean and std over scans and voxels -> [2C]."""
    return torch.cat([z.mean((0, 2, 3, 4)), z.std((0, 2, 3, 4))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--brain-index", required=True, help="results/e0_data/brain_index.csv (anatomy features)")
    p.add_argument("--coupling", required=True, choices=["random", "latent", "anatomy"])
    p.add_argument("--eps", type=float, default=0.05, help="Sinkhorn regularization, relative to the median cost")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--label-dropout", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--channels", nargs=3, type=int, default=[64, 128, 256])
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anatomy-net", default=None, help="checkpoint from train_anatomy_net.py (also used for monitoring)")
    p.add_argument("--labels-dir", default=None, help="anatomy targets from anatomy_labels.py")
    p.add_argument("--anatomy-weight", type=float, default=0.0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if args.anatomy_weight > 0 and not (args.anatomy_net and args.labels_dir):
        raise SystemExit("--anatomy-weight needs --anatomy-net and --labels-dir")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ---- data: latents, scaling, sources vs NYU
    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    is_tgt = (meta["site"] == TARGET).to_numpy()
    stats = LatentStats.fit(z[tr])
    src_tr, tgt_tr = np.where(tr & ~is_tgt)[0], np.where(tr & is_tgt)[0]
    src_va, tgt_va = np.where(va & ~is_tgt)[0], np.where(va & is_tgt)[0]
    n_sites = int(meta["site_id"].max()) + 1
    print(f"train: {len(src_tr)} source scans, {len(tgt_tr)} NYU; val: {len(src_va)} source, {len(tgt_va)} NYU", flush=True)

    def batch_tensor(idx: np.ndarray) -> torch.Tensor:
        return stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))

    # ---- coupling between training sources and training NYU scans
    brain = pd.read_csv(args.brain_index).set_index("subject_id").loc[meta["subject_id"]].reset_index()
    feats = anatomy_features(brain)
    feats = standardize(feats, feats[tr]).astype(np.float32)
    t0 = time.time()
    plan = make_coupling(args.coupling, z[src_tr].astype(np.float32), z[tgt_tr].astype(np.float32),
                         feats[src_tr], feats[tgt_tr], args.eps)
    if not torch.isfinite(plan).all():
        raise SystemExit("coupling has non-finite entries; raise --eps")
    plan_np = plan.double().numpy()
    plan_np /= plan_np.sum(1, keepdims=True)
    eff_partners = float(torch.exp(-(plan * torch.log(plan + 1e-12)).sum(1)).mean())
    print(f"coupling {args.coupling}: {time.time() - t0:.0f} s, about {eff_partners:.1f} likely partners per source scan", flush=True)
    sites_tr = meta["site_id"].to_numpy()[src_tr]
    by_site = {s: np.where(sites_tr == s)[0] for s in np.unique(sites_tr)}

    # ---- model, optimizer, averaged copy
    net = VelocityNet(n_sites, channels=tuple(args.channels)).to(device)
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"velocity net: {sum(q.numel() for q in net.parameters()) / 1e6:.1f} M parameters", flush=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "n_sites": n_sites, "partners": eff_partners,
                                                 "n_src_train": len(src_tr), "n_tgt_train": len(tgt_tr)}, indent=2))

    # ---- frozen anatomy reader and the source scans' anatomy targets
    anat = None
    if args.anatomy_net:
        ack = torch.load(args.anatomy_net, map_location="cpu", weights_only=False)
        anat = AnatomyNet().to(device).eval()
        anat.load_state_dict(ack["state_dict"])
        for q in anat.parameters():
            q.requires_grad_(False)
        anat_stats = LatentStats(ack["stats"]["mean"], ack["stats"]["std"])
        labels = {i: np.load(Path(args.labels_dir) / f"{meta.at[i, 'subject_id']}.npy")
                  for i in np.concatenate([src_tr, src_va])}

        def anatomy_target(idx: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.stack([labels[i] for i in idx]).astype(np.float32) / 255).to(device)

        def read_anatomy(z_norm: torch.Tensor) -> torch.Tensor:
            """Flow-normalized latents -> anatomy logits (the reader has its own scaling)."""
            return anat(anat_stats.normalize(stats.denormalize(z_norm)))

    z_src_va, z_tgt_va = batch_tensor(src_va), batch_tensor(tgt_va)
    site_va = torch.from_numpy(meta["site_id"].to_numpy()[src_va]).to(device)
    gap_before = float((channel_stats(z_src_va) - channel_stats(z_tgt_va)).abs().mean())
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "loss", "loss_anatomy", "seconds_per_step", "peak_gb", "val_moved", "val_gap_before",
                     "val_gap_after", "val_anatomy_dice_before", "val_anatomy_dice_after"])
    dice_before = None
    if anat is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = torch.cat([torch.softmax(read_anatomy(z_src_va[i:i + 8]).float(), 1) for i in range(0, len(src_va), 8)])
        dice_before = float(soft_dice(probs, anatomy_target(src_va))[1:].mean())

    losses, anat_losses, tic = [], [], time.time()
    for step in range(1, args.steps + 1):
        # 1. site-balanced source scans
        sites = rng.choice(list(by_site), size=args.batch)
        rows = np.array([rng.choice(by_site[s]) for s in sites])
        # 2. NYU partners drawn from the coupling
        partners = np.array([rng.choice(len(tgt_tr), p=plan_np[r]) for r in rows])
        z0, z1 = batch_tensor(src_tr[rows]), batch_tensor(tgt_tr[partners])
        site = torch.from_numpy(meta["site_id"].to_numpy()[src_tr[rows]]).to(device)
        site = torch.where(torch.rand(args.batch, device=device) < args.label_dropout, torch.full_like(site, n_sites), site)
        # 3. a point on the straight line between each pair
        t = torch.rand(args.batch, device=device)
        zt = (1 - t)[:, None, None, None, None] * z0 + t[:, None, None, None, None] * z1
        # 4. regress the velocity onto the line's direction
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(zt, t, site)
        loss = F.mse_loss(v.float(), z1 - z0)
        losses.append(loss.item())
        # 5. anatomy consistency: where this velocity would take the scan should keep its anatomy
        if args.anatomy_weight > 0:
            z_end = zt + (1 - t)[:, None, None, None, None] * v.float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = read_anatomy(z_end)
            loss_anat = soft_cross_entropy(logits.float(), anatomy_target(src_tr[rows]))
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = [transport(ema, z_src_va[i:i + 8], site_va[i:i + 8], args.ode_steps).float()
                        for i in range(0, len(z_src_va), 8)]
            zh = torch.cat(outs)
            val_moved = float(((zh - z_src_va) ** 2).mean())
            gap_after = float((channel_stats(zh) - channel_stats(z_tgt_va)).abs().mean())
            dice_after = None
            if anat is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    probs = torch.cat([torch.softmax(read_anatomy(zh[i:i + 8]).float(), 1) for i in range(0, len(zh), 8)])
                dice_after = float(soft_dice(probs, anatomy_target(src_va))[1:].mean())
            peak = torch.cuda.max_memory_allocated() / 1e9
            la = float(np.mean(anat_losses)) if anat_losses else None
            writer.writerow([step, np.mean(losses), la, round(sec, 3), round(peak, 2), round(val_moved, 4),
                             round(gap_before, 4), round(gap_after, 4), dice_before, dice_after])
            log.flush()
            anat_txt = f" anatomy dice {dice_before:.3f} -> {dice_after:.3f}" if anat is not None else ""
            print(f"step {step} loss {np.mean(losses):.4f} ({sec:.2f} s/step, {peak:.1f} GB) "
                  f"val moved {val_moved:.3f} gap {gap_before:.3f} -> {gap_after:.3f}{anat_txt}", flush=True)
            ckpt = {"step": step, "net": net.state_dict(), "ema": ema.state_dict(),
                    "stats": {"mean": stats.mean, "std": stats.std}, "config": vars(args)}
            torch.save(ckpt, out / "checkpoints" / "latest.pt")
            if step % 10000 == 0 or step == args.steps:  # keep a few for checkpoint selection on validation
                torch.save({k: v for k, v in ckpt.items() if k != "net"}, out / "checkpoints" / f"ema_{step:06d}.pt")
            losses, anat_losses, tic = [], [], time.time()
    log.close()


if __name__ == "__main__":
    main()
