"""E2 evaluation of a trained flow, on the validation split only.

1. Transport every training and validation source latent to NYU with the averaged (EMA) weights.
2. Latent site probes (the E1 linear probe on the central crop), three ways:
   - before: trained and tested on the original latents (what the flow starts from);
   - frozen: trained on original latents, applied to transported ones (how the audit's frozen
     probes would see it; low accuracy alone proves nothing);
   - output-trained: trained on transported training latents plus the NYU training latents, tested
     on transported validation latents (the audit's real test, in latent space).
3. Decoded volumes: transported latents are decoded, put back on each scan's grid (outside the grid
   the input is kept), and compared with the input (PSNR) and with NYU (Wasserstein-1 distance of
   foreground intensities to the NYU training reference; negative change = closer to NYU). The plain
   reconstruction decode(encode(x)) is scored the same way, so the flow's own effect can be told
   apart from the autoencoder's.

``--save-nifti`` writes the harmonized and reconstructed 1 mm volumes for FastSurfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from hdev.flow import LatentStats, VelocityNet, transport
from hdev.latents import load_latents
from hdev.metrics import FG_THR, psnr_fg, quantile_profile, w1
from hdev.preprocess import Placement, from_grid
from hdev.probes import coarse, probe
from hdev.vae import decode, load_vae

TARGET = "NYU"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-nifti", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    n_sites = int(meta["site_id"].max()) + 1
    net = VelocityNet(n_sites, channels=tuple(cfg["channels"])).to(device).eval()
    net.load_state_dict(ck["ema"])
    stats = LatentStats(ck["stats"]["mean"], ck["stats"]["std"])

    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    is_tgt = (meta["site"] == TARGET).to_numpy()
    y = meta["site_id"].to_numpy()
    zh = z.astype(np.float32).copy()  # transported latents; NYU scans stay as they are
    src = np.where(~is_tgt)[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(src), 8):
            idx = src[i:i + 8]
            z0 = stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))
            site = torch.from_numpy(y[idx]).to(device)
            zh[idx] = stats.denormalize(transport(net, z0, site, args.ode_steps).float()).cpu().numpy()
    moved = ((stats.normalize(torch.from_numpy(zh[src])) - stats.normalize(torch.from_numpy(z[src].astype(np.float32))))
             ** 2).mean((1, 2, 3, 4)).numpy()
    res = {"checkpoint": args.checkpoint, "step": ck["step"], "coupling": cfg["coupling"],
           "latent_moved_val": float(moved[va[src]].mean())}

    f0, fh = coarse(z), coarse(zh)
    res["probe_before"], before = probe(f0[tr], y[tr], f0[va], y[va])
    res["probe_frozen"] = float(balanced_accuracy_score(y[va & ~is_tgt], before.predict(fh[va & ~is_tgt])))
    res["frozen_predicts_nyu"] = float((before.predict(fh[va & ~is_tgt]) == y[is_tgt][0]).mean())
    res["probe_output_trained"], _ = probe(fh[tr], y[tr], fh[va], y[va])
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)

    # NYU training reference: average foreground intensity quantiles
    ref_dir = Path(args.ref_dir)
    q_nyu = np.mean([quantile_profile(x, x > FG_THR) for x in
                     (np.asarray(nib.load(str(ref_dir / f"{s}.nii.gz")).dataobj, dtype=np.float32)
                      for s in meta.loc[tr & is_tgt, "subject_id"])], 0)

    lat_index = pd.read_csv(Path(args.latent_dir) / "index.csv").set_index("subject_id")
    vae = load_vae(args.vae, args.vae_ckpt, device)
    rows = []
    for i in np.where(va & ~is_tgt)[0]:
        sid = meta.at[i, "subject_id"]
        img = nib.load(str(ref_dir / f"{sid}.nii.gz"))
        x = np.asarray(img.dataobj, dtype=np.float32)
        li = lat_index.loc[sid]
        pl = Placement((zh.shape[2] * 4, zh.shape[3] * 4, zh.shape[4] * 4), (li.offset_x, li.offset_y, li.offset_z))
        vols = {}
        for name, lat in (("harmonized", zh[i]), ("reconstruction", z[i].astype(np.float32))):
            g = decode(vae, torch.from_numpy(lat).to(device)[None])[0, 0].clamp(0, 1).cpu().numpy()
            vols[name] = from_grid(g, pl, x.shape, fill=x)
        q_in = quantile_profile(x, x > FG_THR)
        rec = {"subject_id": sid, "site": meta.at[i, "site"], "w1_nyu_input": w1(q_in, q_nyu)}
        for name, v in vols.items():
            rec[f"psnr_{name}"] = psnr_fg(x, v)
            rec[f"w1_nyu_{name}"] = w1(quantile_profile(v, x > FG_THR), q_nyu)
            if args.save_nifti:
                (out / "nifti" / name).mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(v.astype(np.float32), img.affine), out / "nifti" / name / f"{sid}.nii.gz")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "val_subjects.csv", index=False)
    for c in ["psnr_harmonized", "psnr_reconstruction", "w1_nyu_input", "w1_nyu_reconstruction", "w1_nyu_harmonized"]:
        res[f"median_{c}"] = float(df[c].median())
    res["median_delta_w1_nyu"] = float((df["w1_nyu_harmonized"] - df["w1_nyu_input"]).median())
    res["share_of_distance_closed"] = float(1 - df["w1_nyu_harmonized"].median() / df["w1_nyu_input"].median())
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
