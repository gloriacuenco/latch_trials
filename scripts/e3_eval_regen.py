"""E3 evaluation of a trained regenerator, on the validation split only, at several ``--keep`` levels.

For each keep level (share of the source latent kept; the rest is noise):
1. Regenerate every training and validation scan as NYU, NYU scans included, so every scan has been
   through the same process (a user would harmonize the whole dataset). Each scan gets the same noise
   at every keep level, so differences between levels come from ``keep`` alone.
2. Latent site probes, as in E2 (hdev/probes.py):
   - before: trained and tested on the original latents;
   - frozen: trained on original latents, applied to regenerated validation source scans;
   - output-trained: trained on regenerated training latents, tested on regenerated validation ones
     (the audit's real test, in latent space);
   - output-trained, NYU original: the same, but NYU scans left as they are (E2's protocol, so the
     numbers compare directly with the E2 pilots).
3. Decoded validation source volumes (the same 92 as E2), put back on each scan's grid, compared with
   the input (PSNR) and with NYU (Wasserstein-1 distance of foreground intensities to the NYU training
   reference). The plain reconstruction is not repeated: it is the same latents as E2.

``--save-nifti`` writes the harmonized 1 mm volumes as ``nifti/keep_<k>/<subject>.nii.gz`` and, with
``--ref-seg-dir``, links the input segmentations as ``fastsurfer/input/<subject>`` so FastSurfer
(slurm/fastsurfer_seg.sbatch, INPUT_DIR mode, SEG_DIR=<out>/fastsurfer) and scripts/seg_agreement.py
can run on the result directly.
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

from hdev.flow import LatentStats
from hdev.latents import load_latents
from hdev.metrics import FG_THR, psnr_fg, quantile_profile, w1
from hdev.preprocess import Placement, from_grid
from hdev.probes import coarse, probe
from hdev.regen import RegenNet, regenerate
from hdev.vae import decode, load_vae

TARGET = "NYU"
SEG = "aparc.DKTatlas+aseg.deep.mgz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--keep", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-nifti", action="store_true")
    p.add_argument("--ref-seg-dir", default=None, help="FastSurfer segmentations of the inputs (fastsurfer/reference_1mm)")
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    z, meta = load_latents(args.latent_dir, splits=("train", "val"))
    n_sites = int(meta["site_id"].max()) + 1
    net = RegenNet(n_sites, channels=tuple(ck["config"]["channels"])).to(device).eval()
    net.load_state_dict(ck["ema"])
    stats = LatentStats(ck["stats"]["mean"], ck["stats"]["std"])
    maps = np.stack([np.load(Path(args.labels_dir) / f"{s}.npy") for s in meta["subject_id"]])

    tr, va = (meta["split"] == "train").to_numpy(), (meta["split"] == "val").to_numpy()
    is_tgt = (meta["site"] == TARGET).to_numpy()
    y = meta["site_id"].to_numpy()
    nyu = int(y[is_tgt][0])
    val_src = np.where(va & ~is_tgt)[0]

    f0 = coarse(z)
    probe_before, before = probe(f0[tr], y[tr], f0[va], y[va])

    # NYU training reference: average foreground intensity quantiles
    ref_dir = Path(args.ref_dir)
    q_nyu = np.mean([quantile_profile(x, x > FG_THR) for x in
                     (np.asarray(nib.load(str(ref_dir / f"{s}.nii.gz")).dataobj, dtype=np.float32)
                      for s in meta.loc[tr & is_tgt, "subject_id"])], 0)
    lat_index = pd.read_csv(Path(args.latent_dir) / "index.csv").set_index("subject_id")
    vae = load_vae(args.vae, args.vae_ckpt, device)

    if args.save_nifti and args.ref_seg_dir:
        (out / "fastsurfer" / "input").mkdir(parents=True, exist_ok=True)
        for i in val_src:
            sid = meta.at[i, "subject_id"]
            source, target = Path(args.ref_seg_dir) / sid, out / "fastsurfer" / "input" / sid
            if not (source / SEG).exists():
                raise FileNotFoundError(source / SEG)
            if not target.is_symlink():
                target.symlink_to(source)

    summaries = []
    for keep in args.keep:
        name = f"keep_{keep:.2f}"
        zh = np.empty(z.shape, dtype=np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(z), 8):
                idx = np.arange(s, min(s + 8, len(z)))
                z0 = stats.normalize(torch.from_numpy(z[idx].astype(np.float32)).to(device))
                noise = torch.stack([torch.randn(z0.shape[1:], generator=torch.Generator().manual_seed(args.seed * 100_000 + int(i)))
                                     for i in idx]).to(device)
                a = torch.from_numpy(maps[idx].astype(np.float32) / 255).to(device)
                site = torch.full((len(idx),), nyu, device=device)
                zr = regenerate(net, z0, noise, site, a, keep, args.ode_steps, args.guidance).float()
                zh[idx] = stats.denormalize(zr).cpu().numpy()
        moved = ((stats.normalize(torch.from_numpy(zh)) - stats.normalize(torch.from_numpy(z.astype(np.float32))))
                 ** 2).mean((1, 2, 3, 4)).numpy()
        res = {"checkpoint": args.checkpoint, "step": ck["step"], "keep": keep, "guidance": args.guidance,
               "latent_moved_val_source": float(moved[val_src].mean()), "probe_before": probe_before}

        fh = coarse(zh)
        res["probe_frozen"] = float(balanced_accuracy_score(y[val_src], before.predict(fh[val_src])))
        res["frozen_predicts_nyu"] = float((before.predict(fh[val_src]) == nyu).mean())
        res["probe_output_trained"], _ = probe(fh[tr], y[tr], fh[va], y[va])
        fk = np.where(is_tgt[:, None], f0, fh)  # NYU scans left as they are (E2's protocol)
        res["probe_output_trained_nyu_original"], _ = probe(fk[tr], y[tr], fk[va], y[va])
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)

        rows = []
        for i in val_src:
            sid = meta.at[i, "subject_id"]
            img = nib.load(str(ref_dir / f"{sid}.nii.gz"))
            x = np.asarray(img.dataobj, dtype=np.float32)
            li = lat_index.loc[sid]
            pl = Placement((zh.shape[2] * 4, zh.shape[3] * 4, zh.shape[4] * 4), (li.offset_x, li.offset_y, li.offset_z))
            g = decode(vae, torch.from_numpy(zh[i]).to(device)[None])[0, 0].clamp(0, 1).cpu().numpy()
            v = from_grid(g, pl, x.shape, fill=x)
            rows.append({"subject_id": sid, "site": meta.at[i, "site"], "psnr_harmonized": psnr_fg(x, v),
                         "w1_nyu_input": w1(quantile_profile(x, x > FG_THR), q_nyu),
                         "w1_nyu_harmonized": w1(quantile_profile(v, x > FG_THR), q_nyu)})
            if args.save_nifti:
                (out / "nifti" / name).mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(v.astype(np.float32), img.affine), out / "nifti" / name / f"{sid}.nii.gz")
        df = pd.DataFrame(rows)
        (out / name).mkdir(exist_ok=True)
        df.to_csv(out / name / "val_subjects.csv", index=False)
        res["median_psnr_harmonized"] = float(df["psnr_harmonized"].median())
        res["median_w1_nyu_input"] = float(df["w1_nyu_input"].median())
        res["median_w1_nyu_harmonized"] = float(df["w1_nyu_harmonized"].median())
        res["share_of_distance_closed"] = float(1 - df["w1_nyu_harmonized"].median() / df["w1_nyu_input"].median())
        (out / name / "summary.json").write_text(json.dumps(res, indent=2))
        summaries.append(res)
        print(json.dumps(res, indent=2), flush=True)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
