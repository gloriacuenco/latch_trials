"""Summarize scripts/ae_fidelity.py runs: tables (markdown, CSV) and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "maisi": "MAISI VAE",
    "nvmr": "NV-Generate-MR VAE",
    "hcld": "Adapted HCLD AE",
    "hcld_resize": "HCLD resize only",
    "blur05": "Blur 0.5 mm",
    "blur10": "Blur 1.0 mm",
}
ORDER = ["nvmr", "maisi", "hcld", "hcld_resize", "blur05", "blur10"]
METRICS = ["psnr_fg", "ssim_fg", "grad_ratio", "highpass_corr", "w1_fg"]


def q(s: pd.Series) -> str:
    return f"{s.median():.3f} [{s.quantile(0.25):.3f}, {s.quantile(0.75):.3f}]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run = Path(args.run_dir)
    df = pd.concat([pd.read_csv(f) for f in sorted(run.glob("*/per_subject.csv"))], ignore_index=True)
    models = [m for m in ORDER if m in set(df["model"])]

    lines = [f"# Autoencoder fidelity on ABIDE I validation ({df['subject_id'].nunique()} subjects, "
             f"{df['site'].nunique()} sites)", "",
             "Median [IQR] per subject, in the 1 mm reference space, foreground = reference > 0.02.", "",
             "| Model | PSNR fg (dB) | SSIM fg | Gradient ratio | High-pass corr | W1 fg | s/volume | peak GB |",
             "|---|---|---|---|---|---|---|---|"]
    for m in models:
        d = df[df["model"] == m]
        lines.append(f"| {LABELS[m]} | " + " | ".join(q(d[c]) for c in METRICS)
                     + f" | {d['seconds'].median():.1f} | {d['peak_gb'].max():.1f} |")
    lines += ["", "Gradient ratio < 1 means the output is smoother than the input; high-pass correlation "
              "measures how much sub-millimetre detail survives.", ""]

    site_tab = (df.pivot_table(index="site", columns="model", values="psnr_fg", aggfunc="median")[models]
                .round(2))
    site_tab.to_csv(run / "psnr_by_site.csv")
    lines += ["## PSNR fg (dB) by site, median", "", "| Site | " + " | ".join(LABELS[m] for m in models) + " |",
              "|---|" + "---|" * len(models)]
    for site, r in site_tab.iterrows():
        lines.append(f"| {site} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
    modes = df.groupby("model")["mode"].value_counts().to_dict()
    lines += ["", f"Inference modes: {modes}", ""]
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    df.groupby("model")[METRICS + ["seconds", "peak_gb"]].describe().to_csv(run / "summary_describe.csv")
    print("\n".join(lines))

    # Figures: one row per example subject, axial view with a zoomed inset.
    sl = {}
    for f in run.glob("*/slices_*.npz"):
        sl[f.stem.replace("slices_", "")] = dict(np.load(f))
    shown = [m for m in ["nvmr", "maisi", "hcld"] if m in sl]
    preferred = ["NYU", "USM", "Pitt", "UCLA", "Stanford", "KKI"]
    subs = []
    for site in preferred:
        cands = [s for s in sl["input"] if s.startswith(site + "_")]
        if cands:
            subs.append(sorted(cands)[0])
    for view, vname in ((0, "axial"), (1, "coronal")):
        cols = ["input"] + shown
        fig, axes = plt.subplots(len(subs), len(cols) * 2, figsize=(2.0 * len(cols) * 2, 2.1 * len(subs)))
        for i, s in enumerate(subs):
            ref = sl["input"][s][view].astype(np.float32)
            for j, m in enumerate(cols):
                img = np.rot90(sl[m][s][view].astype(np.float32))
                diff = np.rot90(np.abs(sl[m][s][view].astype(np.float32) - ref))
                ax = axes[i, 2 * j]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                h, w = img.shape
                zoom = img[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40]
                ax2 = axes[i, 2 * j + 1]
                if m == "input":
                    ax2.imshow(zoom, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax2.imshow(diff, cmap="magma", vmin=0, vmax=0.2, interpolation="nearest")
                for a in (ax, ax2):
                    a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    ax.set_title(LABELS.get(m, "Input"), fontsize=8)
                    ax2.set_title("zoom" if m == "input" else "|error| (0-0.2)", fontsize=8)
            axes[i, 0].set_ylabel(s, fontsize=7)
        fig.tight_layout()
        fig.savefig(run / f"fig_{vname}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in models:
        d = df[df["model"] == m]
        ax[0].scatter(d["psnr_fg"], d["ssim_fg"], s=8, label=LABELS[m])
        ax[1].scatter(d["grad_ratio"], d["highpass_corr"], s=8, label=LABELS[m])
    ax[0].set_xlabel("PSNR fg (dB)"); ax[0].set_ylabel("SSIM fg")
    ax[1].set_xlabel("gradient ratio (output / input)"); ax[1].set_ylabel("high-pass correlation")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(run / "fig_metrics.png", dpi=130)


if __name__ == "__main__":
    main()
