# TRDP2 gate: pretrained 3D autoencoder fidelity on ABIDE I (2026-10-01)

Question: can a pretrained 3D autoencoder reconstruct ABIDE I T1 volumes without losing
anatomy? A latent harmonizer cannot preserve more than its autoencoder reconstructs.

Run on cl (A100), validation split only (113 subjects, 17 sites); the test split was not used.
Reference space: raw T1 reoriented to RAS, `robust_normalize` ([p1, p99] to [0, 1]) as in the
ISBI 2027 audit, trilinear to 1 mm isotropic. Foreground: reference > 0.02.

| Autoencoder | Weights | Licence of weights |
| --- | --- | --- |
| MAISI VAE | `MONAI/maisi_ct_generative` `models/autoencoder.pt` | NVIDIA, non-commercial research |
| NV-Generate-MR VAE | `nvidia/NV-Generate-MR` `models/autoencoder_v2.pt` | NVIDIA OneWay non-commercial research |
| Adapted HCLD AE | TRDP1 checkpoint `outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt` (cl) | ours |

SHA-256 of weights and script: `sha256.txt`. Blur 0.5/1.0 mm and "HCLD resize only" are calibration
references, not models.

## Results

Reconstruction, 113 subjects, medians (full table with IQRs: `summary.md`, per subject:
`*/per_subject.csv`, by site: `psnr_by_site.csv`):

| Reconstruction | PSNR fg (dB) | SSIM fg | PSNR fg after 1 mm smoothing | Gradient ratio |
| --- | ---: | ---: | ---: | ---: |
| NV-Generate-MR VAE | 24.9 | 0.84 | 32.7 | 0.95 |
| MAISI VAE | 24.9 | 0.83 | 34.0 | 0.98 |
| Adapted HCLD AE | 18.1 | 0.53 | 21.0 | 0.61 |
| HCLD resize only | 22.9 | 0.80 | 30.0 | 0.64 |
| Blur 0.5 mm | 33.3 | 0.98 | 44.3 | 0.83 |
| Blur 1.0 mm | 24.4 | 0.83 | 32.6 | 0.58 |

FastSurferCNN v2.5.4 segmentation agreement with the input's segmentation, one subject per site
(17), medians (`seg_agreement.csv`):

| Reconstruction | Dice, all labels | Cortical parcels | Subcortical groups | Median abs. volume change |
| --- | ---: | ---: | ---: | ---: |
| NV-Generate-MR VAE | 0.894 | 0.880 | 0.943-0.971 | 2.8% |
| MAISI VAE | 0.888 | 0.870 | 0.939-0.968 | 2.9% |
| Adapted HCLD AE | 0.544 | 0.480 | 0.678-0.852 | 16.8% |
| HCLD resize only | 0.845 | 0.824 | 0.917-0.962 | 4.9% |
| Blur 0.5 mm | 0.959 | 0.956 | 0.977-0.985 | 2.2% |
| Blur 1.0 mm | 0.856 | 0.842 | 0.927-0.963 | 8.6% |

Verdict: both pretrained VAEs pass (between a 0.5 mm and a 1 mm blur; sharpness kept; errors look
like noise and texture, `fig_zoom.png`). The adapted TRDP1 HCLD autoencoder fails, and its
192x192x64 input resize alone already loses detail. Agreement is consistency with the input's
segmentation, not accuracy.

Caveats: inputs used our robust scaling rather than NVIDIA's [0, p99.5] scaling; VAE outputs
overshoot [0, 1] (about -0.23 to 1.57) and are scored unclipped (NIfTIs for segmentation were
clipped to [0, 1]); segmentation covers 17 subjects only.

## Reproduce

Code: `scripts/ae_fidelity.py`, `ae_fidelity_report.py`, `make_blur_nifti.py`, `seg_agreement.py`;
launchers `slurm/ae_fidelity.sbatch`, `slurm/fastsurfer_seg.sbatch`. The run itself used these scripts
from a HarmonIt working copy (`scripts/trdp2/`), with HarmonIt's `src/` on the path; the only change
since is the import of `robust_normalize`, so `sha256.txt` lists the script as run.

Large artifacts stay on cl under `/home/muqsitamir/trdp2/runs/ae_fidelity_val_20261001`
(1 mm NIfTIs for one subject per site, FastSurfer outputs, slice NPZs, full-size figures).
Environments on cl: `~/envs/trdp2_ae` (MONAI 1.5.1), conda `hcld` (HCLD AE), `~/envs/fastsurfer`
with FastSurfer v2.5.4 in `~/trdp2/tools/FastSurfer`. Weights in `~/trdp2/models`.
