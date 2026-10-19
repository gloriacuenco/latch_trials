# LATCH: plan

Updated 2026-10-19.

## Direction

A frozen pretrained 3D VAE maps each T1 volume to a latent; a model working on the latent makes it
look like NYU; the decoder returns the harmonized volume, which is put back on the scan's own grid
and judged with the ISBI 2027 audit protocol.

**E2 (done): a deterministic flow cannot remove the site.** A site-conditioned flow, trained by flow
matching, transports each source latent to NYU. With the anatomy-consistency term it keeps anatomy
(FastSurfer Dice 0.80) and fools a frozen probe completely, but probes trained on its outputs still
find the site (0.93-0.97). An ODE flow is invertible, so its output carries everything about the
source; it can only move site information where a frozen probe does not look. Anatomy-matched
pairing (pseudo-travelling subjects) gave no preservation advantage over random pairing.

**E3 (now): regenerate the scan as NYU from its own anatomy** (`hdev/regen.py`). A generator
learns, on all 17 sites, to turn noise into a latent given the scan's site and its FastSurfer
tissue-fraction maps. To harmonize, part of the source latent is replaced by noise and the generator
re-draws it as NYU with the source's anatomy maps and the anatomy-consistency term. The noise
destroys site detail for good; what comes back is NYU's look on this scan's anatomy. One knob, the
share of the source kept, trades site removal against faithfulness, so the result is a curve. At
zero kept, the only site information left is what the anatomy maps carry (about 0.58 by the E1
anatomy probe), which bounds what any anatomy-preserving method should reach.

Closest prior work (to check in full before writing): HCLD (Wu et al., MedIA 2025, 3D conditional
latent diffusion, an audit baseline); UMH (MICCAI 2025, two-stage latent diffusion whose fine stage
noises partway and denoises to the target style); SGCD (semantic-guided conditional diffusion);
IHF-Harmony 2026 (an invertible flow); CALAMITI (information-bottleneck disentangling). None
conditions generation on segmentation maps, reports the site/anatomy trade-off over the noise level,
or evaluates against output-trained probes. Partial noising alone is not new; the contribution is
anatomy-conditioned regeneration with that trade-off and the invertibility argument, which also
explains the audit's finding.

Why this direction (from the audit): probes trained on harmonized outputs still decoded the site
for every method tested (brain-only 0.74-0.91 against 0.52-0.54 from brain masks alone); the only
near-chance site score came from destroyed anatomy (adapted HCLD, 13.9 dB); 2D diffusion closed
about 1% of the distance to NYU and two sampling draws differed more than from the input. The
earlier 2.5D diffusion plan was dropped (adjacent-slice harmonization exists in MURD 2024 and
IHF-Harmony 2026; YODA needs paired data; slice inconsistency is a symptom). Fallback: site-invariant
adapters for 3D brain foundation models, scored on OpenBHB.

## Fixed choices

- **Preprocessing** (`hdev/preprocess.py`): RAS, the audit's robust [0, 1] scaling, 1 mm, then a
  192 x 224 x 192 grid centred on the FastSurfer brain box; QC-excluded scans fall back to
  head-box placement (top of the head kept, neck cut first). Undone
  before evaluation: the grid exactly, the resampling up to slight smoothing. No N4 or registration,
  so outputs are scored on the audit's frozen test slices without re-running any baseline.
- **Autoencoders:** both NVIDIA VAEs passed the gate (`results/ae_fidelity_val_20261001`: FastSurfer
  Dice 0.89 against the input, between a 0.5 and a 1 mm blur; the adapted TRDP1 HCLD autoencoder
  0.54). NV-Generate-MR is better on segmentation agreement, MAISI on intensities; the choice is made
  at E2a on validation. Weights are licensed for non-commercial research only.
- **Selection:** every setting on the 113 validation subjects; the test split is used once.
  Flow training and FastSurfer-based validation leave out the fixed QC exclusions, giving 870
  training scans and 110 validation scans (92 non-NYU source scans).
- **Latent cache:** current MONAI 1.5.1 `encode_stage_2_inputs` samples the posterior, despite
  the original posterior-mean description. The current E2 comparison holds those cached latents
  fixed. Flow transport is deterministic for a fixed latent; repeatability after re-encoding
  remains to be measured or enforced in a separate change.

## Experiments

| Step | Question | What runs | Status |
| --- | --- | --- | --- |
| E0 | Data ready | Reference volumes; grid size; latents with both VAEs; FastSurfer on all 1,112 scans | Done |
| E1 | How much site is in the latents? | Site probe on latents | Done: linear BA 1.00 for both VAEs; anatomy alone 0.58 |
| E2 | Can a deterministic flow remove the site? | Random / latent-OT / anatomy-OT pairing x anatomy term off / on (`results/e2_pairing_ablation_20261014`) | Done (10k-step pilots): anatomy term Dice 0.58-0.60 -> 0.80-0.81; site BA 0.93-0.97 everywhere |
| E3 | Does regeneration remove it? | Regenerator pilot (20k steps, NV-Generate-MR); keep 0.3 / 0.5 / 0.7; probes, PSNR, W1, FastSurfer | Code in review |
| E4 | Final comparison | ISBI test slices, once, against the ten audited outputs | |

E3 then needs: site guidance (>1) as a second knob; a seed spread (outputs depend on the noise);
the autoencoder choice (MAISI); and re-encoding with the posterior mean (`gloria/e0-encoding-check`)
before final runs. Until then all E2 and E3 runs share the current cache, so they compare directly.

Later: a whole-volume comparison (2D baselines applied slice by slice with their existing
checkpoints; HCLD and IGUANe natively), the only evidence for a 3D claim, and a test on unseen
sites (ABIDE II), which keeps the work from being single-dataset.

## Where things run

cl: reference volumes in `~/trdp2/data/reference_1mm`, FastSurfer in `~/trdp2/data/fastsurfer`,
weights in `~/trdp2/models`, environments `~/envs/trdp2_ae` and `~/envs/fastsurfer`. vpulab: clone
at `/mnt/rhome/mmi/projects/harmonit-dev` (pull only); flow training runs there.
