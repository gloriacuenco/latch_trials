# HarmonIt method: plan

Updated 2026-10-01.

## Direction

Harmonize with the smallest change that reaches the target site. A frozen pretrained 3D VAE maps
each T1 volume to a latent; a site-conditioned flow, trained by flow matching with
optimal-transport pairing, moves the latent from the source site toward NYU; the decoder returns
the harmonized volume, which is put back on the scan's own grid and judged with the ISBI 2027
audit protocol.

**Novel element: anatomy-matched pairing.** Flow matching learns from (source scan, NYU scan)
pairs. Without travelling subjects these are different people, so the flow can learn to change
anatomy. We pair each source scan with NYU scans of similar anatomy (FastSurfer regional volumes,
age, sex), so the systematic difference within a pair is the site: pseudo-travelling subjects
from unpaired data. A first search found no prior use for unpaired multi-site harmonization; a
full literature check comes before writing. Training with the source-site label sometimes dropped
should also let the flow harmonize scans from unseen sites.

Why this direction (from the audit): probes trained on harmonized outputs still decoded the site
for every method tested (brain-only 0.74-0.91 against 0.52-0.54 from brain masks alone); the only
near-chance site score came from destroyed anatomy (adapted HCLD, 13.9 dB); 2D diffusion closed
about 1% of the distance to NYU and two sampling draws differed more than from the input. The
earlier 2.5D diffusion plan was dropped (adjacent-slice harmonization exists in MURD 2024 and
IHF-Harmony 2026; YODA needs paired data; slice inconsistency is a symptom). Fallback: site-invariant
adapters for 3D brain foundation models, scored on OpenBHB.

## Fixed choices

- **Preprocessing** (`hdev/preprocess.py`): RAS, the audit's robust [0, 1] scaling, 1 mm, then a
  fixed grid placed on the head (centred in x and y, top of the head kept, neck cut first). Undone
  before evaluation: the grid exactly, the resampling up to slight smoothing. No N4 or registration,
  so outputs are scored on the audit's frozen test slices without re-running any baseline.
- **Autoencoders:** both NVIDIA VAEs passed the gate (`results/ae_fidelity_val_20261001`: FastSurfer
  Dice 0.89 against the input, between a 0.5 and a 1 mm blur; the adapted TRDP1 HCLD autoencoder
  0.54). NV-Generate-MR is better on segmentation agreement, MAISI on intensities; the choice is made
  at E2a on validation. Weights are licensed for non-commercial research only.
- **Selection:** every setting on the 113 validation subjects; the test split is used once.

## Experiments

| Step | Question | What runs | Status |
| --- | --- | --- | --- |
| E0 | Data ready for the flow | Reference volumes; grid size; latents with both VAEs; FastSurfer on all 1,112 scans | Running |
| E1 | How much site is in the latents? | Site probe on latents | |
| E2a | Which autoencoder? | One latent-OT flow per VAE, compared on validation | |
| E2b | Does anatomy-matched pairing help? | Random vs latent-OT vs anatomy-matched pairing | |
| E3 | Final comparison | ISBI test slices, once, against the ten audited outputs | |

Later: a whole-volume comparison (2D baselines applied slice by slice with their existing
checkpoints; HCLD and IGUANe natively), the only evidence for a 3D claim, and a test on unseen
sites (ABIDE II), which keeps the work from being single-dataset. Open: compare the direct bridge
with the inversion route, or start with the bridge only.

## Where things run

cl: reference volumes in `~/trdp2/data/reference_1mm`, FastSurfer in `~/trdp2/data/fastsurfer`,
weights in `~/trdp2/models`, environments `~/envs/trdp2_ae` and `~/envs/fastsurfer`. vpulab: clone
at `/mnt/rhome/mmi/projects/harmonit-dev` (pull only); flow training runs there.
