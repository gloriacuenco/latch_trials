# LATCH: plan

Updated 2026-10-15.

## Direction

Harmonize with the smallest change that reaches the target site. A frozen pretrained 3D VAE maps
each T1 volume to a latent; a site-conditioned flow, trained by flow matching with
optimal-transport pairing, moves the latent from the source site toward NYU; the decoder returns
the harmonized volume, which is put back on the scan's own grid and judged with the ISBI 2027
audit protocol.

**Proposed element: anatomy-matched pairing.** Flow matching learns from (source scan, NYU scan)
pairs. Without travelling subjects these are different people, so the flow can learn to change
anatomy. The hypothesis is that pairing source scans with NYU scans of similar anatomy (FastSurfer
regional volumes, age, sex) reduces anatomical differences within pairs: approximate
pseudo-travelling subjects from unpaired data. The six-condition pilot does not show a preservation
advantage for this pairing. A first search found no prior use for unpaired multi-site harmonization; a
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
| E0 | Data ready for the flow | Reference volumes; grid size; latents with both VAEs; FastSurfer on all 1,112 scans | Complete on cl/vpulab |
| E1 | How much site is in the latents? | Site probe on latents | Complete; linear BA 1.00 for both VAEs |
| E2a | Which autoencoder? | One latent-OT flow per VAE, compared on validation | Pending |
| E2b | Does anatomy-matched pairing help? | Random / latent-OT / anatomy-OT, each with anatomy weight 0 / 1 | Six-condition 10k-step pilot complete and verified; longer runs and multiple seeds pending |
| E3 | Final comparison | ISBI test slices, once, against the ten audited outputs | |

The two existing NV-Generate-MR anatomy-pairing pilots each ran for 10,000 steps with seed 0.
Against the input's FastSurfer segmentation on 92 source-validation volumes: no anatomy term
gave Dice 0.58 and median volume change 13.6%; anatomy weight 1 gave Dice 0.80 and 6.4%.
Output-trained latent site probes remained high (0.95 and 0.93); foreground-intensity distance
closed toward NYU was 24% and 16%. These are pilot validation results, not final test evidence.

The completed comparison (`results/e2_pairing_ablation_20261014`) adds four random/latent conditions
under the same settings and cached latents. All six share the exact 92 source-validation subjects
and complete FastSurfer outputs. Anatomy consistency raises Dice from 0.581-0.599 to 0.798-0.807
across all pairings; anatomy pairing does not lead preservation at either loss setting. Latent OT
with the loss has the highest Dice (0.807) and lowest regional-volume change (6.0%). Anatomy OT
with the loss has the lowest output-trained site BA (0.926), but the least NYU intensity distance
closed among loss-enabled runs (16.1%, versus 18.3-19.4%). Site BA stays 0.926-0.970 across all
conditions: the central site-removal problem remains.

The updated progress report is `output/pdf/LATCH Progress Report - Updated.pdf`. Small results and
verification are versionable in the results folder; the frozen source, hashes, and large artifacts
stay at `/mnt/rhome/mmi/projects/harmonit-data/e2/pairing_ablation_20261014`. These are single-seed
validation pilots, not final test evidence. Confirm the pairing effect with longer runs and multiple
seeds; then investigate a term targeting recoverable site information. A weight sweep (3 and 10)
and a matched MAISI comparison remain pending. Resolve posterior-mean encoding/repeatability in
a separate controlled change, without altering the cache used for this ablation.

Later: a whole-volume comparison (2D baselines applied slice by slice with their existing
checkpoints; HCLD and IGUANe natively), the only evidence for a 3D claim, and a test on unseen
sites (ABIDE II), which keeps the work from being single-dataset. Open: compare the direct bridge
with the inversion route, or start with the bridge only.

## Where things run

cl: reference volumes in `~/trdp2/data/reference_1mm`, FastSurfer in `~/trdp2/data/fastsurfer`,
weights in `~/trdp2/models`, environments `~/envs/trdp2_ae` and `~/envs/fastsurfer`. vpulab: clone
at `/mnt/rhome/mmi/projects/harmonit-dev` (pull only); flow training runs there.
