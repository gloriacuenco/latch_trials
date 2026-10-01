# HarmonIt method (TRDP2): plan

Updated 2026-10-01.

## Direction

Harmonize with the smallest change that reaches the target site. A pretrained 3D autoencoder
maps each T1 volume to a latent; a site-conditioned flow, trained by flow matching with
optimal-transport pairing, moves the latent from the source site toward NYU; the decoder returns a
harmonized volume.

Why this direction, from the ISBI 2027 audit:

- Probes trained on harmonized outputs still decoded the source site for every method tested
  (brain-only 0.74-0.91, against 0.52-0.54 from brain masks alone).
- The only near-chance site score came from destroyed anatomy (adapted HCLD, 13.9 dB).
- 2D diffusion closed about 1% of the distance to NYU, and two sampling draws differed more from
  each other than from the input.

Intended properties: deterministic (an ODE, no sampling draw), minimal and reversible in latent
space, volumetric (no slice stacking).

Fallback (direction B): site-invariant adapters for 3D brain foundation models, scored on
OpenBHB.

The earlier 2.5D diffusion plan was dropped: adjacent-slice harmonization already exists (MURD
2024, IHF-Harmony 2026; YODA 2026 for translation), YODA needs paired data that ABIDE lacks, and
slice inconsistency is a symptom rather than the audit's main finding.

## Gate: does a pretrained autoencoder keep anatomy? (passed 2026-10-01)

`results/ae_fidelity_val_20261001/`. On 113 validation subjects, the MAISI and NV-Generate-MR VAEs
reconstruct between a 0.5 mm and a 1 mm blur (FastSurfer Dice against the input 0.89, subcortical
0.94-0.97, median volume change about 3%) and keep sharpness. The adapted TRDP1 HCLD autoencoder
fails (Dice 0.54). Both NVIDIA weight sets are licensed for non-commercial research only.

## Evaluation

The ISBI audit protocol on whole volumes: converged site probes trained on outputs (head and
brain-only), target alignment to NYU, fixed-scale fidelity, FastSurfer segmentation agreement and
regional volumes. Baselines re-exported as volumes with the same preprocessing: histogram
matching, CycleGAN, StarGAN and 2D diffusion applied slice by slice; HCLD and IGUANe run natively.

## Next steps

1. Encode all ABIDE I volumes with the chosen VAE (NV-Generate-MR unless the meeting decides
   otherwise) and save the latents.
2. Train a first site-conditioned flow on the training latents; check target alignment and site
   probes on validation.
3. Decide whether the decoder needs fine-tuning on ABIDE.

Open decisions for the 1 October meeting with J. M. Martínez and P. Coupé: method and fallback,
latent space, evaluation protocol, venue (MICCAI 2027 or a journal).
