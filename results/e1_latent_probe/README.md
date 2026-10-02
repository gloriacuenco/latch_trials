# E1: site information in the latents (2026-10-02)

Site probes trained on the training latents and scored once on the validation latents (QC exclusions
out: 870 training, 110 validation scans), balanced accuracy over 17 sites (chance 0.06). CNN: mean
of 3 seeds, fixed 60 epochs, no checkpoint picked on validation. Central = latent with 6 latent
voxels (24 mm) cut from each side, roughly the brain region.

| Latents | Linear, full | Linear, central | CNN, full | CNN, central | Shuffled-label CNN |
| --- | ---: | ---: | ---: | ---: | ---: |
| NV-Generate-MR | 1.00 | 1.00 | 0.97 | 0.96 | 0.02-0.09 |
| MAISI | 1.00 | 1.00 | 0.97 | 0.96 | 0.02-0.09 |

Anatomy alone, for reference (logistic regression on `results/e0_data/brain_index.csv`, same split):
FastSurfer regional volumes 0.45, brain-box size 0.08, age and sex 0.17, all together 0.58.

Reading:
- The site is perfectly decodable from either VAE's latents, and the central crop changes nothing:
  the signal is in the brain region's appearance, not only in field of view or padding.
- About 0.58 is a floor that anatomy and demographics already reach (sites recruited different
  ages, and FastSurfer volumes themselves vary with the scanner). A harmonizer that pushes
  output-trained probes far below it is probably changing anatomy, the audit's HCLD failure.
- E2's target: bring the output-trained latent probe from 1.00 toward that floor while the
  decoded scans keep their anatomy.

Code: `scripts/e1_latent_probe.py` (commit 4cc4f68). Results: `latent_probe_{nvmr,maisi}.json`.
