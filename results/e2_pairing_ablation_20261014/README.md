# E2 pairing comparison (2026-10-14 to 2026-10-15)

Status: complete and independently verified. The four new conditions finished on 2026-10-15
at 08:17 Europe/Paris; the two earlier anatomy-pairing pilots were reused. All six conditions
use the same 92 source-validation subjects. No test scans were used.

## Results

Medians over the 92 source-validation volumes, except the output-trained linear latent site
probe (balanced accuracy over all 17 sites, 110 validation latents, NYU unchanged).

| Pairing | Anatomy weight | Dice | Cortical Dice | Abs. regional-volume change | NYU intensity distance closed | Output-trained site BA | PSNR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Autoencoder only | - | 0.901 | 0.886 | 2.5% | -1.2% | 1.000 | 25.7 |
| Random | 0 | 0.597 | 0.572 | 14.6% | 22.8% | 0.942 | 14.0 |
| Random | 1 | 0.799 | 0.778 | 6.2% | 19.4% | 0.969 | 14.9 |
| Latent OT | 0 | 0.599 | 0.580 | 13.5% | 23.5% | 0.958 | 14.1 |
| Latent OT | 1 | 0.807 | 0.781 | 6.0% | 18.3% | 0.970 | 14.8 |
| Anatomy OT | 0 | 0.581 | 0.560 | 13.6% | 24.4% | 0.950 | 14.2 |
| Anatomy OT | 1 | 0.798 | 0.777 | 6.4% | 16.1% | 0.926 | 14.8 |

The anatomy-consistency loss drives the preservation gain: enabling it raises Dice by 0.202
(random), 0.208 (latent OT), and 0.218 (anatomy OT), while reducing median absolute regional-volume
change by 7.2-8.3 percentage points. All three pairings reach similar Dice, 0.798-0.807, with the
loss. Anatomy-matched pairing does not lead Dice at either loss setting in this pilot.

Latent OT with the loss has the highest Dice (0.807) and lowest regional-volume change (6.0%).
Anatomy OT with the loss has the lowest output-trained site BA (0.926 versus 0.969-0.970 for the
other loss-enabled runs), but also closes less of the NYU intensity distance (16.1% versus
18.3-19.4%). The autoencoder reconstruction reference still has higher Dice (0.901).

No condition removes site information successfully by this probe: output-trained BA remains
0.926-0.970 even though frozen-probe BA is only 0.000-0.013. Enabling the anatomy loss decreases
NYU intensity distance closed by 3.3, 5.2, and 8.3 percentage points for random, latent OT, and
anatomy OT respectively. Alignment, preservation, and recoverable site information are distinct
outcomes; a near-zero frozen-probe score is not sufficient evidence of harmonization.

These are single-seed, 10,000-step pilots. The close preservation rankings and the apparent site-BA
advantage of anatomy pairing require longer runs and multiple seeds. The comparison supports
anatomy consistency more clearly than the proposed anatomy-pairing contribution. Next, confirm
the pairing effect and test constraints on recoverable site information before a final test run.

## Protocol

- NV-Generate-MR, using the existing cached latents; no scans are re-encoded for this comparison.
- Pairing: random, latent optimal transport, anatomy optimal transport.
- Anatomy-consistency weight: 0 or 1, independently of the pairing strategy.
- Same 10,000 training steps, batch 4, seed 0, learning rate 1e-4, EMA 0.999, label dropout 0.1,
  Sinkhorn epsilon 0.05, validation every 1,000 steps, 20 midpoint integration steps.
- 870 eligible training scans: 723 sources and 147 NYU targets.
- Original/output-trained site probes: 110 validation scans across 17 sites, including unchanged NYU latents.
- Frozen probes: 92 source-validation scans across 16 sites, excluding NYU; their cohort differs
  from the original/output-trained probe cohort, identically for all six conditions.
- Decoded metrics and FastSurfer agreement: the same 92 source-validation scans for each condition.
- The test split is not used. These are single-seed pilot results, not final reported test results.

The existing `pilot_anatomy_nvmr` and `pilot_anatomy_term_nvmr` are reused after checking their
configuration, complete evaluation, and segmentation subject sets. The four random/latent runs
are new. No earlier artifact is overwritten. The original pilot-1 segmentation CSV also contains
the plain autoencoder reconstruction baseline for the same 92 scans.

Code and launch settings are recorded by `scripts/e2_pairing_comparison.py`. Training, evaluation,
and FastSurfer run sequentially on the vpulab RTX A5000; FastSurfer uses two workers as in pilot 2.
A frozen source snapshot and hashes are saved with the remote run. Completed stages can be reused;
interrupted training restarts that condition because the original trainer cannot resume optimizer
state. Missing or invalid segmentations prevent the comparison from being marked complete.

## Artifacts

Large artifacts and the live `status.json` stay on vpulab:
`/mnt/rhome/mmi/projects/harmonit-data/e2/pairing_ablation_20261014`.

This folder contains `comparison.json`, `comparison.csv`, `provenance.json`, `status.json`,
`verification.json`, and each condition's small evaluation/configuration files under `small_results`.
The status embedded in `comparison.json` was refreshed from the final canonical `status.json`;
the measured values are unchanged. The runner's original JSON is preserved locally under
`runs/e2_pairing_ablation_20261014/comparison_as_written.json`.

The updated progress report is `output/pdf/LATCH Progress Report - Updated.pdf` at the repository
root. The current 16-slide version ends with Next experiments; the timeline and discussion slides
were removed at the user's request. It preserves the original pilot history and figures, adds the
six-condition results and interpretation, and corrects the encoding-repeatability claim and the
use of 0.58 as a hard floor.

Independent verification recomputed image/anatomy aggregates from per-subject files, checked
the shared subject IDs and configurations, confirmed 110 original/output-trained probe-validation subjects across 17
sites, and checked the cached-index and frozen-code hashes. All six sets of harmonized volumes
are present on vpulab; the reused pilot-1 segmentations and reconstruction baseline were checked
on cl, and the other segmentation sets on vpulab. There are 92 complete segmentations per
condition. See `verification.json` for the verification record.

## Interpretation

Higher Dice and lower regional-volume change indicate better agreement with FastSurfer on the
input. Lower output-trained site balanced accuracy indicates less recoverable site information;
a fooled frozen probe alone does not establish harmonization. Greater foreground-intensity
distance closed indicates movement toward the NYU training reference. Each answers a different
question and should be interpreted together.

The VAE encoding API samples a posterior latent in MONAI 1.5.1, despite the original code's
posterior-mean description. Holding the existing cache fixed makes this comparison consistent;
it does not establish repeatability after re-encoding. The anatomy/demographics probe's 0.58
score is an empirical comparison point, not a mathematical lower bound.
