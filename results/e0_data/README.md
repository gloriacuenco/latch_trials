# E0 data products (2026-10-01)

| File | Contents |
| --- | --- |
| `brain_index.csv` | Per scan (1,112): FastSurfer brain box on the 1 mm reference grid, regional volumes (mm^3), age, sex, diagnosis (ABIDE I phenotypes, `DX_GROUP`: 1 autism, 2 control). Made by `scripts/brain_index.py`. |
| `qc_exclusions.csv` | 23 scans left out of flow training and of FastSurfer-based metrics (`scripts/qc_exclusions.py`): 10 UM scans whose header orientation is wrong (sagittal data labelled axial), 10 thick-slice UCLA scans (1.5 x 1.5 x 4 mm) and 3 others where segmentation failed. 20 train, 3 validation, 0 test. |

The rule was fixed before any flow result. None of the excluded scans is in the test split; the
15 UM test scans are correctly oriented, so the ISBI 2027 test results are unaffected.

FastSurferCNN v2.5.4 (segmentation only) ran on the 1 mm reference volumes on cl in 2 h 40 min
(4 GPUs). Large outputs stay on cl: `~/trdp2/data/reference_1mm`, `~/trdp2/data/fastsurfer`.

Brain extent of the 1,089 scans passing QC (mm): median 136 x 172 x 135, max 156 x 196 x 158.
