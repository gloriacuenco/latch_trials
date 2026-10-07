# LATCH

**Latent Anatomy-preserving Transport for Cross-site Harmonization.** Private development of our
MRI harmonization method (TRDP2). It moves a scan's latent, in a frozen pretrained 3D autoencoder,
from its acquisition site toward a reference site (NYU) with a flow, while keeping the anatomy.

The public [HarmonIt](https://github.com/muqsitamir/HarmonIt) repository holds the ISBI 2027
evaluation audit and stays frozen at the tag `isbi2027-submission`; it is also the evaluation
LATCH is judged by. This repository stays private until the method paper is submitted, then goes
public with it.

Plan and decisions: [docs/PLAN.md](docs/PLAN.md).

## Layout

| Path | Contents |
| --- | --- |
| `hdev/` | Shared code: preprocessing and its inverse, the frozen VAEs |
| `scripts/` | Experiment entry points |
| `tests/` | Unit tests (`PYTHONPATH=. python -m pytest tests`) |
| `slurm/` | Launchers for the cl cluster |
| `results/<run>/` | Small versioned outputs (CSV, Markdown, small figures) with a README per run; large artifacts stay on cl or vpulab |
| `docs/` | Plan, protocol and decisions |

## Dependency on HarmonIt

Data loading, preprocessing and the audit evaluator come from HarmonIt pinned to the ISBI tag,
so the audit's numbers stay reproducible while this code changes:

```bash
pip install "harmonit @ git+https://github.com/muqsitamir/HarmonIt.git@isbi2027-submission"
```

Or point `HARMONIT_SRC` at a checkout's `src/` (the Slurm launchers prepend it to `PYTHONPATH`).
Other requirements: `requirements.txt`.

## Rules

- Every comparison gives all methods the same preprocessing and the same evaluation unit
  (slices, or whole volumes with 2D baselines applied slice by slice).
- Settings are chosen on the validation split; the test split is used once per reported result.
- No AI co-author trailers in commit messages, no tool-named branches, no assistant instruction
  files in the repository.
