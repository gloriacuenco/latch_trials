# harmonit-dev

Private development of the HarmonIt harmonization method (TRDP2). The public
[HarmonIt](https://github.com/muqsitamir/HarmonIt) repository holds the ISBI 2027 evaluation audit
and stays frozen at the tag `isbi2027-submission`. Finished, cleaned code from here is merged
into HarmonIt as a tagged release when the method paper is submitted; until then this repository
stays private.

Plan and decisions: [docs/PLAN.md](docs/PLAN.md).

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | Experiment code |
| `slurm/` | Launchers for the cl cluster |
| `results/<run>/` | Small versioned outputs (CSV, Markdown, small figures) with a README per run; large artifacts stay on cl or vpulab |
| `docs/` | Plan, protocol and decisions |

## Dependency on HarmonIt

Data loading, preprocessing and the audit evaluator come from HarmonIt pinned to the ISBI tag,
so the audit's numbers stay reproducible while this code changes:

```bash
pip install "harmonit @ git+ssh://git@github.com/muqsitamir/HarmonIt.git@isbi2027-submission"
```

Or point `HARMONIT_SRC` at a checkout's `src/` (the Slurm launchers prepend it to `PYTHONPATH`).
Other requirements: `requirements.txt`.

## Rules

- Every comparison gives all methods the same preprocessing and the same evaluation unit
  (slices, or whole volumes with 2D baselines applied slice by slice).
- Settings are chosen on the validation split; the test split is used once per reported result.
- No AI co-author trailers in commit messages, no tool-named branches, no assistant instruction
  files in the repository.
