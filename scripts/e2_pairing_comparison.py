"""Run the six-condition E2 pairing pilot on the existing NV-Generate-MR latents.

Reuses the original anatomy-pairing pilots after checking their settings and subject sets.
Four new runs use the same 10,000-step protocol. Training, decoding and FastSurfer run
sequentially on one GPU. Status is written atomically; completed stages are checked before
reuse. No test-split scan is loaded. Large products remain beside the existing remote data.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone

SEG = "aparc.DKTatlas+aseg.deep.mgz"
EXPECTED = {"steps": 10000, "batch": 4, "lr": 1e-4, "label_dropout": 0.1,
            "ema": 0.999, "channels": [64, 128, 256], "val_every": 1000,
            "ode_steps": 20, "seed": 0, "eps": 0.05}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows, expected_ids, label):
    ids = [r["subject_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError(f"{label}: expected {len(expected_ids)} unique source-validation subjects, got {len(ids)}")


def check_config(path, coupling, weight, data):
    cfg = json.loads(Path(path).read_text())
    expected = {**EXPECTED, "coupling": coupling, "latent_dir": str(data / "latents_nvmr")}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"{path}: {key} is {cfg.get(key)!r}, expected {value!r}")
    if cfg.get("anatomy_weight", 0.0) != weight:
        raise ValueError(f"{path}: wrong anatomy weight")
    if weight and (cfg.get("anatomy_net") != str(data / "anatomy_net/nvmr.pt")
                   or cfg.get("labels_dir") != str(data / "anatomy_labels")):
        raise ValueError(f"{path}: different anatomy reader or labels")


def validate_eval(run, expected_ids):
    summary = json.loads((run / "eval/summary.json").read_text())
    if summary["step"] != EXPECTED["steps"]:
        raise ValueError(f"{run}: wrong evaluation checkpoint step")
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{run}: nonfinite evaluation metric {key}")
    validate_rows(read_csv(run / "eval/val_subjects.csv"), expected_ids, str(run))
    return summary


def validate_seg(path, expected_ids):
    rows = [r for r in read_csv(path) if r["model"] == "harmonized"]
    validate_rows(rows, expected_ids, str(path))
    for row in rows:
        for key in ("dice_mean_all_labels", "dice_cortical_parcels", "abs_volume_change_median"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{path}: invalid {key} for {row['subject_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pilot1-seg", type=Path, required=True)
    parser.add_argument("--python", default="/home/mmi/envs/hdev/bin/python")
    parser.add_argument("--fastsurfer-env", default="/home/mmi/envs/fastsurfer")
    parser.add_argument("--fastsurfer-root", default="/home/mmi/tools/FastSurfer")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    code, data, out = args.code_root.resolve(), args.data_root.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # This lock is on local disk because the data directory is an NFS mount without flock.
    lock_path = Path("/tmp") / ("latch-comparison-" + hashlib.sha256(str(out).encode()).hexdigest()[:16] + ".lock")
    lock = lock_path.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = read_csv(data / "latents_nvmr/index.csv")
    eligible = [r for r in meta if r.get("qc_exclude", "False").lower() not in ("true", "1")]
    ids = {r["subject_id"] for r in eligible if r["split"] == "val" and r["site"] != "NYU"}
    if len(ids) != 92 or sum(r["split"] == "train" for r in eligible) != 870:
        raise ValueError("Cached latent split differs from the original pilot protocol")
    sources = {
        ("anatomy", 0): data / "e2/pilot_anatomy_nvmr",
        ("anatomy", 1): data / "e2/pilot_anatomy_term_nvmr",
    }
    pilot_segments = {
        ("anatomy", 0): args.pilot1_seg.resolve(),
        ("anatomy", 1): sources[("anatomy", 1)] / "eval/seg_agreement.csv",
    }
    for key, run in sources.items():
        check_config(run / "config.json", *key, data)
        validate_eval(run, ids)
        validate_seg(pilot_segments[key], ids)
    baseline = [r for r in read_csv(args.pilot1_seg) if r["model"] == "reconstruction"]
    validate_rows(baseline, ids, "autoencoder reconstruction baseline")
    old = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    state = {"status": "preflight", "started_at": old.get("started_at", now()), "updated_at": now(),
             "protocol": EXPECTED, "n_val_source": len(ids), "n_probe_train": 870, "n_probe_val": 110,
             "n_sites_probe": 17, "vae": "nvmr", "test_used": False,
             "code_root": str(code), "data_root": str(data), "conditions": old.get("conditions", {}),
             "note": "Single-seed 10,000-step pilot ablation; cached sampled latents held fixed across conditions."}
    tracked = ["hdev/flow.py", "hdev/anatomy.py", "hdev/vae.py", "scripts/e2_train_flow.py",
               "scripts/e2_eval_flow.py", "scripts/seg_agreement.py", "slurm/fastsurfer_seg.sbatch",
               "results/e0_data/brain_index.csv", "results/e0_data/qc_exclusions.csv"]
    commit = ((code / "COMMIT").read_text().strip() if (code / "COMMIT").exists()
              else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip())
    provenance = {"created_at": now(), "commit": commit,
                  "source_sha256": {p: sha256(code / p) for p in tracked},
                  "latent_index_sha256": sha256(data / "latents_nvmr/index.csv"),
                  "anatomy_reader_sha256": sha256(data / "anatomy_net/nvmr.pt"),
                  "vae_weights_sha256": sha256(data / "models/nvidia__NV-Generate-MR/models/autoencoder_v2.pt")}
    if (out / "provenance.json").exists():
        previous = json.loads((out / "provenance.json").read_text())
        for key in ("source_sha256", "latent_index_sha256", "anatomy_reader_sha256", "vae_weights_sha256"):
            if previous[key] != provenance[key]:
                raise ValueError(f"Provenance changed while comparison was running: {key}")
    else:
        write_json(out / "provenance.json", provenance)
    write_json(out / "status.json", state)
    if args.check_only:
        print("Preflight passed: two complete pilots, reconstruction baseline, 92 shared validation subjects", flush=True)
        return
    env = {**os.environ, "PYTHONPATH": str(code), "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    records = []

    def stage(condition, name, command, log_path, extra_env=None):
        state.update(status="running", current_condition=condition, current_stage=name, updated_at=now())
        state["conditions"].setdefault(condition, {}).update(stage=name, updated_at=now())
        write_json(out / "status.json", state)
        print(f"{now()} {condition} {name}", flush=True)
        with log_path.open("a") as log:
            log.write(f"\n{now()} {command!r}\n")
            log.flush()
            subprocess.run(command, cwd=code, env={**env, **(extra_env or {})}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        for coupling in ("anatomy", "random", "latent"):
            for weight in (0, 1):
                key = (coupling, weight)
                name = f"{coupling}_weight{weight}"
                reused = key in sources
                run = sources[key] if reused else out / name
                run.mkdir(parents=True, exist_ok=True)
                if not reused:
                    if not (run / "training_complete.json").exists():
                        if (run / "config.json").exists():
                            check_config(run / "config.json", coupling, weight, data)
                        command = [args.python, "scripts/e2_train_flow.py", "--latent-dir", str(data / "latents_nvmr"),
                                   "--brain-index", "results/e0_data/brain_index.csv", "--coupling", coupling,
                                   "--batch", "4", "--steps", "10000", "--val-every", "1000", "--seed", "0",
                                   "--out-dir", str(run)]
                        if weight:
                            command += ["--anatomy-net", str(data / "anatomy_net/nvmr.pt"), "--labels-dir",
                                        str(data / "anatomy_labels"), "--anatomy-weight", "1.0"]
                        stage(name, "train", command, run / "train.stdout.log")
                        check_config(run / "config.json", coupling, weight, data)
                        logged = read_csv(run / "log.csv")
                        if not logged or int(logged[-1]["step"]) != 10000:
                            raise ValueError(f"{name}: training did not reach 10,000 steps")
                        write_json(run / "training_complete.json", {"completed_at": now()})
                    if not (run / "eval/summary.json").exists():
                        command = [args.python, "scripts/e2_eval_flow.py", "--checkpoint", str(run / "checkpoints/latest.pt"),
                                   "--latent-dir", str(data / "latents_nvmr"), "--ref-dir", str(data / "reference_1mm"),
                                   "--vae", "nvmr", "--vae-ckpt", str(data / "models/nvidia__NV-Generate-MR/models/autoencoder_v2.pt"),
                                   "--out-dir", str(run / "eval"), "--save-nifti"]
                        stage(name, "evaluate", command, run / "eval.stdout.log")
                    summary = validate_eval(run, ids)
                    nifti = run / "eval/nifti/harmonized"
                    if {p.name.removesuffix(".nii.gz") for p in nifti.glob("*.nii.gz")} != ids:
                        raise ValueError(f"{name}: harmonized NIfTIs do not match the 92 validation subjects")
                    seg_root = run / "eval/fastsurfer"
                    (seg_root / "input").mkdir(parents=True, exist_ok=True)
                    for sid in sorted(ids):
                        source = data / "fastsurfer/reference_1mm" / sid
                        if not (source / SEG).exists():
                            raise FileNotFoundError(source / SEG)
                        target = seg_root / "input" / sid
                        if not target.is_symlink():
                            target.symlink_to(source)
                    seg_path = run / "eval/seg_agreement.csv"
                    if not seg_path.exists():
                        state.update(status="running", current_condition=name, current_stage="fastsurfer", updated_at=now())
                        state["conditions"].setdefault(name, {}).update(stage="fastsurfer", updated_at=now())
                        write_json(out / "status.json", state)
                        print(f"{now()} {name} fastsurfer", flush=True)
                        workers = []
                        with (run / "fastsurfer.stdout.log").open("a") as log:
                            for worker in (0, 1):
                                worker_env = {**env, "INPUT_DIR": str(nifti), "SEG_DIR": str(seg_root),
                                              "FS_DIR": args.fastsurfer_root, "FS_ENV": args.fastsurfer_env,
                                              "SLURM_ARRAY_TASK_COUNT": "2", "SLURM_ARRAY_TASK_ID": str(worker),
                                              "SLURM_ARRAY_JOB_ID": "latch-comparison"}
                                workers.append(subprocess.Popen(["bash", "slurm/fastsurfer_seg.sbatch"], cwd=code,
                                                                env=worker_env, stdout=log, stderr=subprocess.STDOUT))
                            exits = [p.wait() for p in workers]
                        if any(exits):
                            raise RuntimeError(f"{name}: FastSurfer launcher exit codes {exits}")
                        missing = [sid for sid in ids if not (seg_root / "harmonized" / sid / SEG).is_file()]
                        if missing:
                            raise RuntimeError(f"{name}: {len(missing)} missing segmentations: {missing[:5]}")
                        stage(name, "segmentation-agreement", [str(Path(args.fastsurfer_env) / "bin/python"),
                              "scripts/seg_agreement.py", "--seg-dir", str(seg_root), "--out", str(seg_path)],
                              run / "agreement.stdout.log")
                else:
                    summary = validate_eval(run, ids)
                    seg_path = pilot_segments[key]
                rows = validate_seg(seg_path, ids)
                dest = out / "small_results" / name
                dest.mkdir(parents=True, exist_ok=True)
                for source, target_name in ((run / "config.json", "config.json"), (run / "eval/summary.json", "summary.json"),
                                            (run / "eval/val_subjects.csv", "val_subjects.csv"), (seg_path, "seg_agreement.csv")):
                    (dest / target_name).write_bytes(source.read_bytes())
                rec = {"condition": name, "coupling": coupling, "anatomy_weight": weight, "reused_pilot": reused,
                       "n_val_source": len(rows), "dice": statistics.median(float(r["dice_mean_all_labels"]) for r in rows),
                       "cortical_dice": statistics.median(float(r["dice_cortical_parcels"]) for r in rows),
                       "volume_change": statistics.median(float(r["abs_volume_change_median"]) for r in rows),
                       "probe_output_trained": summary["probe_output_trained"], "probe_frozen": summary["probe_frozen"],
                       "distance_closed": summary["share_of_distance_closed"], "psnr": summary["median_psnr_harmonized"]}
                records.append(rec)
                state["conditions"][name] = {"stage": "complete", "completed_at": now(), "reused_pilot": reused, "run": str(run)}
                write_json(out / "status.json", state)
        original_summary = validate_eval(sources[("anatomy", 0)], ids)
        baseline_metrics = {"n": len(baseline), "dice": statistics.median(float(r["dice_mean_all_labels"]) for r in baseline),
                            "cortical_dice": statistics.median(float(r["dice_cortical_parcels"]) for r in baseline),
                            "volume_change": statistics.median(float(r["abs_volume_change_median"]) for r in baseline),
                            "psnr": original_summary["median_psnr_reconstruction"], "probe_output_trained": 1.0,
                            "distance_closed": 1 - original_summary["median_w1_nyu_reconstruction"] / original_summary["median_w1_nyu_input"]}
        completed_state = {**state, "status": "complete", "completed_at": now(), "updated_at": now(),
                           "current_stage": "complete"}
        write_json(out / "comparison.json", {"protocol": completed_state, "baseline": baseline_metrics, "conditions": records})
        with (out / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        state = completed_state
        write_json(out / "status.json", state)
        print(f"{now()} COMPARISON COMPLETE", flush=True)
    except Exception as exc:
        state.update(status="failed", updated_at=now(), error=str(exc), traceback=traceback.format_exc())
        write_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
