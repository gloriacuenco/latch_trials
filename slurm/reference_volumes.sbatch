#!/bin/bash
#SBATCH --job-name=hdev_reference_volumes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E0 step 1: every ABIDE I scan in the 1 mm reference space, plus index.csv (scripts/make_reference_volumes.py).
set -euo pipefail
CODE_DIR="${CODE_DIR:-$HOME/repos/latch}"
PYTHON="${PYTHON:-$HOME/envs/trdp2_ae/bin/python}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR (abide_manifest.csv, splits.json, ABIDE/)}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
git -C "${CODE_DIR}" rev-parse HEAD 2>/dev/null || cat "${CODE_DIR}/COMMIT" 2>/dev/null || true
"${PYTHON}" scripts/make_reference_volumes.py --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}" --workers "${SLURM_CPUS_PER_TASK:-8}"
