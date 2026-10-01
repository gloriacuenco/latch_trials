#!/bin/bash
#SBATCH --job-name=hdev_brain_index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E0 step 2: brain boxes and anatomy features from FastSurfer (scripts/brain_index.py).
set -euo pipefail
CODE_DIR="${CODE_DIR:-$HOME/repos/harmonit-dev}"
PYTHON="${PYTHON:-$HOME/envs/trdp2_ae/bin/python}"
D="${DATA_ROOT:-$HOME/trdp2/data}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cat "${CODE_DIR}/COMMIT" 2>/dev/null || true
"${PYTHON}" scripts/brain_index.py --index "$D/reference_1mm/index.csv" --ref-dir "$D/reference_1mm" \
  --seg-dir "$D/fastsurfer/reference_1mm" --phenotypes "$D/Phenotypic_V1_0b.csv" \
  --out "$D/brain_index.csv" --workers "${SLURM_CPUS_PER_TASK:-8}"
