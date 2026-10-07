#!/bin/bash
#SBATCH --job-name=trdp2_ae_fidelity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Feasibility gate: reconstruction fidelity of pretrained 3D autoencoders on the
# ABIDE I validation split (scripts/ae_fidelity.py). Needs HarmonIt (tag isbi2027-submission)
# importable: pip-installed, or HARMONIT_SRC pointing at its src/. The HCLD autoencoder needs
# the hcld conda env (MONAI 1.3 + generative); MAISI/NV-Generate-MR need MONAI 1.5.
set -euo pipefail
CODE_DIR="${CODE_DIR:-/home/muqsitamir/repos/latch}"
DATA_DIR="${DATA_DIR:-/home/muqsitamir/repos/HarmonIt/data}"
MODEL_DIR="${MODEL_DIR:-/home/muqsitamir/trdp2/models}"
HCLD_CKPT="${HCLD_CKPT:-/home/muqsitamir/repos/HarmonIt/outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR}"
LIMIT="${LIMIT:-}"
EXTRA=()
[ -n "${LIMIT}" ] && EXTRA=(--limit "${LIMIT}")

cd "${CODE_DIR}"
if [ -n "${HARMONIT_SRC:-}" ]; then export PYTHONPATH="${HARMONIT_SRC}:${PYTHONPATH:-}"; fi
mkdir -p "${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum scripts/ae_fidelity.py "${HCLD_CKPT}" \
  "${MODEL_DIR}/MONAI__maisi_ct_generative/models/autoencoder.pt" \
  "${MODEL_DIR}/nvidia__NV-Generate-MR/models/autoencoder_v2.pt" | tee "${OUT_DIR}/sha256.txt"

/home/muqsitamir/miniconda3/envs/hcld/bin/python scripts/ae_fidelity.py \
  --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}/hcld" --models hcld hcld_resize --save-nifti ${EXTRA[@]+"${EXTRA[@]}"} \
  --hcld-root /home/muqsitamir/repos/HCLD --hcld-ckpt "${HCLD_CKPT}"

/home/muqsitamir/envs/trdp2_ae/bin/python scripts/ae_fidelity.py \
  --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}/maisi" --models maisi nvmr blur05 blur10 --save-nifti ${EXTRA[@]+"${EXTRA[@]}"} \
  --maisi-ckpt "${MODEL_DIR}/MONAI__maisi_ct_generative/models/autoencoder.pt" \
  --nvmr-ckpt "${MODEL_DIR}/nvidia__NV-Generate-MR/models/autoencoder_v2.pt"
