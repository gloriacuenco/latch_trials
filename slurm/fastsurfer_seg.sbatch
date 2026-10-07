#!/bin/bash
#SBATCH --job-name=trdp2_fs_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# FastSurferCNN (v2.5.4, segmentation only). Either INPUT_DIR (every *.nii.gz in it; outputs under
# SEG_DIR/<basename of INPUT_DIR>/<subject>) or RUN_DIR (volumes saved by scripts/ae_fidelity.py
# --save-nifti, laid out as <part>/nifti/<model>/<subject>.nii.gz). Array task i takes every N-th
# volume; volumes that already have a segmentation are skipped.
set -euo pipefail
if [ -n "${INPUT_DIR:-}" ]; then
  SEG_DIR="${SEG_DIR:?Set SEG_DIR}"
else
  : "${RUN_DIR:?Set INPUT_DIR or RUN_DIR}"
  SEG_DIR="${SEG_DIR:-${RUN_DIR}/fastsurfer}"
fi
FILE_LIST="${SEG_DIR}/files_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.txt"
mkdir -p "${SEG_DIR}"
if [ -n "${INPUT_DIR:-}" ]; then
  find "${INPUT_DIR}" -maxdepth 1 -name '*.nii.gz' | sort > "${FILE_LIST}"
else
  # Inputs are saved by both parts of the run; segment the copy under hcld/ only.
  find "${RUN_DIR}" -path '*/nifti/*' -name '*.nii.gz' ! -path '*/maisi/nifti/input/*' | sort > "${FILE_LIST}"
fi
# FastSurfer install and environment: cl defaults; on vpulab FS_DIR=/home/mmi/tools/FastSurfer FS_ENV=/home/mmi/envs/fastsurfer.
FS_DIR="${FS_DIR:-/home/muqsitamir/trdp2/tools/FastSurfer}"
FS_ENV="${FS_ENV:-/home/muqsitamir/envs/fastsurfer}"
N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
source "${FS_ENV}/bin/activate"
export PYTHONPATH="${FS_DIR}"
cd "${FS_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader
awk -v n="${N}" -v i="${I}" '(NR - 1) % n == i' "${FILE_LIST}" | while read -r f; do
  model=$(basename "$(dirname "${f}")")
  sid=$(basename "${f}" .nii.gz)
  out="${SEG_DIR}/${model}"
  seg="${out}/${sid}/aparc.DKTatlas+aseg.deep.mgz"
  mkdir -p "${out}/${sid}"
  [ -s "${seg}" ] && continue
  python FastSurferCNN/run_prediction.py --t1 "${f}" --sd "${out}" --sid "${sid}" \
    --asegdkt_segfile "${seg}" --conformed_name "${out}/${sid}/orig.mgz" \
    --device cuda --viewagg_device cuda --threads 8 --vox_size 1 < /dev/null > "${out}_${sid}.log" 2>&1 \
    && echo "done ${model} ${sid}" || echo "FAILED ${model} ${sid}"
done
