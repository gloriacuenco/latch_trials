#!/bin/bash
#SBATCH --job-name=latch_assemblynet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# AssemblyNet (volbrain/assemblynet:1.0.0 as a Singularity image) on every volume listed in TASKS
# ("<set> <path>" per line). Array task i takes every N-th line; volumes already segmented are
# skipped. Only the native-space structures and tissues are kept: OUT_DIR/<set>/native_*_<subject>.nii.gz.
set -euo pipefail
: "${TASKS:?Set TASKS}" "${OUT_DIR:?Set OUT_DIR}"
SIF="${SIF:-$HOME/trdp2/tools/assemblynet_1.0.0.sif}"
N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/assemblynet_XXXXXX")
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${WORK}/in" "${WORK}/out" "${WORK}/tmp"
n=0
while read -r set path; do
  sid=$(basename "${path}" .nii.gz)
  [ -s "${OUT_DIR}/${set}/native_tissues_${sid}.nii.gz" ] && continue
  mkdir -p "${WORK}/in/${set}"
  cp "${path}" "${WORK}/in/${set}/"
  n=$((n + 1))
done < <(awk -v n="${N}" -v i="${I}" '(NR - 1) % n == i' "${TASKS}")
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true): ${n} volumes"
[ "${n}" -eq 0 ] && exit 0
singularity run --nv -B "${WORK}/tmp:/tmp" -B "${WORK}/in:/data" -B "${WORK}/out:/data_out" "${SIF}" \
  -recursive -no-pdf-report -batch-size 3 /data/ /data_out/
for d in "${WORK}"/out/*/; do
  set=$(basename "${d}")
  mkdir -p "${OUT_DIR}/${set}"
  cp "${d}"native_tissues_* "${d}"native_str*ctures_* "${OUT_DIR}/${set}/"
done
echo "done $(date)"
