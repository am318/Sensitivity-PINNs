#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep centered on the defaults in
# asrnn_mexican_hat_symmetry_sensitivity.py:
#   learning_rate = 1e-3
#   weight_decay  = 0.0
#   l1_weight     = 1e-5
#
# This script sweeps only those three knobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SCRIPT="${SCRIPT_DIR}/asrnn_mexican_hat_symmetry_sensitivity.py"
OUTPUT_ROOT="${SCRIPT_DIR}/outputs/sweeps/mexican_hat_centered"

# Centered, log-spaced grid around the defaults.
LRS=(5e-4 1e-3 5e-3)
WEIGHT_DECAYS=(0 1e-7 1e-6 1e-5 1e-4)
L1S=(1e-6 1e-5 1e-4)

# Use up to four GPUs by default; edit this list if needed.
GPU_IDS=(0 1 2 3)

mkdir -p "$OUTPUT_ROOT"

# Build the job queue as tab-separated records:
# script <tab> lr <tab> weight_decay <tab> l1
job_queue=()
for lr in "${LRS[@]}"; do
  for weight_decay in "${WEIGHT_DECAYS[@]}"; do
    for l1 in "${L1S[@]}"; do
      job_queue+=("${MODEL_SCRIPT}"$'\t'"${lr}"$'\t'"${weight_decay}"$'\t'"${l1}")
    done
  done
done

job_count=${#job_queue[@]}
next_job_file=$(mktemp)
lock_file=$(mktemp)
trap 'rm -f "$next_job_file" "$lock_file"' EXIT
printf '0\n' > "$next_job_file"

get_next_job() {
  local idx

  exec 9>"$lock_file"
  flock 9

  idx=$(<"$next_job_file")
  if (( idx >= job_count )); then
    flock -u 9
    return 1
  fi

  printf '%s\n' "${job_queue[idx]}"
  printf '%s\n' "$((idx + 1))" > "$next_job_file"

  flock -u 9
}

run_job() {
  local gpu="$1"
  local script="$2"
  local lr="$3"
  local weight_decay="$4"
  local l1="$5"

  local run_dir cfg
  run_dir="lr_${lr}/weight_decay_${weight_decay}/l1_${l1}"
  cfg="${OUTPUT_ROOT}/${run_dir}/config.json"
  mkdir -p "$(dirname "$cfg")"

  cat > "$cfg" <<EOF_CFG
{
  "learning_rate": ${lr},
  "weight_decay": ${weight_decay},
  "l1_weight": ${l1}
}
EOF_CFG

  echo "[GPU ${gpu}] lr=${lr} wd=${weight_decay} l1=${l1}"
  CUDA_VISIBLE_DEVICES="$gpu" python "$script" \
    --config "$cfg" \
    --output-dir "${OUTPUT_ROOT}/${run_dir}"
}

worker() {
  local gpu="$1"
  local job

  while job="$(get_next_job)"; do
    IFS=$'\t' read -r script lr weight_decay l1 <<< "$job"
    run_job "$gpu" "$script" "$lr" "$weight_decay" "$l1"
  done

  echo "[GPU ${gpu}] done"
}

for gpu in "${GPU_IDS[@]}"; do
  worker "$gpu" &
done

wait