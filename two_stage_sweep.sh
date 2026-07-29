#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_sweep.sh
# Runs Stage 2 only, at a fixed width/depth, sweeping LR and weight decay.

SCRIPTS=(
  # "asrnn_double_well_bifurcation_sensitivity.py"
  # "asrnn_henon_heiles_symmetry_sensitivity.py"
  "asrnn_mexican_hat_symmetry_sensitivity.py"
)

# Fixed architecture for Stage 2.
FIXED_WIDTH=32
FIXED_DEPTH=3

# Fixed L1 value for the sweep.
MAIN_L1=0

# LR / weight-decay grid.
LRS=(1e-2 1e-3 1e-4)
WEIGHT_DECAYS=(1e-6 1e-5 1e-4 1e-3 1e-2 1e-1)

mkdir -p outputs/sweeps

# ----------------------------------------------------------------------------
# build_queue: fills the global task_queue array for Stage 2 only.
# ----------------------------------------------------------------------------
build_queue() {
  task_queue=()

  for script in "${SCRIPTS[@]}"; do
    base=$(basename "$script" .py)

    for lr in "${LRS[@]}"; do
      for weight_decay in "${WEIGHT_DECAYS[@]}"; do
        task_queue+=(
          "$script"$'\t'"$base"$'\t'"$MAIN_L1"$'\t'"$lr"$'\t'"$FIXED_WIDTH"$'\t'"$FIXED_DEPTH"$'\t'"$weight_decay"
        )
      done
    done
  done
}

run_job() {
  local stage="$1"
  local gpu="$2"
  local script="$3"
  local base="$4"
  local l1="$5"
  local lr="$6"
  local width="$7"
  local depth="$8"
  local weight_decay="$9"

  local run outdir cfg
  run="${base}/${stage}/l1_${l1}/weight_decay_${weight_decay}/lr_${lr}/w${width}_d${depth}"
  outdir="outputs/sweeps/${run}"
  mkdir -p "$outdir"

  cfg="${outdir}/config.json"

  cat > "$cfg" <<EOF
{
  "learning_rate": $lr,
  "l1_weight": $l1,
  "weight_decay": $weight_decay,
  "kinetic_hidden_dim": $width,
  "kinetic_hidden_layers": $depth,
  "potential_hidden_dim": $width,
  "potential_hidden_layers": $depth
}
EOF

  echo "[GPU ${gpu}] Running ${run}"
  CUDA_VISIBLE_DEVICES="$gpu" python "$script" \
    --config "$cfg" \
    --output-dir "$outdir"
}

# ----------------------------------------------------------------------------
# run_stage: builds the queue for Stage 2 and runs it to completion across
# 4 GPU workers before returning.
# ----------------------------------------------------------------------------
run_stage() {
  local stage="$1"

  build_queue
  job_count=${#task_queue[@]}
  echo "=== STAGE=${stage}: ${job_count} jobs queued. ==="

  next_job_file=$(mktemp)
  lock_file=$(mktemp)
  echo 0 > "$next_job_file"

  get_next_job() {
    local idx

    exec 9>"$lock_file"
    flock 9

    idx=$(<"$next_job_file")
    if (( idx >= job_count )); then
      flock -u 9
      return 1
    fi

    printf '%s\n' "${task_queue[idx]}"
    echo $((idx + 1)) > "$next_job_file"

    flock -u 9
  }

  worker() {
    local gpu="$1"
    local job

    while job="$(get_next_job)"; do
      IFS=$'\t' read -r script base l1 lr width depth weight_decay <<< "$job"
      run_job "$stage" "$gpu" "$script" "$base" "$l1" "$lr" "$width" "$depth" "$weight_decay"
    done

    echo "[GPU ${gpu}] Done"
  }

  for gpu in 0 1 2 3; do
    worker "$gpu" &
  done
  wait

  rm -f "$next_job_file" "$lock_file"
  echo "=== STAGE=${stage} complete. ==="
}

run_stage "main"