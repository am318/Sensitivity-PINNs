#!/usr/bin/env bash
set -euo pipefail

# Usage: STAGE=arch_pilot ./run_sweep.sh   (coarse width/depth search, reg fixed at defaults)
#        STAGE=main       ./run_sweep.sh   (default) (full L1/LR/weight-decay grid at
#                                            representative "small" and "large" capacities)
STAGE="${STAGE:-main}"

SCRIPTS=(
"asrnn_mexican_hat_symmetry_sensitivity.py"
)

# --- Stage 1: architecture pilot -------------------------------------------
# Coarse width/depth search at fixed, "no regularization" defaults, to pick
# representative capacities before spending budget on the full reg/LR grid.
PILOT_LR=1e-3
PILOT_L1=1e-5
PILOT_WEIGHT_DECAY=1e-5
WIDTHS=(8 16 32 64)
DEPTHS=(1 2 3 4)

# --- Stage 2: main regularization/LR grid -----------------------------------
# Run the full L1/LR/weight-decay grid only at a small number of representative
# capacities (chosen from the Stage 1 pilot), rather than the full width/depth
# cross product. Edit these once the pilot results are in.
SIZES=(
"16 2"   # representative "small" capacity: width depth
"64 4"   # representative "large" capacity: width depth
)

# Trimmed grids: dropped values that are empirically indistinguishable from 0
# (e.g. weight_decay=1e-7, l1=1e-5/1e-6) and log-spaced the rest so each point
# is expected to actually change behavior.
L1S=(0 1e-1 1e-2 1e-3 1e-4)
LRS=(1e-2 1e-3 1e-4)
WEIGHT_DECAYS=(0 1e-4 1e-3 1e-2)

mkdir -p outputs/sweeps

# Build the job queue for the selected stage.
task_queue=()
for script in "${SCRIPTS[@]}"; do
base=$(basename "$script" .py)

if [[ "$STAGE" == "arch_pilot" ]]; then
for width in "${WIDTHS[@]}"; do
for depth in "${DEPTHS[@]}"; do
task_queue+=("$script"$'\t'"$base"$'\t'"$PILOT_L1"$'\t'"$PILOT_LR"$'\t'"$width"$'\t'"$depth"$'\t'"$PILOT_WEIGHT_DECAY")
done
done

elif [[ "$STAGE" == "main" ]]; then
for size in "${SIZES[@]}"; do
read -r width depth <<< "$size"
for l1 in "${L1S[@]}"; do
for lr in "${LRS[@]}"; do
for weight_decay in "${WEIGHT_DECAYS[@]}"; do
task_queue+=("$script"$'\t'"$base"$'\t'"$l1"$'\t'"$lr"$'\t'"$width"$'\t'"$depth"$'\t'"$weight_decay")
done
done
done
done

else
echo "Unknown STAGE: $STAGE (expected 'arch_pilot' or 'main')" >&2
exit 1
fi
done

job_count=${#task_queue[@]}
echo "STAGE=${STAGE}: ${job_count} jobs queued."

next_job_file=$(mktemp)
lock_file=$(mktemp)
trap 'rm -f "$next_job_file" "$lock_file"' EXIT
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

run_job() {
local gpu="$1"
local script="$2"
local base="$3"
local l1="$4"
local lr="$5"
local width="$6"
local depth="$7"
local weight_decay="$8"

local run outdir cfg
run="${base}/${STAGE}/l1_${l1}/weight_decay_${weight_decay}/lr_${lr}/w${width}_d${depth}"
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

worker() {
local gpu="$1"
local job

while job="$(get_next_job)"; do
IFS=$'\t' read -r script base l1 lr width depth weight_decay <<< "$job"
run_job "$gpu" "$script" "$base" "$l1" "$lr" "$width" "$depth" "$weight_decay"
done

echo "[GPU ${gpu}] Done"
}

# Start 4 GPU workers.
for gpu in 0 1 2 3; do
worker "$gpu" &
done

wait