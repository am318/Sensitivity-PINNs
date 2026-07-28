#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=(
    # "asrnn_double_well_bifurcation_sensitivity.py"
    # "asrnn_henon_heiles_symmetry_sensitivity.py"
    "asrnn_mexican_hat_symmetry_sensitivity.py"
)

ARCHITECTURES=(direct_mlp hamiltonian)
L1S=(0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
AUGMENT=(true false)
LRS=(1e-1 1e-2 1e-3 1e-4)
WEIGHT_DECAYS=(0 1e-7 1e-5 1e-3)

WIDTHS=(8 16 32 64)
DEPTHS=(1 2 3 4)

mkdir -p outputs/sweeps

# Build the full queue of jobs.
task_queue=()
for script in "${SCRIPTS[@]}"; do
    base=$(basename "$script" .py)

    for arch in "${ARCHITECTURES[@]}"; do
        for l1 in "${L1S[@]}"; do
            for aug in "${AUGMENT[@]}"; do
                for lr in "${LRS[@]}"; do
                    for width in "${WIDTHS[@]}"; do
                        for depth in "${DEPTHS[@]}"; do
                            for weight_decay in "${WEIGHT_DECAYS[@]}"; do
                                task_queue+=("$script"$'\t'"$base"$'\t'"$arch"$'\t'"$l1"$'\t'"$aug"$'\t'"$lr"$'\t'"$width"$'\t'"$depth"$'\t'"$weight_decay")
                            done
                        done
                    done
                done
            done
        done
    done
done

job_count=${#task_queue[@]}
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
    local arch="$4"
    local l1="$5"
    local aug="$6"
    local lr="$7"
    local width="$8"
    local depth="$9"
    local weight_decay="${10}"

    local run outdir cfg
    run="${base}/${arch}/l1_${l1}/weight_decay_${weight_decay}/aug_${aug}/lr_${lr}/w${width}_d${depth}"
    outdir="outputs/sweeps/${run}"
    mkdir -p "$outdir"

    cfg="${outdir}/config.json"

    if [[ "$arch" == "direct_mlp" ]]; then
        cat > "$cfg" <<EOF
{
  "architecture": "$arch",
  "learning_rate": $lr,
  "l1_weight": $l1,
  "weight_decay": $weight_decay,
  "augment_dataset": $aug,
  "direct_mlp_hidden_dim": $width,
  "direct_mlp_hidden_layers": $depth
}
EOF
    else
        cat > "$cfg" <<EOF
{
  "architecture": "$arch",
  "learning_rate": $lr,
  "l1_weight": $l1,
  "weight_decay": $weight_decay,
  "augment_dataset": $aug,
  "kinetic_hidden_dim": $width,
  "kinetic_hidden_layers": $depth,
  "potential_hidden_dim": $width,
  "potential_hidden_layers": $depth
}
EOF
    fi

    echo "[GPU ${gpu}] Running ${run}"
    CUDA_VISIBLE_DEVICES="$gpu" python "$script" \
        --config "$cfg" \
        --output-dir "$outdir"
}

worker() {
    local gpu="$1"
    local job

    while job="$(get_next_job)"; do
        IFS=$'\t' read -r script base arch l1 aug lr width depth weight_decay <<< "$job"
        run_job "$gpu" "$script" "$base" "$arch" "$l1" "$aug" "$lr" "$width" "$depth" "$weight_decay"
    done

    echo "[GPU ${gpu}] Done"
}

# Start 4 GPU workers.
for gpu in 0 1 2 3; do
    worker "$gpu" &
done

wait
