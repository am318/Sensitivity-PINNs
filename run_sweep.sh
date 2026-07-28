#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=(
    "asrnn_double_well_bifurcation_sensitivity.py"
    "asrnn_henon_heiles_symmetry_sensitivity.py"
    "asrnn_mexican_hat_symmetry_sensitivity.py"
)

ARCHITECTURES=(direct_mlp hamiltonian)
L1S=(0 1e-3 1e-5)
AUGMENT=(true false)
LRS=(1e-2 1e-3 1e-4)

WIDTHS=(16 32 64)
DEPTHS=(1 2 3 4)

mkdir -p outputs/sweeps

for script in "${SCRIPTS[@]}"; do
    base=$(basename "$script" .py)

    for arch in "${ARCHITECTURES[@]}"; do
        for l1 in "${L1S[@]}"; do
            for aug in "${AUGMENT[@]}"; do
                for lr in "${LRS[@]}"; do
                    for width in "${WIDTHS[@]}"; do
                        for depth in "${DEPTHS[@]}"; do

                            run="${base}/${arch}/l1_${l1}/aug_${aug}/lr_${lr}/w${width}_d${depth}"
                            outdir="outputs/sweeps/${run}"
                            mkdir -p "$outdir"

                            cfg="${outdir}/config.json"

                            if [[ "$arch" == "direct_mlp" ]]; then
                                cat > "$cfg" <<EOF
{
  "architecture": "$arch",
  "learning_rate": $lr,
  "l1_regularization": $l1,
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
  "l1_regularization": $l1,
  "augment_dataset": $aug,
  "kinetic_hidden_dim": $width,
  "kinetic_hidden_layers": $depth,
  "potential_hidden_dim": $width,
  "potential_hidden_layers": $depth
}
EOF
                            fi

                            echo "Running $run"

                            python "$script" \
                                --config "$cfg" \
                                --output-dir "$outdir"

                        done
                    done
                done
            done
        done
    done
done