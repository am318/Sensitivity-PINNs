#!/bin/bash
# L1-strength sweep for the Mexican-hat hamiltonian architecture.
# Same 3x50 architecture and seed=42 (script defaults) at each l1_weight;
# only l1_weight and output_dir vary. Reruns are safe (each writes to its
# own directory). Run sequentially -- see project convention against
# concurrent training jobs contending for CPU.
set -euo pipefail
cd "$(dirname "$0")"

WEIGHTS=(0.0 3e-5 1e-4 3e-4 1e-3)

for w in "${WEIGHTS[@]}"; do
    echo "=== L1 sweep: l1_weight=${w} ==="
    python asrnn_mexican_hat_symmetry_sensitivity.py \
        --device cpu \
        --l1-weight "${w}" \
        --output-dir "outputs/l1_sweep/l1_${w}"
done

echo "=== L1 sweep complete ==="
