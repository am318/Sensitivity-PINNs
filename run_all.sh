#!/usr/bin/env bash

set -euo pipefail

# Activate virtual environment
source venv/bin/activate

echo "Running double well experiment..."
python asrnn_double_well_bifurcation_sensitivity.py

echo "Running Henon-Heiles experiment..."
python asrnn_henon_heiles_symmetry_sensitivity.py

echo "Running Mexican hat experiment..."
python asrnn_mexican_hat_symmetry_sensitivity.py

echo "All experiments completed."