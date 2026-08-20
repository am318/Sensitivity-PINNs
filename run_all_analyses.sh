#!/bin/bash
# Run every observational analysis on a finished causal_symmetry_control run directory.
# Usage: ./run_all_analyses.sh outputs/causal_symmetry_control_hamiltonian_augmented [...]
set -u
PY=venv/bin/python
for dir in "$@"; do
  npz="$dir/analysis_inputs.npz"
  [ -f "$npz" ] || { echo "SKIP $dir (no analysis_inputs.npz)"; continue; }
  echo; echo "##################### $dir #####################"
  echo; echo "----- conditioning decomposition -----"
  $PY conditioning_decomposition.py "$npz"
  echo; echo "----- gauge dependence -----"
  $PY gauge_dependence.py "$npz"
  echo; echo "----- reparametrisation covariance -----"
  $PY reparametrisation_covariance.py "$npz"
  echo; echo "----- identifiability under probe resampling -----"
  $PY attribution_stability.py "$npz" --draws 40
  if [ -f "$dir/model.pt" ]; then
    echo; echo "----- probe grid refinement -----"
    $PY probe_grid_sweep.py "$dir"
  fi
done
