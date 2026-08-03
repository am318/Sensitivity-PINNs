"""Regenerate every plot for a completed asrnn_mexican_hat_symmetry_sensitivity.py run from its
saved outputs -- no retraining needed.

Usage:
    python3 replot_mexican_hat.py outputs/asrnn_mexican_hat_symmetry_1e-5

Everything a plot needs is already written to disk by the original run:
``config.json`` (to rebuild the Config and the model architecture), ``all_checkpoint_results.json``
(the exact per-checkpoint ``all_results`` list every plot_* function consumes),
``training_history.npz`` (for the loss-curve plot), and ``final_model.pt`` (the trained weights,
needed only by the two model-dependent plots: learned force field / potential). Nothing here
re-runs training or re-generates data.

To change how the plots look, edit PLOT_STYLE below and rerun this script -- it overrides
matplotlib's rcParams *after* importing the analysis script (whose own module-level
``set_paper_style()`` call already set the defaults you're seeing now), so every plot_*
function picks up the new values without any other code changes. To turn specific plots
on/off, comment lines in/out of the call list at the bottom of ``main()``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Edit these, then rerun this script -- no need to touch the main experiment
# script or retrain anything. Anything not listed here keeps whatever
# experiment_common.set_paper_style() set as the default.
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}

import matplotlib.pyplot as plt  # noqa: E402

import asrnn_mexican_hat_symmetry_sensitivity as m  # noqa: E402
from experiment_common import parameter_layout, plot_training_history, select_device, select_dtype  # noqa: E402

plt.rcParams.update(PLOT_STYLE)


def load_run(output_dir: Path):
    saved_fields = json.loads((output_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    unknown = sorted(set(saved_fields) - known)
    if unknown:
        print(f"Note: ignoring config fields no longer recognised by Config: {unknown}")
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})

    all_results = json.loads((output_dir / "all_checkpoint_results.json").read_text())
    history = dict(np.load(output_dir / "training_history.npz"))

    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    model, _ = m.build_model(cfg, device, dtype)
    model.load_state_dict(torch.load(output_dir / "final_model.pt", map_location=device))
    flat_names, parameter_slices = parameter_layout(model)

    return cfg, all_results, history, model, flat_names, parameter_slices, device, dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Existing run's output directory to replot in place.")
    args = parser.parse_args()
    output_dir = args.output_dir

    cfg, all_results, history, model, flat_names, parameter_slices, device, dtype = load_run(output_dir)
    print(f"Loaded {len(all_results)} checkpoints from {output_dir} (architecture={cfg.architecture}).")

    plot_training_history(history, output_dir)
    m.plot_summary(all_results, output_dir)
    m.plot_equivariance_by_module(all_results, output_dir)
    m.plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    m.plot_magnitude_diagnostics(all_results, parameter_slices, output_dir)
    m.plot_module_attribution(all_results, parameter_slices, output_dir)
    m.plot_attribution_scatter(all_results, parameter_slices, output_dir)
    m.plot_module_symmetry_comparison(all_results, parameter_slices, output_dir)
    m.plot_symmetry_attribution_scatter(all_results, parameter_slices, output_dir)
    m.plot_learned_force_field(model, cfg, device, dtype, output_dir)
    m.plot_learned_potential(model, cfg, device, dtype, output_dir)

    print(f"Replotted everything in {output_dir}")


if __name__ == "__main__":
    main()
