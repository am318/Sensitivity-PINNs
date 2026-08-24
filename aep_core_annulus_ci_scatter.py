"""Per-parameter attribution c_i at random init vs. after training, in the ASRNN scripts' style.

Reuses ``experiment_common.plot_ei_initial_vs_final`` -- the same figure the
double-well / Henon-Heiles / Mexican-hat experiments produce for the
sensitivity-equivariance defect E_i, here fed the attribution coefficients
``c_i`` (the ``log_scale=True`` path that function documents). One point per
parameter, coloured by module: on the ``y = x`` line means training left that
parameter's role in realising the symmetry direction untouched, above means
training gave it a larger role, below a smaller one.

Both targets from ``aep_core_annulus_tangent.py`` are plotted, for the wedge
dataset (90-degree outer arc, so the training set is *not* rotation-symmetric)
and the fully invariant task (full annulus). Nothing here is regularised: the
underlying runs optimise plain BCE with no weight decay, no L1, no augmentation
and no projection penalty, so any structure in these scatters is what
unconstrained training did on its own.

Reads the coefficients ``aep_core_annulus_tangent.py`` already wrote, so it
never recomputes a Jacobian.

Usage:
    python aep_core_annulus_ci_scatter.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from aep_core_annulus_mlp import SimpleMLP
from experiment_common import (
    parameter_layout,
    plot_ei_initial_vs_final,
    plot_magnitude_vs_quantity,
)
from sensitivity_tools import parameter_magnitude_ci_correlation


@dataclass
class Config:
    coefficients: str = "outputs/aep_core_annulus_tangent/attribution_coefficients.npz"
    output_dir: str = "outputs/aep_core_annulus_tangent"
    targets: list[str] = field(default_factory=lambda: ["xrot", "delta"])
    run_dirs: dict[str, str] = field(
        default_factory=lambda: {
            "quarter": "outputs/aep_core_annulus_mlp",
            "full": "outputs/aep_core_annulus_mlp_full",
        }
    )


TARGET_LABEL = {"xrot": r"$X_{\rm rot}f$", "delta": r"$\Delta = f - P(f)$"}
RUN_LABEL = {"quarter": "wedge dataset (90 deg arc)", "full": "fully invariant task (full annulus)"}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--run-dirs", nargs="+", default=None,
                        help="label=output_dir pairs matching the coefficient keys")
    args = parser.parse_args()
    cfg = Config()
    for name in ("coefficients", "output_dir"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.run_dirs:
        cfg.run_dirs = dict(pair.split("=", 1) for pair in args.run_dirs)
    return cfg


def flat_magnitudes(model: SimpleMLP, state_dict: dict) -> np.ndarray:
    """``|theta_i|`` in ``model.parameters()`` order -- the order the Jacobian columns use."""
    model.load_state_dict(state_dict)
    return torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()]).numpy()


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stored = np.load(cfg.coefficients)

    for run, run_dir in cfg.run_dirs.items():
        checkpoint = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
        run_cfg = checkpoint["config"]
        model = SimpleMLP(hidden_dim=run_cfg["hidden_dim"], depth=run_cfg["depth"])
        _, parameter_slices = parameter_layout(model)
        magnitude_initial = flat_magnitudes(model, checkpoint["initial_state_dict"])
        magnitude_final = flat_magnitudes(model, checkpoint["final_state_dict"])

        for target in cfg.targets:
            initial = np.abs(stored[f"{target}/{run}/initial"])
            final = np.abs(stored[f"{target}/{run}/trained"])
            caption = f"{TARGET_LABEL[target]}, {RUN_LABEL[run]}"

            plot_ei_initial_vs_final(
                initial, final, parameter_slices, title=caption,
                output_stem=output_dir / f"ci_scatter_{target}_{run}",
                quantity_label="$|c_i|$", log_scale=True,
            )
            # Their companion diagnostic: is a small c_i genuine alignment, or just a
            # parameter that L1 drove to near-zero magnitude?
            plot_magnitude_vs_quantity(
                magnitude_initial, magnitude_final, initial, final, parameter_slices,
                quantity_label="$|c_i|$", title=caption,
                output_stem=output_dir / f"ci_vs_magnitude_{target}_{run}",
            )

            grew = int(np.sum(final > initial))
            ratio = final / np.maximum(initial, 1e-30)
            log_corr = np.corrcoef(
                np.log10(np.maximum(magnitude_final, 1e-12)), np.log10(np.maximum(final, 1e-12))
            )[0, 1]
            print(
                f"[{target:>5s}] {RUN_LABEL[run]:<38s}: "
                f"||c||: {np.linalg.norm(initial):.3e} -> {np.linalg.norm(final):.3e}  "
                f"median ratio = {np.median(ratio):.3f}  "
                f"grew {100 * grew / initial.size:>5.1f}%  "
                f"corr(|theta|,|c|) = {parameter_magnitude_ci_correlation(magnitude_final, final):+.3f}  "
                f"log-log corr = {log_corr:+.3f}"
            )


if __name__ == "__main__":
    main()
