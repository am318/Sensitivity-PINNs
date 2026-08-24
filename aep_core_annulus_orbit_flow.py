"""Run the decision boundary all the way around the orbit by integrating the X_rot f direction.

``aep_core_annulus_equivariance_step.py`` takes a single straight-line step
``theta_0 + t z`` with ``z = argmin ||J z - X_rot f||``. That only reproduces
``f_0 . R_t`` to first order, and the fidelity decays with the angle.

The orbit direction is a *vector field* on parameter space, not a fixed vector:
at every point of the path the model has a different ``X_rot f`` and a different
Jacobian. So the honest way to transport the model a finite angle is to
integrate it,

    theta_{k+1} = theta_k + dt * z_k,    z_k = argmin_z ||J_{theta_k} z - X_rot f_{theta_k}||,

which is what this script does, all the way around a full turn. If the tangent
space really represents the SO(2) action, then after integrating an angle
``alpha`` the model should equal ``f_0 . R_alpha``, and after a full ``2 pi``
turn it should come back to ``f_0`` -- a closure test with no free parameters.

Reported along the flow: distance to the rotated reference ``f_0 . R_alpha``
against the do-nothing baseline, the non-equivariant component (which pure
transport must leave *unchanged*), the task metrics, and the distance travelled
in parameter space. The decision boundary is drawn every 60 degrees, and the
single-step walk is overlaid for comparison where it exists.

Usage:
    python aep_core_annulus_orbit_flow.py
    python aep_core_annulus_orbit_flow.py --step-degrees 1 --runs quarter=outputs/aep_core_annulus_mlp
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aep_core_annulus_mlp import SimpleMLP, decision_grid, draw_decision_panel
from aep_core_annulus_equivariance_step import (
    flat_parameters,
    set_flat_parameters,
    task_metrics,
)
from aep_core_annulus_tangent import (
    build_probe_grid,
    condition_style,
    equivariance_defect,
    orbit_direction,
    sensitivity_jacobian,
)
from experiment_common import apply_config_overrides, select_device
from sensitivity_tools import tangent_projection


@dataclass
class Config:
    device: str = "auto"
    output_dir: str = "outputs/aep_core_annulus_orbit_flow"
    runs: dict[str, str] = field(
        default_factory=lambda: {
            "quarter": "outputs/aep_core_annulus_mlp",
            "full": "outputs/aep_core_annulus_mlp_full",
        }
    )
    single_step_summary: str = "outputs/aep_core_annulus_equivariance_step/equivariance_step_summary.json"

    probe_extent: float = 3.5
    probe_grid_points_per_axis: int = 24
    probe_max_radius: float = 3.5
    projection_group_order: int = 64
    tangent_svd_relative_cutoff: float = 1e-3

    # Integration: forward Euler on the orbit vector field. 2 degrees keeps the
    # per-step angle well inside the linear regime measured by the single-step
    # walk (which is still ~0.1x faithful at 15 degrees).
    step_degrees: float = 2.0
    total_degrees: float = 360.0

    boundary_degrees: list[float] = field(
        default_factory=lambda: [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]
    )
    boundary_limit: float = 4.0
    boundary_grid: int = 220
    boundary_levels: int = 8


@torch.no_grad()
def rotated_reference(
    base_model: torch.nn.Module, points: torch.Tensor, angle: float
) -> torch.Tensor:
    """``f_0(R_alpha x)`` on the probe grid -- the exact object the flow should reach."""
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=points.dtype, device=points.device
    )
    return base_model(points @ rotation.T).squeeze(-1)


def integrate_orbit(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    base_parameters: torch.Tensor,
    points: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> tuple[dict[str, list[float]], dict[float, torch.Tensor]]:
    """Euler-integrate the orbit direction field; return the record and boundary snapshots."""
    step = math.radians(cfg.step_degrees)
    n_steps = int(round(cfg.total_degrees / cfg.step_degrees))
    base_values = base_model(points).squeeze(-1).detach()
    base_norm = torch.linalg.vector_norm(base_values).clamp_min(1e-15)
    parameter_norm = torch.linalg.vector_norm(base_parameters).clamp_min(1e-15)

    wanted = {round(d, 6) for d in cfg.boundary_degrees}
    snapshots: dict[float, torch.Tensor] = {}
    record: dict[str, list[float]] = {
        "degrees": [], "transport_error": [], "transport_baseline": [],
        "relative_defect": [], "defect_norm": [], "accuracy": [], "loss": [],
        "parameter_distance": [], "path_length": [], "solve_residual": [],
    }

    path_length = 0.0
    for index in range(n_steps + 1):
        degrees = index * cfg.step_degrees
        angle = math.radians(degrees)

        values, defect = equivariance_defect(model, points, cfg.projection_group_order)
        reference = rotated_reference(base_model, points, angle)
        loss, accuracy = task_metrics(model, x, y)
        parameters = flat_parameters(model)

        record["degrees"].append(degrees)
        record["transport_error"].append(
            float(torch.linalg.vector_norm(values - reference) / base_norm)
        )
        record["transport_baseline"].append(
            float(torch.linalg.vector_norm(base_values - reference) / base_norm)
        )
        record["relative_defect"].append(
            float(torch.linalg.vector_norm(defect) / torch.linalg.vector_norm(values).clamp_min(1e-15))
        )
        record["defect_norm"].append(float(torch.linalg.vector_norm(defect)))
        record["accuracy"].append(accuracy)
        record["loss"].append(loss)
        record["parameter_distance"].append(
            float(torch.linalg.vector_norm(parameters - base_parameters) / parameter_norm)
        )
        record["path_length"].append(path_length / float(parameter_norm))

        if round(degrees, 6) in wanted:
            snapshots[degrees] = parameters.clone()
        if index == n_steps:
            break

        jacobian = sensitivity_jacobian(model, points)
        _, target = orbit_direction(model, points)
        residual, _, direction, _, _ = tangent_projection(
            jacobian, target, cfg.tangent_svd_relative_cutoff
        )
        direction = direction.to(device=points.device, dtype=points.dtype)
        record["solve_residual"].append(residual)
        path_length += step * float(torch.linalg.vector_norm(direction))
        set_flat_parameters(model, parameters + step * direction)

    return record, snapshots


def load_single_step(path: Path) -> dict[str, dict[str, list[float]]]:
    """The straight-line xrot walk, for comparison. Absent file just disables the overlay."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        label: result["walk"]
        for label, result in payload.get("targets", {}).get("xrot", {}).items()
    }


def plot_flow(
    results: dict[str, dict[str, Any]], single_step: dict, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), constrained_layout=True)
    axes = axes.ravel()

    for label, result in results.items():
        colour, style, line_width = condition_style(label)
        record = result["flow"]
        degrees = np.asarray(record["degrees"])

        axes[0].plot(degrees, record["transport_error"], linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        axes[0].plot(degrees, record["transport_baseline"], linewidth=1.0,
                     linestyle=(0, (1, 2)), color=colour, alpha=0.8)
        axes[1].plot(degrees, record["relative_defect"], linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        axes[2].plot(degrees, record["accuracy"], linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        axes[3].plot(degrees, record["loss"], linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        axes[4].plot(degrees, record["parameter_distance"], linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        axes[4].plot(degrees, record["path_length"], linewidth=1.0,
                     linestyle=(0, (1, 2)), color=colour, alpha=0.8)

        walk = single_step.get(label)
        if walk:
            steps = np.degrees(np.asarray(walk["steps"]))
            keep = (steps >= 0) & (steps <= 90.0)
            axes[5].plot(steps[keep], np.asarray(walk["reference_distance"])[keep],
                         linewidth=1.2, linestyle=(0, (4, 2)), color=colour,
                         label=f"{label} (single step)")
        keep = degrees <= 90.0
        axes[5].plot(degrees[keep], np.asarray(record["transport_error"])[keep],
                     linewidth=line_width, linestyle=style, color=colour,
                     label=f"{label} (flow)")

    panel_setup = [
        ("distance to $f_0 \\circ R_\\alpha$\n(dotted: same distance from $f_0$, i.e. not moving)",
         r"$\|f_\alpha - f_0 \circ R_\alpha\| / \|f_0\|$", "log"),
        ("non-equivariant component\n(pure transport must leave this fixed)",
         r"$\|(I-P)f\| / \|f\|$", "log"),
        ("task accuracy", "accuracy", "linear"),
        ("task loss", "BCE on training data", "log"),
        ("distance travelled in parameter space\n(dotted: path length)",
         r"$\|\theta_\alpha - \theta_0\| / \|\theta_0\|$", "linear"),
        ("flow vs single straight-line step", r"$\|f - f_0 \circ R_\alpha\| / \|f_0\|$", "log"),
    ]
    for ax, (title, ylabel, yscale) in zip(axes, panel_setup):
        ax.set(xlabel=r"transported angle $\alpha$ (degrees)", ylabel=ylabel,
               title=title, yscale=yscale)
        ax.legend(fontsize=7)
    for ax in axes[:5]:
        ax.axvline(360.0, color="black", linestyle=":", linewidth=1)
    # The transport error is exactly zero at both ends of a closed turn (and so is
    # the baseline), which on a log axis stretches the scale over 16 decades of
    # nothing; clip to the range where the curves actually live.
    axes[0].set_ylim(1e-4, 3.0)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"orbit_flow_diagnostics.{suffix}")
    plt.close(fig)


def plot_flow_boundaries(
    states: dict[str, dict[str, Any]], points: torch.Tensor, cfg: Config,
    device: torch.device, output_dir: Path,
) -> None:
    """Decision boundary every 60 degrees of transported angle."""
    mesh = decision_grid(cfg.boundary_limit, cfg.boundary_grid, device, torch.float64)
    columns = cfg.boundary_degrees
    fig, axes = plt.subplots(
        len(states), len(columns), figsize=(3.4 * len(columns), 3.7 * len(states)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for i, (label, state) in enumerate(states.items()):
        model = state["model"]
        for j, degrees in enumerate(columns):
            set_flat_parameters(model, state["snapshots"][degrees])
            draw_decision_panel(
                axes[i, j], model, mesh, state["x_numpy"], state["y_numpy"],
                n_levels=cfg.boundary_levels,
            )
            values, defect = equivariance_defect(model, points, cfg.projection_group_order)
            relative = float(
                torch.linalg.vector_norm(defect)
                / torch.linalg.vector_norm(values).clamp_min(1e-15)
            )
            _, accuracy = task_metrics(model, state["x"], state["y"])
            axes[i, j].text(
                0.5, -0.02,
                f"$\\|(I-P)f\\|/\\|f\\|$ = {relative:.3f},  acc = {accuracy:.2f}",
                ha="center", va="top", fontsize=12, transform=axes[i, j].transAxes,
            )
        set_flat_parameters(model, state["snapshots"][columns[0]])
        axes[i, 0].set_ylabel(label, fontsize=17, labelpad=12)

    for j, degrees in enumerate(columns):
        axes[-1, j].set_xlabel(f"$\\alpha = {degrees:g}^\\circ$", fontsize=17, labelpad=28)
    axes[0, 0].set_title("start", fontsize=16, pad=8)
    axes[0, -1].set_title("after a full turn", fontsize=16, pad=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"orbit_flow_boundary.{suffix}", dpi=200)
    plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--step-degrees", type=float)
    parser.add_argument("--total-degrees", type=float)
    parser.add_argument("--tangent-svd-relative-cutoff", type=float)
    args = parser.parse_args()

    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.runs:
        cfg.runs = dict(pair.split("=", 1) for pair in args.runs)
    for name in ("device", "output_dir", "step_degrees", "total_degrees",
                 "tangent_svd_relative_cutoff"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    return cfg


def main() -> None:
    cfg = parse_args()
    device = select_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = build_probe_grid(cfg, device)
    steps_per_turn = int(round(cfg.total_degrees / cfg.step_degrees))
    print(f"probe grid: {points.shape[0]} points; integrating {cfg.total_degrees:g} deg "
          f"in {steps_per_turn} steps of {cfg.step_degrees:g} deg")

    single_step = load_single_step(Path(cfg.single_step_summary))
    results: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, Any]] = {}

    for run_label, run_dir in cfg.runs.items():
        checkpoint = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
        run_cfg = checkpoint["config"]
        x = torch.tensor(checkpoint["x"], device=device, dtype=torch.float64)
        y = torch.tensor((checkpoint["y"] > 0).astype(np.float64), device=device)

        for state_label in ("initial", "trained"):
            key = "initial_state_dict" if state_label == "initial" else "final_state_dict"
            models = []
            for _ in range(2):  # one to flow, one frozen copy for the reference
                model = SimpleMLP(hidden_dim=run_cfg["hidden_dim"], depth=run_cfg["depth"])
                model.load_state_dict(checkpoint[key])
                models.append(model.to(device=device, dtype=torch.float64).eval())
            model, base_model = models

            label = f"{run_label}/{state_label}"
            base_parameters = flat_parameters(model)
            record, snapshots = integrate_orbit(
                model, base_model, base_parameters, points, x, y, cfg
            )
            states[label] = {
                "model": model, "snapshots": snapshots, "x": x, "y": y,
                "x_numpy": checkpoint["x"], "y_numpy": checkpoint["y"],
                "base": base_parameters,
            }

            degrees = np.asarray(record["degrees"])
            at_90 = int(np.argmin(np.abs(degrees - 90.0)))
            results[label] = {
                "dataset_outer_arc_degrees": run_cfg["outer_arc_degrees"],
                "transport_error_at_90": record["transport_error"][at_90],
                "transport_baseline_at_90": record["transport_baseline"][at_90],
                "closure_error": record["transport_error"][-1],
                "closure_parameter_distance": record["parameter_distance"][-1],
                "path_length": record["path_length"][-1],
                "relative_defect_start": record["relative_defect"][0],
                "relative_defect_end": record["relative_defect"][-1],
                "accuracy_start": record["accuracy"][0],
                "accuracy_min": min(record["accuracy"]),
                "accuracy_end": record["accuracy"][-1],
                "max_solve_residual": max(record["solve_residual"]),
                "flow": record,
            }
            print(
                f"{label:>16s}: at 90 deg |f-f0.R|/|f0| = {record['transport_error'][at_90]:.4f}"
                f" (not moving: {record['transport_baseline'][at_90]:.4f})"
                f"  | closure after 360 deg: {record['transport_error'][-1]:.4f}"
                f", param dist {record['parameter_distance'][-1]:.4f} (path {record['path_length'][-1]:.3f})"
                f"  | rel defect {record['relative_defect'][0]:.4f} -> {record['relative_defect'][-1]:.4f}"
                f"  | acc {record['accuracy'][0]:.3f}, min {min(record['accuracy']):.3f},"
                f" end {record['accuracy'][-1]:.3f}"
            )

    plot_flow(results, single_step, output_dir)
    plot_flow_boundaries(states, points, cfg, device, output_dir)
    # Persist the parameter vectors along the path: the flow re-solves z at every
    # step, so its cumulative displacement is not recoverable from the scalars.
    np.savez_compressed(
        output_dir / "orbit_flow_parameters.npz",
        **{f"{label}/base": state["base"].cpu().numpy() for label, state in states.items()},
        **{f"{label}/{degrees:g}": parameters.cpu().numpy()
           for label, state in states.items()
           for degrees, parameters in state["snapshots"].items()},
    )
    (output_dir / "orbit_flow_summary.json").write_text(
        json.dumps({"config": asdict(cfg), "conditions": results}, indent=2)
    )
    print(f"wrote {output_dir}/orbit_flow_boundary.png, orbit_flow_diagnostics.png, "
          f"orbit_flow_summary.json")


if __name__ == "__main__":
    main()
