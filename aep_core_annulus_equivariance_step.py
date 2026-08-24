"""Walk the parameters along a tangent-space solution and watch what the model does.

For a function-space direction ``g`` and the Jacobian ``J_theta`` on the probe
grid, let

    z = argmin_z ||J_theta z - g||   (minimum-norm solution)

be the parameter direction that best realises ``g``. To first order,
``f_{theta + t z} = f_theta + t J z + O(t^2) ~ f + t g``. Two choices of ``g``
make *different* predictions, and this script walks both:

``delta``  (``g = Delta = f - P(f)``, the non-equivariant component)
    ``P`` is linear and idempotent, so ``(I - P) f_{theta + t z} ~ (1 + t)
    Delta``: the non-equivariant part should shrink linearly along ``-z`` and
    vanish at ``t = -1``, i.e. one full step lands on ``P(f)``. Note the sign:
    with ``Delta = f - P(f)`` the *descent* direction is ``-z``.

``xrot``  (``g = X_rot f``, the infinitesimal orbit direction)
    ``f + t X_rot f`` is the first-order expansion of ``f . R_t``, so this step
    should *transport the model along the group orbit* -- rotating the decision
    boundary by ``t`` radians -- rather than symmetrising it. The non-equivariant
    component is carried along unchanged, so ``||(I - P) f||`` should stay flat.
    Being able to move along the orbit and being able to move *towards* the
    equivariant subspace are different properties of the same tangent space,
    and this is where they separate.

At every stop the walk records the true non-equivariance ``||(I - P) f||``
against its prediction, how faithfully the step realised the intended function
change, the distance to the exact object the step aims at (``P(f_0)`` or
``f_0 . R_t``), the reference repo's ``E(T)`` on the data points, and the task
BCE and accuracy -- so the cost of the move is visible next to its effect. The
decision boundary is drawn along the same path.

Usage:
    python aep_core_annulus_equivariance_step.py
    python aep_core_annulus_equivariance_step.py --targets xrot
    python aep_core_annulus_equivariance_step.py --tangent-svd-relative-cutoff 1e-6
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
import torch.nn.functional as functional

from aep_core_annulus_mlp import (
    SimpleMLP,
    decision_grid,
    draw_decision_panel,
    invariance_error,
)
from aep_core_annulus_tangent import (
    build_probe_grid,
    condition_style,
    equivariance_defect,
    group_average,
    orbit_direction,
    sensitivity_jacobian,
)
from experiment_common import apply_config_overrides, select_device
from sensitivity_tools import tangent_projection


# Step grids differ by target because the two predictions live on different
# scales: ``delta`` has a distinguished point at t = -1 (the projection), while
# for ``xrot`` the step *is* an angle in radians, so the interesting stops are
# fractions of a turn.
WALK_GRIDS = {
    "delta": {
        "step_min": -1.5, "step_max": 0.5, "step_count": 81,
        "boundary_steps": [0.5, 0.0, -0.25, -0.5, -0.75, -1.0, -1.25],
    },
    "xrot": {
        "step_min": -1.6, "step_max": 1.6, "step_count": 65,
        "boundary_steps": [math.radians(d) for d in (-45, 0, 15, 30, 45, 60, 90)],
    },
}


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/aep_core_annulus_equivariance_step"
    runs: dict[str, str] = field(
        default_factory=lambda: {
            "quarter": "outputs/aep_core_annulus_mlp",
            "full": "outputs/aep_core_annulus_mlp_full",
        }
    )
    targets: list[str] = field(default_factory=lambda: ["delta", "xrot"])
    walk_grids: dict[str, dict] = field(default_factory=lambda: WALK_GRIDS)

    # Probe geometry and projection group: kept identical to the tangent script so
    # z is solved for on exactly the grid the diagnostics were computed on.
    probe_extent: float = 3.5
    probe_grid_points_per_axis: int = 24
    probe_max_radius: float = 3.5
    projection_group_order: int = 64

    tangent_svd_relative_cutoff: float = 1e-3

    invariance_rotations: int = 40
    boundary_limit: float = 4.0
    boundary_grid: int = 220
    boundary_levels: int = 8


TARGET_LABEL = {"delta": r"$\Delta = f - P(f)$", "xrot": r"$X_{\rm rot}f$"}
REFERENCE_LABEL = {"delta": r"$P(f_0)$", "xrot": r"$f_0 \circ R_t$"}


# ------------------------------ parameter walk ------------------------------
def set_flat_parameters(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into the model, in ``model.parameters()`` order.

    That is the same order ``sensitivity_tools.parameter_gradient_row`` uses to
    build the Jacobian columns, so index i of ``z`` really is the parameter that
    column i of J differentiates with respect to.
    """
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count
    assert offset == flat.numel(), "flat vector length does not match the model"


def flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def logit_metrics(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    loss = functional.binary_cross_entropy_with_logits(logits, y)
    return float(loss), float(((logits > 0).to(y.dtype) == y).to(torch.float64).mean())


@torch.no_grad()
def task_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    return logit_metrics(model(x).squeeze(-1), y)


@torch.no_grad()
def projected_task_metrics(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, order: int
) -> tuple[float, float]:
    """Loss and accuracy of the *exactly* projected model ``P(f)`` on the data.

    The walk only reaches ``P(f)`` to first order, so this is the control that
    separates "the linear step is inexact" from "the equivariant component of
    this model genuinely cannot do the task".
    """
    return logit_metrics(group_average(model, x, order), y)


def target_field(
    model: torch.nn.Module, points: torch.Tensor, target: str, cfg: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(f, g)`` on the probe grid for the requested target direction."""
    if target == "delta":
        return equivariance_defect(model, points, cfg.projection_group_order)
    if target == "xrot":
        return orbit_direction(model, points)
    raise ValueError(f"unknown target: {target}")


@torch.no_grad()
def reference_values(
    base_model: torch.nn.Module, points: torch.Tensor, step: float, target: str, cfg: Config
) -> torch.Tensor:
    """The exact function the step at ``t = step`` is aiming at, evaluated on the probe grid.

    For ``delta`` that is the projection ``P(f_0)`` (independent of ``t``; the
    step only aims at it at ``t = -1``, and the distance to it is what the walk
    is trying to shrink). For ``xrot`` it is the genuinely rotated model
    ``f_0(R_t x)``, which the first-order step approximates.
    """
    if target == "delta":
        return group_average(base_model, points, cfg.projection_group_order)
    cosine, sine = math.cos(step), math.sin(step)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=points.dtype, device=points.device
    )
    return base_model(points @ rotation.T).squeeze(-1)


def walk(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    base_parameters: torch.Tensor,
    direction: torch.Tensor,
    base_values: torch.Tensor,
    base_target: torch.Tensor,
    points: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: str,
    grid: dict,
    cfg: Config,
    device: torch.device,
) -> dict[str, list[float]]:
    """Step ``theta = theta_0 + t z`` across the grid of ``t`` and measure at each stop."""
    steps = np.linspace(grid["step_min"], grid["step_max"], grid["step_count"])
    record: dict[str, list[float]] = {
        "steps": steps.tolist(), "defect_norm": [], "relative_defect": [],
        "invariance_error_max": [], "loss": [], "accuracy": [], "parameter_step": [],
        "realisation_error": [], "reference_distance": [], "reference_baseline": [],
    }
    parameter_norm = float(torch.linalg.vector_norm(base_parameters))
    base_norm = torch.linalg.vector_norm(base_values).clamp_min(1e-15)

    for step in steps:
        set_flat_parameters(model, base_parameters + float(step) * direction)
        values, defect = equivariance_defect(model, points, cfg.projection_group_order)
        loss, accuracy = task_metrics(model, x, y)
        # Same rotations at every stop: E(T) draws its angles at random, and
        # resampling them per step would add noise on top of the effect measured.
        torch.manual_seed(cfg.seed)
        _, invariance_max = invariance_error(model, x.cpu().numpy(), cfg, device, torch.float64)

        predicted = base_values + float(step) * base_target
        reference = reference_values(base_model, points, float(step), target, cfg)

        record["defect_norm"].append(float(torch.linalg.vector_norm(defect)))
        record["relative_defect"].append(
            float(torch.linalg.vector_norm(defect) / torch.linalg.vector_norm(values).clamp_min(1e-15))
        )
        record["invariance_error_max"].append(invariance_max)
        record["loss"].append(loss)
        record["accuracy"].append(accuracy)
        record["parameter_step"].append(
            abs(float(step)) * float(torch.linalg.vector_norm(direction)) / parameter_norm
        )
        record["realisation_error"].append(
            float(torch.linalg.vector_norm(values - predicted) / base_norm)
        )
        record["reference_distance"].append(
            float(torch.linalg.vector_norm(values - reference) / base_norm)
        )
        record["reference_baseline"].append(
            float(torch.linalg.vector_norm(base_values - reference) / base_norm)
        )

    set_flat_parameters(model, base_parameters)
    return record


# --------------------------------- plotting ---------------------------------
def plot_walk(
    results: dict[str, dict[str, Any]], target: str, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22.0, 8.8), constrained_layout=True)
    axes = axes.ravel()

    for label, result in results.items():
        colour, style, line_width = condition_style(label)
        record = result["walk"]
        steps = np.asarray(record["steps"])
        prediction = (
            np.abs(1.0 + steps) * result["defect_norm_at_base"] if target == "delta"
            else np.full_like(steps, result["defect_norm_at_base"])
        )

        axes[0].plot(steps, record["defect_norm"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[0].plot(steps, prediction, linewidth=1.0, linestyle=(0, (1, 2)), color=colour, alpha=0.8)
        axes[1].plot(steps, record["relative_defect"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[2].plot(steps, record["invariance_error_max"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[3].plot(steps, record["loss"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[4].plot(steps, record["accuracy"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[5].plot(steps, record["parameter_step"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[6].plot(steps, record["realisation_error"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[7].plot(steps, record["reference_distance"], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[7].plot(steps, record["reference_baseline"], linewidth=1.0, linestyle=(0, (1, 2)),
                     color=colour, alpha=0.8)

        if target == "delta":
            # Star at t = -1: what the *exact* projection P(f) scores, i.e. where
            # the walk would land if the first-order step were exact.
            axes[3].scatter([-1.0], [result["projected_loss"]], marker="*", s=140,
                            color=colour, edgecolors="black", linewidths=0.5, zorder=10)
            axes[4].scatter([-1.0], [result["projected_accuracy"]], marker="*", s=140,
                            color=colour, edgecolors="black", linewidths=0.5, zorder=10)

    prediction_note = (
        r"dotted: $|1+t|\,\|\Delta_0\|$" if target == "delta"
        else r"dotted: $\|\Delta_0\|$ (transport leaves it fixed)"
    )
    star_note = " (star: exact $P(f)$)" if target == "delta" else ""
    panel_setup = [
        (f"$\\|(I-P)f\\|$ along the walk\n({prediction_note})", r"$\|(I-P)f\|$", "log"),
        ("relative non-equivariance", r"$\|(I-P)f\| / \|f\|$", "log"),
        ("rotation-invariance error on the data", r"$\mathcal{E}(T)$", "log"),
        (f"task loss{star_note}", "BCE on training data", "log"),
        (f"task accuracy{star_note}", "accuracy", "linear"),
        ("size of the parameter step", r"$\|t z\| / \|\theta_0\|$", "log"),
        (f"how faithfully the step realised {TARGET_LABEL[target]}",
         r"$\|f_t - (f_0 + t g)\| / \|f_0\|$", "log"),
        (f"distance to {REFERENCE_LABEL[target]}\n(dotted: same distance from $f_0$)",
         r"$\|f_t - f_{\rm ref}\| / \|f_0\|$", "log"),
    ]
    marker = -1.0 if target == "delta" else math.pi / 2
    for ax, (title, ylabel, yscale) in zip(axes, panel_setup):
        ax.axvline(marker, color="black", linestyle=":", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set(xlabel="step $t$ along $z$", ylabel=ylabel, title=title, yscale=yscale)
        ax.legend(fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"equivariance_step_{target}.{suffix}")
    plt.close(fig)


def step_label(step: float, target: str) -> str:
    """Column label for the boundary walk; for ``xrot`` the step is an angle in radians."""
    if target != "xrot":
        return f"$t = {step:g}$"
    return f"$t = {step:.2f}$\n(${math.degrees(step):.0f}^\\circ$)"


def plot_decision_boundary_walk(
    states: dict[str, dict[str, Any]],
    points: torch.Tensor,
    target: str,
    grid: dict,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Decision boundary at each stop of the walk: rows = condition, columns = step t.

    Drawn with the same panel style as the before/after figure, so these are
    directly comparable with ``decision_boundary_before_after.png``.
    """
    mesh = decision_grid(cfg.boundary_limit, cfg.boundary_grid, device, torch.float64)
    boundary_steps = grid["boundary_steps"]
    rows, columns = len(states), len(boundary_steps)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.4 * columns, 3.7 * rows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, (label, state) in enumerate(states.items()):
        model = state["model"]
        for j, step in enumerate(boundary_steps):
            set_flat_parameters(model, state["base"] + step * state["z"])
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
        set_flat_parameters(model, state["base"])
        axes[i, 0].set_ylabel(label, fontsize=17, labelpad=12)

    highlight = {"delta": (-1.0, r"$\approx P(f_0)$"), "xrot": (math.pi / 2, r"$\approx f_0 \circ R_{90^\circ}$")}
    for j, step in enumerate(boundary_steps):
        axes[-1, j].set_xlabel(step_label(step, target), fontsize=17, labelpad=28)
        if abs(step) < 1e-12:
            axes[0, j].set_title("start", fontsize=16, pad=8)
        elif abs(step - highlight[target][0]) < 1e-6:
            axes[0, j].set_title(highlight[target][1], fontsize=16, pad=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"decision_boundary_walk_{target}.{suffix}", dpi=200)
    plt.close(fig)


# ----------------------------------- main -----------------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--targets", nargs="+", default=None, choices=list(WALK_GRIDS))
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--tangent-svd-relative-cutoff", type=float)
    args = parser.parse_args()

    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.runs:
        cfg.runs = dict(pair.split("=", 1) for pair in args.runs)
    if args.targets:
        cfg.targets = list(args.targets)
    for name in ("device", "output_dir", "tangent_svd_relative_cutoff"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    return cfg


def run_target(cfg: Config, target: str, points: torch.Tensor, device: torch.device,
               output_dir: Path) -> dict[str, dict[str, Any]]:
    grid = cfg.walk_grids[target]
    print(f"\n=== target {target}: stepping t in "
          f"[{grid['step_min']:g}, {grid['step_max']:g}] ({grid['step_count']} stops) ===")

    results: dict[str, dict[str, Any]] = {}
    walk_states: dict[str, dict[str, Any]] = {}
    for run_label, run_dir in cfg.runs.items():
        checkpoint = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
        run_cfg = checkpoint["config"]
        x = torch.tensor(checkpoint["x"], device=device, dtype=torch.float64)
        y = torch.tensor((checkpoint["y"] > 0).astype(np.float64), device=device)

        for state_label in ("initial", "trained"):
            key = "initial_state_dict" if state_label == "initial" else "final_state_dict"
            models = []
            for _ in range(2):  # one to walk, one frozen copy for the reference
                model = SimpleMLP(hidden_dim=run_cfg["hidden_dim"], depth=run_cfg["depth"])
                model.load_state_dict(checkpoint[key])
                models.append(model.to(device=device, dtype=torch.float64).eval())
            model, base_model = models

            label = f"{run_label}/{state_label}"
            base_parameters = flat_parameters(model)
            base_values, base_target = target_field(model, points, target, cfg)
            jacobian = sensitivity_jacobian(model, points)
            residual, angle, z, resolved_rank, _ = tangent_projection(
                jacobian, base_target, cfg.tangent_svd_relative_cutoff
            )
            z = z.to(device=device, dtype=torch.float64)
            projected_loss, projected_accuracy = projected_task_metrics(
                model, x, y, cfg.projection_group_order
            )

            record = walk(model, base_model, base_parameters, z, base_values, base_target,
                          points, x, y, target, grid, cfg, device)
            walk_states[label] = {
                "model": model, "base": base_parameters, "z": z,
                "x": x, "y": y, "x_numpy": checkpoint["x"], "y_numpy": checkpoint["y"],
            }
            steps = np.asarray(record["steps"])
            best = int(np.argmin(record["defect_norm"]))
            at_base = int(np.argmin(np.abs(steps)))
            extreme = int(np.argmax(steps))

            results[label] = {
                "dataset_outer_arc_degrees": run_cfg["outer_arc_degrees"],
                "defect_norm_at_base": record["defect_norm"][at_base],
                "solve_residual": residual,          # ||Jz - g|| / ||g||
                "solve_angle_degrees": angle,
                "resolved_rank": resolved_rank,
                "direction_norm": float(torch.linalg.vector_norm(z)),
                "relative_direction_norm": float(
                    torch.linalg.vector_norm(z) / torch.linalg.vector_norm(base_parameters)
                ),
                "best_step": float(steps[best]),
                "defect_norm_at_best": record["defect_norm"][best],
                "defect_reduction_factor": record["defect_norm"][at_base] / max(record["defect_norm"][best], 1e-300),
                "accuracy_at_base": record["accuracy"][at_base],
                "accuracy_at_best": record["accuracy"][best],
                "accuracy_at_max_step": record["accuracy"][extreme],
                "reference_distance_at_max_step": record["reference_distance"][extreme],
                "reference_baseline_at_max_step": record["reference_baseline"][extreme],
                "invariance_error_at_base": record["invariance_error_max"][at_base],
                "invariance_error_at_best": record["invariance_error_max"][best],
                "projected_loss": projected_loss,
                "projected_accuracy": projected_accuracy,
                "walk": record,
            }

            print(
                f"{label:>16s}: ||Jz-g||/||g|| = {residual:.2e}"
                f"  ||z||/||theta|| = {results[label]['relative_direction_norm']:.3e}"
                f"  best t = {steps[best]:+.3f}"
                f"  ||(I-P)f||: {record['defect_norm'][at_base]:.3e} -> {record['defect_norm'][best]:.3e}"
                f"  ({results[label]['defect_reduction_factor']:.1f}x)"
                f"  acc: {record['accuracy'][at_base]:.3f} -> {record['accuracy'][best]:.3f}"
                f"  | at t={steps[extreme]:+.2f}: acc {record['accuracy'][extreme]:.3f},"
                f" dist to ref {record['reference_distance'][extreme]:.3f}"
                f" (vs {record['reference_baseline'][extreme]:.3f} without moving)"
            )

    plot_walk(results, target, output_dir)
    plot_decision_boundary_walk(walk_states, points, target, grid, cfg, device, output_dir)
    return results


def main() -> None:
    cfg = parse_args()
    device = select_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = build_probe_grid(cfg, device)
    print(f"probe grid: {points.shape[0]} points")

    summary = {target: run_target(cfg, target, points, device, output_dir)
               for target in cfg.targets}
    (output_dir / "equivariance_step_summary.json").write_text(
        json.dumps({"config": asdict(cfg), "targets": summary}, indent=2)
    )
    print(f"\nwrote {output_dir}/equivariance_step_<target>.png, "
          f"decision_boundary_walk_<target>.png, equivariance_step_summary.json")


if __name__ == "__main__":
    main()
