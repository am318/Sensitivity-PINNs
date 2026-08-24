"""Do the symmetry directions lie in the plain MLP's functional tangent space?

Companion to ``aep_core_annulus_mlp.py``. For each saved model state (before /
after training, on the 90-degree-wedge and the full-annulus dataset) this
computes, on a shared probe grid, the functional sensitivity Jacobian
``J_theta = df/dtheta`` (shape ``[N, P]``, column span = the functional tangent
space ``T_theta``) and two different function-space directions:

``xrot``
    The *infinitesimal* orbit direction ``X_rot f = grad_x f . (-x2, x1)``, the
    velocity of ``f`` under an infinitesimal rotation of its input. Vanishes
    pointwise iff ``f`` is exactly rotation-invariant.

``delta``
    The *finite* non-equivariant component ``Delta = f - P(f)`` of Berndt &
    Stuhmer sec. 3, where ``P(f)(x) = |G|^-1 sum_g f(g.x)`` is the group
    average over ``G = C_m`` (``projection_group_order``), a quadrature for the
    SO(2) Haar integral. ``P`` is an exact idempotent projection for finite
    ``G`` -- ``P(Delta) = 0`` pointwise and identically, whatever the probe
    grid -- so ``Delta`` is exactly the component the paper's ``lambda_perp``
    penalty acts on, and ``f - Delta`` is exactly invariant.

Both are pushed through ``sensitivity_tools.tangent_projection``: projection
error ``eps = ||(I - Pi_T) target|| / ||target||``, principal angle, and the
minimum-norm coefficients ``c`` solving ``target = sum_i c_i S_i``.

Scale warning, which the diagnostics here are built around: with P >> N the
tangent space generically fills the whole probe space, so ``eps = 0`` for *any*
target at zero truncation, and every number below is really a statement about
the *numerically resolved* tangent space at a given SVD cutoff. Both ``eps``
and ``||c||`` are therefore also reported as curves over a cutoff sweep,
together with the cutoff-free version of the same question -- how much of the
target's energy lies in the top-k left singular directions of J.

``aep_core_annulus_equivariance_step.py`` then takes the ``delta`` coefficients
and actually walks the parameters along them, to test whether this linear
picture predicts what happens to the model.

Usage:
    python aep_core_annulus_tangent.py
    python aep_core_annulus_tangent.py --runs quarter=outputs/aep_core_annulus_mlp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aep_core_annulus_mlp import SimpleMLP
from experiment_common import (
    apply_config_overrides,
    parameter_layout,
    prettify_parameter_name,
    select_device,
    write_csv_rows,
)
from sensitivity_tools import (
    aggregate_by_module,
    parameter_gradient_row,
    participation_ratio,
    tangent_projection,
)


@dataclass
class Config:
    device: str = "auto"
    output_dir: str = "outputs/aep_core_annulus_tangent"
    runs: dict[str, str] = field(
        default_factory=lambda: {
            "quarter": "outputs/aep_core_annulus_mlp",
            "full": "outputs/aep_core_annulus_mlp_full",
        }
    )

    # Probe grid: same Cartesian convention as the ASRNN scripts' build_probe_grid,
    # with the far corners dropped so the diagnostic is not dominated by the
    # region beyond every data point, where the model purely extrapolates.
    probe_extent: float = 3.5
    probe_grid_points_per_axis: int = 24
    probe_max_radius: float = 3.5

    # Order m of the cyclic group C_m used as the quadrature for the SO(2) Haar
    # average in P(f). P is an exact projection for every m; m only controls how
    # well C_m-invariance approximates SO(2)-invariance, and 64 rotations is far
    # past convergence for a smooth 2-input MLP (checked in projection_residual).
    projection_group_order: int = 64

    # Headline cutoff; the same default as asrnn_mexican_hat_symmetry_sensitivity.py.
    tangent_svd_relative_cutoff: float = 1e-3
    cutoff_sweep: list[float] = field(
        default_factory=lambda: [1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    )
    top_parameters_to_report: int = 25


# ------------------------------ probe geometry ------------------------------
def build_probe_grid(cfg: Config, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(
        -cfg.probe_extent, cfg.probe_extent, cfg.probe_grid_points_per_axis,
        device=device, dtype=torch.float64,
    )
    mesh_x, mesh_y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)
    return points[points.norm(dim=1) <= cfg.probe_max_radius]


def orbit_direction(model: torch.nn.Module, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``f`` and ``X_rot f = grad_x f . (Omega x)`` at every probe point.

    ``Omega = [[0, -1], [1, 0]]`` is the SO(2) Lie generator, so ``Omega x =
    (-x2, x1)`` is the orbit velocity at ``x``. The model output is a scalar
    logit carrying the trivial representation (the task is *invariance*, not
    equivariance), so the ``- rho'(g) f`` term present in the vector-field
    version of this generator in the ASRNN scripts vanishes here.
    """
    x = points.clone().requires_grad_(True)
    values = model(x).squeeze(-1)
    (spatial_gradient,) = torch.autograd.grad(values.sum(), x)
    omega_x = torch.stack([-x[:, 1], x[:, 0]], dim=1)
    return values.detach(), (spatial_gradient * omega_x).sum(dim=1).detach()


@torch.no_grad()
def group_average(model: torch.nn.Module, points: torch.Tensor, order: int) -> torch.Tensor:
    """``P(f)(x) = |G|^-1 sum_{g in C_m} f(g.x)`` -- the paper's finite-group Haar sum.

    The output carries the trivial representation (invariance), so no
    ``rho(g)^-1`` factor is needed in front of ``f(g.x)``.
    """
    angles = 2 * np.pi * torch.arange(order, dtype=points.dtype, device=points.device) / order
    total = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    for angle in angles:
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack([
            torch.stack([cosine, -sine]),
            torch.stack([sine, cosine]),
        ])
        total += model(points @ rotation.T).squeeze(-1)
    return total / order


def equivariance_defect(
    model: torch.nn.Module, points: torch.Tensor, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``f`` and its non-equivariant component ``Delta = f - P(f)``.

    Because ``P`` averages over a finite group it is exactly idempotent, so
    ``P(Delta) = P(f) - P(P(f)) = 0`` identically: ``Delta`` really is the
    orthogonal residual, and ``f - Delta = P(f)`` is exactly C_m-invariant --
    both facts hold pointwise and therefore independently of the probe grid.
    """
    with torch.no_grad():
        values = model(points).squeeze(-1)
    return values, values - group_average(model, points, order)


@torch.no_grad()
def projection_residual(model: torch.nn.Module, points: torch.Tensor, order: int) -> float:
    """How far ``P(f)`` still is from *continuous* SO(2)-invariance, relative to ``||f||``.

    Guards the ``C_m ~ SO(2)`` quadrature assumption: it compares ``P(f)`` at
    the probe points with ``P(f)`` at points rotated by half a group step, the
    worst case for a C_m average.
    """
    half_step = np.pi / order
    rotation = torch.tensor(
        [[np.cos(half_step), -np.sin(half_step)], [np.sin(half_step), np.cos(half_step)]],
        dtype=points.dtype, device=points.device,
    )
    averaged = group_average(model, points, order)
    rotated = group_average(model, points @ rotation.T, order)
    values = model(points).squeeze(-1)
    return float(
        torch.linalg.vector_norm(rotated - averaged)
        / torch.linalg.vector_norm(values).clamp_min(1e-15)
    )


def sensitivity_jacobian(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    """``J[n, i] = d f(x_n) / d theta_i`` -- one backward pass per probe point."""
    params = tuple(model.parameters())
    rows = []
    for n in range(points.shape[0]):
        value = model(points[n : n + 1]).squeeze()
        rows.append(parameter_gradient_row(value, params))
    return torch.stack(rows)


# ------------------------------ the diagnostic ------------------------------
def cutoff_sweep(
    jacobian: torch.Tensor, target: torch.Tensor, cutoffs: list[float]
) -> dict[str, list[float]]:
    """``eps(X)``, principal angle and ``||c||`` as a function of the SVD cutoff.

    Computed from a single SVD of J so the whole sweep costs no more than one
    ``tangent_projection`` call; the shared-cutoff entry is cross-checked
    against ``tangent_projection`` in ``probe_state`` to keep this honest.
    """
    u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-15)
    coordinates = u.T @ target  # target's energy per singular direction

    errors, angles, coefficient_norms, ranks = [], [], [], []
    for cutoff in cutoffs:
        keep = singular_values >= cutoff * singular_values[0]
        rank = int(keep.sum())
        projected = u[:, keep] @ coordinates[keep]
        residual = torch.linalg.vector_norm(target - projected) / target_norm
        cosine = torch.dot(target, projected) / (
            target_norm * torch.linalg.vector_norm(projected).clamp_min(1e-15)
        )
        coefficients = vh[keep, :].T @ (coordinates[keep] / singular_values[keep])
        errors.append(float(residual))
        angles.append(float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))))
        coefficient_norms.append(float(torch.linalg.vector_norm(coefficients)))
        ranks.append(rank)

    # Cutoff-free version: cumulative fraction of the target's energy captured by
    # the leading k singular directions of J, k = 1 ... rank(J).
    captured = torch.cumsum(coordinates**2, dim=0) / target_norm**2
    return {
        "cutoffs": list(cutoffs),
        "projection_errors": errors,
        "principal_angles_degrees": angles,
        "coefficient_norms": coefficient_norms,
        "resolved_ranks": ranks,
        "singular_values": singular_values.tolist(),
        "captured_energy_fraction": captured.tolist(),
    }


def probe_target(
    jacobian: torch.Tensor,
    values: torch.Tensor,
    target: torch.Tensor,
    cfg: Config,
    slices: dict[str, slice],
) -> tuple[dict[str, Any], np.ndarray]:
    """Project one function-space direction onto ``Im(J)`` and summarise the result.

    ``jacobian`` and ``values`` depend only on the model, so both targets of a
    given state share them -- the Jacobian is built once per state.
    """
    error, angle, coefficients, resolved_rank, _ = tangent_projection(
        jacobian, target, cfg.tangent_svd_relative_cutoff
    )
    sweep = cutoff_sweep(jacobian.double(), target.double(), cfg.cutoff_sweep)

    index = cfg.cutoff_sweep.index(cfg.tangent_svd_relative_cutoff)
    assert abs(sweep["projection_errors"][index] - error) < 1e-8, "sweep disagrees with tangent_projection"

    target_norm = float(torch.linalg.vector_norm(target))
    coefficient_norm = float(torch.linalg.vector_norm(coefficients))
    summary = {
        "target_norm": target_norm,
        "function_norm": float(torch.linalg.vector_norm(values)),
        # Scale-free "how far from symmetric is f": the raw target norm grows with
        # the logit scale, which differs by orders of magnitude before/after training.
        "relative_defect": target_norm / float(torch.linalg.vector_norm(values).clamp_min(1e-15)),
        "projection_error": error,
        "principal_angle_degrees": angle,
        "resolved_rank": resolved_rank,
        "jacobian_norm": float(torch.linalg.matrix_norm(jacobian)),
        "coefficient_norm": coefficient_norm,
        # ||c|| for the *unit-norm* target: c is linear in the target, so this
        # divides out the logit scale and measures parameter cost per unit of
        # function-space motion along the direction.
        "coefficient_norm_per_unit_target": coefficient_norm / max(target_norm, 1e-15),
        "coefficient_participation_ratio": participation_ratio(coefficients.numpy()),
        "max_abs_coefficient": float(coefficients.abs().max()),
        "module_attribution": aggregate_by_module(coefficients.abs(), slices),
        "sweep": sweep,
    }
    return summary, coefficients.numpy()


# --------------------------------- plotting ---------------------------------
# The two runs share an initialisation by construction, so the two "initial"
# conditions get different dash patterns *and* the one drawn first is drawn
# fatter -- otherwise it hides exactly underneath the other and the figures look
# like they are missing a condition.
CONDITION_STYLE = {
    ("quarter", "initial"): ("tab:orange", ":", 3.2),
    ("quarter", "trained"): ("tab:red", "-", 1.6),
    ("full", "initial"): ("tab:cyan", "--", 1.6),
    ("full", "trained"): ("tab:blue", "-", 1.6),
}


def condition_style(label: str) -> tuple[str, str, float]:
    run, state = label.split("/")
    return CONDITION_STYLE.get((run, state), ("tab:grey", "-", 1.6))


TARGET_LABEL = {"xrot": r"$X_{\rm rot}f$", "delta": r"$\Delta = f - P(f)$"}


def plot_coefficient_distribution(
    coefficients: dict[str, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    module_labels: dict[str, str],
    target_name: str,
    output_dir: Path,
) -> None:
    """Distribution of the attribution coefficients c_i across the four conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)

    all_values = np.concatenate([np.abs(c) for c in coefficients.values()])
    positive = all_values[all_values > 0]
    # Bin geometrically down to the smallest coefficient actually present, so the
    # bin straddling zero stays narrow instead of swallowing a visible slab of
    # the distribution and showing up as a spurious plateau.
    linear_threshold = float(positive.min()) if positive.size else 1e-12
    largest = float(all_values.max())
    edges = np.concatenate([
        -np.geomspace(largest, linear_threshold, 70),
        [0.0],
        np.geomspace(linear_threshold, largest, 70),
    ])

    for label, c in coefficients.items():
        colour, style, line_width = condition_style(label)
        axes[0].hist(c, bins=edges, histtype="step", linewidth=line_width, linestyle=style,
                     color=colour, label=label)
        axes[1].plot(np.sort(np.abs(c))[::-1], linewidth=line_width, linestyle=style,
                     color=colour, label=label)
    axes[0].set_xscale("symlog", linthresh=linear_threshold)
    axes[0].set_yscale("log")
    # Label only well-separated decades: the symlog linear region is narrower than
    # the tick labels themselves, so ticks nearer zero than this just collide.
    decades = [10.0**e for e in range(-6, 1, 2) if 10.0**e <= largest]
    axes[0].set_xticks([-t for t in reversed(decades)] + decades)
    axes[0].xaxis.set_minor_locator(plt.NullLocator())
    axes[0].set(xlabel="$c_i$", ylabel="parameters per bin",
                title=f"attribution of {TARGET_LABEL[target_name]}")
    axes[0].legend(fontsize=8)

    axes[1].set(xscale="log", yscale="log", xlabel="rank of $|c_i|$", ylabel="$|c_i|$",
                title="sorted magnitudes")
    axes[1].legend(fontsize=8)

    modules = list(next(iter(summaries.values()))["module_attribution"])
    x_positions = np.arange(len(modules))
    width = 0.8 / len(summaries)
    for offset, (label, summary) in enumerate(summaries.items()):
        colour = condition_style(label)[0]
        heights = [summary["module_attribution"][m] for m in modules]
        axes[2].bar(x_positions + offset * width, heights, width=width, color=colour,
                    label=label, edgecolor="black", linewidth=0.4)
    axes[2].set_xticks(x_positions + width * (len(summaries) - 1) / 2)
    axes[2].set_xticklabels([module_labels[m] for m in modules], rotation=30, ha="right")
    axes[2].set(yscale="log", ylabel=r"$\|c\|$ within module",
                title="attribution by layer")
    axes[2].legend(fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"ci_distribution_{target_name}.{suffix}")
    plt.close(fig)


def plot_tangent_diagnostics(
    summaries: dict[str, dict[str, Any]], target_name: str, cfg: Config, output_dir: Path
) -> None:
    """Cutoff dependence of eps and ||c||, plus the cutoff-free energy-capture curve."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)

    for label, summary in summaries.items():
        colour, style, line_width = condition_style(label)
        sweep = summary["sweep"]
        axes[0].plot(sweep["cutoffs"], sweep["projection_errors"], marker="o", markersize=3,
                     linewidth=line_width, linestyle=style, color=colour, label=label)
        captured = np.asarray(sweep["captured_energy_fraction"])
        axes[1].plot(np.arange(1, captured.size + 1), 1.0 - captured, linewidth=line_width,
                     linestyle=style, color=colour, label=label)
        singular_values = np.asarray(sweep["singular_values"])
        axes[2].plot(np.arange(1, singular_values.size + 1), singular_values / singular_values[0],
                     linewidth=line_width, linestyle=style, color=colour, label=label)

    axes[0].axvline(cfg.tangent_svd_relative_cutoff, color="black", linestyle=":", linewidth=1)
    axes[0].set(xscale="log", xlabel="SVD relative cutoff", ylabel=r"$\varepsilon$",
                title=f"projection error of {TARGET_LABEL[target_name]}")
    axes[0].legend(fontsize=8)

    axes[1].set(xscale="log", yscale="log", xlabel="retained singular directions $k$",
                ylabel=f"unexplained energy of {TARGET_LABEL[target_name]}",
                title="cutoff-free capture curve")
    axes[1].legend(fontsize=8)

    axes[2].set(xscale="log", yscale="log", xlabel="index", ylabel=r"$\sigma_k/\sigma_1$",
                title="Jacobian spectrum")
    axes[2].legend(fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tangent_projection_diagnostics_{target_name}.{suffix}")
    plt.close(fig)


# ----------------------------------- main -----------------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="label=output_dir pairs pointing at aep_core_annulus_mlp runs")
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--tangent-svd-relative-cutoff", type=float)
    args = parser.parse_args()

    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.runs:
        cfg.runs = dict(pair.split("=", 1) for pair in args.runs)
    for name in ("device", "output_dir", "tangent_svd_relative_cutoff"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if cfg.tangent_svd_relative_cutoff not in cfg.cutoff_sweep:
        cfg.cutoff_sweep = sorted(cfg.cutoff_sweep + [cfg.tangent_svd_relative_cutoff])
    return cfg


def main() -> None:
    cfg = parse_args()
    device = select_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = build_probe_grid(cfg, device)
    print(f"probe grid: {points.shape[0]} points, |x| <= {cfg.probe_max_radius:g}")

    targets = list(TARGET_LABEL)
    summaries: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in targets}
    coefficients: dict[str, dict[str, np.ndarray]] = {t: {} for t in targets}
    flat_names: list[str] = []
    module_labels: dict[str, str] = {}

    for run_label, run_dir in cfg.runs.items():
        checkpoint = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
        run_cfg = checkpoint["config"]
        for state_label in ("initial", "trained"):
            key = "initial_state_dict" if state_label == "initial" else "final_state_dict"
            model = SimpleMLP(hidden_dim=run_cfg["hidden_dim"], depth=run_cfg["depth"])
            model.load_state_dict(checkpoint[key])
            model = model.to(device=device, dtype=torch.float64).eval()

            if not flat_names:
                flat_names, slices = parameter_layout(model)
                names = list(slices)
                module_labels = {n: prettify_parameter_name(n, names) for n in names}

            label = f"{run_label}/{state_label}"
            jacobian = sensitivity_jacobian(model, points)
            values, orbit = orbit_direction(model, points)
            _, defect = equivariance_defect(model, points, cfg.projection_group_order)
            quadrature_residual = projection_residual(model, points, cfg.projection_group_order)

            for target_name, target in (("xrot", orbit), ("delta", defect)):
                summary, c = probe_target(jacobian, values, target, cfg, slices)
                summary["dataset_outer_arc_degrees"] = run_cfg["outer_arc_degrees"]
                summary["projection_quadrature_residual"] = quadrature_residual
                summaries[target_name][label] = summary
                coefficients[target_name][label] = c

                print(
                    f"[{target_name:>5s}] {label:>16s}: ||target|| = {summary['target_norm']:.3e}"
                    f"  rel = {summary['relative_defect']:.3e}"
                    f"  eps = {summary['projection_error']:.3e}"
                    f"  angle = {summary['principal_angle_degrees']:.2f} deg"
                    f"  rank = {summary['resolved_rank']:4d}"
                    f"  ||c|| = {summary['coefficient_norm']:.3e}"
                    f"  ||c||/||target|| = {summary['coefficient_norm_per_unit_target']:.3e}"
                    f"  PR = {summary['coefficient_participation_ratio']:.1f}"
                )

                order = np.argsort(np.abs(c))[::-1][: cfg.top_parameters_to_report]
                write_csv_rows(
                    [{"flat_index": int(i), "name": flat_names[i], "coefficient": float(c[i])}
                     for i in order],
                    ["flat_index", "name", "coefficient"],
                    output_dir / f"top_{target_name}_parameters_{run_label}_{state_label}.csv",
                )

    # The two runs seed the model identically before touching their data, so the
    # two "initial" conditions are literally the same network; say so rather than
    # letting two identical curves silently overlap in the figures.
    initial_labels = [label for label in summaries[targets[0]] if label.endswith("/initial")]
    for first, second in zip(initial_labels, initial_labels[1:]):
        if np.array_equal(coefficients[targets[0]][first], coefficients[targets[0]][second]):
            print(f"note: {first} and {second} are the same parameters (identical seed), "
                  f"so their curves coincide by construction.")

    for target_name in targets:
        plot_coefficient_distribution(
            coefficients[target_name], summaries[target_name], module_labels,
            target_name, output_dir,
        )
        plot_tangent_diagnostics(summaries[target_name], target_name, cfg, output_dir)

    np.savez_compressed(
        output_dir / "attribution_coefficients.npz",
        **{f"{t}/{label}": c for t, per_state in coefficients.items()
           for label, c in per_state.items()},
    )
    (output_dir / "tangent_projection_summary.json").write_text(
        json.dumps({"config": asdict(cfg), "targets": summaries}, indent=2)
    )
    print(f"wrote {output_dir}/ci_distribution_{{{','.join(targets)}}}.png, "
          f"tangent_projection_diagnostics_*.png, tangent_projection_summary.json, "
          f"attribution_coefficients.npz")


if __name__ == "__main__":
    main()
