"""Functional-sensitivity experiment for the 2D isotropic double well ("Mexican hat").

The physical system is

    V(q1, q2; alpha) = 0.5*alpha*r^2 + 0.25*r^4,   r^2 = q1^2 + q2^2,
    dq/dt = p,
    dp1/dt = F1 = -q1*(alpha + r^2),
    dp2/dt = F2 = -q2*(alpha + r^2).

This system has an *exact continuous* SO(2) rotational symmetry for every alpha.

Two continuous generators are tested, both via the same tangent_projection
machinery as the other two scripts:

1. The bifurcation direction dF/dalpha = -(q1, q2).
2. X_rot F: the network's own current infinitesimal rotation
   generator since this system genuinely has continuous rotational symmetry
   at every alpha, so X_rot F is exactly zero for the true dynamics.

Both are combined via principal_angles_and_dimension into a
2-generator "representation dimension" (0, 1, or 2), and a discrete
finite-angle rotation check (at an arbitrary, non-special angle) is also
run as an independent cross-check.
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
from tqdm import tqdm

from direct_dynamics import DirectDynamicsMLP, DirectLeapfrogIntegrator, generic_residuals
from experiment_common import (
    apply_config_overrides,
    parameter_layout,
    plot_ei_initial_vs_final,
    plot_training_history,
    run_training_loop,
    select_device,
    select_dtype,
    write_csv_rows,
    write_result_json,
)
from mexican_hat_dynamics import (
    F,
    EquivariantHamiltonianMLP,
    GenericHamiltonianMLP,
    VerletIntegrator,
    generate_data,
    residuals,
    train_test_split,
)
from sensitivity_tools import (
    aggregate_by_module,
    aggregate_by_module_mean,
    finite_transform_residual,
    parameter_gradient_row,
    per_parameter_equivariance_error,
    principal_angles_and_dimension,
    relative_energy_drift,
    sensitivity_transform_residual,
    tangent_projection,
)


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "outputs/asrnn_mexican_hat_symmetry"

    architecture: str = "hamiltonian"  # hamiltonian, direct_mlp, or equivariant

    kinetic_hidden_dim: int = 50
    kinetic_hidden_layers: int = 2
    potential_hidden_dim: int = 50
    potential_hidden_layers: int = 2
    direct_mlp_hidden_dim: int = 50
    direct_mlp_hidden_layers: int = 2

    training_alphas: list[float] = field(
        default_factory=lambda: [-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4]
    )
    trajectory_window: int = 5
    trajectory_splits: int = 5
    sampled_instants: int = 5
    initial_conditions_per_alpha: int = 10
    integration_dt: float = 0.1
    coarsening_factor: int = 50
    validation_fraction: float = 0.25
    augment_dataset: bool = True

    optimizer: str = "adam"
    training_steps: int = 50000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    l1_regularization: float = 1e-5
    # Checkpoints are specified as fractions of training_steps (each in
    # [0, 1]) so they scale automatically if training_steps changes, e.g.
    # under --quick. Resolved to absolute step indices in validate_config.
    checkpoint_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
    checkpoint_steps: list[int] = field(default_factory=list)
    lbfgs_history_size: int = 10

    q_extent: float = 1.0
    q_grid_points_per_axis: int = 7
    # Deliberately not a multiple of 120 degrees, to make clear this is a
    # genuinely continuous symmetry, not a discrete point-group artefact.
    rotation_angle_degrees: float = 73.0

    analysis_alphas: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )

    tangent_svd_relative_cutoff: float = 1e-3
    representation_angle_threshold_degrees: float = 10.0
    top_parameters_to_report: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON file overriding fields in Config.")
    parser.add_argument("--quick", action="store_true", help="Run a very small end-to-end smoke test.")
    parser.add_argument("--device", help="Override Config.device.")
    parser.add_argument("--output-dir", help="Override Config.output_dir.")
    parser.add_argument(
        "--architecture",
        choices=["hamiltonian", "direct_mlp", "equivariant"],
        help="Override Config.architecture.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.architecture:
        cfg.architecture = args.architecture
    if args.quick:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.direct_mlp_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.coarsening_factor = 2
        cfg.training_steps = 2
        cfg.checkpoint_fractions = [0.0, 0.5, 1.0]
        cfg.analysis_alphas = [-0.5, 0.0, 0.5]
        cfg.q_grid_points_per_axis = 4
        cfg.top_parameters_to_report = 5
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.optimizer.lower() not in {"adam", "lbfgs"}:
        raise ValueError("optimizer must be 'adam' or 'lbfgs'.")
    if cfg.sampled_instants > cfg.trajectory_window:
        raise ValueError("sampled_instants cannot exceed trajectory_window.")
    if cfg.trajectory_splits < 1:
        raise ValueError("trajectory_splits must be at least 1.")
    if cfg.q_grid_points_per_axis < 2:
        raise ValueError("q_grid_points_per_axis must be at least 2.")
    if not 0.0 < cfg.tangent_svd_relative_cutoff < 1.0:
        raise ValueError("tangent_svd_relative_cutoff must lie strictly between 0 and 1.")
    if not all(0.0 <= f <= 1.0 for f in cfg.checkpoint_fractions):
        raise ValueError("checkpoint_fractions must all lie in [0, 1].")
    cfg.checkpoint_steps = sorted(
        {int(round(f * cfg.training_steps)) for f in cfg.checkpoint_fractions}
        | {0, cfg.training_steps}
    )

def make_dataset(cfg: Config, device: torch.device, dtype: torch.dtype):
    alphas = torch.tensor(cfg.training_alphas, device=device, dtype=dtype)
    total_length = cfg.trajectory_splits + cfg.trajectory_window - 1
    trajectories, params, indices = generate_data(
        alphas, F, total_length, cfg.trajectory_window, cfg.sampled_instants,
        dt=cfg.integration_dt, in_conds=cfg.initial_conditions_per_alpha,
        coarsening_factor=cfg.coarsening_factor, device=device, dtype=dtype,
        augment_dataset=cfg.augment_dataset,
    )
    return train_test_split(trajectories, params, indices, val_size=cfg.validation_fraction)


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    if cfg.architecture == "direct_mlp":
        model = DirectDynamicsMLP(
            p_dim=2, q_dim=2, param_dim=1,
            hidden_dim=cfg.direct_mlp_hidden_dim, n_hidden=cfg.direct_mlp_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        return model, DirectLeapfrogIntegrator(model=model, dt=cfg.integration_dt)
    if cfg.architecture == "equivariant":
        model = EquivariantHamiltonianMLP(
            kin_hidden_dim=cfg.kinetic_hidden_dim, kin_n_hidden=cfg.kinetic_hidden_layers,
            pot_hidden_dim=cfg.potential_hidden_dim, pot_n_hidden=cfg.potential_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        return model, VerletIntegrator(model=model, dt=cfg.integration_dt)
    model = GenericHamiltonianMLP(
        q_dim=2, p_dim=2, param_dim=1,
        kin_hidden_dim=cfg.kinetic_hidden_dim, kin_n_hidden=cfg.kinetic_hidden_layers,
        pot_hidden_dim=cfg.potential_hidden_dim, pot_n_hidden=cfg.potential_hidden_layers,
        device=device,
    ).to(device=device, dtype=dtype)
    return model, VerletIntegrator(model=model, dt=cfg.integration_dt)


def make_optimizer(cfg: Config, model: torch.nn.Module):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.LBFGS(
        model.parameters(), lr=cfg.learning_rate, history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-12,
    )


def build_probe_grid(cfg: Config, device: torch.device, dtype: torch.dtype):
    axis = torch.linspace(-cfg.q_extent, cfg.q_extent, cfg.q_grid_points_per_axis, device=device, dtype=dtype)
    q1_mesh, q2_mesh = torch.meshgrid(axis, axis, indexing="ij")
    return q1_mesh.reshape(-1), q2_mesh.reshape(-1)


def rotation_matrix(theta_degrees: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = math.radians(theta_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def transform_points(q1: torch.Tensor, q2: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.stack([q1, q2], dim=1)
    qt = q @ matrix.T
    return qt[:, 0], qt[:, 1]


def bifurcation_generator_target(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """dF_true/dalpha = -(q1, q2), the direct 2D analogue of the double well's dF/dalpha = -q."""
    return -torch.stack([q1, q2], dim=1)


ROTATION_LIE_GENERATOR = torch.tensor([[0.0, -1.0], [1.0, 0.0]])  # d/dtheta R_theta |_0


def rotation_generator_target(
    q1: torch.Tensor, q2: torch.Tensor, force: torch.Tensor, spatial_jacobian: torch.Tensor
) -> torch.Tensor:
    """X_rot F = (dF/dq)(Omega q) - Omega F(q) (PDF sec. 0.2, extended to a vector field).

    Verified numerically against the exact analytic Mexican-hat force before
    use (relative residual ~1e-16 at every alpha) -- unlike Henon-Heiles,
    this system genuinely has continuous rotational symmetry, so this is a
    valid generator here.
    """
    omega_q = torch.stack([-q2, q1], dim=1)
    directional = torch.einsum("nkj,nj->nk", spatial_jacobian, omega_q)
    omega = ROTATION_LIE_GENERATOR.to(device=force.device, dtype=force.dtype)
    return directional - force @ omega.T


def evaluate_at_points(
    model: torch.nn.Module, q1_pts: torch.Tensor, q2_pts: torch.Tensor, alpha: float, architecture: str,
    *, device: torch.device, dtype: torch.dtype, need_spatial_jacobian: bool = False,
):
    """Evaluate learned potential, force, force-parameter Jacobian (and optionally dF/dq) at each point.

    Returns ``(V, F, J[, spatial_J])`` with shapes ``[N]``, ``[N, 2]``,
    ``[N, 2, P]`` (``[N, 2, 2]`` for spatial_J). No V_net for direct_mlp (V is
    NaN there); force read from the model's own dp/dt output at p=(0,0).
    """
    params = tuple(model.parameters())
    v_values, f_values, jac_rows, spatial_jacs = [], [], [], []
    for q1_val, q2_val in zip(q1_pts.detach().cpu().tolist(), q2_pts.detach().cpu().tolist()):
        q = torch.tensor([[q1_val, q2_val]], device=device, dtype=dtype, requires_grad=True)
        alpha_t = torch.tensor([[alpha]], device=device, dtype=dtype)
        if architecture == "direct_mlp":
            p = torch.zeros_like(q)
            dpdt, _ = model(p, q, alpha_t)
            force = dpdt.squeeze(0)
            v_values.append(float("nan"))
        else:
            potential = model.V_net(torch.cat((q, alpha_t), dim=1))
            force = -torch.autograd.grad(potential.sum(), q, create_graph=True)[0].squeeze(0)
            v_values.append(float(potential.detach().cpu()))
        row0 = parameter_gradient_row(force[0], params)
        row1 = parameter_gradient_row(force[1], params)
        f_values.append(force.detach())
        jac_rows.append(torch.stack([row0, row1]))
        if need_spatial_jacobian:
            d_f1_dq = torch.autograd.grad(force[0], q, retain_graph=True)[0].squeeze(0)
            d_f2_dq = torch.autograd.grad(force[1], q, retain_graph=False)[0].squeeze(0)
            spatial_jacs.append(torch.stack([d_f1_dq, d_f2_dq]).detach())
    v = torch.tensor(v_values, device=device, dtype=dtype)
    f = torch.stack(f_values)
    j = torch.stack(jac_rows)
    if need_spatial_jacobian:
        return v, f, j, torch.stack(spatial_jacs)
    return v, f, j


def learned_dqdt(
    model: torch.nn.Module, p1_values: torch.Tensor, p2_values: torch.Tensor, alpha: float, architecture: str
) -> torch.Tensor:
    p = torch.stack([p1_values, p2_values], dim=1)
    if architecture == "direct_mlp":
        q = torch.zeros_like(p)
        alpha_t = torch.full((p.shape[0], 1), float(alpha), device=p.device, dtype=p.dtype)
        with torch.no_grad():
            _, dqdt = model(p, q, alpha_t)
        return dqdt
    p = p.clone().requires_grad_(True)
    kinetic = model.K_net(p)
    dqdt = torch.autograd.grad(kinetic.sum(), p)[0]
    return dqdt.detach()


def analyse_checkpoint(
    model: torch.nn.Module, step: int, cfg: Config, device: torch.device, dtype: torch.dtype,
    flat_names: list[str], parameter_slices: dict[str, slice],
) -> dict[str, Any]:
    model.eval()
    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    rot_mat = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    q1_rot, q2_rot = transform_points(q1_grid, q2_grid, rot_mat)
    p1_grid, p2_grid = build_probe_grid(cfg, device, dtype)
    grad_e_p = torch.stack([p1_grid, p2_grid], dim=1)
    n_points = q1_grid.shape[0]

    alpha_results = []
    for alpha in tqdm(cfg.analysis_alphas, desc=f"analysing step {step}", leave=False):
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_grid, q2_grid, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
        )
        v_rot, f_rot, j_rot = evaluate_at_points(
            model, q1_rot, q2_rot, alpha, cfg.architecture, device=device, dtype=dtype
        )

        rotation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_rot.unsqueeze(-1))
        rotation_force_residual = finite_transform_residual(f_x, f_rot, rot_mat)
        rotation_sensitivity_residual = sensitivity_transform_residual(j_x, j_rot, rot_mat)
        rotation_equivariance_error = per_parameter_equivariance_error(j_x, j_rot, rot_mat)

        sensitivity = torch.sqrt(torch.mean(j_x.detach().cpu().double().square(), dim=(0, 1)))
        jac_flat = j_x.reshape(n_points * 2, -1)

        bifurcation_target = bifurcation_generator_target(q1_grid, q2_grid)
        bifurcation_target_flat = bifurcation_target.reshape(n_points * 2)
        (
            bifurcation_projection_error, bifurcation_principal_angle,
            bifurcation_coefficients, bifurcation_resolved_rank, bifurcation_singular_values,
        ) = tangent_projection(jac_flat, bifurcation_target_flat, cfg.tangent_svd_relative_cutoff)

        xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
        xrot_target_flat = xrot_target.reshape(n_points * 2)
        (
            xrot_projection_error, xrot_principal_angle,
            xrot_coefficients, xrot_resolved_rank, xrot_singular_values,
        ) = tangent_projection(jac_flat, xrot_target_flat, cfg.tangent_svd_relative_cutoff)

        generator_matrix = torch.stack([bifurcation_target_flat, xrot_target_flat], dim=1)
        dimension_info = principal_angles_and_dimension(
            jac_flat, generator_matrix, cfg.tangent_svd_relative_cutoff,
            angle_threshold_degrees=cfg.representation_angle_threshold_degrees,
        )

        learned_dqdt_values = learned_dqdt(model, p1_grid, p2_grid, alpha, cfg.architecture)
        grad_e_q = float(alpha) * torch.stack([q1_grid, q2_grid], dim=1) + (
            (q1_grid**2 + q2_grid**2).unsqueeze(-1) * torch.stack([q1_grid, q2_grid], dim=1)
        )
        energy_conservation_violation = relative_energy_drift(
            grad_e_p.detach().cpu().double(), grad_e_q.detach().cpu().double(),
            f_x.detach().cpu().double(), learned_dqdt_values.detach().cpu().double()
        )

        alpha_results.append(
            {
                "alpha": float(alpha),
                "rotation_potential_residual": rotation_potential_residual,
                "rotation_force_residual": rotation_force_residual,
                "rotation_sensitivity_residual": rotation_sensitivity_residual,
                "energy_conservation_violation": energy_conservation_violation,
                "sensitivity": sensitivity.cpu().tolist(),
                "rotation_equivariance_error_by_parameter": rotation_equivariance_error.cpu().tolist(),
                "module_sensitivity": aggregate_by_module(sensitivity, parameter_slices),
                "module_rotation_equivariance_error": aggregate_by_module_mean(
                    rotation_equivariance_error, parameter_slices
                ),
                "bifurcation_projection_error": bifurcation_projection_error,
                "bifurcation_principal_angle_degrees": bifurcation_principal_angle,
                "bifurcation_resolved_rank": bifurcation_resolved_rank,
                "bifurcation_singular_values": bifurcation_singular_values,
                "bifurcation_attribution_coefficients": bifurcation_coefficients.cpu().tolist(),
                "module_bifurcation_attribution": aggregate_by_module(
                    bifurcation_coefficients.abs(), parameter_slices
                ),
                "xrot_projection_error": xrot_projection_error,
                "xrot_principal_angle_degrees": xrot_principal_angle,
                "xrot_resolved_rank": xrot_resolved_rank,
                "xrot_singular_values": xrot_singular_values,
                "xrot_attribution_coefficients": xrot_coefficients.cpu().tolist(),
                "module_xrot_attribution": aggregate_by_module(xrot_coefficients.abs(), parameter_slices),
                "joint_representation_dimension": dimension_info["representation_dimension"],
                "joint_principal_angles_degrees": dimension_info["principal_angles_degrees"],
            }
        )

    coefficient_matrix = np.asarray([r["bifurcation_attribution_coefficients"] for r in alpha_results])
    bifurcation_score = np.sqrt(np.mean(coefficient_matrix**2, axis=0))
    top_bifurcation = np.argsort(bifurcation_score)[::-1][: cfg.top_parameters_to_report]

    xrot_coefficient_matrix = np.asarray([r["xrot_attribution_coefficients"] for r in alpha_results])
    xrot_score = np.sqrt(np.mean(xrot_coefficient_matrix**2, axis=0))
    top_xrot = np.argsort(xrot_score)[::-1][: cfg.top_parameters_to_report]

    equivariance_matrix = np.asarray([r["rotation_equivariance_error_by_parameter"] for r in alpha_results])
    equivariance_score = np.sqrt(np.mean(equivariance_matrix**2, axis=0))
    top_equivariance_violators = np.argsort(equivariance_score)[::-1][: cfg.top_parameters_to_report]

    return {
        "step": step,
        "alpha_results": alpha_results,
        "top_bifurcation_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(bifurcation_score[i])}
            for i in top_bifurcation
        ],
        "top_xrot_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xrot_score[i])}
            for i in top_xrot
        ],
        "top_equivariance_violating_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_equivariance_error": float(equivariance_score[i])}
            for i in top_equivariance_violators
        ],
    }


def write_checkpoint_outputs(result: dict[str, Any], output_dir: Path) -> None:
    step = result["step"]
    write_result_json(result, output_dir, step)
    write_csv_rows(
        result["top_bifurcation_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_bifurcation_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_xrot_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_xrot_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_equivariance_violating_parameters"], ["flat_index", "name", "rms_equivariance_error"],
        output_dir / f"top_equivariance_violations_step_{step:06d}.csv",
    )


def plot_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Core Experiment-1 story: random init vs trained, across alpha (including the bifurcation)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(all_results)))

    for colour, checkpoint in zip(colours, all_results):
        rows = checkpoint["alpha_results"]
        alphas = np.asarray([r["alpha"] for r in rows])
        label = f"step {checkpoint['step']}"
        axes[0, 0].plot(
            alphas, np.maximum([r["rotation_force_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[0, 1].plot(
            alphas, np.maximum([r["rotation_sensitivity_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[0, 2].plot(
            alphas, [r["bifurcation_projection_error"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 0].plot(
            alphas, [r["xrot_projection_error"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 1].plot(
            alphas, [r["joint_representation_dimension"] for r in rows],
            marker="o", ms=5, color=colour, label=label,
        )
        axes[1, 2].plot(
            alphas, np.maximum([r["energy_conservation_violation"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )

    axes[0, 0].set(title="Finite rotation (73 deg): force equivariance", ylabel="relative residual", yscale="log")
    axes[0, 1].set(title="Finite rotation: sensitivity equivariance", ylabel="relative Frobenius residual", yscale="log")
    axes[0, 2].set(title=r"Bifurcation generator $\partial_\alpha F=-(q_1,q_2)$", ylabel="relative projection error")
    axes[1, 0].set(title=r"$X_{\rm rot}F$ (genuine continuous generator)", ylabel="relative projection error")
    axes[1, 1].set(
        title="Joint representation dimension (bifurcation + rotation)",
        ylabel="# generators represented (of 2)", yticks=[0, 1, 2],
    )
    axes[1, 2].set(title="Noether energy-conservation violation", ylabel="relative energy drift", yscale="log")
    for ax in axes.flat:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "mexican_hat_summary.png", dpi=200)
    fig.savefig(output_dir / "mexican_hat_summary.pdf")
    plt.close(fig)


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    first, last = all_results[0], all_results[-1]

    def central_row(checkpoint):
        rows = checkpoint["alpha_results"]
        return rows[len(rows) // 2]

    first_row, last_row = central_row(first), central_row(last)
    modules = list(first_row["module_rotation_equivariance_error"].keys())
    before = [first_row["module_rotation_equivariance_error"][m] for m in modules]
    after = [last_row["module_rotation_equivariance_error"][m] for m in modules]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    ax.bar(x - width / 2, before, width, label=f"step {first['step']} (random init)")
    ax.bar(x + width / 2, after, width, label=f"step {last['step']} (trained)")
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=60, ha="right", fontsize=8)
    ax.set(title="Rotation sensitivity-equivariance error $E_i$ by module", ylabel="mean $E_i$ within module")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200)
    fig.savefig(output_dir / "equivariance_by_module.pdf")
    plt.close(fig)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    first, last = all_results[0], all_results[-1]
    first_row = first["alpha_results"][len(first["alpha_results"]) // 2]
    last_row = last["alpha_results"][len(last["alpha_results"]) // 2]
    ei_initial = np.asarray(first_row["rotation_equivariance_error_by_parameter"])
    ei_final = np.asarray(last_row["rotation_equivariance_error_by_parameter"])
    plot_ei_initial_vs_final(
        ei_initial, ei_final, parameter_slices,
        title="Rotation sensitivity-equivariance $E_i$: init vs. trained",
        output_stem=output_dir / "equivariance_scatter",
    )


def train_and_analyse(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"Device: {device}; dtype: {dtype}; output: {output_dir}")
    print("Generating sparse Mexican-hat trajectories...")
    train_data, validation_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}")

    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *args, **kwargs: generic_residuals(*args, **kwargs, p_dim=2)
    )
    history, checkpoint_states = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=validation_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_regularization=cfg.l1_regularization,
    )

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    np.savez(output_dir / "training_history.npz", **history)
    plot_training_history(history, output_dir)

    all_results = []
    for step in tqdm(cfg.checkpoint_steps, desc="checkpoint analysis", unit="checkpoint"):
        model.load_state_dict(checkpoint_states[step])
        result = analyse_checkpoint(model, step, cfg, device, dtype, flat_names, parameter_slices)
        write_checkpoint_outputs(result, output_dir)
        all_results.append(result)

    plot_summary(all_results, output_dir)
    plot_equivariance_by_module(all_results, output_dir)
    plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()