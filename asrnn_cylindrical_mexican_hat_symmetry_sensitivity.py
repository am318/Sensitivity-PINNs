"""Functional-sensitivity experiment for the 3D "cylindrical Mexican hat".

The physical system is

    V(q1, q2, q3; alpha) = 0.5*alpha*r^2 + 0.25*r^4,   r^2 = q1^2 + q2^2,
    dq/dt = p,
    dp1/dt = F1 = -q1*(alpha + r^2),
    dp2/dt = F2 = -q2*(alpha + r^2),
    dp3/dt = F3 = 0.

Unlike the 2D Mexican-hat (which has only rotation), this system has *two*
independent, exact continuous symmetries at once: SO(2) rotation in the
(q1, q2) plane, and continuous translation in q3 (V has no q3-dependence, so
p3 is exactly conserved). Both generators are verified numerically against
the exact analytic force before use (X_rot F ~ 1e-15, X_trans F exactly 0.0
-- see the verification run in the session that built this file).

Unlike the earlier rotation-vs-energy-conservation comparison (which needed
a structurally different "project the violation onto its own Jacobian"
construction for energy, since time-translation isn't a domain group action),
translation here IS a genuine group action on the input domain, exactly like
rotation -- so both generators are tested with the *same* machinery:
finite-transform residuals, per-parameter sensitivity-equivariance E_i, and
attribution c_i via tangent_projection of the network's own current
X_rot F / X_trans F onto its own force-sensitivity basis. This is the first
directly apples-to-apples two-symmetry comparison in this project (PDF sec.
0.3's own example pairing: "which layers primarily represent translations?
which represent rotations?").

Only the hamiltonian and direct_mlp architectures are implemented (the
escnn-equivariant architecture would need a mixed-representation setup --
trivial on q3, SO(2) on (q1,q2) -- left for future work).
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

from cylindrical_mexican_hat_dynamics import F, generate_data, train_test_split
from direct_dynamics import DirectDynamicsMLP, DirectLeapfrogIntegrator, generic_residuals
from experiment_common import (
    apply_config_overrides,
    module_colours,
    parameter_layout,
    plot_ei_initial_vs_final,
    plot_magnitude_vs_quantity,
    plot_training_history,
    prettify_parameter_name,
    run_training_loop,
    select_device,
    select_dtype,
    write_csv_rows,
    write_result_json,
)
from mexican_hat_dynamics import GenericHamiltonianMLP, VerletIntegrator
from sensitivity_tools import (
    aggregate_by_module,
    aggregate_by_module_mean,
    finite_transform_residual,
    parameter_gradient_row,
    participation_ratio,
    per_parameter_equivariance_error,
    relative_energy_drift,
    sensitivity_transform_residual,
    tangent_projection,
)


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "outputs/asrnn_cylindrical_mexican_hat_symmetry"

    architecture: str = "mlp"  # hamiltonian or direct_mlp

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
    # Doubles the training set with a copy transformed by a random rotation (in q1,q2)
    # and an independent random shift (in q3) -- both exact symmetries of this system's
    # true dynamics for every alpha (verified numerically before use), so this is free,
    # exactly-valid additional data teaching both symmetries implicitly through data
    # diversity, as an alternative/complement to architectural constraints or L1.
    augment_dataset: bool = False

    optimizer: str = "adam"
    training_steps: int = 20000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    l1_weight: float = 1e-4
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [0, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    )
    lbfgs_history_size: int = 10

    q_extent: float = 1.0
    q_grid_points_per_axis: int = 5
    z_grid_points: int = 3
    # Deliberately not a multiple of 120 degrees, matching the 2D Mexican-hat convention.
    rotation_angle_degrees: float = 73.0
    translation_shift: float = 0.6

    analysis_alphas: list[float] = field(
        default_factory=lambda: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )

    tangent_svd_relative_cutoff: float = 1e-3
    top_parameters_to_report: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON file overriding fields in Config.")
    parser.add_argument("--quick", action="store_true", help="Run a very small end-to-end smoke test.")
    parser.add_argument("--device", help="Override Config.device.")
    parser.add_argument("--output-dir", help="Override Config.output_dir.")
    parser.add_argument("--architecture", choices=["hamiltonian", "direct_mlp"], help="Override Config.architecture.")
    parser.add_argument("--l1-weight", type=float, help="Override Config.l1_weight.")
    parser.add_argument("--augment-dataset", dest="augment_dataset", action="store_true", default=None, help="Override Config.augment_dataset to True.")
    parser.add_argument("--no-augment-dataset", dest="augment_dataset", action="store_false", help="Override Config.augment_dataset to False.")
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
    if args.l1_weight is not None:
        cfg.l1_weight = args.l1_weight
    if args.augment_dataset is not None:
        cfg.augment_dataset = args.augment_dataset
    if args.quick:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.direct_mlp_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.coarsening_factor = 2
        cfg.training_steps = 2
        cfg.checkpoint_steps = [0, 1, 2]
        cfg.analysis_alphas = [-0.5, 0.0, 0.5]
        cfg.q_grid_points_per_axis = 3
        cfg.z_grid_points = 2
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
    if cfg.q_grid_points_per_axis < 2 or cfg.z_grid_points < 2:
        raise ValueError("q_grid_points_per_axis and z_grid_points must be at least 2.")
    if not 0.0 < cfg.tangent_svd_relative_cutoff < 1.0:
        raise ValueError("tangent_svd_relative_cutoff must lie strictly between 0 and 1.")
    cfg.checkpoint_steps = sorted(
        {int(s) for s in cfg.checkpoint_steps if 0 <= int(s) <= cfg.training_steps}
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
            p_dim=3, q_dim=3, param_dim=1,
            hidden_dim=cfg.direct_mlp_hidden_dim, n_hidden=cfg.direct_mlp_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        return model, DirectLeapfrogIntegrator(model=model, dt=cfg.integration_dt)
    model = GenericHamiltonianMLP(
        q_dim=3, p_dim=3, param_dim=1,
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
    z_axis = torch.linspace(-cfg.q_extent, cfg.q_extent, cfg.z_grid_points, device=device, dtype=dtype)
    q1_mesh, q2_mesh, q3_mesh = torch.meshgrid(axis, axis, z_axis, indexing="ij")
    return q1_mesh.reshape(-1), q2_mesh.reshape(-1), q3_mesh.reshape(-1)


def rotation_matrix_3d(theta_degrees: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = math.radians(theta_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)


def rotate_points(q1, q2, q3, matrix: torch.Tensor):
    q = torch.stack([q1, q2, q3], dim=1)
    qt = q @ matrix.T
    return qt[:, 0], qt[:, 1], qt[:, 2]


def translate_points(q1, q2, q3, shift: float):
    return q1, q2, q3 + shift


ROTATION_LIE_GENERATOR = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # d/dtheta R_theta |_0


def rotation_generator_target(q1, q2, q3, force, spatial_jacobian) -> torch.Tensor:
    """X_rot F = (dF/dq)(Omega q) - Omega F(q), rotation acting only on (q1, q2).

    Verified numerically against the exact analytic force before use
    (relative residual ~1e-16 at every alpha).
    """
    omega_q = torch.stack([-q2, q1, torch.zeros_like(q3)], dim=1)
    directional = torch.einsum("nkj,nj->nk", spatial_jacobian, omega_q)
    omega = ROTATION_LIE_GENERATOR.to(device=force.device, dtype=force.dtype)
    return directional - force @ omega.T


def translation_generator_target(spatial_jacobian) -> torch.Tensor:
    """X_trans F = -dF/dq3 (translation in q3, PDF sec. 0.2's X_trans f = -d_x f).

    Verified numerically against the exact analytic force before use
    (exactly 0.0, since F has no q3-dependence at all).
    """
    return -spatial_jacobian[:, :, 2]


def evaluate_at_points(
    model: torch.nn.Module, q1_pts, q2_pts, q3_pts, alpha: float, architecture: str,
    *, device: torch.device, dtype: torch.dtype, need_spatial_jacobian: bool = False,
):
    """Evaluate learned potential, force, force-parameter Jacobian (and optionally dF/dq) at each point.

    Returns ``(V, F, J[, spatial_J])`` with shapes ``[N]``, ``[N, 3]``,
    ``[N, 3, P]`` (``[N, 3, 3]`` for spatial_J).
    """
    params = tuple(model.parameters())
    v_values, f_values, jac_rows, spatial_jacs = [], [], [], []
    for q1_v, q2_v, q3_v in zip(
        q1_pts.detach().cpu().tolist(), q2_pts.detach().cpu().tolist(), q3_pts.detach().cpu().tolist(),
    ):
        q = torch.tensor([[q1_v, q2_v, q3_v]], device=device, dtype=dtype, requires_grad=True)
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
        rows = [parameter_gradient_row(force[k], params) for k in range(3)]
        f_values.append(force.detach())
        jac_rows.append(torch.stack(rows))
        if need_spatial_jacobian:
            spatial_rows = [
                torch.autograd.grad(force[k], q, retain_graph=(k < 2), create_graph=False)[0].squeeze(0)
                for k in range(3)
            ]
            spatial_jacs.append(torch.stack(spatial_rows).detach())
    v = torch.tensor(v_values, device=device, dtype=dtype)
    f = torch.stack(f_values)
    j = torch.stack(jac_rows)
    if need_spatial_jacobian:
        return v, f, j, torch.stack(spatial_jacs)
    return v, f, j


def learned_dqdt(model: torch.nn.Module, p1, p2, p3, alpha: float, architecture: str) -> torch.Tensor:
    p = torch.stack([p1, p2, p3], dim=1)
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
    parameter_magnitude = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()]).cpu().tolist()
    q1_grid, q2_grid, q3_grid = build_probe_grid(cfg, device, dtype)
    rot_mat = rotation_matrix_3d(cfg.rotation_angle_degrees, device, dtype)
    q1_rot, q2_rot, q3_rot = rotate_points(q1_grid, q2_grid, q3_grid, rot_mat)
    q1_shift, q2_shift, q3_shift = translate_points(q1_grid, q2_grid, q3_grid, cfg.translation_shift)
    p1_grid, p2_grid, p3_grid = build_probe_grid(cfg, device, dtype)
    grad_e_p = torch.stack([p1_grid, p2_grid, p3_grid], dim=1)
    n_points = q1_grid.shape[0]

    alpha_results = []
    for alpha in tqdm(cfg.analysis_alphas, desc=f"analysing step {step}", leave=False):
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_grid, q2_grid, q3_grid, alpha, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        v_rot, f_rot, j_rot = evaluate_at_points(
            model, q1_rot, q2_rot, q3_rot, alpha, cfg.architecture, device=device, dtype=dtype
        )
        v_shift, f_shift, j_shift = evaluate_at_points(
            model, q1_shift, q2_shift, q3_shift, alpha, cfg.architecture, device=device, dtype=dtype
        )

        rotation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_rot.unsqueeze(-1))
        rotation_force_residual = finite_transform_residual(f_x, f_rot, rot_mat)
        rotation_sensitivity_residual = sensitivity_transform_residual(j_x, j_rot, rot_mat)
        rotation_equivariance_error = per_parameter_equivariance_error(j_x, j_rot, rot_mat)

        translation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_shift.unsqueeze(-1))
        translation_force_residual = finite_transform_residual(f_x, f_shift, None)
        translation_sensitivity_residual = sensitivity_transform_residual(j_x, j_shift, None)
        translation_equivariance_error = per_parameter_equivariance_error(j_x, j_shift, None)

        sensitivity = torch.sqrt(torch.mean(j_x.double().square(), dim=(0, 1)))
        jac_flat = j_x.reshape(n_points * 3, -1)

        xrot_target = rotation_generator_target(q1_grid, q2_grid, q3_grid, f_x, spatial_jac_x)
        xrot_target_flat = xrot_target.reshape(n_points * 3)
        (
            xrot_projection_error, xrot_principal_angle,
            xrot_coefficients, xrot_resolved_rank, xrot_singular_values,
        ) = tangent_projection(jac_flat, xrot_target_flat, cfg.tangent_svd_relative_cutoff)

        xtrans_target = translation_generator_target(spatial_jac_x)
        xtrans_target_flat = xtrans_target.reshape(n_points * 3)
        (
            xtrans_projection_error, xtrans_principal_angle,
            xtrans_coefficients, xtrans_resolved_rank, xtrans_singular_values,
        ) = tangent_projection(jac_flat, xtrans_target_flat, cfg.tangent_svd_relative_cutoff)

        learned_dqdt_values = learned_dqdt(model, p1_grid, p2_grid, p3_grid, alpha, cfg.architecture)
        grad_e_q = float(alpha) * torch.stack([q1_grid, q2_grid, torch.zeros_like(q3_grid)], dim=1) + (
            (q1_grid**2 + q2_grid**2).unsqueeze(-1)
            * torch.stack([q1_grid, q2_grid, torch.zeros_like(q3_grid)], dim=1)
        )
        energy_conservation_violation = relative_energy_drift(
            grad_e_p.double(), grad_e_q.double(), f_x.double(), learned_dqdt_values.double()
        )

        alpha_results.append(
            {
                "alpha": float(alpha),
                "rotation_potential_residual": rotation_potential_residual,
                "rotation_force_residual": rotation_force_residual,
                "rotation_sensitivity_residual": rotation_sensitivity_residual,
                "translation_potential_residual": translation_potential_residual,
                "translation_force_residual": translation_force_residual,
                "translation_sensitivity_residual": translation_sensitivity_residual,
                "energy_conservation_violation": energy_conservation_violation,
                "sensitivity": sensitivity.cpu().tolist(),
                "rotation_equivariance_error_by_parameter": rotation_equivariance_error.cpu().tolist(),
                "translation_equivariance_error_by_parameter": translation_equivariance_error.cpu().tolist(),
                "module_sensitivity": aggregate_by_module(sensitivity, parameter_slices),
                "module_rotation_equivariance_error": aggregate_by_module_mean(
                    rotation_equivariance_error, parameter_slices
                ),
                "module_translation_equivariance_error": aggregate_by_module_mean(
                    translation_equivariance_error, parameter_slices
                ),
                "xrot_projection_error": xrot_projection_error,
                "xrot_principal_angle_degrees": xrot_principal_angle,
                "xrot_resolved_rank": xrot_resolved_rank,
                "xrot_singular_values": xrot_singular_values,
                "xrot_attribution_coefficients": xrot_coefficients.cpu().tolist(),
                "module_xrot_attribution": aggregate_by_module(xrot_coefficients.abs(), parameter_slices),
                "xtrans_projection_error": xtrans_projection_error,
                "xtrans_principal_angle_degrees": xtrans_principal_angle,
                "xtrans_resolved_rank": xtrans_resolved_rank,
                "xtrans_singular_values": xtrans_singular_values,
                "xtrans_attribution_coefficients": xtrans_coefficients.cpu().tolist(),
                "module_xtrans_attribution": aggregate_by_module(xtrans_coefficients.abs(), parameter_slices),
            }
        )

    xrot_coefficient_matrix = np.asarray([r["xrot_attribution_coefficients"] for r in alpha_results])
    xrot_score = np.sqrt(np.mean(xrot_coefficient_matrix**2, axis=0))
    top_xrot = np.argsort(xrot_score)[::-1][: cfg.top_parameters_to_report]

    xtrans_coefficient_matrix = np.asarray([r["xtrans_attribution_coefficients"] for r in alpha_results])
    xtrans_score = np.sqrt(np.mean(xtrans_coefficient_matrix**2, axis=0))
    top_xtrans = np.argsort(xtrans_score)[::-1][: cfg.top_parameters_to_report]

    rotation_equivariance_matrix = np.asarray(
        [r["rotation_equivariance_error_by_parameter"] for r in alpha_results]
    )
    rotation_equivariance_score = np.sqrt(np.mean(rotation_equivariance_matrix**2, axis=0))

    translation_equivariance_matrix = np.asarray(
        [r["translation_equivariance_error_by_parameter"] for r in alpha_results]
    )
    translation_equivariance_score = np.sqrt(np.mean(translation_equivariance_matrix**2, axis=0))

    vnet_mask = np.array([name.startswith("V_net.") for name in flat_names])
    return {
        "step": step,
        "parameter_magnitude": parameter_magnitude,
        "alpha_results": alpha_results,
        "xrot_score": xrot_score.tolist(),
        "xtrans_score": xtrans_score.tolist(),
        "rotation_equivariance_score": rotation_equivariance_score.tolist(),
        "translation_equivariance_score": translation_equivariance_score.tolist(),
        "xrot_participation_ratio": participation_ratio(xrot_score[vnet_mask]) if vnet_mask.any() else participation_ratio(xrot_score),
        "xtrans_participation_ratio": participation_ratio(xtrans_score[vnet_mask]) if vnet_mask.any() else participation_ratio(xtrans_score),
        "top_xrot_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xrot_score[i])}
            for i in top_xrot
        ],
        "top_xtrans_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xtrans_score[i])}
            for i in top_xtrans
        ],
    }


def write_checkpoint_outputs(result: dict[str, Any], output_dir: Path) -> None:
    step = result["step"]
    write_result_json(result, output_dir, step)
    write_csv_rows(
        result["top_xrot_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_xrot_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_xtrans_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_xtrans_parameters_step_{step:06d}.csv",
    )


def plot_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Core Experiment-1 story: random init vs trained, across alpha, for both generators."""
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
            alphas, np.maximum([r["translation_force_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[0, 2].plot(
            alphas, np.maximum([r["energy_conservation_violation"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 0].plot(
            alphas, [r["xrot_projection_error"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 1].plot(
            alphas, [r["xtrans_projection_error"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 2].plot(
            alphas, np.maximum([r["rotation_sensitivity_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label="rotation " + label,
        )
        axes[1, 2].plot(
            alphas, np.maximum([r["translation_sensitivity_residual"] for r in rows], 1e-12),
            marker="s", ms=3, color=colour, linestyle="--", label="translation " + label,
        )

    axes[0, 0].set(title="Finite rotation (73 deg): force equivariance", ylabel="relative residual", yscale="log")
    axes[0, 1].set(title="Finite translation (z-shift): force invariance", ylabel="relative residual", yscale="log")
    axes[0, 2].set(title="Noether energy-conservation violation", ylabel="relative energy drift", yscale="log")
    axes[1, 0].set(title=r"$X_{\rm rot}F$ projection error", ylabel="relative projection error")
    axes[1, 1].set(title=r"$X_{\rm trans}F$ (z) projection error", ylabel="relative projection error")
    axes[1, 2].set(title="Finite-transform sensitivity equivariance", ylabel="relative Frobenius residual", yscale="log")
    for ax in axes.flat:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "cylindrical_mexican_hat_summary.png", dpi=200)
    fig.savefig(output_dir / "cylindrical_mexican_hat_summary.pdf")
    plt.close(fig)


def _alpha_averaged(checkpoint: dict[str, Any], key: str) -> np.ndarray:
    rows = checkpoint["alpha_results"]
    return np.mean([np.asarray(row[key]) for row in rows], axis=0)


def _vnet_only(parameter_slices: dict[str, slice]) -> dict[str, slice]:
    return {name: sl for name, sl in parameter_slices.items() if not name.startswith("K_net.")}


def plot_module_symmetry_comparison(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Rotation vs. translation attribution, side by side by module (PDF sec. 0.3/0.5's own example)."""
    first, last = all_results[0], all_results[-1]
    plotting_slices = _vnet_only(parameter_slices)
    modules = list(plotting_slices.keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, checkpoint, subtitle in ((axes[0], first, "random init"), (axes[1], last, "trained")):
        rotation = np.asarray(checkpoint["xrot_score"])
        translation = np.asarray(checkpoint["xtrans_score"])
        rotation_by_module = [np.linalg.norm(rotation[plotting_slices[m]]) for m in modules]
        translation_by_module = [np.linalg.norm(translation[plotting_slices[m]]) for m in modules]
        ax.bar(x - width / 2, rotation_by_module, width, label=r"rotation ($X_{\rm rot}F$)")
        ax.bar(x + width / 2, translation_by_module, width, label=r"translation ($X_{\rm trans}F$, in $q_3$)")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
        ax.set(title=f"Symmetry attribution by module ({subtitle})", ylabel="attribution norm within module")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "module_symmetry_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_symmetry_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_symmetry_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter rotation attribution vs. translation attribution.

    Points hugging one axis mean a parameter realises one generator but not
    the other (localised); points on the diagonal mean the same parameters
    implement both (entangled), exactly the question PDF sec. 0.5 asks.
    """
    plotting_slices = _vnet_only(parameter_slices)
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
        rotation = np.asarray(checkpoint["xrot_score"])
        translation = np.asarray(checkpoint["xtrans_score"])
        for name in modules:
            sl = plotting_slices[name]
            ax.scatter(
                np.maximum(rotation[sl], 1e-12), np.maximum(translation[sl], 1e-12),
                s=14, color=colours[name], label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(
            xlabel=r"rotation attribution $|c_i|$", ylabel=r"translation attribution $|c_i|$",
            title=f"Rotation vs. translation attribution ({subtitle})",
        )
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_dir / "symmetry_attribution_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "symmetry_attribution_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    first, last = all_results[0], all_results[-1]

    def central_row(checkpoint):
        rows = checkpoint["alpha_results"]
        return rows[len(rows) // 2]

    first_row, last_row = central_row(first), central_row(last)
    modules = [
        m for m in first_row["module_rotation_equivariance_error"].keys() if not m.startswith("K_net.")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, row, subtitle in ((axes[0], first_row, "random init"), (axes[1], last_row, "trained")):
        rotation = [row["module_rotation_equivariance_error"][m] for m in modules]
        translation = [row["module_translation_equivariance_error"][m] for m in modules]
        ax.bar(x - width / 2, rotation, width, label="rotation")
        ax.bar(x + width / 2, translation, width, label="translation")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
        ax.set(title=f"Sensitivity-equivariance error $E_i$ by module ({subtitle})", ylabel="mean $E_i$ within module")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend()
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "equivariance_by_module.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    first, last = all_results[0], all_results[-1]
    plotting_slices = _vnet_only(parameter_slices)
    for key, stem, label in (
        ("rotation_equivariance_score", "equivariance_scatter_rotation", "Rotation"),
        ("translation_equivariance_score", "equivariance_scatter_translation", "Translation"),
    ):
        plot_ei_initial_vs_final(
            np.asarray(first[key]), np.asarray(last[key]), plotting_slices,
            title=f"{label} sensitivity-equivariance $E_i$: init vs. trained",
            output_stem=output_dir / stem,
        )


def plot_magnitude_diagnostics(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    first, last = all_results[0], all_results[-1]
    magnitude_initial = np.asarray(first["parameter_magnitude"])
    magnitude_final = np.asarray(last["parameter_magnitude"])
    plotting_slices = _vnet_only(parameter_slices)

    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
        plotting_slices,
        quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
        title=r"Parameter magnitude vs. sensitivity $S_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_sensitivity",
    )


def report_sparsity(model: torch.nn.Module, thresholds: tuple[float, ...] = (1e-4, 1e-3, 1e-2)) -> dict[str, Any]:
    all_params = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()])
    return {
        "n_parameters": int(all_params.numel()),
        "mean_abs_weight": float(all_params.mean()),
        "median_abs_weight": float(all_params.median()),
        "fraction_below_threshold": {str(t): float((all_params < t).float().mean()) for t in thresholds},
    }


def train_and_analyse(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"Device: {device}; dtype: {dtype}; output: {output_dir}")
    print("Generating sparse cylindrical Mexican-hat trajectories...")
    train_data, validation_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}")

    residuals_fn = lambda *args, **kwargs: generic_residuals(*args, **kwargs, p_dim=3)  # noqa: E731
    history, checkpoint_states = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=validation_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    np.savez(output_dir / "training_history.npz", **history)
    plot_training_history(history, output_dir)

    all_results = []
    for step in tqdm(sorted(checkpoint_states.keys()), desc="checkpoint analysis", unit="checkpoint"):
        model.load_state_dict(checkpoint_states[step])
        result = analyse_checkpoint(model, step, cfg, device, dtype, flat_names, parameter_slices)
        write_checkpoint_outputs(result, output_dir)
        all_results.append(result)

    plot_summary(all_results, output_dir)
    plot_equivariance_by_module(all_results, output_dir)
    plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    plot_magnitude_diagnostics(all_results, parameter_slices, output_dir)
    plot_module_symmetry_comparison(all_results, parameter_slices, output_dir)
    plot_symmetry_attribution_scatter(all_results, parameter_slices, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))

    model.load_state_dict(checkpoint_states[max(checkpoint_states.keys())])
    sparsity = report_sparsity(model)
    (output_dir / "sparsity_report.json").write_text(json.dumps(sparsity, indent=2))
    print(f"Sparsity (final checkpoint): {sparsity['fraction_below_threshold']}")
    print(
        f"Final rotation attribution PR: {all_results[-1]['xrot_participation_ratio']:.2f}; "
        f"translation attribution PR: {all_results[-1]['xtrans_participation_ratio']:.2f}"
    )
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()
