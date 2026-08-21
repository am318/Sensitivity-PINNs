"""POOLED-DIRECTION variant of asrnn_mexican_hat_symmetry_sensitivity.py, for direct comparison.

The only difference from the un-suffixed sibling script: c_i (rotation-generator attribution)
and b_i (equivariance-defect attribution) are each solved ONCE per checkpoint from a single
combined least-squares system that stacks Jacobian/target rows across every alpha in
cfg.training_alphas union cfg.analysis_alphas, instead of being solved independently at each
analysis alpha and then RMS-aggregated into a magnitude-only "score" afterward. See
analyse_checkpoint's "Pooled c_i/b_i" block for the full rationale (in short: a per-alpha c_i,
if ever used as an actual parameter-space step -- as in run_potential_walk_experiment.py -- gives
a different perturbed model at each alpha, which isn't one coherent direction; pooling first
avoids that). Run with the SAME config.json as an existing un-suffixed run (same seed => same
training trajectory, since training itself never touches c_i/b_i) to get a directly comparable
set of plots differing only in this one definitional choice.

Functional-sensitivity experiment for the 2D isotropic double well ("Mexican hat").

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
    module_colours,
    parameter_layout,
    plot_ei_initial_vs_final,
    plot_magnitude_vs_quantity,
    plot_signed_initial_vs_final,
    plot_both_signed_quantity,
    plot_training_history,
    robust_linthresh,
    prettify_parameter_name,
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
    participation_ratio,
    per_parameter_equivariance_error,
    principal_angles_and_dimension,
    relative_energy_drift,
    sensitivity_transform_residual,
    tangent_projection,
    parameter_magnitude_ci_correlation,
)


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "outputs/asrnn_mexican_hat_symmetry_pooled"

    architecture: str = "hamiltonian"  # hamiltonian, direct_mlp, or equivariant

    kinetic_hidden_dim: int = 32
    kinetic_hidden_layers: int = 3
    potential_hidden_dim: int = 32
    potential_hidden_layers: int = 3
    direct_mlp_hidden_dim: int = 32
    direct_mlp_hidden_layers: int = 3

    training_alphas: list[float] = field(
        default_factory=lambda: [-1.1, -0.8, -0.6, -0.2, 0.2, 0.6, 0.8, 1.1]
    )
    trajectory_window: int = 20
    trajectory_splits: int = 100
    sampled_instants: int = 10
    initial_conditions_per_alpha: int = 10
    integration_dt: float = 0.1
    coarsening_factor: int = 100
    validation_fraction: float = 0.25
    # Doubles the training set with a random-SO(2)-rotated copy of every trajectory --
    # since this system's dynamics is exactly rotationally equivariant for every alpha
    # (verified throughout this project), this is free, exactly-valid additional data
    # that teaches rotational consistency implicitly, through data diversity, as an
    # alternative/complement to architectural equivariance (sec. 8) or L1 (sec. 8b/e).
    augment_dataset: bool = False

    optimizer: str = "adam"  # adam or lbfgs
    training_steps: int = 20000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    # L1 penalty on all parameters, added to the trajectory-fitting loss --
    # tests whether an explicit sparsity pressure makes a non-equivariant
    # architecture's parameters align more with the equivariant directions
    # (i.e. improve sensitivity equivariance E_i) by squeezing out redundant
    # capacity that has no reason to respect the symmetry on its own.
    l1_weight: float = 1e-5
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
    # "square" (default) samples a Cartesian q_grid_points_per_axis x q_grid_points_per_axis
    # grid over [-q_extent, q_extent]^2 -- NOT invariant under continuous rotation. The exact
    # algebraic identity <delta, g> = 0 (g = X_rot F tangent to the orbit, delta = F - Pi F the
    # equivariance defect; see compute_orbit_average's docstring) relies on integrating over a
    # domain closed under rotation so the integration-by-parts boundary term vanishes -- a
    # square has corners the rotation flow exits, so the discrete inner product computed on it
    # picks up a genuine, grid-refinement-stable ~0.1 cosine between g and delta instead of the
    # expected ~1e-15 (verified: refining the square from 7x7 to 81x81 leaves cos(g,delta)
    # stuck around -0.11, not shrinking). "polar" samples a rotationally-symmetric disk of
    # radius q_extent instead (polar_n_radii concentric rings x polar_n_angles points each,
    # equally spaced in angle) -- verified to recover cos(g,delta) ~ 1e-14 even with far fewer
    # points (320 vs 1681) than any square grid tried, confirming the square grid's nonzero
    # cosine is a domain-boundary artefact, not floating-point noise or a genuine model property.
    probe_grid_shape: str = "square"  # "square" or "polar"
    polar_n_radii: int = 15
    polar_n_angles: int = 32
    # Deliberately not a multiple of 120 degrees, to make clear this is a
    # genuinely continuous symmetry, not a discrete point-group artefact.
    rotation_angle_degrees: float = 73.0
    # Number of angles used to numerically approximate the SO(2) orbit-average
    # Pi f_theta = (1/2pi) int R_phi^{-1} f_theta(R_phi q) dphi (the g-vs-delta
    # decomposition below). Verified (session that added this) against both an
    # exactly-equivariant analytic force (Pi F = F to ~1e-16) and an equivariance
    # check on Pi F itself (~1e-16) in float64 -- 64 angles is already essentially
    # exact for a smooth force field; not worth tuning down for speed given how
    # cheap each forward-only evaluation is compared to the existing autograd-heavy
    # per-parameter Jacobian computation.
    orbit_average_n_phi: int = 64

    analysis_alphas: list[float] = field(
        default_factory=lambda: [-1.0, -0.8, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    )

    plotting_alphas: list[float] = field(
            default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0]
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
    if cfg.probe_grid_shape not in {"square", "polar"}:
        raise ValueError("probe_grid_shape must be 'square' or 'polar'.")
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
    if cfg.probe_grid_shape == "polar":
        radii = torch.linspace(
            cfg.q_extent / cfg.polar_n_radii, cfg.q_extent, cfg.polar_n_radii, device=device, dtype=dtype
        )
        angles = torch.linspace(0.0, 2 * math.pi, cfg.polar_n_angles + 1, device=device, dtype=dtype)[:-1]
        r_mesh, theta_mesh = torch.meshgrid(radii, angles, indexing="ij")
        q1 = (r_mesh * torch.cos(theta_mesh)).reshape(-1)
        q2 = (r_mesh * torch.sin(theta_mesh)).reshape(-1)
        return q1, q2
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


def evaluate_energy_drift_jacobian(
    model: torch.nn.Module, q1_pts: torch.Tensor, q2_pts: torch.Tensor,
    p1_pts: torch.Tensor, p2_pts: torch.Tensor, alpha: float, architecture: str,
    *, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-probe-point value and parameter-gradient of the energy-conservation violation.

    D_theta = grad_E_true_p . dp/dt_theta + grad_E_true_q . dq/dt_theta (the
    same quantity as ``noether_energy_drift``/``relative_energy_drift``, here
    kept differentiable end-to-end). Unlike ``evaluate_at_points`` (which
    only differentiates V_net, force being independent of p for a separable
    Hamiltonian), this keeps *both* V_net (via force) and K_net (via dq/dt)
    connected to autograd -- D_theta genuinely depends on both, so this is
    the first diagnostic in this file where K_net parameters get a
    non-trivial attribution. Analytic grad_E_true_p = p, grad_E_true_q =
    (alpha + r^2) q (kinetic energy 0.5|p|^2, potential 0.5*alpha*r^2 +
    0.25*r^4). Returns ``(drift [N], jacobian [N, P])`` -- feed both into
    ``tangent_projection`` (target=drift, jacobian=jacobian) to get a proper
    attribution ``c_i`` that shrinks toward zero as the violation itself
    shrinks toward zero, rather than taking the raw parameter-gradient's
    magnitude directly (which is the *local slope* of the violation, not
    tied to its *size* -- large even for an already near-conserving network,
    and not a meaningful "attribution" of a small or zero violation).
    """
    params = tuple(model.parameters())
    drift_values, rows = [], []
    for q1_val, q2_val, p1_val, p2_val in zip(
        q1_pts.detach().cpu().tolist(), q2_pts.detach().cpu().tolist(),
        p1_pts.detach().cpu().tolist(), p2_pts.detach().cpu().tolist(),
    ):
        q = torch.tensor([[q1_val, q2_val]], device=device, dtype=dtype, requires_grad=True)
        p = torch.tensor([[p1_val, p2_val]], device=device, dtype=dtype, requires_grad=True)
        alpha_t = torch.tensor([[alpha]], device=device, dtype=dtype)
        if architecture == "direct_mlp":
            dpdt, dqdt = model(p, q, alpha_t)
            force = dpdt.squeeze(0)
            dqdt = dqdt.squeeze(0)
        else:
            potential = model.V_net(torch.cat((q, alpha_t), dim=1))
            force = -torch.autograd.grad(potential.sum(), q, create_graph=True)[0].squeeze(0)
            kinetic = model.K_net(p)
            dqdt = torch.autograd.grad(kinetic.sum(), p, create_graph=True)[0].squeeze(0)
        grad_e_p = p.squeeze(0)
        grad_e_q = (alpha + q1_val**2 + q2_val**2) * q.squeeze(0)
        drift = torch.dot(grad_e_p, force) + torch.dot(grad_e_q, dqdt)
        drift_values.append(drift.detach())
        rows.append(parameter_gradient_row(drift, params))
    return torch.stack(drift_values), torch.stack(rows)


def compute_orbit_average(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype,
    alpha: float, q1_flat: torch.Tensor, q2_flat: torch.Tensor,
) -> torch.Tensor:
    r"""The SO(2) orbit average (Reynolds operator) of the learned force,

        Pi F_theta(q) = (1 / 2*pi) * int_0^{2*pi} R_phi^{-1} F_theta(R_phi q) dphi,

    approximated via a uniform grid of ``cfg.orbit_average_n_phi`` angles. This is the
    g-vs-delta decomposition raised in review of an earlier draft: since Pi projects onto
    ker(X_rot) *by construction* -- averaging over the whole group annihilates any
    phi-dependence -- Pi F is exactly rotationally equivariant regardless of how equivariant
    F_theta itself currently is. delta = F_theta - Pi F_theta (computed at the call site) is
    then the genuine "distance from the equivariant subspace" that g = X_rot F_theta is
    provably *orthogonal* to (g is tangent to the constant-||delta|| orbit through F_theta,
    not radial) -- so g and delta answer different attribution questions and need separate
    projections, not one standing in for the other.

    Verified (session that added this function), in float64 on a random-weight model:
    (a) Pi F itself satisfies exact finite-rotation equivariance for an arbitrary test
    angle, to ~1e-15 relative error; (b) applied to the *true*, already-exactly-equivariant
    analytic force, Pi F recovers that force exactly (~1e-15); (c) the resulting delta has
    zero orbit-average itself (Pi(delta) ~= 0, ~1e-7), confirming Pi is a genuine projector
    onto the invariant subspace, not an approximation that leaks some of delta back in.
    """
    n_phi = cfg.orbit_average_n_phi
    total = torch.zeros(q1_flat.shape[0], 2, device=device, dtype=dtype)
    for k in range(n_phi):
        phi_degrees = 360.0 * k / n_phi
        R = rotation_matrix(phi_degrees, device, dtype)
        q1_rot, q2_rot = transform_points(q1_flat, q2_flat, R)
        _, _, _, force_rot = _evaluate_force_and_potential(model, cfg, device, dtype, alpha, q1_rot, q2_rot)
        f1_back, f2_back = transform_points(force_rot[:, 0], force_rot[:, 1], R.T)
        total = total + torch.stack([f1_back, f2_back], dim=1)
    return total / n_phi


def analyse_checkpoint(
    model: torch.nn.Module, step: int, cfg: Config, device: torch.device, dtype: torch.dtype,
    flat_names: list[str], parameter_slices: dict[str, slice],
) -> dict[str, Any]:
    model.eval()
    parameter_magnitude = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()]).cpu().tolist()
    parameter_value = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().tolist()
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
        # Signed companion to the RMS-magnitude "sensitivity" above: mean (not RMS) of the raw
        # signed per-point, per-component Jacobian entries. Can be near zero from cancellation
        # (a parameter whose local effect flips sign across the probe grid), which is itself
        # informative -- distinct from small |sensitivity|, which means the effect is small
        # everywhere. Used only for the signed diagnostic plots (paired with signed theta_i,
        # c_i, b_i); the regression itself still uses the full, unaggregated jac_flat below.
        signed_sensitivity = torch.mean(j_x.detach().cpu().double(), dim=(0, 1))
        jac_flat = j_x.reshape(n_points * 2, -1)

        bifurcation_target = bifurcation_generator_target(q1_grid, q2_grid)
        bifurcation_target_flat = bifurcation_target.reshape(n_points * 2)
        (
            bifurcation_projection_error, bifurcation_principal_angle,
            bifurcation_coefficients, bifurcation_resolved_rank, bifurcation_singular_values,
        ) = tangent_projection(jac_flat, bifurcation_target_flat, cfg.tangent_svd_relative_cutoff)

        # xrot_target_flat is still needed below for generator_matrix/dimension_info; the
        # per-alpha tangent_projection solve against it (and against delta) is NOT done here
        # any more -- this script pools rows across alpha into one combined solve instead (see
        # the "Pooled c_i/b_i" block after this loop) rather than solving independently at each
        # alpha and RMS-aggregating afterward (that original definition lives in the un-suffixed
        # sibling script, asrnn_mexican_hat_symmetry_sensitivity.py).
        xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
        xrot_target_flat = xrot_target.reshape(n_points * 2)

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

        energy_drift_values, energy_drift_jacobian = evaluate_energy_drift_jacobian(
            model, q1_grid, q2_grid, p1_grid, p2_grid, alpha, cfg.architecture, device=device, dtype=dtype,
        )
        (
            energy_projection_error, energy_principal_angle,
            energy_coefficients, energy_resolved_rank, energy_singular_values,
        ) = tangent_projection(energy_drift_jacobian, energy_drift_values, cfg.tangent_svd_relative_cutoff)

        alpha_results.append(
            {
                "alpha": float(alpha),
                "rotation_potential_residual": rotation_potential_residual,
                "rotation_force_residual": rotation_force_residual,
                "rotation_sensitivity_residual": rotation_sensitivity_residual,
                "energy_conservation_violation": energy_conservation_violation,
                "energy_projection_error": energy_projection_error,
                "energy_principal_angle_degrees": energy_principal_angle,
                "energy_resolved_rank": energy_resolved_rank,
                "energy_singular_values": energy_singular_values,
                "energy_attribution_by_parameter": energy_coefficients.cpu().tolist(),
                "module_energy_attribution": aggregate_by_module(energy_coefficients.abs(), parameter_slices),
                "sensitivity": sensitivity.cpu().tolist(),
                "signed_sensitivity": signed_sensitivity.cpu().tolist(),
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
                # xrot_*/delta_* (c_i/b_i) fields are filled in uniformly below, after the
                # pooled solve -- see the "Pooled c_i/b_i" block.
                "joint_representation_dimension": dimension_info["representation_dimension"],
                "joint_principal_angles_degrees": dimension_info["principal_angles_degrees"],
            }
        )

    # Pooled c_i/b_i: ONE combined least-squares solve stacking Jacobian/target rows across
    # every alpha in cfg.training_alphas union cfg.analysis_alphas (alpha is just a network
    # input feature here, not something requiring "data" at that value, so both sets are fair
    # game -- see run_potential_walk_experiment.py's combined_direction for the identical
    # construction and the full rationale). This is the direct generalisation of how
    # tangent_projection already pools rows across every probe point q within one alpha to also
    # pool across alpha itself, giving one c/b vector valid across every probed regime at once,
    # rather than a separate vector per alpha (RMS-aggregated only for magnitude-ranking
    # purposes in the un-suffixed sibling script). The identical pooled vector is written into
    # every alpha_results row below, so every downstream consumer (RMS "score" aggregation,
    # CSV export, signed plots) picks it up unchanged -- RMS of an alpha-constant vector is
    # just its own magnitude, so xrot_score/b_score end up being exactly |pooled c|/|pooled b|.
    direction_alphas = sorted(set(round(a, 6) for a in cfg.training_alphas) | set(round(a, 6) for a in cfg.analysis_alphas))
    jac_blocks, xrot_target_blocks, delta_target_blocks = [], [], []
    for alpha in direction_alphas:
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_grid, q2_grid, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
        )
        jac_blocks.append(j_x.reshape(n_points * 2, -1))
        xrot_target_blocks.append(rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x).reshape(-1))
        pi_f_x = compute_orbit_average(model, cfg, device, dtype, alpha, q1_grid, q2_grid)
        delta_target_blocks.append((f_x - pi_f_x).reshape(-1))
    jac_pooled = torch.cat(jac_blocks, dim=0)
    xrot_target_pooled = torch.cat(xrot_target_blocks, dim=0)
    delta_target_pooled = torch.cat(delta_target_blocks, dim=0)
    (
        pooled_xrot_projection_error, pooled_xrot_principal_angle,
        pooled_xrot_coefficients, pooled_xrot_resolved_rank, pooled_xrot_singular_values,
    ) = tangent_projection(jac_pooled, xrot_target_pooled, cfg.tangent_svd_relative_cutoff)
    (
        pooled_delta_projection_error, pooled_delta_principal_angle,
        pooled_delta_coefficients, pooled_delta_resolved_rank, pooled_delta_singular_values,
    ) = tangent_projection(jac_pooled, delta_target_pooled, cfg.tangent_svd_relative_cutoff)

    for row in alpha_results:
        row["xrot_projection_error"] = pooled_xrot_projection_error
        row["xrot_principal_angle_degrees"] = pooled_xrot_principal_angle
        row["xrot_resolved_rank"] = pooled_xrot_resolved_rank
        row["xrot_singular_values"] = pooled_xrot_singular_values
        row["xrot_attribution_coefficients"] = pooled_xrot_coefficients.cpu().tolist()
        row["module_xrot_attribution"] = aggregate_by_module(pooled_xrot_coefficients.abs(), parameter_slices)
        row["delta_projection_error"] = pooled_delta_projection_error
        row["delta_principal_angle_degrees"] = pooled_delta_principal_angle
        row["delta_resolved_rank"] = pooled_delta_resolved_rank
        row["delta_singular_values"] = pooled_delta_singular_values
        row["delta_attribution_coefficients"] = pooled_delta_coefficients.cpu().tolist()
        row["module_delta_attribution"] = aggregate_by_module(pooled_delta_coefficients.abs(), parameter_slices)

    coefficient_matrix = np.asarray([r["bifurcation_attribution_coefficients"] for r in alpha_results])
    bifurcation_score = np.sqrt(np.mean(coefficient_matrix**2, axis=0))
    top_bifurcation = np.argsort(bifurcation_score)[::-1][: cfg.top_parameters_to_report]

    xrot_coefficient_matrix = np.asarray([r["xrot_attribution_coefficients"] for r in alpha_results])
    xrot_score = np.sqrt(np.mean(xrot_coefficient_matrix**2, axis=0))
    top_xrot = np.argsort(xrot_score)[::-1][: cfg.top_parameters_to_report]

    # b_i: RMS-over-alpha magnitude of the delta attribution (c_i's analogue for g -- see
    # compute_orbit_average's docstring for why these are two genuinely different questions,
    # not the same one computed twice).
    delta_coefficient_matrix = np.asarray([r["delta_attribution_coefficients"] for r in alpha_results])
    b_score = np.sqrt(np.mean(delta_coefficient_matrix**2, axis=0))
    top_b = np.argsort(b_score)[::-1][: cfg.top_parameters_to_report]

    equivariance_matrix = np.asarray([r["rotation_equivariance_error_by_parameter"] for r in alpha_results])
    equivariance_score = np.sqrt(np.mean(equivariance_matrix**2, axis=0))
    top_equivariance_violators = np.argsort(equivariance_score)[::-1][: cfg.top_parameters_to_report]

    energy_attribution_matrix = np.asarray([r["energy_attribution_by_parameter"] for r in alpha_results])
    energy_attribution_score = np.sqrt(np.mean(energy_attribution_matrix**2, axis=0))

    vnet_mask = np.array([name.startswith("V_net.") for name in flat_names])
    xrot_mag_corr = parameter_magnitude_ci_correlation(
        parameter_magnitude,
        xrot_score,
        mask=vnet_mask,
    )

    energy_mag_corr = parameter_magnitude_ci_correlation(
        parameter_magnitude,
        energy_attribution_score,
    )

    equivariance_mag_corr = parameter_magnitude_ci_correlation(
        parameter_magnitude,
        equivariance_score,
        mask=vnet_mask,
    )

    # Pearson correlation between the two primary attributions themselves (not either one
    # against |theta_i|) -- the central "do c_i and b_i share support" question driving the
    # module/scatter comparison plots, but as a single reported number rather than something
    # only ever eyeballed off a scatter plot or computed ad hoc. V_net-only (vnet_mask): K_net
    # is architecturally zero for both c_i and b_i (see plot_c_vs_b_scatter's docstring), so
    # including it would just add a cluster of trivial (0,0) pairs, not real correlation signal.
    xrot_b_corr = parameter_magnitude_ci_correlation(
        xrot_score,
        b_score,
        mask=vnet_mask,
    )
    return {
        "step": step,
        "parameter_magnitude": parameter_magnitude,
        "parameter_value": parameter_value,
        "alpha_results": alpha_results,
        "bifurcation_score": bifurcation_score.tolist(),
        "xrot_score": xrot_score.tolist(),
        "b_score": b_score.tolist(),
        "equivariance_score": equivariance_score.tolist(),
        "energy_attribution_score": energy_attribution_score.tolist(),
        "xrot_participation_ratio": participation_ratio(xrot_score[vnet_mask]),
        "b_participation_ratio": participation_ratio(b_score[vnet_mask]),
        "energy_participation_ratio": participation_ratio(energy_attribution_score),
        "xrot_magnitude_correlation": xrot_mag_corr,
        "xrot_b_correlation": xrot_b_corr,
        # "energy_magnitude_correlation": energy_mag_corr,
        # "equivariance_magnitude_correlation": equivariance_mag_corr,
        "top_bifurcation_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(bifurcation_score[i])}
            for i in top_bifurcation
        ],
        "top_xrot_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xrot_score[i])}
            for i in top_xrot
        ],
        "top_b_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(b_score[i])}
            for i in top_b
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
        result["top_b_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_b_parameters_step_{step:06d}.csv",
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

    axes[0, 0].set(title="finite rotation: force", ylabel="relative residual", yscale="log")
    axes[0, 1].set(title="finite rotation: sensitivity", ylabel="relative Frobenius residual", yscale="log")
    axes[0, 2].set(title=r"bifurcation generator $\partial_\alpha F$", ylabel="relative projection error", yscale="log")
    axes[1, 0].set(title=r"$X_{\rm rot}F$ generator", ylabel="relative projection error", yscale="log")
    axes[1, 1].set(
        title="joint representation dimension",
        ylabel="# generators represented (of 2)", yticks=[0, 1, 2],
    )
    axes[1, 2].set(title="Noether energy-conservation violation", ylabel="relative energy drift", yscale="log")
    for ax in axes.flat:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
    axes[0, 0].legend(ncol=2)
    fig.savefig(output_dir / "mexican_hat_summary.png", dpi=200)
    fig.savefig(output_dir / "mexican_hat_summary.pdf")
    plt.close(fig)


def _exclude_structurally_zero(names: list[str]) -> list[str]:
    """Drop parameters whose rotation S_i/E_i/c_i are exactly zero by construction, not training.

    K_net never enters the force/sensitivity Jacobian (evaluate_at_points only
    differentiates V_net), so all of it is trivially zero. V_net's own output-layer
    bias is zero for the same underlying reason: force = -dV/dq, and
    V = W_out @ h(q) + b_out, so d(dV/dq)/d(b_out) = 0 identically -- an additive
    constant in V never survives differentiation. Both are analytic certainties at
    every checkpoint (verified: at random init, the only exactly-zero rotation
    scores are all of K_net plus this one V_net bias), not a genuine result, and
    otherwise pin every log-log plot's floor, crushing the visible range for every
    other parameter. Drop both from diagnostic plots to avoid a misleading point/cluster.
    """
    v_net_bias_layers = [
        int(n.split(".")[2]) for n in names if n.startswith("V_net.net.") and n.endswith(".bias")
    ]
    output_bias_name = f"V_net.net.{max(v_net_bias_layers)}.bias" if v_net_bias_layers else None
    return [n for n in names if not n.startswith("K_net.") and n != output_bias_name]


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    first, last = all_results[0], all_results[-1]

    def central_row(checkpoint):
        rows = checkpoint["alpha_results"]
        return rows[len(rows) // 2]

    first_row, last_row = central_row(first), central_row(last)
    modules = _exclude_structurally_zero(list(first_row["module_rotation_equivariance_error"].keys()))
    before = [first_row["module_rotation_equivariance_error"][m] for m in modules]
    after = [last_row["module_rotation_equivariance_error"][m] for m in modules]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    ax.bar(x - width / 2, before, width, label="random init")
    ax.bar(x + width / 2, after, width, label="after training")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
    ax.set(ylabel="mean $E_i$ within module")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200)
    fig.savefig(output_dir / "equivariance_by_module.pdf")
    plt.close(fig)


def _alpha_averaged(checkpoint: dict[str, Any], key: str) -> np.ndarray:
    """Mean of a per-parameter quantity (S_i or E_i) across all analysis alphas.

    A single representative alpha (e.g. the literal middle of analysis_alphas,
    which happens to be alpha=0) is not safe to use alone: any V_net first-layer
    weight multiplying the alpha input channel has S_i = |alpha * (downstream
    grad)|, which is exactly zero whenever that probe alpha is exactly zero --
    a calculus certainty, not a trained or magnitude-driven effect. Averaging
    over every analysis alpha (which spans both signs and excludes only the
    single alpha=0 slice from dominating) removes this probe-point artefact.
    """
    rows = checkpoint["alpha_results"]
    return np.mean([np.asarray(row[key]) for row in rows], axis=0)


def _signed_like_score(checkpoint: dict[str, Any], raw_key: str, unsigned_score: np.ndarray) -> np.ndarray:
    """Signed companion to an alpha-aggregated unsigned score (RMS or mean-of-RMS), with
    matching magnitude -- |signed_i| == unsigned_score_i exactly, always.

    Earlier this project paired an RMS-aggregated (all analysis_alphas) unsigned magnitude
    with a single representative alpha's raw sign, which was an inconsistency: the two
    numbers came from different underlying statistics (18-alpha RMS vs. 1-alpha raw), so
    their magnitudes didn't match and a reader comparing signed vs. unsigned plots side by
    side was implicitly comparing two different quantities. This instead keeps the exact
    unsigned magnitude already used elsewhere, and attaches a sign derived from the mean
    (across all analysis_alphas) of the raw per-alpha signed values -- the sign each
    parameter's effect agrees on most often/strongly, not an arbitrary single alpha's sign.
    """
    raw_matrix = np.asarray([r[raw_key] for r in checkpoint["alpha_results"]])
    sign = np.sign(raw_matrix.mean(axis=0))
    sign[sign == 0] = 1.0  # deterministic tie-break for exact zeros (e.g. structurally-zero params)
    return sign * np.asarray(unsigned_score)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    first, last = all_results[0], all_results[-1]
    ei_initial = _alpha_averaged(first, "rotation_equivariance_error_by_parameter")
    ei_final = _alpha_averaged(last, "rotation_equivariance_error_by_parameter")
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    plot_ei_initial_vs_final(
        ei_initial, ei_final, plotting_slices,
        title="Rotation sensitivity-equivariance $E_i$: init vs. trained (mean over $\\alpha$)",
        output_stem=output_dir / "equivariance_scatter",
    )


def plot_magnitude_diagnostics(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Check whether small S_i / E_i / c_i values at specific parameters are a genuine
    effect or simply an artefact of those parameters having small |theta_i| (e.g. from L1)."""
    first, last = all_results[0], all_results[-1]
    magnitude_initial = np.asarray(first["parameter_magnitude"])
    magnitude_final = np.asarray(last["parameter_magnitude"])
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}

    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
        plotting_slices,
        quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
        title=r"Parameter magnitude vs. sensitivity $S_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_sensitivity",
    )
    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        _alpha_averaged(first, "rotation_equivariance_error_by_parameter"),
        _alpha_averaged(last, "rotation_equivariance_error_by_parameter"),
        plotting_slices,
        quantity_label=r"$E_i$ (rotation sensitivity-equivariance error, mean over $\alpha$)",
        title=r"Parameter magnitude vs. equivariance error $E_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_equivariance",
    )
    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        np.asarray(first["xrot_score"]), np.asarray(last["xrot_score"]),
        plotting_slices,
        quantity_label=r"$|c_i|$ (rotation-generator attribution, RMS over $\alpha$)",
        title=r"Parameter magnitude vs. rotation-generator attribution $|c_i|$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_attribution",
    )
    plot_magnitude_vs_quantity(
        np.asarray(first["xrot_score"]), np.asarray(last["xrot_score"]),
        _alpha_averaged(first, "rotation_equivariance_error_by_parameter"),
        _alpha_averaged(last, "rotation_equivariance_error_by_parameter"),
        plotting_slices,
        quantity_label=r"$E_i$ (rotation sensitivity-equivariance error, mean over $\alpha$)",
        title=r"Rotation-generator attribution $|c_i|$ vs. equivariance error $E_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "attribution_vs_equivariance",
        x_label=r"$|c_i|$ (rotation-generator attribution, RMS over $\alpha$)",
    )
    plot_magnitude_vs_quantity(
        np.asarray(first["xrot_score"]), np.asarray(last["xrot_score"]),
        _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
        plotting_slices,
        quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
        title=r"Rotation-generator attribution $|c_i|$ vs. sensitivity $S_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "attribution_vs_sensitivity",
        x_label=r"$|c_i|$ (rotation-generator attribution, RMS over $\alpha$)",
    )

    # b_i (delta = F - Pi F attribution) analogues of the three magnitude/attribution
    # diagnostics above -- see compute_orbit_average's docstring for why b_i is a genuinely
    # distinct question from c_i, not the same one recomputed.
    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        np.asarray(first["b_score"]), np.asarray(last["b_score"]),
        plotting_slices,
        quantity_label=r"$|b_i|$ (equivariance-defect attribution, RMS over $\alpha$)",
        title=r"Parameter magnitude vs. defect attribution $|b_i|$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_b_attribution",
    )
    plot_magnitude_vs_quantity(
        np.asarray(first["b_score"]), np.asarray(last["b_score"]),
        _alpha_averaged(first, "rotation_equivariance_error_by_parameter"),
        _alpha_averaged(last, "rotation_equivariance_error_by_parameter"),
        plotting_slices,
        quantity_label=r"$E_i$ (rotation sensitivity-equivariance error, mean over $\alpha$)",
        title=r"Defect attribution $|b_i|$ vs. equivariance error $E_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "b_attribution_vs_equivariance",
        x_label=r"$|b_i|$ (equivariance-defect attribution, RMS over $\alpha$)",
    )
    plot_magnitude_vs_quantity(
        np.asarray(first["b_score"]), np.asarray(last["b_score"]),
        _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
        plotting_slices,
        quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
        title=r"Defect attribution $|b_i|$ vs. sensitivity $S_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "b_attribution_vs_sensitivity",
        x_label=r"$|b_i|$ (equivariance-defect attribution, RMS over $\alpha$)",
    )

    # Signed companions. Same magnitude as the unsigned score already used above (RMS for
    # c_i/b_i, mean-of-RMS for S_i) -- not an arbitrary single alpha's raw magnitude -- with
    # sign from the mean (across all analysis_alphas) of the raw per-alpha signed values. Not
    # |theta_i| but the raw signed theta_i, since neither axis in any of these pairs is a
    # genuine non-negative magnitude.
    theta_signed_initial = np.asarray(first["parameter_value"])
    theta_signed_final = np.asarray(last["parameter_value"])
    c_signed_initial = _signed_like_score(first, "xrot_attribution_coefficients", first["xrot_score"])
    c_signed_final = _signed_like_score(last, "xrot_attribution_coefficients", last["xrot_score"])
    b_signed_initial = _signed_like_score(first, "delta_attribution_coefficients", first["b_score"])
    b_signed_final = _signed_like_score(last, "delta_attribution_coefficients", last["b_score"])
    s_signed_initial = _signed_like_score(first, "signed_sensitivity", _alpha_averaged(first, "sensitivity"))
    s_signed_final = _signed_like_score(last, "signed_sensitivity", _alpha_averaged(last, "sensitivity"))

    plot_both_signed_quantity(
        theta_signed_initial, theta_signed_final, c_signed_initial, c_signed_final,
        plotting_slices,
        x_label=r"$\theta_i$ (signed parameter value)",
        y_label=r"$c_i$ (rotation-generator attribution, central $\alpha$)",
        output_stem=output_dir / "magnitude_vs_attribution_signed",
    )
    plot_both_signed_quantity(
        theta_signed_initial, theta_signed_final, b_signed_initial, b_signed_final,
        plotting_slices,
        x_label=r"$\theta_i$ (signed parameter value)",
        y_label=r"$b_i$ (equivariance-defect attribution, central $\alpha$)",
        output_stem=output_dir / "magnitude_vs_b_attribution_signed",
    )
    plot_both_signed_quantity(
        theta_signed_initial, theta_signed_final, s_signed_initial, s_signed_final,
        plotting_slices,
        x_label=r"$\theta_i$ (signed parameter value)",
        y_label=r"$S_i$ (signed sensitivity, mean over probe grid, central $\alpha$)",
        output_stem=output_dir / "magnitude_vs_sensitivity_signed",
    )
    plot_both_signed_quantity(
        c_signed_initial, c_signed_final, s_signed_initial, s_signed_final,
        plotting_slices,
        x_label=r"$c_i$ (rotation-generator attribution, central $\alpha$)",
        y_label=r"$S_i$ (signed sensitivity, central $\alpha$)",
        output_stem=output_dir / "attribution_vs_sensitivity_signed",
    )
    plot_both_signed_quantity(
        b_signed_initial, b_signed_final, s_signed_initial, s_signed_final,
        plotting_slices,
        x_label=r"$b_i$ (equivariance-defect attribution, central $\alpha$)",
        y_label=r"$S_i$ (signed sensitivity, central $\alpha$)",
        output_stem=output_dir / "b_attribution_vs_sensitivity_signed",
    )


def plot_module_attribution(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Experiment 3 (PDF sec. 0.4-0.5): which subnetworks realise the rotation generator?

    X_rot F = sum_i c_i S_i is solved via tangent_projection's truncated SVD
    pseudo-inverse (see docstring there); xrot_score is |c_i|, RMS-aggregated
    over alpha. Rotation only -- the bifurcation direction dF/dalpha is a
    problem-specific parameter-family direction, not a symmetry generator, so
    it isn't included here (it's still tracked separately via
    bifurcation_score / the Experiment-1 summary plot).
    """
    first, last = all_results[0], all_results[-1]
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    modules = list(plotting_slices.keys())

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    before = [np.linalg.norm(np.asarray(first["xrot_score"])[plotting_slices[m]]) for m in modules]
    after = [np.linalg.norm(np.asarray(last["xrot_score"])[plotting_slices[m]]) for m in modules]
    ax.bar(x - width / 2, before, width, label="random init")
    ax.bar(x + width / 2, after, width, label="after training")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
    ax.set(ylabel=r"$\|c_i\|$ within module")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "module_attribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_attribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter rotation-generator attribution |c_i|: random init vs. trained.

    Same story as equivariance_scatter.png but for c_i rather than E_i --
    does training concentrate the rotation generator onto fewer parameters,
    or leave its attribution pattern essentially unchanged?
    """
    first, last = all_results[0], all_results[-1]
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    plot_ei_initial_vs_final(
        np.asarray(first["xrot_score"]), np.asarray(last["xrot_score"]), plotting_slices,
        title=r"Rotation-generator attribution $|c_i|$: init vs. trained",
        output_stem=output_dir / "attribution_scatter",
        quantity_label="$|c_i|$",
        log_scale=True,
    )
    c_signed_initial = _signed_like_score(first, "xrot_attribution_coefficients", first["xrot_score"])
    c_signed_final = _signed_like_score(last, "xrot_attribution_coefficients", last["xrot_score"])
    plot_signed_initial_vs_final(
        c_signed_initial, c_signed_final, plotting_slices,
        output_stem=output_dir / "attribution_scatter_signed",
        quantity_label="$c_i$",
    )


def plot_module_b_attribution(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """b_i analogue of plot_module_attribution: which subnetworks realise delta = F - Pi F,
    the genuine equivariance defect (as opposed to c_i's g = X_rot F, which is tangent to
    the orbit rather than radial toward the equivariant subspace -- see
    compute_orbit_average's docstring)."""
    first, last = all_results[0], all_results[-1]
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    modules = list(plotting_slices.keys())

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    before = [np.linalg.norm(np.asarray(first["b_score"])[plotting_slices[m]]) for m in modules]
    after = [np.linalg.norm(np.asarray(last["b_score"])[plotting_slices[m]]) for m in modules]
    ax.bar(x - width / 2, before, width, label="random init")
    ax.bar(x + width / 2, after, width, label="after training")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
    ax.set(ylabel=r"$\|b_i\|$ within module")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "module_b_attribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_b_attribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_b_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """b_i analogue of plot_attribution_scatter: per-parameter defect attribution |b_i|,
    random init vs. trained -- plus its signed companion."""
    first, last = all_results[0], all_results[-1]
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    plot_ei_initial_vs_final(
        np.asarray(first["b_score"]), np.asarray(last["b_score"]), plotting_slices,
        title=r"Equivariance-defect attribution $|b_i|$: init vs. trained",
        output_stem=output_dir / "b_attribution_scatter",
        quantity_label="$|b_i|$",
        log_scale=True,
    )
    b_signed_initial = _signed_like_score(first, "delta_attribution_coefficients", first["b_score"])
    b_signed_final = _signed_like_score(last, "delta_attribution_coefficients", last["b_score"])
    plot_signed_initial_vs_final(
        b_signed_initial, b_signed_final, plotting_slices,
        output_stem=output_dir / "b_attribution_scatter_signed",
        quantity_label="$b_i$",
    )


def plot_module_symmetry_comparison(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Two genuine symmetries side by side: rotation (spatial, SO(2)) vs. time-translation
    (energy conservation). Unlike the earlier bifurcation-vs-rotation comparison (sec. 8f
    of the notes -- bifurcation isn't a symmetry at all), both quantities here are
    attributions for real, verified continuous symmetries of this system. Crucially,
    rotation attribution is *architecturally* zero for K_net (the rotation diagnostic only
    ever differentiates V_net), while the energy-conservation attribution genuinely depends
    on *both* V_net (via force) and K_net (via dq/dt) -- so K_net is not excluded here, and
    its energy-only participation is itself part of the finding, not a diagnostic gap.
    """
    first, last = all_results[0], all_results[-1]
    modules = list(parameter_slices.keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, checkpoint, subtitle in ((axes[0], first, "random init"), (axes[1], last, "trained")):
        rotation = np.asarray(checkpoint["xrot_score"])
        energy = np.asarray(checkpoint["energy_attribution_score"])
        rotation_by_module = [np.linalg.norm(rotation[parameter_slices[m]]) for m in modules]
        energy_by_module = [np.linalg.norm(energy[parameter_slices[m]]) for m in modules]
        ax.bar(x - width / 2, rotation_by_module, width, label=r"rotation ($X_{\rm rot}F$)")
        ax.bar(x + width / 2, energy_by_module, width, label="time-translation (energy conservation)")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
        ax.set(ylabel="attribution norm within module")
        ax.set_title(subtitle)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend()
    fig.savefig(output_dir / "module_symmetry_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_symmetry_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_symmetry_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter rotation attribution vs. energy-conservation attribution.

    Points hugging the energy axis with zero rotation attribution are (by
    construction) K_net parameters -- confirming the architectural split.
    The interesting question is *within* V_net: do its parameters also split
    into rotation-only / energy-only populations, or are the two symmetries
    entangled there too (as bifurcation and rotation were, sec. 8f)?
    """
    modules = list(parameter_slices.keys())
    colours = module_colours(modules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
        rotation = np.asarray(checkpoint["xrot_score"])
        energy = np.asarray(checkpoint["energy_attribution_score"])
        for name in modules:
            sl = parameter_slices[name]
            ax.scatter(
                np.maximum(rotation[sl], 1e-12), np.maximum(energy[sl], 1e-12),
                s=14, color=colours[name], label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=r"rotation attribution $|c_i|$", ylabel=r"energy-conservation attribution $|c_i|$")
        ax.set_title(subtitle)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_dir / "symmetry_attribution_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "symmetry_attribution_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_module_c_vs_b_comparison(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Primary c_i-vs-b_i comparison, module level: g = X_rot F (tangent to the orbit) vs.
    delta = F - Pi F (the actual equivariance defect). Proposition 1 of feedback.pdf shows
    these are orthogonal as functions (<delta, g> = 0), so a priori there is no reason their
    per-parameter attributions have to agree -- this is the module-aggregated first look at
    whether they nonetheless share support. K_net is excluded (unlike
    plot_module_symmetry_comparison): both c_i and b_i are architecturally zero there for the
    same reason (the rotation diagnostic only ever differentiates V_net), so unlike the
    rotation-vs-energy comparison there's no split to see, only a redundant floor cluster."""
    first, last = all_results[0], all_results[-1]
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    modules = list(plotting_slices.keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, checkpoint, subtitle in ((axes[0], first, "random init"), (axes[1], last, "trained")):
        c = np.asarray(checkpoint["xrot_score"])
        b = np.asarray(checkpoint["b_score"])
        c_by_module = [np.linalg.norm(c[plotting_slices[m]]) for m in modules]
        b_by_module = [np.linalg.norm(b[plotting_slices[m]]) for m in modules]
        ax.bar(x - width / 2, c_by_module, width, label=r"$c_i$ (orbit-tangent $g$)")
        ax.bar(x + width / 2, b_by_module, width, label=r"$b_i$ (defect $\delta$)")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
        ax.set(ylabel="attribution norm within module")
        ax.set_title(subtitle)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend()
    fig.savefig(output_dir / "module_c_vs_b_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_c_vs_b_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_c_vs_b_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Primary c_i-vs-b_i comparison, per parameter: |c_i| (RMS over alpha) against |b_i|
    (RMS over alpha). Since g and delta are provably orthogonal as functions, agreement here
    reflects a genuine (if a priori unnecessary) alignment of *which* parameters move each
    quantity, not a restatement of the same computation. K_net is excluded -- c_i and b_i are
    architecturally zero there for the identical reason (see plot_module_c_vs_b_comparison),
    so including it only adds a redundant floor cluster at the axis minimum, not a genuine
    zero-vs-nonzero split the way it does in the rotation-vs-energy comparison."""
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
        c = np.asarray(checkpoint["xrot_score"])
        b = np.asarray(checkpoint["b_score"])
        for name in modules:
            sl = plotting_slices[name]
            ax.scatter(
                np.maximum(c[sl], 1e-12), np.maximum(b[sl], 1e-12),
                s=14, color=colours[name], label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=r"$|c_i|$ (orbit-tangent $g$ attribution)", ylabel=r"$|b_i|$ (defect $\delta$ attribution)")
        ax.set_title(subtitle)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_dir / "c_vs_b_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "c_vs_b_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_c_vs_b_scatter_signed(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Signed companion of plot_c_vs_b_scatter: same magnitude as the unsigned |c_i|/|b_i|
    RMS scores there, signed by the mean (across all analysis_alphas) of the raw per-alpha
    coefficients. Sign here indicates the direction a parameter would move to realise g /
    delta respectively -- agreement in sign is a stronger, more specific form of alignment
    than agreement in magnitude alone. K_net is excluded, same reasoning as plot_c_vs_b_scatter."""
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    plotting_slices = {name: sl for name, sl in parameter_slices.items() if name in kept_names}
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
        c = _signed_like_score(checkpoint, "xrot_attribution_coefficients", checkpoint["xrot_score"])
        b = _signed_like_score(checkpoint, "delta_attribution_coefficients", checkpoint["b_score"])
        kept_mask = np.zeros_like(c, dtype=bool)
        for sl in plotting_slices.values():
            kept_mask[sl] = True
        linthresh = robust_linthresh(np.concatenate([c[kept_mask], b[kept_mask]]))
        for name in modules:
            sl = plotting_slices[name]
            ax.scatter(
                c[sl], b[sl], s=14, color=colours[name],
                label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
            )
        ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.6", linewidth=0.8, zorder=0)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set(xlabel=r"$c_i$ (orbit-tangent $g$ attribution, RMS-magnitude/mean-sign)",
               ylabel=r"$b_i$ (defect $\delta$ attribution, RMS-magnitude/mean-sign)")
        ax.set_title(subtitle)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_dir / "c_vs_b_scatter_signed.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "c_vs_b_scatter_signed.pdf", bbox_inches="tight")
    plt.close(fig)


def report_sparsity(model: torch.nn.Module, thresholds: tuple[float, ...] = (1e-4, 1e-3, 1e-2)) -> dict[str, Any]:
    """Fraction of parameters with |theta_i| below each threshold -- a direct sparsity readout,
    complementing E_i (does an L1 penalty actually induce sparsity, and by how much)."""
    all_params = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()])
    return {
        "n_parameters": int(all_params.numel()),
        "mean_abs_weight": float(all_params.mean()),
        "median_abs_weight": float(all_params.median()),
        "fraction_below_threshold": {
            str(t): float((all_params < t).float().mean()) for t in thresholds
        },
    }


def _evaluate_force_and_potential(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype,
    alpha: float, q1_flat: torch.Tensor, q2_flat: torch.Tensor,
):
    q = torch.stack([q1_flat, q2_flat], dim=1)
    alpha_col = torch.full((q.shape[0], 1), float(alpha), device=device, dtype=dtype)
    r2 = q1_flat**2 + q2_flat**2
    true_v = 0.5 * alpha * r2 + 0.25 * r2**2
    true_force = torch.stack([-q1_flat * (alpha + r2), -q2_flat * (alpha + r2)], dim=1)
    if cfg.architecture == "direct_mlp":
        p_zero = torch.zeros_like(q)
        with torch.no_grad():
            dpdt, _ = model(p_zero, q, alpha_col)
        learned_force = dpdt
        learned_v = None
    else:
        q_grad = q.clone().requires_grad_(True)
        with torch.enable_grad():
            potential = model.V_net(torch.cat((q_grad, alpha_col), dim=1))
            learned_force = -torch.autograd.grad(potential.sum(), q_grad)[0].detach()
        learned_v = potential.detach().squeeze(-1)
    return true_v, true_force, learned_v, learned_force


def _make_grid(cfg: Config, n: int, device: torch.device, dtype: torch.dtype):
    axis = torch.linspace(-cfg.q_extent, cfg.q_extent, n, device=device, dtype=dtype)
    q1_mesh, q2_mesh = torch.meshgrid(axis, axis, indexing="ij")
    return q1_mesh.reshape(-1), q2_mesh.reshape(-1)


def plot_learned_force_field(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype, output_dir: Path
) -> None:
    """Quiver comparison of the learned vs. true force field, one column per analysis alpha,
    true on top / learned on the bottom (horizontal layout -- easier to scan across alpha).

    True and learned quivers within a column share the same matplotlib ``scale``
    (derived from the true force's own magnitude at that alpha), so arrow
    length is directly comparable between the two rows -- a visibly
    shorter/longer learned arrow means the learned force is actually
    weaker/stronger, not an artefact of independent auto-scaling.
    """
    model.eval()
    alphas = cfg.plotting_alphas
    q1_coarse, q2_coarse = _make_grid(cfg, 9, device, dtype)

    fig, axes = plt.subplots(
        2, len(alphas), figsize=(3.4 * len(alphas), 7.2), constrained_layout=True, squeeze=False,
    )
    for col, alpha in enumerate(alphas):
        _, true_force_c, _, learned_force_c = _evaluate_force_and_potential(
            model, cfg, device, dtype, alpha, q1_coarse, q2_coarse
        )
        true_mag = torch.linalg.vector_norm(true_force_c, dim=-1).max().clamp_min(1e-12).item()
        scale = true_mag * 12  # matplotlib quiver "scale": larger = shorter arrows: fixed reference for both rows.
        for row, force, title in (
            (0, true_force_c, "true"),
            (1, learned_force_c, "learned"),
        ):
            ax = axes[row, col]
            ax.quiver(
                q1_coarse.cpu().numpy(), q2_coarse.cpu().numpy(),
                force[:, 0].cpu().numpy(), force[:, 1].cpu().numpy(),
                scale=scale, scale_units="width", angles="xy", color="tab:blue", width=0.008,
            )
            ax.set(xlabel="$q_1$", ylabel="$q_2$" if col == 0 else "")
            if row == 0:
                ax.set_title(rf"$\alpha={alpha:g}$")
            if col == 0:
                ax.annotate(
                    title, xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90,
                    ha="center", va="center",
                )
            ax.set_aspect("equal")

    fig.savefig(output_dir / "learned_force_field_final.png", dpi=200)
    fig.savefig(output_dir / "learned_force_field_final.pdf")
    plt.close(fig)


def plot_learned_potential(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype, output_dir: Path
) -> None:
    """Heatmap/contour comparison of the learned vs. true potential, one column per analysis
    alpha, true on top / learned on the bottom (only for architectures with a V_net --
    direct_mlp has no potential to plot).

    V_theta is only physically meaningful up to an arbitrary additive constant: force = -grad V
    never constrains V's absolute offset, and the trajectory-fitting loss never sees V directly,
    only the force it induces -- so comparing *absolute* V levels between true and learned is
    not meaningful until this gauge freedom is fixed. Both panels are baseline-shifted to their
    own minimum (V -> V - V.min()) before plotting, which is the correct, physically-motivated
    fix (rather than a cosmetic one) -- this is exactly analogous to choosing a common zero of
    potential energy. After that shift, both panels share the same colour scale *and* the same
    explicit contour level values (drawn from the true, shifted potential's own range), so a
    contour line at a given value/colour means the same shifted-V level in both panels.
    """
    if cfg.architecture == "direct_mlp":
        return
    model.eval()
    alphas = cfg.plotting_alphas
    q1_fine, q2_fine = _make_grid(cfg, 60, device, dtype)
    n = int(math.isqrt(q1_fine.shape[0]))
    q1_grid_np = q1_fine.cpu().numpy().reshape(n, n)
    q2_grid_np = q2_fine.cpu().numpy().reshape(n, n)

    fig, axes = plt.subplots(
        2, len(alphas), figsize=(3.6 * len(alphas), 7.6), constrained_layout=True, squeeze=False,
    )
    for col, alpha in enumerate(alphas):
        true_v_f, _, learned_v_f, _ = _evaluate_force_and_potential(
            model, cfg, device, dtype, alpha, q1_fine, q2_fine
        )
        true_shifted = true_v_f - true_v_f.min()
        learned_shifted = learned_v_f - learned_v_f.min()
        vmin, vmax = 0.0, true_shifted.max().clamp_min(1e-12).item()
        levels = np.linspace(vmin, vmax, 9)
        for row, v, title in (
            (0, true_shifted, "true"),
            (1, learned_shifted, "learned"),
        ):
            ax = axes[row, col]
            v_np = v.cpu().numpy().reshape(n, n)
            mesh = ax.pcolormesh(
                q1_grid_np, q2_grid_np, v_np, vmin=vmin, vmax=vmax, shading="gouraud", cmap="viridis",
            )
            cs = ax.contour(q1_grid_np, q2_grid_np, v_np, levels=levels, colors="white", linewidths=0.7)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
            ax.set(xlabel="$q_1$", ylabel="$q_2$" if col == 0 else "")
            if row == 0:
                ax.set_title(rf"$\alpha={alpha:g}$")
            if col == 0:
                ax.annotate(
                    title, xy=(-0.4, 0.5), xycoords="axes fraction", rotation=90,
                    ha="center", va="center",
                )
            ax.set_aspect("equal")
        # One colorbar per alpha column (each alpha has its own vmax after baseline-shifting,
        # so a single figure-wide colorbar would misrepresent all but one column).
        fig.colorbar(mesh, ax=list(axes[:, col]), fraction=0.046, pad=0.04, label="$V - V_{\\min}$")

    fig.savefig(output_dir / "learned_potential_final.png", dpi=200)
    fig.savefig(output_dir / "learned_potential_final.pdf")
    plt.close(fig)


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
    plot_module_attribution(all_results, parameter_slices, output_dir)
    plot_attribution_scatter(all_results, parameter_slices, output_dir)
    plot_module_b_attribution(all_results, parameter_slices, output_dir)
    plot_b_attribution_scatter(all_results, parameter_slices, output_dir)
    plot_module_c_vs_b_comparison(all_results, parameter_slices, output_dir)
    plot_c_vs_b_scatter(all_results, parameter_slices, output_dir)
    plot_c_vs_b_scatter_signed(all_results, parameter_slices, output_dir)
    plot_module_symmetry_comparison(all_results, parameter_slices, output_dir)
    plot_symmetry_attribution_scatter(all_results, parameter_slices, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))

    model.load_state_dict(checkpoint_states[max(checkpoint_states.keys())])
    plot_learned_force_field(model, cfg, device, dtype, output_dir)
    plot_learned_potential(model, cfg, device, dtype, output_dir)
    sparsity = report_sparsity(model)
    (output_dir / "sparsity_report.json").write_text(json.dumps(sparsity, indent=2))
    # Training Quality Report
    summary = {
    "xrot_magnitude_correlation": all_results[-1]["xrot_magnitude_correlation"],
    "xrot_b_correlation": all_results[-1]["xrot_b_correlation"],
    "xrot_participation_ratio": all_results[-1]["xrot_participation_ratio"],
    "sparsity_report": sparsity,
}

    (output_dir / "xrot_sparsity_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Sparsity (final checkpoint): {sparsity['fraction_below_threshold']}")
    print(
        f"Final rotation attribution PR: {all_results[-1]['xrot_participation_ratio']:.2f}; "
        f"energy attribution PR: {all_results[-1]['energy_participation_ratio']:.2f}"
    )
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()
