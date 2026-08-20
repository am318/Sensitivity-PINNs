"""Functional-sensitivity experiment for the ASRNN Henon-Heiles model.

The physical system is

    V(q1, q2; alpha1, alpha2) = (q1**2 + q2**2)/2 + alpha1*q1**2*q2 - alpha2*q2**3/3,
    dq/dt = p,
    dp1/dt = F1 = -q1 - 2*alpha1*q1*q2,
    dp2/dt = F2 = -q2 - alpha1*q1**2 + alpha2*q2**2.

Two exact symmetries are used as the physical generators (PDF sec. 0.2):

* A reflection q1 -> -q1 (with p1 -> -p1) is an exact symmetry for *any*
  (alpha1, alpha2).
* A 120-degree rotation of (q1, q2) is an exact symmetry only on the
  symmetric diagonal alpha1 = alpha2 (the classical Henon-Heiles C3v point
  group); moving off that diagonal continuously breaks it, giving a graded
  symmetry-breaking test.
* The coupling-constant family direction dF/dalpha along the symmetric
  diagonal, (-2*q1*q2, q2**2 - q1**2), is a continuous generator
  and is projected onto the functional tangent
  space Im(J_theta).

All frequently changed experiment choices live in the Config dataclass below.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    prettify_parameter_name,
    resolve_attribution_coefficients,
    run_training_loop,
    select_device,
    select_dtype,
    write_csv_rows,
    write_result_json,
)
from sensitivity_tools import (
    aggregate_by_module,
    aggregate_by_module_mean,
    finite_transform_residual,
    parameter_gradient_row,
    participation_ratio,
    per_parameter_equivariance_error,
    relative_energy_drift,
    sensitivity_transform_residual,
)

ROOT = Path(__file__).resolve().parent
ASRNN_HENON_HEILES = ROOT / "ASRNN_Sparse_Data" / "Henon_Heiles_Code"
if not ASRNN_HENON_HEILES.exists():
    raise FileNotFoundError(
        f"Expected the cloned ASRNN repository at {ASRNN_HENON_HEILES}"
    )
sys.path.insert(0, str(ASRNN_HENON_HEILES))

from helper import (  # noqa: E402
    F,
    Hamiltonian_MLP_Network,
    VerletIntegrator,
    generate_data,
    residuals,
    train_test_split,
)


@dataclass
class Config:
    # Reproducibility and outputs
    seed: int = 0
    device: str = "auto"  # auto, cpu, cuda, or mps
    dtype: str = "float32"
    output_dir: str = "outputs/asrnn_henon_heiles_symmetry"

    # "hamiltonian": the ASRNN V_net/K_net split (architecturally guarantees
    # conservation of the network's own energy via the symplectic Verlet
    # step). "direct_mlp": one plain MLP outputting (dp/dt, dq/dt) directly,
    # integrated with the same kick-drift-kick step shape but with no
    # conservation structure at all (Experiment 2's architecture comparison).
    architecture: str = "hamiltonian"  # hamiltonian or direct_mlp

    # ASRNN Hamiltonian network. V_net receives (q1, q2, alpha1, alpha2); K_net receives (p1, p2).
    kinetic_hidden_dim: int = 50
    kinetic_hidden_layers: int = 3
    potential_hidden_dim: int = 50
    potential_hidden_layers: int = 3

    # direct_mlp architecture only.
    direct_mlp_hidden_dim: int = 50
    direct_mlp_hidden_layers: int = 2

    # Sparse/noisy trajectory generation. Training is restricted to the
    # symmetric diagonal alpha1=alpha2 so the trained system has the exact
    # C3v symmetry over its whole training distribution; the network is only
    # ever probed off-diagonal at analysis time (a generalisation test).
    training_alpha_pairs: list[list[float]] = field(
        default_factory=lambda: [[0.6, 0.6], [0.8, 0.8], [1.0, 1.0], [1.2, 1.2], [1.4, 1.4]]
    )
    trajectory_window: int = 5
    trajectory_splits: int = 5
    sampled_instants: int = 5
    initial_conditions_per_alpha: int = 10
    integration_dt: float = 0.1
    coarsening_factor: int = 50
    noise_standard_deviation: float = 0
    noise_correlation_time: float = 0
    validation_fraction: float = 0.25
    # Doubles the training set with a q1-reflected copy of every trajectory --
    # (p1,q1)->(-p1,-q1), (p2,q2) unchanged, is exact for *every* (alpha1,alpha2)
    # pair (V depends on q1 only through q1**2, unlike the C3 rotation, which is
    # only exact on the symmetric diagonal alpha1=alpha2) -- verified numerically
    # before use, so only this always-valid symmetry is used for augmentation.
    augment_dataset: bool = False

    # Training. Adam is convenient for resolving the evolution in time.
    optimizer: str = "adam"  # adam or lbfgs
    training_steps: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    l1_weight: float = 0
    # Checkpoints are specified as fractions of training_steps (each in
    # [0, 1]) so they scale automatically if training_steps changes, e.g.
    # under --quick. Resolved to absolute step indices in validate_config.
    checkpoint_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
    checkpoint_steps: list[int] = field(default_factory=list)
    lbfgs_history_size: int = 10
    
    # Functional sensitivity probe: a 2D (q1, q2) grid.
    q_extent: float = 1.0
    q_grid_points_per_axis: int = 7
    rotation_angle_degrees: float = 120.0

    # Two analysis sweeps over (alpha1, alpha2). "Symmetric" sweeps the
    # common coupling alpha1=alpha2=alpha (mirrors the double well's
    # continuous-alpha story: random init vs trained). "Breaking" fixes
    # alpha1 and varies alpha2 away from it, to show residuals grow with
    # |alpha1-alpha2| for the rotation generator (but not the reflection
    # generator, which is exact everywhere -- a built-in control).
    analysis_symmetric_alphas: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    analysis_breaking_alpha1: float = 1.0
    analysis_breaking_alpha2_offsets: list[float] = field(
        default_factory=lambda: [-0.5, -0.25, 0.0, 0.25, 0.5]
    )

    # Only singular directions above this fraction of sigma_max define the
    # numerically resolved tangent space (see double-well script for why).
    tangent_svd_relative_cutoff: float = 1e-3
    top_parameters_to_report: int = 20
    # How the per-parameter attribution coefficients c_i (tangent_projection_auto,
    # sensitivity_tools.py) are solved: "l2" (min-norm, dense), "l1" (min-1-norm,
    # sparse but can arbitrarily zero one of several collinear parameters), or
    # "elastic_net" (default -- sparse like L1, but with the least ridge admixture
    # needed to still tie-break collinear columns; see choose_l1_ratio_for_sparsity).
    attribution_method: str = "elastic_net"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Optional JSON file overriding fields in Config."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a very small end-to-end smoke test."
    )
    parser.add_argument("--device", help="Override Config.device.")
    parser.add_argument("--output-dir", help="Override Config.output_dir.")
    parser.add_argument(
        "--architecture", choices=["hamiltonian", "direct_mlp"], help="Override Config.architecture."
    )
    parser.add_argument("--augment-dataset", dest="augment_dataset", action="store_true", default=None, help="Override Config.augment_dataset to True.")
    parser.add_argument("--no-augment-dataset", dest="augment_dataset", action="store_false", help="Override Config.augment_dataset to False.")
    parser.add_argument("--l1-weight", type=float, help="Override Config.l1_weight.")
    parser.add_argument(
        "--attribution-method",
        choices=["l2", "l1", "elastic_net"],
        help="Override Config.attribution_method (how c_i is solved for; default elastic_net).",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.augment_dataset is not None:
        cfg.augment_dataset = args.augment_dataset
    if args.l1_weight is not None:
        cfg.l1_weight = args.l1_weight
    if args.architecture:
        cfg.architecture = args.architecture
    if args.attribution_method:
        cfg.attribution_method = args.attribution_method
    if args.quick:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.direct_mlp_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.coarsening_factor = 2
        cfg.training_steps = 2
        cfg.checkpoint_fractions = [0.0, 0.5, 1.0]
        cfg.analysis_symmetric_alphas = [0.8, 1.0, 1.2]
        cfg.analysis_breaking_alpha2_offsets = [-0.3, 0.0, 0.3]
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
    if cfg.attribution_method not in {"l2", "l1", "elastic_net"}:
        raise ValueError("attribution_method must be 'l2', 'l1', or 'elastic_net'.")
    cfg.checkpoint_steps = sorted(
        {int(round(f * cfg.training_steps)) for f in cfg.checkpoint_fractions}
        | {0, cfg.training_steps}
    )

def augment_with_reflection(trajectories: torch.Tensor) -> torch.Tensor:
    """Double a trajectory batch with a q1-reflected copy: (p1,q1)->(-p1,-q1), (p2,q2) unchanged.

    ``trajectories`` has layout ``[T, batch, 4]`` = ``[p1, p2, q1, q2]``. Exact
    for every (alpha1, alpha2) pair, since V depends on q1 only through
    q1**2 -- unlike the C3 rotation, which is only exact on the symmetric
    diagonal alpha1=alpha2, so only this always-valid symmetry is used here.
    """
    reflected = trajectories.clone()
    reflected[:, :, 0] = -trajectories[:, :, 0]
    reflected[:, :, 2] = -trajectories[:, :, 2]
    return reflected


def make_dataset(cfg: Config, device: torch.device, dtype: torch.dtype):
    alphas = torch.tensor(cfg.training_alpha_pairs, device=device, dtype=dtype)
    total_length = cfg.trajectory_splits + cfg.trajectory_window - 1
    trajectories, params, indices = generate_data(
        alphas,
        F,
        total_length,
        cfg.trajectory_window,
        cfg.sampled_instants,
        dt=cfg.integration_dt,
        in_conds=cfg.initial_conditions_per_alpha,
        coarsening_factor=cfg.coarsening_factor,
        nsr=cfg.noise_standard_deviation**2,
        theta=(1.0 / cfg.noise_correlation_time if cfg.noise_standard_deviation > 0 else 0.0),
        burn_in_var=0,
        apply_ou_noise=cfg.noise_standard_deviation > 0,
        device=device,
        dtype=dtype,
        seed_ou=cfg.seed,
    )
    if cfg.augment_dataset:
        trajectories = torch.cat((trajectories, augment_with_reflection(trajectories)), dim=1)
        params = torch.cat((params, params), dim=0)
        indices = torch.cat((indices, indices), dim=0)
        perm = torch.randperm(trajectories.size(1), device=trajectories.device)
        trajectories, params, indices = trajectories[:, perm, :], params[perm], indices[perm]
    return train_test_split(trajectories, params, indices, val_size=cfg.validation_fraction)


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    if cfg.architecture == "direct_mlp":
        model = DirectDynamicsMLP(
            p_dim=2, q_dim=2, param_dim=2,
            hidden_dim=cfg.direct_mlp_hidden_dim, n_hidden=cfg.direct_mlp_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        return model, DirectLeapfrogIntegrator(model=model, dt=cfg.integration_dt)
    model = Hamiltonian_MLP_Network(
        kin_hidden_dim=cfg.kinetic_hidden_dim,
        kin_n_hidden=cfg.kinetic_hidden_layers,
        pot_hidden_dim=cfg.potential_hidden_dim,
        pot_n_hidden=cfg.potential_hidden_layers,
        device=device,
    ).to(device=device, dtype=dtype)
    return model, VerletIntegrator(model=model, dt=cfg.integration_dt)


def make_optimizer(cfg: Config, model: torch.nn.Module):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.learning_rate,
        history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )


def build_probe_grid(cfg: Config, device: torch.device, dtype: torch.dtype):
    axis = torch.linspace(-cfg.q_extent, cfg.q_extent, cfg.q_grid_points_per_axis, device=device, dtype=dtype)
    q1_mesh, q2_mesh = torch.meshgrid(axis, axis, indexing="ij")
    return q1_mesh.reshape(-1), q2_mesh.reshape(-1)


def rotation_matrix(theta_degrees: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = math.radians(theta_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def reflection_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([[-1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)


def transform_points(q1: torch.Tensor, q2: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.stack([q1, q2], dim=1)
    qt = q @ matrix.T
    return qt[:, 0], qt[:, 1]


def coupling_generator_target(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """dF_true/dalpha along the symmetric diagonal alpha1=alpha2=alpha: (-2 q1 q2, q2^2 - q1^2)."""
    return torch.stack([-2.0 * q1 * q2, q2**2 - q1**2], dim=1)


ROTATION_LIE_GENERATOR = torch.tensor([[0.0, -1.0], [1.0, 0.0]])  # d/dtheta R_theta |_0


def rotation_generator_target_NEGATIVE_CONTROL(
    q1: torch.Tensor, q2: torch.Tensor, force: torch.Tensor, spatial_jacobian: torch.Tensor
) -> torch.Tensor:
    """X_rot F = (dF/dq)(Omega q) - Omega F(q) -- a DELIBERATE negative control, not a symmetry test.

    This is the same infinitesimal-rotation construction sec. 0's erratum
    found invalid for this system (verified again here: relative magnitude
    ~2.7 at alpha1=alpha2=1, nonzero even on the symmetric diagonal where the
    *discrete* C3 rotation is exact -- a continuous group has no valid
    generator unless the symmetry is genuinely continuous, which this system
    is not). It is reintroduced here *on purpose*, precisely because it is
    known to be invalid: unlike the erratum (which mistakenly treated a small
    X_rot F as evidence of learned rotational symmetry), the point here is to
    test whether the *attribution* c_i built from this generator -- via the
    exact same tangent_projection machinery used for the Mexican-hat's
    genuine rotation attribution -- shows the same "concentrated on a few
    parameters" pattern even though there is no real symmetry underneath. If
    it does, that pattern is a generic property of the projection method
    (tangent-space redundancy/rank structure), not evidence of symmetry
    learning specifically -- see SYMMETRY_SENSITIVITY_NOTES.md.
    """
    omega_q = torch.stack([-q2, q1], dim=1)
    directional = torch.einsum("nkj,nj->nk", spatial_jacobian, omega_q)
    omega = ROTATION_LIE_GENERATOR.to(device=force.device, dtype=force.dtype)
    return directional - force @ omega.T


def evaluate_at_points(
    model: torch.nn.Module,
    q1_pts: torch.Tensor,
    q2_pts: torch.Tensor,
    alpha1: float,
    alpha2: float,
    architecture: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    need_spatial_jacobian: bool = False,
):
    """Evaluate learned potential, force, force-parameter Jacobian (and optionally dF/dq) at each point.

    Returns ``(V, F, J[, spatial_J])`` with shapes ``[N]``, ``[N, 2]``,
    ``[N, 2, P]`` (``[N, 2, 2]`` for spatial_J). There is no V_net for
    ``architecture="direct_mlp"``, so V is NaN there (force is read directly
    from the model's own dp/dt output at a fixed reference momentum
    p=(0,0), the same architectural fact that makes the Hamiltonian net's
    force independent of p).
    """
    params = tuple(model.parameters())
    v_values, f_values, jac_rows, spatial_jacs = [], [], [], []
    for q1_val, q2_val in zip(q1_pts.detach().cpu().tolist(), q2_pts.detach().cpu().tolist()):
        q = torch.tensor([[q1_val, q2_val]], device=device, dtype=dtype, requires_grad=True)
        alpha = torch.tensor([[alpha1, alpha2]], device=device, dtype=dtype)
        if architecture == "direct_mlp":
            p = torch.zeros_like(q)
            dpdt, _ = model(p, q, alpha)
            force = dpdt.squeeze(0)
            v_values.append(float("nan"))
        else:
            potential = model.V_net(torch.cat((q, alpha), dim=1))
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
    model: torch.nn.Module,
    p1_values: torch.Tensor,
    p2_values: torch.Tensor,
    alpha1: float,
    alpha2: float,
    architecture: str,
) -> torch.Tensor:
    """Return the model's own (dq1/dt, dq2/dt) at each probe momentum, at a fixed reference q=(0,0).

    For the Hamiltonian net, dq/dt = dK_theta/d(p1,p2) is architecturally
    independent of both q and alpha. For direct_mlp there is no such
    guarantee, so alpha must be supplied and q is held at a reference value.
    """
    p = torch.stack([p1_values, p2_values], dim=1)
    if architecture == "direct_mlp":
        q = torch.zeros_like(p)
        alpha = torch.tensor([[alpha1, alpha2]], device=p.device, dtype=p.dtype).expand(p.shape[0], -1)
        with torch.no_grad():
            _, dqdt = model(p, q, alpha)
        return dqdt
    p = p.clone().requires_grad_(True)
    kinetic = model.K_net(p)
    dqdt = torch.autograd.grad(kinetic.sum(), p)[0]
    return dqdt.detach()


def analyse_checkpoint(
    model: torch.nn.Module,
    step: int,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    flat_names: list[str],
    parameter_slices: dict[str, slice],
    attribution_l1_ratio_cache: dict[str, float] | None = None,
) -> dict[str, Any]:
    model.eval()
    if attribution_l1_ratio_cache is None:
        attribution_l1_ratio_cache = {}
    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    rot_mat = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    refl_mat = reflection_matrix(device, dtype)
    q1_rot, q2_rot = transform_points(q1_grid, q2_grid, rot_mat)
    q1_refl, q2_refl = transform_points(q1_grid, q2_grid, refl_mat)

    alpha_pairs: list[tuple[float, float, str]] = [
        (alpha, alpha, "symmetric") for alpha in cfg.analysis_symmetric_alphas
    ] + [
        (
            cfg.analysis_breaking_alpha1,
            cfg.analysis_breaking_alpha1 + offset,
            "breaking",
        )
        for offset in cfg.analysis_breaking_alpha2_offsets
    ]

    coupling_target = coupling_generator_target(q1_grid, q2_grid)
    n_points = q1_grid.shape[0]

    # Reuse the same grid shape for a paired (p1, p2) probe (not a full
    # phase-space mesh, to keep this diagnostic's cost the same order as the
    # existing force probe).
    p1_grid, p2_grid = build_probe_grid(cfg, device, dtype)
    grad_e_p = torch.stack([p1_grid, p2_grid], dim=1)

    alpha_results = []
    for alpha1, alpha2, tag in tqdm(alpha_pairs, desc=f"analysing step {step}", leave=False):
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_grid, q2_grid, alpha1, alpha2, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        v_rot, f_rot, j_rot = evaluate_at_points(
            model, q1_rot, q2_rot, alpha1, alpha2, cfg.architecture, device=device, dtype=dtype
        )
        v_refl, f_refl, j_refl = evaluate_at_points(
            model, q1_refl, q2_refl, alpha1, alpha2, cfg.architecture, device=device, dtype=dtype
        )
        learned_dqdt_values = learned_dqdt(model, p1_grid, p2_grid, alpha1, alpha2, cfg.architecture)

        rotation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_rot.unsqueeze(-1))
        rotation_force_residual = finite_transform_residual(f_x, f_rot, rot_mat)
        rotation_sensitivity_residual = sensitivity_transform_residual(j_x, j_rot, rot_mat)
        rotation_equivariance_error = per_parameter_equivariance_error(j_x, j_rot, rot_mat)

        reflection_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_refl.unsqueeze(-1))
        reflection_force_residual = finite_transform_residual(f_x, f_refl, refl_mat)
        reflection_sensitivity_residual = sensitivity_transform_residual(j_x, j_refl, refl_mat)
        reflection_equivariance_error = per_parameter_equivariance_error(j_x, j_refl, refl_mat)

        sensitivity = torch.sqrt(torch.mean(j_x.detach().cpu().double().square(), dim=(0, 1)))
        jac_flat = j_x.reshape(n_points * 2, -1)

        # Noether energy-conservation diagnostic (see the double-well script
        # for the important caveat: this is NOT architecturally pinned near
        # zero, only "conserves its own H_theta" is -- so it starts large and
        # falls only as far as training makes H_theta approach the truth).
        grad_e_q = torch.stack(
            [
                q1_grid + 2 * alpha1 * q1_grid * q2_grid,
                q2_grid + alpha1 * q1_grid**2 - alpha2 * q2_grid**2,
            ],
            dim=1,
        )
        energy_conservation_violation = relative_energy_drift(
            grad_e_p.detach().cpu().double(), grad_e_q.detach().cpu().double(), f_x.detach().cpu().double(), learned_dqdt_values.detach().cpu().double()
        )

        result = {
            "alpha1": float(alpha1),
            "alpha2": float(alpha2),
            "tag": tag,
            "energy_conservation_violation": energy_conservation_violation,
            "rotation_potential_residual": rotation_potential_residual,
            "rotation_force_residual": rotation_force_residual,
            "rotation_sensitivity_residual": rotation_sensitivity_residual,
            "reflection_potential_residual": reflection_potential_residual,
            "reflection_force_residual": reflection_force_residual,
            "reflection_sensitivity_residual": reflection_sensitivity_residual,
            "sensitivity": sensitivity.cpu().tolist(),
            "rotation_equivariance_error_by_parameter": rotation_equivariance_error.cpu().tolist(),
            "reflection_equivariance_error_by_parameter": reflection_equivariance_error.cpu().tolist(),
            "module_sensitivity": aggregate_by_module(sensitivity, parameter_slices),
            "module_rotation_equivariance_error": aggregate_by_module_mean(
                rotation_equivariance_error, parameter_slices
            ),
            "module_reflection_equivariance_error": aggregate_by_module_mean(
                reflection_equivariance_error, parameter_slices
            ),
        }

        if tag == "symmetric":
            target_flat = coupling_target.reshape(n_points * 2)
            (
                coupling_projection_error,
                coupling_principal_angle,
                coupling_coefficients,
                coupling_resolved_rank,
                coupling_singular_values,
            ) = resolve_attribution_coefficients(
                cfg, attribution_l1_ratio_cache, "coupling", jac_flat, target_flat
            )
            result.update(
                {
                    "coupling_projection_error": coupling_projection_error,
                    "coupling_principal_angle_degrees": coupling_principal_angle,
                    "coupling_resolved_rank": coupling_resolved_rank,
                    "coupling_singular_values": coupling_singular_values,
                    "coupling_attribution_coefficients": coupling_coefficients.cpu().tolist(),
                    "module_coupling_attribution": aggregate_by_module(
                        coupling_coefficients.abs(), parameter_slices
                    ),
                }
            )

            # NEGATIVE CONTROL (see rotation_generator_target_NEGATIVE_CONTROL's
            # docstring): this system has no continuous rotational symmetry, so
            # X_rot F is large and nonzero even here, on the symmetric diagonal.
            # Attributed via the identical tangent_projection machinery as the
            # valid coupling generator above, purely to test whether a
            # "concentrated on a few parameters" attribution pattern appears
            # regardless of whether a real symmetry exists.
            xrot_target = rotation_generator_target_NEGATIVE_CONTROL(q1_grid, q2_grid, f_x, spatial_jac_x)
            xrot_target_flat = xrot_target.reshape(n_points * 2)
            (
                xrot_control_projection_error,
                xrot_control_principal_angle,
                xrot_control_coefficients,
                xrot_control_resolved_rank,
                xrot_control_singular_values,
            ) = resolve_attribution_coefficients(
                cfg, attribution_l1_ratio_cache, "xrot_control", jac_flat, xrot_target_flat
            )
            result.update(
                {
                    "xrot_control_projection_error": xrot_control_projection_error,
                    "xrot_control_principal_angle_degrees": xrot_control_principal_angle,
                    "xrot_control_resolved_rank": xrot_control_resolved_rank,
                    "xrot_control_singular_values": xrot_control_singular_values,
                    "xrot_control_attribution_coefficients": xrot_control_coefficients.cpu().tolist(),
                    "module_xrot_control_attribution": aggregate_by_module(
                        xrot_control_coefficients.abs(), parameter_slices
                    ),
                }
            )
        alpha_results.append(result)

    symmetric_results = [r for r in alpha_results if r["tag"] == "symmetric"]
    coefficient_matrix = np.asarray([r["coupling_attribution_coefficients"] for r in symmetric_results])
    coupling_score = np.sqrt(np.mean(coefficient_matrix**2, axis=0))
    top_coupling = np.argsort(coupling_score)[::-1][: cfg.top_parameters_to_report]

    xrot_control_matrix = np.asarray([r["xrot_control_attribution_coefficients"] for r in symmetric_results])
    xrot_control_score = np.sqrt(np.mean(xrot_control_matrix**2, axis=0))

    equivariance_matrix = np.asarray(
        [r["rotation_equivariance_error_by_parameter"] for r in alpha_results]
    )
    equivariance_score = np.sqrt(np.mean(equivariance_matrix**2, axis=0))
    top_equivariance_violators = np.argsort(equivariance_score)[::-1][: cfg.top_parameters_to_report]

    return {
        "step": step,
        "alpha_results": alpha_results,
        "coupling_score": coupling_score.tolist(),
        "xrot_control_score": xrot_control_score.tolist(),
        "coupling_participation_ratio": participation_ratio(coupling_score),
        "xrot_control_participation_ratio": participation_ratio(xrot_control_score),
        "top_coupling_generator_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(coupling_score[i])}
            for i in top_coupling
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
        result["top_coupling_generator_parameters"],
        ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_coupling_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_equivariance_violating_parameters"],
        ["flat_index", "name", "rms_equivariance_error"],
        output_dir / f"top_equivariance_violations_step_{step:06d}.csv",
    )


def plot_symmetric_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Core Experiment-1 story: random init vs trained, along the symmetric diagonal alpha1=alpha2=alpha."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(all_results)))

    for colour, checkpoint in zip(colours, all_results):
        rows = [r for r in checkpoint["alpha_results"] if r["tag"] == "symmetric"]
        alphas = np.asarray([r["alpha1"] for r in rows])
        label = f"step {checkpoint['step']}"
        axes[0, 0].plot(
            alphas, np.maximum([r["rotation_force_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[0, 1].plot(
            alphas, np.maximum([r["rotation_sensitivity_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 0].plot(
            alphas, [r["coupling_projection_error"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1, 1].plot(
            alphas, [r["coupling_principal_angle_degrees"] for r in rows],
            marker="o", ms=3, color=colour, label=label,
        )

    axes[0, 0].set(title="C3 rotation: force equivariance", ylabel="relative residual", yscale="log")
    axes[0, 1].set(title="C3 rotation: sensitivity equivariance", ylabel="relative Frobenius residual", yscale="log")
    axes[1, 0].set(
        title=r"Representation of coupling generator $\partial_\alpha F$",
        ylabel="relative projection error",
    )
    axes[1, 1].set(title="Coupling-generator tangent-space angle", ylabel="degrees")
    for ax in axes.flat:
        ax.set_xlabel(r"$\alpha=\alpha_1=\alpha_2$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "symmetric_diagonal_summary.png", dpi=200)
    fig.savefig(output_dir / "symmetric_diagonal_summary.pdf")
    plt.close(fig)


def central_symmetric_row(checkpoint: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in checkpoint["alpha_results"] if r["tag"] == "symmetric"]
    return rows[len(rows) // 2]


def plot_energy_conservation(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Noether energy-conservation violation vs. training step (mean over all analysis alpha pairs)."""
    steps = [c["step"] for c in all_results]
    mean_drift = [
        float(np.mean([r["energy_conservation_violation"] for r in c["alpha_results"]]))
        for c in all_results
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(steps, np.maximum(mean_drift, 1e-12), marker="o", ms=4)
    ax.set(
        title="Noether energy-conservation violation vs. training step",
        xlabel="training step", ylabel="relative energy drift (mean over alpha pairs)",
        yscale="log",
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "energy_conservation.png", dpi=200)
    fig.savefig(output_dir / "energy_conservation.pdf")
    plt.close(fig)


def plot_module_attribution_control(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Continuous-rotation attribution by module, random init vs. trained -- styled identically
    to the Mexican-hat's own module_attribution.png for direct comparison. NEGATIVE CONTROL: this
    system has no continuous rotational symmetry (see rotation_generator_target_NEGATIVE_CONTROL's
    docstring), so this tests whether the same attribution machinery still shows a "few large,
    most small" concentration pattern under L1 even when there's no real symmetry to represent."""
    first, last = all_results[0], all_results[-1]
    modules = list(parameter_slices.keys())

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    before = [np.linalg.norm(np.asarray(first["xrot_control_score"])[parameter_slices[m]]) for m in modules]
    after = [np.linalg.norm(np.asarray(last["xrot_control_score"])[parameter_slices[m]]) for m in modules]
    ax.bar(x - width / 2, before, width, label=f"step {first['step']} (random init)")
    ax.bar(x + width / 2, after, width, label=f"step {last['step']} (trained)")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
    ax.set(
        title=r"$X_{\rm rot}F$ attribution by module (NEGATIVE CONTROL -- no real rotational symmetry)",
        ylabel=r"$\|c_i\|$ within module",
    )
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "module_attribution_control.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_attribution_control.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_attribution_scatter_control(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter continuous-rotation attribution, random init vs. trained -- styled
    identically to the Mexican-hat's own attribution_scatter.png (log-log, init vs. trained)
    for direct comparison, rather than against this system's own valid coupling generator."""
    first, last = all_results[0], all_results[-1]
    plot_ei_initial_vs_final(
        np.asarray(first["xrot_control_score"]), np.asarray(last["xrot_control_score"]), parameter_slices,
        title=r"$X_{\rm rot}F$ attribution (NEGATIVE CONTROL): init vs. trained",
        output_stem=output_dir / "attribution_scatter_control",
        quantity_label="$|c_i|$",
        log_scale=True,
    )


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Localise the sensitivity-equivariance defect (sec. 0.6's E_i) to individual layers.

    Compares mean per-parameter E_i within each named module at random
    initialisation versus the final trained checkpoint, at the central
    symmetric alpha, for both the rotation and reflection generators.
    """
    first, last = all_results[0], all_results[-1]
    first_row, last_row = central_symmetric_row(first), central_symmetric_row(last)
    modules = list(first_row["module_rotation_equivariance_error"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.7 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, key, title in (
        (axes[0], "module_rotation_equivariance_error", "C3 rotation"),
        (axes[1], "module_reflection_equivariance_error", "q1-reflection"),
    ):
        before = [first_row[key][m] for m in modules]
        after = [last_row[key][m] for m in modules]
        ax.bar(x - width / 2, before, width, label=f"step {first['step']} (random init)")
        ax.bar(x + width / 2, after, width, label=f"step {last['step']} (trained)")
        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=60, ha="right", fontsize=8)
        ax.set(title=rf"{title}: mean $E_i$ by module", ylabel="mean $E_i$ within module")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200)
    fig.savefig(output_dir / "equivariance_by_module.pdf")
    plt.close(fig)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter E_i at random init vs. final checkpoint, one point per parameter, for both generators."""
    first, last = all_results[0], all_results[-1]
    first_row, last_row = central_symmetric_row(first), central_symmetric_row(last)
    for key, label, stem in (
        ("rotation_equivariance_error_by_parameter", "C3 rotation", "rotation_equivariance_scatter"),
        ("reflection_equivariance_error_by_parameter", "q1-reflection", "reflection_equivariance_scatter"),
    ):
        ei_initial = np.asarray(first_row[key])
        ei_final = np.asarray(last_row[key])
        plot_ei_initial_vs_final(
            ei_initial, ei_final, parameter_slices,
            title=rf"{label} sensitivity-equivariance $E_i$: init vs. trained",
            output_stem=output_dir / stem,
        )


def plot_breaking_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Symmetry-breaking sweep: rotation residual should grow with |alpha1-alpha2|; reflection should not (control)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(all_results)))

    for colour, checkpoint in zip(colours, all_results):
        rows = [r for r in checkpoint["alpha_results"] if r["tag"] == "breaking"]
        rows = sorted(rows, key=lambda r: r["alpha2"] - r["alpha1"])
        detuning = np.asarray([r["alpha2"] - r["alpha1"] for r in rows])
        label = f"step {checkpoint['step']}"
        axes[0].plot(
            detuning, np.maximum([r["rotation_force_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )
        axes[1].plot(
            detuning, np.maximum([r["reflection_force_residual"] for r in rows], 1e-12),
            marker="o", ms=3, color=colour, label=label,
        )

    axes[0].set(
        title="C3 rotation residual vs symmetry breaking",
        xlabel=r"$\alpha_2-\alpha_1$", ylabel="relative residual", yscale="log",
    )
    axes[1].set(
        title="q1-reflection residual (exact for all alpha1, alpha2 -- control)",
        xlabel=r"$\alpha_2-\alpha_1$", ylabel="relative residual", yscale="log",
    )
    for ax in axes:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "symmetry_breaking_summary.png", dpi=200)
    fig.savefig(output_dir / "symmetry_breaking_summary.pdf")
    plt.close(fig)


def plot_learned_force_field(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype, output_dir: Path
) -> None:
    """Quiver comparison of the learned vs true force field at the central symmetric alpha."""
    model.eval()
    alpha = cfg.analysis_symmetric_alphas[len(cfg.analysis_symmetric_alphas) // 2]
    axis = torch.linspace(-cfg.q_extent, cfg.q_extent, 9, device=device, dtype=dtype)
    q1_mesh, q2_mesh = torch.meshgrid(axis, axis, indexing="ij")
    q1_flat, q2_flat = q1_mesh.reshape(-1), q2_mesh.reshape(-1)
    q = torch.stack([q1_flat, q2_flat], dim=1)
    alpha_col = torch.full((q.shape[0], 2), float(alpha), device=device, dtype=dtype)
    if cfg.architecture == "direct_mlp":
        p_zero = torch.zeros_like(q)
        with torch.no_grad():
            dpdt, _ = model(p_zero, q, alpha_col)
        learned_force = dpdt
    else:
        q_grad = q.clone().requires_grad_(True)
        with torch.enable_grad():
            potential = model.V_net(torch.cat((q_grad, alpha_col), dim=1))
            learned_force = -torch.autograd.grad(potential.sum(), q_grad)[0].detach()
    true_force = torch.stack(
        [
            -q1_flat - 2 * alpha * q1_flat * q2_flat,
            -q2_flat - alpha * q1_flat**2 + alpha * q2_flat**2,
        ],
        dim=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, force, title in (
        (axes[0], true_force, "true force"),
        (axes[1], learned_force, "learned force"),
    ):
        ax.quiver(
            q1_flat.cpu().numpy(), q2_flat.cpu().numpy(),
            force[:, 0].cpu().numpy(), force[:, 1].cpu().numpy(),
        )
        ax.set(title=rf"{title} ($\alpha_1=\alpha_2={alpha:g}$)", xlabel="$q_1$", ylabel="$q_2$")
        ax.set_aspect("equal")
    fig.savefig(output_dir / "learned_force_field_final.png", dpi=200)
    fig.savefig(output_dir / "learned_force_field_final.pdf")
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
    print("Generating sparse Henon-Heiles trajectories...")
    train_data, validation_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}")

    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *args, **kwargs: generic_residuals(*args, **kwargs, p_dim=2)
    )
    history, checkpoint_states = run_training_loop(
        training_steps=cfg.training_steps,
        checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer,
        optimizer_name=cfg.optimizer,
        model=model,
        train_data=train_data,
        val_data=validation_data,
        trajectory_window=cfg.trajectory_window,
        residuals_fn=residuals_fn,
        integrator=integrator,
        l1_weight=cfg.l1_weight,
        weight_decay=cfg.weight_decay,
    )

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    np.savez(output_dir / "training_history.npz", **history)
    plot_training_history(history, output_dir)

    all_results = []
    attribution_l1_ratio_cache: dict[str, float] = {}
    for step in tqdm(sorted(checkpoint_states.keys()), desc="checkpoint analysis", unit="checkpoint"):
        model.load_state_dict(checkpoint_states[step])
        result = analyse_checkpoint(
            model, step, cfg, device, dtype, flat_names, parameter_slices, attribution_l1_ratio_cache
        )
        write_checkpoint_outputs(result, output_dir)
        all_results.append(result)

    plot_symmetric_summary(all_results, output_dir)
    plot_breaking_summary(all_results, output_dir)
    plot_energy_conservation(all_results, output_dir)
    plot_equivariance_by_module(all_results, output_dir)
    plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    plot_module_attribution_control(all_results, parameter_slices, output_dir)
    plot_attribution_scatter_control(all_results, parameter_slices, output_dir)
    model.load_state_dict(checkpoint_states[max(checkpoint_states.keys())])
    plot_learned_force_field(model, cfg, device, dtype, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))
    print(
        f"Final X_rot F (negative control -- no real rotational symmetry) attribution PR: "
        f"{all_results[-1]['xrot_control_participation_ratio']:.2f} "
        f"(coupling generator PR, unrelated pre-existing diagnostic: {all_results[-1]['coupling_participation_ratio']:.2f})"
    )
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()
