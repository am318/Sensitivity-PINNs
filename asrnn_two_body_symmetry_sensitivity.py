"""Functional-sensitivity experiment for the planar two-body problem, kept in the full 8D lab frame.

The physical system is

    q = (q1x, q1y, q2x, q2y), p = (p1x, p1y, p2x, p2y), unit masses m1 = m2 = 1,
    V(q; alpha) = -alpha / r,   r = |q1 - q2|,
    dq/dt = p,
    dp1/dt = F1 = -alpha * (q1 - q2) / r**3,
    dp2/dt = -F1.

Deliberately *not* reduced to relative coordinates: this system has two
independent, exact continuous symmetries the network must discover directly
in its raw 8D input representation -- SO(2) rotation (acting on (q1,q2) and
(p1,p2) simultaneously by the same matrix) and 2D translation (acting on
(q1,q2) by the same shift vector, momenta untouched). Both generators are
verified numerically against the exact analytic force before use (residual
~1e-14 at every alpha -- see the verification run in the session that built
this file). For an exact 1/r potential, Bertrand's theorem guarantees every
bound orbit is a closed ellipse (or circle) for free, no special parameter
tuning needed -- training_alphas are all positive (attractive) couplings.

Same machinery as the cylindrical Mexican-hat script (its two-generator
comparison is the closest existing template): finite-transform residuals,
per-parameter sensitivity-equivariance E_i, and attribution c_i via
tangent_projection of the network's own current X_rot F / X_trans F onto its
own force-sensitivity basis, for both generators side by side.
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

from direct_dynamics import DirectDynamicsMLP, generic_residuals
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
from mexican_hat_dynamics import GenericHamiltonianMLP
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
from two_body_dynamics import (
    F,
    SubsteppedDirectLeapfrogIntegrator,
    SubsteppedVerletIntegrator,
    generate_data,
    generate_data_analytic,
    train_test_split,
)


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "outputs/asrnn_two_body_symmetry"

    architecture: str = "hamiltonian"  # hamiltonian or direct_mlp

    kinetic_hidden_dim: int = 32
    kinetic_hidden_layers: int = 3
    potential_hidden_dim: int = 32
    potential_hidden_layers: int = 3
    direct_mlp_hidden_dim: int = 50
    direct_mlp_hidden_layers: int = 2

    # alpha is the attractive coupling strength (-alpha/r); must stay positive --
    # a negative alpha is repulsive and has no bound (closed) orbits at all.
    training_alphas: list[float] = field(
        default_factory=lambda: [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    )
    trajectory_window: int = 10
    trajectory_splits: int = 10
    sampled_instants: int = 10
    initial_conditions_per_alpha: int = 10
    integration_dt: float = 0.1
    # The training-time rollout (SubsteppedVerletIntegrator/SubsteppedDirectLeapfrogIntegrator)
    # takes this many sub-steps of size integration_dt/integrator_substeps per outer
    # trajectory-window index. IMPORTANT compute-cost note (session that added the analytic
    # Kepler generator below): this cost is roughly *linear* in integrator_substeps and in
    # (trajectory_window - 1) -- verified empirically (substeps 10->40 measured ~5x wall-clock
    # cost, matching the ~4x step-count increase). substeps=80-100 gives excellent accuracy
    # even at r_peri_min as low as 0.08 (0% of periapsis-centred windows over 10% error), but
    # was measured to cost ~25-35 hours for just 5000 steps at a realistic batch size on this
    # machine's MPS backend -- impractical. substeps=30 with the milder r_peri_min=0.1 below
    # gives a much more usable ~1.3% failure rate (still far better than the old box-sampler,
    # which avoided close approaches almost entirely rather than resolving them -- see the
    # module-level note on generate_data_analytic in two_body_dynamics.py). Benchmark a few
    # steps on your own machine before committing to a long run: this is the single biggest
    # lever on wall-clock cost, and worth tuning down further (or up, for more accuracy) based
    # on your own patience.
    integrator_substeps: int = 20
    validation_fraction: float = 0.25

    # Which data generator to use -- see two_body_dynamics.py for both. "analytic" (default,
    # recommended) uses the exact Kepler solution with periapsis/apoapsis deliberately
    # stratified; "box_sampler" is the original numerical-integration + random-box-rejection
    # approach, kept available (not deleted) so the two can be compared directly rather than
    # forcing a single choice.
    data_generation: str = "analytic"  # "analytic" or "box_sampler"

    # --- "analytic" generator fields (two_body_dynamics.generate_data_analytic) ---
    # Periapsis *and* apoapsis are both stratified log-uniformly (independently), rather than
    # periapsis alone with an independently-sampled eccentricity -- the latter lets apoapsis
    # blow up uncontrollably for wide-periapsis/high-eccentricity combinations (r_apo up to
    # ~15 was observed with r_peri_max=1.2, e_max=0.85), producing huge position/velocity
    # outliers that dominated the trajectory-fitting MSE loss and stalled training entirely.
    # Capping both directly keeps the whole dataset's dynamic range sane while still visiting
    # genuinely close approaches. r_peri_min small is safe as long as integrator_substeps
    # above is large enough to let the *training-time* rollout resolve it (verified together;
    # see the integrator_substeps note above for the accuracy/speed tradeoff this involves).
    r_peri_min: float = 0.1
    r_peri_max: float = 1.2
    r_apo_max: float = 2.0
    # Fraction of windows per orbit deliberately time-shifted so periapsis passage falls near
    # the window's middle (guaranteeing the network actually trains on that orbit's closest
    # approach); the rest use a uniformly random phase across the full orbital period.
    periapsis_centered_fraction: float = 0.1
    cm_box: float = 0.8

    # --- "box_sampler" generator fields (two_body_dynamics.generate_initial_conditions) ---
    # Sampled in relative coordinates via rejection sampling then placed at a random CM
    # offset; ground truth is numerically integrated with this many fine internal substeps
    # (dt/coarsening_factor) for accuracy. These defaults are the ones verified earlier in
    # this project to keep periapsis-in-window incidence low (~2%) at trajectory_window=5.
    box_ic_r_box: float = 1.7
    box_ic_v_box: float = 0.55
    box_ic_r_min: float = 0.01
    box_ic_energy_margin: float = 0.01
    box_ic_l_min: float = 0.7
    box_ic_cm_box: float = 0.8
    box_coarsening_factor: int = 100

    # Doubles the training set with a copy transformed by a random rotation
    # (applied to both particles together) *and* an independent random shift
    # (applied to both particles' positions together) -- both exact symmetries
    # of this system's true dynamics for every alpha > 0 (verified before use),
    # so this is free, exactly-valid additional data teaching both symmetries
    # implicitly through data diversity.
    augment_dataset: bool = True

    optimizer: str = "lbfgs"
    training_steps: int = 5000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    l1_weight: float = 1e-5
    # Clips the total gradient norm before every optimizer step. Needed here specifically:
    # verified in a real run (session that added this field) that Adam took one catastrophic
    # step mid-training (trajectory loss 0.00009 -> 0.011 between two consecutive steps) and
    # never recovered within the remaining budget -- occasional very large gradients are
    # expected for a system whose true force genuinely blows up near collision (1/r**3),
    # which nothing in a plain MLP's architecture prevents it from locally replicating
    # during training. None disables clipping (matches every other script's behaviour).
    max_grad_norm: float | None = 1.0
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [0, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000]
    )
    lbfgs_history_size: int = 10

    # Probe grid for the per-parameter diagnostics: meshgrid over the relative
    # vector (r_x, r_y) x a handful of centre-of-mass offsets (cm_x, cm_y),
    # converted to (q1, q2) via the equal-mass split. q_grid_points_per_axis
    # is deliberately even (e.g. 4 -> [-1, -0.33, 0.33, 1]) so the relative-
    # vector grid never lands exactly on r=(0,0), which would be a genuine
    # 1/r singularity in the true potential, not a numerical artefact.
    q_extent: float = 1.0
    q_grid_points_per_axis: int = 4
    cm_extent: float = 0.8
    cm_grid_points: int = 3
    # Deliberately not a multiple of 90 degrees, to make clear this is a
    # genuinely continuous symmetry, not a discrete point-group artefact.
    rotation_angle_degrees: float = 73.0
    # Direction (in degrees from the x-axis) of the single fixed translation
    # generator tested -- deliberately not axis-aligned, same philosophy as
    # rotation_angle_degrees, since translation invariance holds in every
    # direction and testing an arbitrary one is representative.
    translation_angle_degrees: float = 37.0
    translation_shift: float = 0.6

    analysis_alphas: list[float] = field(
        default_factory=lambda: [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    )
    # Smaller subset of alpha used for the force-field/potential figures below (one column
    # each) -- analysis_alphas is deliberately dense for the per-parameter diagnostics but
    # would make those figures too wide/slow.
    plotting_alphas: list[float] = field(default_factory=lambda: [1.0, 1.6, 2.5])

    tangent_svd_relative_cutoff: float = 1e-3
    top_parameters_to_report: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON file overriding fields in Config.")
    parser.add_argument("--quick", action="store_true", help="Run a very small end-to-end smoke test.")
    parser.add_argument("--device", help="Override Config.device.")
    parser.add_argument("--output-dir", help="Override Config.output_dir.")
    parser.add_argument("--architecture", choices=["hamiltonian", "direct_mlp"], help="Override Config.architecture.")
    parser.add_argument("--data-generation", choices=["analytic", "box_sampler"], help="Override Config.data_generation.")
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
    if args.data_generation:
        cfg.data_generation = args.data_generation
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
        cfg.training_steps = 2
        cfg.checkpoint_steps = [0, 1, 2]
        cfg.analysis_alphas = [0.6, 1.0, 1.6]
        cfg.q_grid_points_per_axis = 2
        cfg.cm_grid_points = 2
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
    if cfg.q_grid_points_per_axis < 2 or cfg.cm_grid_points < 2:
        raise ValueError("q_grid_points_per_axis and cm_grid_points must be at least 2.")
    if any(a <= 0 for a in cfg.training_alphas) or any(a <= 0 for a in cfg.analysis_alphas):
        raise ValueError("alpha must be strictly positive (attractive coupling) for bound closed orbits to exist.")
    if not 0.0 < cfg.tangent_svd_relative_cutoff < 1.0:
        raise ValueError("tangent_svd_relative_cutoff must lie strictly between 0 and 1.")
    if cfg.data_generation not in {"analytic", "box_sampler"}:
        raise ValueError("data_generation must be 'analytic' or 'box_sampler'.")
    if not 0.0 < cfg.r_peri_min < cfg.r_peri_max < cfg.r_apo_max:
        raise ValueError("must have 0 < r_peri_min < r_peri_max < r_apo_max.")
    if not 0.0 <= cfg.periapsis_centered_fraction <= 1.0:
        raise ValueError("periapsis_centered_fraction must lie in [0, 1].")
    cfg.checkpoint_steps = sorted(
        {int(s) for s in cfg.checkpoint_steps if 0 <= int(s) <= cfg.training_steps}
        | {0, cfg.training_steps}
    )


def make_dataset(cfg: Config, device: torch.device, dtype: torch.dtype):
    alphas = torch.tensor(cfg.training_alphas, device=device, dtype=dtype)
    if cfg.data_generation == "box_sampler":
        total_length = cfg.trajectory_splits + cfg.trajectory_window - 1
        ic_kwargs = dict(
            r_box=cfg.box_ic_r_box, v_box=cfg.box_ic_v_box, r_min=cfg.box_ic_r_min,
            energy_margin=cfg.box_ic_energy_margin, l_min=cfg.box_ic_l_min, cm_box=cfg.box_ic_cm_box,
        )
        trajectories, params, indices = generate_data(
            alphas, F, total_length, cfg.trajectory_window, cfg.sampled_instants,
            dt=cfg.integration_dt, in_conds=cfg.initial_conditions_per_alpha,
            coarsening_factor=cfg.box_coarsening_factor, device=device, dtype=dtype,
            augment_dataset=cfg.augment_dataset, ic_kwargs=ic_kwargs,
        )
    else:
        trajectories, params, indices = generate_data_analytic(
            alphas, cfg.trajectory_window, cfg.sampled_instants,
            dt=cfg.integration_dt, in_conds=cfg.initial_conditions_per_alpha, splits=cfg.trajectory_splits,
            r_peri_min=cfg.r_peri_min, r_peri_max=cfg.r_peri_max, r_apo_max=cfg.r_apo_max,
            periapsis_centered_fraction=cfg.periapsis_centered_fraction, cm_box=cfg.cm_box,
            augment_dataset=cfg.augment_dataset, device=device, dtype=dtype,
        )
    return train_test_split(trajectories, params, indices, val_size=cfg.validation_fraction)


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    if cfg.architecture == "direct_mlp":
        model = DirectDynamicsMLP(
            p_dim=4, q_dim=4, param_dim=1,
            hidden_dim=cfg.direct_mlp_hidden_dim, n_hidden=cfg.direct_mlp_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        integrator = SubsteppedDirectLeapfrogIntegrator(
            model=model, dt=cfg.integration_dt, substeps=cfg.integrator_substeps
        )
        return model, integrator
    model = GenericHamiltonianMLP(
        q_dim=4, p_dim=4, param_dim=1,
        kin_hidden_dim=cfg.kinetic_hidden_dim, kin_n_hidden=cfg.kinetic_hidden_layers,
        pot_hidden_dim=cfg.potential_hidden_dim, pot_n_hidden=cfg.potential_hidden_layers,
        device=device,
    ).to(device=device, dtype=dtype)
    integrator = SubsteppedVerletIntegrator(model=model, dt=cfg.integration_dt, substeps=cfg.integrator_substeps)
    return model, integrator


def make_optimizer(cfg: Config, model: torch.nn.Module):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.LBFGS(
        model.parameters(), lr=cfg.learning_rate, history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-12,
    )


def build_probe_grid(cfg: Config, device: torch.device, dtype: torch.dtype):
    """Meshgrid over the relative vector (r_x, r_y) x a few centre-of-mass offsets (cm_x, cm_y),
    converted to lab-frame (q1x, q1y, q2x, q2y) via the equal-mass split. Reused verbatim
    for the momentum probe grid too (arbitrary domain, no physical link to q needed there)."""
    r_axis = torch.linspace(-cfg.q_extent, cfg.q_extent, cfg.q_grid_points_per_axis, device=device, dtype=dtype)
    cm_axis = torch.linspace(-cfg.cm_extent, cfg.cm_extent, cfg.cm_grid_points, device=device, dtype=dtype)
    rx_mesh, ry_mesh, cmx_mesh, cmy_mesh = torch.meshgrid(r_axis, r_axis, cm_axis, cm_axis, indexing="ij")
    rx, ry, cmx, cmy = (t.reshape(-1) for t in (rx_mesh, ry_mesh, cmx_mesh, cmy_mesh))
    q1x, q1y = cmx + 0.5 * rx, cmy + 0.5 * ry
    q2x, q2y = cmx - 0.5 * rx, cmy - 0.5 * ry
    return q1x, q1y, q2x, q2y


def rotation_matrix_2d(theta_degrees: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = math.radians(theta_degrees)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)


def rotation_rep_matrix_4d(rot_mat_2d: torch.Tensor) -> torch.Tensor:
    """Block-diagonal [rot_mat_2d, rot_mat_2d]: the representation both q=(q1,q2) and F=(F1,F2)
    transform under, since both particles' 2-vectors rotate by the same finite angle together."""
    z = torch.zeros_like(rot_mat_2d)
    top = torch.cat((rot_mat_2d, z), dim=1)
    bottom = torch.cat((z, rot_mat_2d), dim=1)
    return torch.cat((top, bottom), dim=0)


def rotate_points(q1x, q1y, q2x, q2y, matrix: torch.Tensor):
    q1 = torch.stack([q1x, q1y], dim=1) @ matrix.T
    q2 = torch.stack([q2x, q2y], dim=1) @ matrix.T
    return q1[:, 0], q1[:, 1], q2[:, 0], q2[:, 1]


def translate_points(q1x, q1y, q2x, q2y, shift: float, direction_degrees: float):
    psi = math.radians(direction_degrees)
    dx, dy = shift * math.cos(psi), shift * math.sin(psi)
    return q1x + dx, q1y + dy, q2x + dx, q2y + dy


# d/dtheta of the block-diagonal rotation acting independently (but by the
# same angle) on (q1x,q1y) and (q2x,q2y).
ROTATION_LIE_GENERATOR = torch.tensor(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]]
)


def rotation_generator_target(q1x, q1y, q2x, q2y, force, spatial_jacobian) -> torch.Tensor:
    """X_rot F = (dF/dq)(Omega q) - Omega F(q), rotation acting on (q1,q2) jointly.

    Verified numerically against the exact analytic two-body force before use
    (relative residual ~1e-14 at every alpha).
    """
    omega_q = torch.stack([-q1y, q1x, -q2y, q2x], dim=1)
    directional = torch.einsum("nkj,nj->nk", spatial_jacobian, omega_q)
    omega = ROTATION_LIE_GENERATOR.to(device=force.device, dtype=force.dtype)
    return directional - force @ omega.T


def translation_generator_target(spatial_jacobian, direction: torch.Tensor) -> torch.Tensor:
    """X_trans F = -(dF/dq) . v, translating (q1,q2) together along a fixed direction v.

    Verified numerically against the exact analytic two-body force before use
    (relative residual ~1e-14 at every alpha).
    """
    return -torch.einsum("nkj,j->nk", spatial_jacobian, direction)


def evaluate_at_points(
    model: torch.nn.Module, q1x_pts, q1y_pts, q2x_pts, q2y_pts, alpha: float, architecture: str,
    *, device: torch.device, dtype: torch.dtype, need_spatial_jacobian: bool = False,
):
    """Evaluate learned potential, force, force-parameter Jacobian (and optionally dF/dq) at each point.

    Returns ``(V, F, J[, spatial_J])`` with shapes ``[N]``, ``[N, 4]``, ``[N, 4, P]`` (``[N, 4, 4]`` for spatial_J).
    """
    params = tuple(model.parameters())
    v_values, f_values, jac_rows, spatial_jacs = [], [], [], []
    for q1x_v, q1y_v, q2x_v, q2y_v in zip(
        q1x_pts.detach().cpu().tolist(), q1y_pts.detach().cpu().tolist(),
        q2x_pts.detach().cpu().tolist(), q2y_pts.detach().cpu().tolist(),
    ):
        q = torch.tensor([[q1x_v, q1y_v, q2x_v, q2y_v]], device=device, dtype=dtype, requires_grad=True)
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
        rows = [parameter_gradient_row(force[k], params) for k in range(4)]
        f_values.append(force.detach())
        jac_rows.append(torch.stack(rows))
        if need_spatial_jacobian:
            spatial_rows = [
                torch.autograd.grad(force[k], q, retain_graph=(k < 3), create_graph=False)[0].squeeze(0)
                for k in range(4)
            ]
            spatial_jacs.append(torch.stack(spatial_rows).detach())
    v = torch.tensor(v_values, device=device, dtype=dtype)
    f = torch.stack(f_values)
    j = torch.stack(jac_rows)
    if need_spatial_jacobian:
        return v, f, j, torch.stack(spatial_jacs)
    return v, f, j


def learned_dqdt(model: torch.nn.Module, p1x, p1y, p2x, p2y, alpha: float, architecture: str) -> torch.Tensor:
    p = torch.stack([p1x, p1y, p2x, p2y], dim=1)
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
    q1x_grid, q1y_grid, q2x_grid, q2y_grid = build_probe_grid(cfg, device, dtype)
    rot_mat = rotation_matrix_2d(cfg.rotation_angle_degrees, device, dtype)
    rot_rep = rotation_rep_matrix_4d(rot_mat)
    q1x_rot, q1y_rot, q2x_rot, q2y_rot = rotate_points(q1x_grid, q1y_grid, q2x_grid, q2y_grid, rot_mat)
    q1x_shift, q1y_shift, q2x_shift, q2y_shift = translate_points(
        q1x_grid, q1y_grid, q2x_grid, q2y_grid, cfg.translation_shift, cfg.translation_angle_degrees
    )
    p1x_grid, p1y_grid, p2x_grid, p2y_grid = build_probe_grid(cfg, device, dtype)
    n_points = q1x_grid.shape[0]

    psi = math.radians(cfg.translation_angle_degrees)
    translation_direction = torch.tensor(
        [math.cos(psi), math.sin(psi), math.cos(psi), math.sin(psi)], device=device, dtype=dtype
    )

    alpha_results = []
    for alpha in tqdm(cfg.analysis_alphas, desc=f"analysing step {step}", leave=False):
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1x_grid, q1y_grid, q2x_grid, q2y_grid, alpha, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        v_rot, f_rot, j_rot = evaluate_at_points(
            model, q1x_rot, q1y_rot, q2x_rot, q2y_rot, alpha, cfg.architecture, device=device, dtype=dtype
        )
        v_shift, f_shift, j_shift = evaluate_at_points(
            model, q1x_shift, q1y_shift, q2x_shift, q2y_shift, alpha, cfg.architecture, device=device, dtype=dtype
        )

        rotation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_rot.unsqueeze(-1))
        rotation_force_residual = finite_transform_residual(f_x, f_rot, rot_rep)
        rotation_sensitivity_residual = sensitivity_transform_residual(j_x, j_rot, rot_rep)
        rotation_equivariance_error = per_parameter_equivariance_error(j_x, j_rot, rot_rep)

        translation_potential_residual = finite_transform_residual(v_x.unsqueeze(-1), v_shift.unsqueeze(-1))
        translation_force_residual = finite_transform_residual(f_x, f_shift, None)
        translation_sensitivity_residual = sensitivity_transform_residual(j_x, j_shift, None)
        translation_equivariance_error = per_parameter_equivariance_error(j_x, j_shift, None)

        sensitivity = torch.sqrt(torch.mean(j_x.detach().cpu().double().square(), dim=(0, 1)))
        jac_flat = j_x.reshape(n_points * 4, -1)

        xrot_target = rotation_generator_target(q1x_grid, q1y_grid, q2x_grid, q2y_grid, f_x, spatial_jac_x)
        xrot_target_flat = xrot_target.reshape(n_points * 4)
        (
            xrot_projection_error, xrot_principal_angle,
            xrot_coefficients, xrot_resolved_rank, xrot_singular_values,
        ) = tangent_projection(jac_flat, xrot_target_flat, cfg.tangent_svd_relative_cutoff)

        xtrans_target = translation_generator_target(spatial_jac_x, translation_direction)
        xtrans_target_flat = xtrans_target.reshape(n_points * 4)
        (
            xtrans_projection_error, xtrans_principal_angle,
            xtrans_coefficients, xtrans_resolved_rank, xtrans_singular_values,
        ) = tangent_projection(jac_flat, xtrans_target_flat, cfg.tangent_svd_relative_cutoff)

        learned_dqdt_values = learned_dqdt(model, p1x_grid, p1y_grid, p2x_grid, p2y_grid, alpha, cfg.architecture)
        grad_e_p = torch.stack([p1x_grid, p1y_grid, p2x_grid, p2y_grid], dim=1)
        rx, ry = q1x_grid - q2x_grid, q1y_grid - q2y_grid
        r = torch.sqrt(rx**2 + ry**2)
        grad_q1x, grad_q1y = float(alpha) * rx / r**3, float(alpha) * ry / r**3
        grad_e_q = torch.stack([grad_q1x, grad_q1y, -grad_q1x, -grad_q1y], dim=1)
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

    axes[0, 0].set(ylabel="relative residual", yscale="log")
    axes[0, 0].set_title("finite rotation: force equivariance", fontsize=10)
    axes[0, 1].set(ylabel="relative residual", yscale="log")
    axes[0, 1].set_title("finite translation: force invariance", fontsize=10)
    axes[0, 2].set(ylabel="relative energy drift", yscale="log")
    axes[0, 2].set_title("Noether energy-conservation violation", fontsize=10)
    axes[1, 0].set(ylabel="relative projection error")
    axes[1, 0].set_title(r"$X_{\rm rot}F$ projection error", fontsize=10)
    axes[1, 1].set(ylabel="relative projection error")
    axes[1, 1].set_title(r"$X_{\rm trans}F$ projection error", fontsize=10)
    axes[1, 2].set(ylabel="relative Frobenius residual", yscale="log")
    axes[1, 2].set_title("finite-transform sensitivity equivariance", fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "two_body_summary.png", dpi=200)
    fig.savefig(output_dir / "two_body_summary.pdf")
    plt.close(fig)


def _alpha_averaged(checkpoint: dict[str, Any], key: str) -> np.ndarray:
    rows = checkpoint["alpha_results"]
    return np.mean([np.asarray(row[key]) for row in rows], axis=0)


def _exclude_structurally_zero(names: list[str]) -> list[str]:
    """Drop parameters whose rotation/translation S_i/E_i/c_i are exactly zero by construction.

    K_net never enters the force/sensitivity Jacobian (evaluate_at_points only
    differentiates V_net), so all of it is trivially zero. V_net's own output-layer
    bias is zero for the same underlying reason: force = -dV/dq, and
    V = W_out @ h(q) + b_out, so d(dV/dq)/d(b_out) = 0 identically -- an additive
    constant in V never survives differentiation (same fact established for the
    Mexican-hat script). Drop both from diagnostic plots to avoid a misleading
    point/cluster pinned to the log-log floor.
    """
    v_net_bias_layers = [
        int(n.split(".")[2]) for n in names if n.startswith("V_net.net.") and n.endswith(".bias")
    ]
    output_bias_name = f"V_net.net.{max(v_net_bias_layers)}.bias" if v_net_bias_layers else None
    return [n for n in names if not n.startswith("K_net.") and n != output_bias_name]


def _plotting_slices(parameter_slices: dict[str, slice]) -> dict[str, slice]:
    kept_names = set(_exclude_structurally_zero(list(parameter_slices.keys())))
    return {name: sl for name, sl in parameter_slices.items() if name in kept_names}


def plot_module_symmetry_comparison(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Rotation vs. translation attribution, side by side by module."""
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
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
        ax.bar(x + width / 2, translation_by_module, width, label=r"translation ($X_{\rm trans}F$)")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
        ax.set(ylabel="attribution norm within module")
        ax.set_title(subtitle, fontsize=10)
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
    implement both (entangled).
    """
    plotting_slices = _plotting_slices(parameter_slices)
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
        ax.set(xlabel=r"rotation attribution $|c_i|$", ylabel=r"translation attribution $|c_i|$")
        ax.set_title(subtitle, fontsize=10)
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
    modules = _exclude_structurally_zero(list(first_row["module_rotation_equivariance_error"].keys()))

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
        ax.set(ylabel="mean $E_i$ within module")
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend()
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "equivariance_by_module.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
    for key, stem, label in (
        ("rotation_equivariance_score", "equivariance_scatter_rotation", "Rotation"),
        ("translation_equivariance_score", "equivariance_scatter_translation", "Translation"),
    ):
        plot_ei_initial_vs_final(
            np.asarray(first[key]), np.asarray(last[key]), plotting_slices,
            title=f"{label} sensitivity-equivariance $E_i$: init vs. trained",
            output_stem=output_dir / stem,
        )


def plot_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter generator attribution |c_i|: random init vs. trained, one plot per generator.

    Same story as equivariance_scatter -- does training concentrate each
    generator's attribution onto fewer parameters, or leave the pattern
    essentially unchanged?
    """
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
    for key, stem, label in (
        ("xrot_score", "attribution_scatter_rotation", r"Rotation-generator attribution $|c_i|$"),
        ("xtrans_score", "attribution_scatter_translation", r"Translation-generator attribution $|c_i|$"),
    ):
        plot_ei_initial_vs_final(
            np.asarray(first[key]), np.asarray(last[key]), plotting_slices,
            title=f"{label}: init vs. trained",
            output_stem=output_dir / stem,
            quantity_label="$|c_i|$",
            log_scale=True,
        )


def plot_magnitude_diagnostics(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Check whether small S_i/E_i/c_i values at specific parameters are a genuine effect
    or simply an artefact of those parameters having small |theta_i| (e.g. from L1)."""
    first, last = all_results[0], all_results[-1]
    magnitude_initial = np.asarray(first["parameter_magnitude"])
    magnitude_final = np.asarray(last["parameter_magnitude"])
    plotting_slices = _plotting_slices(parameter_slices)

    plot_magnitude_vs_quantity(
        magnitude_initial, magnitude_final,
        _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
        plotting_slices,
        quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
        title=r"Parameter magnitude vs. sensitivity $S_i$ ($V_{\rm net}$ only)",
        output_stem=output_dir / "magnitude_vs_sensitivity",
    )
    for key, stem, label in (
        ("xrot_score", "magnitude_vs_attribution_rotation", "rotation-generator"),
        ("xtrans_score", "magnitude_vs_attribution_translation", "translation-generator"),
    ):
        plot_magnitude_vs_quantity(
            magnitude_initial, magnitude_final,
            np.asarray(first[key]), np.asarray(last[key]),
            plotting_slices,
            quantity_label=rf"$|c_i|$ ({label} attribution)",
            title=rf"Parameter magnitude vs. {label} attribution $|c_i|$ ($V_{{\rm net}}$ only)",
            output_stem=output_dir / stem,
        )
    for score_key, error_key, stem, label in (
        ("xrot_score", "rotation_equivariance_score", "attribution_vs_equivariance_rotation", "rotation"),
        ("xtrans_score", "translation_equivariance_score", "attribution_vs_equivariance_translation", "translation"),
    ):
        plot_magnitude_vs_quantity(
            np.asarray(first[score_key]), np.asarray(last[score_key]),
            np.asarray(first[error_key]), np.asarray(last[error_key]),
            plotting_slices,
            quantity_label=rf"$E_i$ ({label} sensitivity-equivariance error)",
            title=rf"{label.capitalize()}-generator attribution $|c_i|$ vs. equivariance error $E_i$",
            output_stem=output_dir / stem,
            x_label=rf"$|c_i|$ ({label}-generator attribution)",
        )
    for score_key, stem, label in (
        ("xrot_score", "attribution_vs_sensitivity_rotation", "rotation"),
        ("xtrans_score", "attribution_vs_sensitivity_translation", "translation"),
    ):
        plot_magnitude_vs_quantity(
            np.asarray(first[score_key]), np.asarray(last[score_key]),
            _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
            plotting_slices,
            quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
            title=rf"{label.capitalize()}-generator attribution $|c_i|$ vs. sensitivity $S_i$",
            output_stem=output_dir / stem,
            x_label=rf"$|c_i|$ ({label}-generator attribution)",
        )


def _evaluate_force_and_potential(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype,
    alpha: float, rx_flat: torch.Tensor, ry_flat: torch.Tensor, cm_x: float = 0.0, cm_y: float = 0.0,
):
    """Evaluate true/learned potential and force *on body 1* as a function of the relative
    separation r_vec = q1 - q2 = (rx, ry), at a fixed centre-of-mass (cm_x, cm_y).

    This is the natural batched (not per-parameter, no autograd-through-parameters) analogue
    of the mexican-hat script's ``_evaluate_force_and_potential`` -- reduced to r_vec since
    that's what the true potential actually depends on, rather than the raw 4D q.
    """
    q1x, q1y = cm_x + 0.5 * rx_flat, cm_y + 0.5 * ry_flat
    q2x, q2y = cm_x - 0.5 * rx_flat, cm_y - 0.5 * ry_flat
    q = torch.stack([q1x, q1y, q2x, q2y], dim=1)
    alpha_col = torch.full((q.shape[0], 1), float(alpha), device=device, dtype=dtype)
    r = torch.sqrt(rx_flat**2 + ry_flat**2)
    true_v = -alpha / r
    true_f1 = torch.stack([-alpha * rx_flat / r**3, -alpha * ry_flat / r**3], dim=1)
    if cfg.architecture == "direct_mlp":
        p_zero = torch.zeros_like(q)
        with torch.no_grad():
            dpdt, _ = model(p_zero, q, alpha_col)
        learned_f1 = dpdt[:, 0:2]
        learned_v = None
    else:
        q_grad = q.clone().requires_grad_(True)
        with torch.enable_grad():
            potential = model.V_net(torch.cat((q_grad, alpha_col), dim=1))
            learned_force = -torch.autograd.grad(potential.sum(), q_grad)[0].detach()
        learned_f1 = learned_force[:, 0:2]
        learned_v = potential.detach().squeeze(-1)
    return true_v, true_f1, learned_v, learned_f1


def _make_relative_grid(cfg: Config, n: int, device: torch.device, dtype: torch.dtype):
    """Regular grid over the relative vector r_vec=(r_x,r_y), at a fixed CM=(0,0), excluding
    points too close to r=0 (a genuine 1/r singularity in the true potential, not an artefact)."""
    axis = torch.linspace(-cfg.q_extent, cfg.q_extent, n, device=device, dtype=dtype)
    rx_mesh, ry_mesh = torch.meshgrid(axis, axis, indexing="ij")
    rx, ry = rx_mesh.reshape(-1), ry_mesh.reshape(-1)
    keep = torch.sqrt(rx**2 + ry**2) > 0.5 * (2 * cfg.q_extent / max(n - 1, 1))
    return rx[keep], ry[keep]


def plot_learned_force_field(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype, output_dir: Path
) -> None:
    """Quiver comparison of the learned vs. true force *on body 1*, as a function of the
    relative separation r_vec = q1 - q2, at a fixed centre of mass (CM = 0) -- the natural
    'reduce it radially' visualisation for a system whose true potential/force only depend
    on r_vec, not the raw 4D q. One column per plotting alpha, true on top / learned below,
    same shared-scale convention as the mexican-hat script's version of this plot.
    """
    model.eval()
    alphas = cfg.plotting_alphas
    rx_coarse, ry_coarse = _make_relative_grid(cfg, 9, device, dtype)

    fig, axes = plt.subplots(
        2, len(alphas), figsize=(3.4 * len(alphas), 7.2), constrained_layout=True, squeeze=False,
    )
    for col, alpha in enumerate(alphas):
        _, true_f1_c, _, learned_f1_c = _evaluate_force_and_potential(
            model, cfg, device, dtype, alpha, rx_coarse, ry_coarse
        )
        true_mag = torch.linalg.vector_norm(true_f1_c, dim=-1).max().clamp_min(1e-12).item()
        scale = true_mag * 12
        for row, force, title in ((0, true_f1_c, "true"), (1, learned_f1_c, "learned")):
            ax = axes[row, col]
            ax.quiver(
                rx_coarse.cpu().numpy(), ry_coarse.cpu().numpy(),
                force[:, 0].cpu().numpy(), force[:, 1].cpu().numpy(),
                scale=scale, scale_units="width", angles="xy", color="tab:blue", width=0.008,
            )
            ax.set(xlabel="$r_x=q_{1x}-q_{2x}$", ylabel="$r_y$" if col == 0 else "")
            if row == 0:
                ax.set_title(rf"$\alpha={alpha:g}$")
            if col == 0:
                ax.annotate(
                    title, xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90,
                    ha="center", va="center", fontsize=11,
                )
            ax.set_aspect("equal")

    fig.savefig(output_dir / "learned_force_field_final.png", dpi=200)
    fig.savefig(output_dir / "learned_force_field_final.pdf")
    plt.close(fig)


def plot_learned_potential_radial(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype, output_dir: Path
) -> None:
    """Radial comparison of the learned vs. true potential: since the true V depends only
    on r=|q1-q2|, V(r) is a complete, exact 1D reduction -- unlike the mexican-hat/
    cylindrical scripts' 2D heatmaps, one curve per alpha suffices (only for architectures
    with a V_net -- direct_mlp has no potential to plot).

    V_theta is only meaningful up to an arbitrary additive constant (same gauge-freedom
    note as the mexican-hat script): both curves are shifted so V(r_max) = 0, the
    least-bound point in the sampled range. Several different (CM offset, orientation)
    contexts are overlaid for the *learned* curve only -- the true potential is exactly
    r-only by construction, so one reference curve suffices -- if the network has learned
    translation+rotation invariance, all the learned context curves should collapse onto
    one another and onto the true curve; visible spread between them is direct evidence it
    hasn't (yet).
    """
    if cfg.architecture == "direct_mlp":
        return
    model.eval()
    alphas = cfg.plotting_alphas
    r_values = torch.linspace(max(cfg.r_peri_min * 0.5, 0.02), cfg.q_extent * 2.0, 60, device=device, dtype=dtype)
    contexts = [(0.0, 0.0, 0.0), (cfg.cm_extent, 0.0, 30.0), (-cfg.cm_extent, cfg.cm_extent, 110.0)]

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.2), constrained_layout=True, squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        for context_idx, (cm_x, cm_y, angle_deg) in enumerate(contexts):
            psi = math.radians(angle_deg)
            rx = r_values * math.cos(psi)
            ry = r_values * math.sin(psi)
            true_v, _, learned_v, _ = _evaluate_force_and_potential(
                model, cfg, device, dtype, alpha, rx, ry, cm_x, cm_y
            )
            if context_idx == 0:
                true_shifted = (true_v - true_v[-1]).cpu().numpy()
                ax.plot(r_values.cpu().numpy(), true_shifted, "k--", linewidth=1.8, label="true", zorder=5)
            learned_shifted = (learned_v - learned_v[-1]).cpu().numpy()
            ax.plot(
                r_values.cpu().numpy(), learned_shifted, alpha=0.85, linewidth=1.3,
                label=rf"learned, cm=({cm_x:.1f},{cm_y:.1f}), $\theta$={angle_deg:.0f}$^\circ$",
            )
        ax.set(xlabel=r"$r=|q_1-q_2|$", ylabel=r"$V(r) - V(r_{\max})$")
        ax.set_title(rf"$\alpha={alpha:g}$", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output_dir / "learned_potential_radial_final.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "learned_potential_radial_final.pdf", bbox_inches="tight")
    plt.close(fig)


def _sample_windows(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], n_sample: int):
    trajectories, _, indices = data
    batch = trajectories.shape[1]
    sample = torch.randperm(batch)[: min(n_sample, batch)]
    return trajectories[:, sample, :], indices[sample, :]


def plot_training_data_trajectories(
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path, n_sample: int = 40,
) -> None:
    """Pure data diagnostic, independent of any model: plot the actual sampled trajectory
    windows used for training/validation as spatial paths of the relative separation
    r_vec = q1 - q2 (centred at the origin -- the random per-window centre-of-mass placement
    would otherwise just scatter absolute (q1, q2) positions around the plot without adding
    anything physically informative here, since translation invariance means only each
    orbit arc's *shape*, not where its centre of mass landed, matters for this diagnostic).

    This is exactly the kind of plot that would have caught the r_apo blow-up bug (session
    that added this function) immediately: sampling eccentricity independently of periapsis
    let some orbits reach apoapsis ~15, wildly outside the intended domain -- a plot like
    this makes a run-away orbit visually obvious rather than requiring a manual magnitude
    check after the fact.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, data, title in ((axes[0], train_data, "training windows"), (axes[1], val_data, "validation windows")):
        trajectories, _ = _sample_windows(data, n_sample)
        r_vec = (trajectories[:, :, 4:6] - trajectories[:, :, 6:8]).cpu().numpy()  # [N, n_sample, 2]
        for b in range(r_vec.shape[1]):
            ax.plot(r_vec[:, b, 0], r_vec[:, b, 1], marker="o", ms=3, linewidth=1, alpha=0.6)
        ax.scatter([0], [0], marker="x", color="k", s=50, zorder=5, label="other body")
        ax.set(xlabel="$r_x = q_{1x}-q_{2x}$", ylabel="$r_y$", title=title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "training_data_trajectories.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "training_data_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_data_radial(
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path, n_sample: int = 40,
) -> None:
    """Pure data diagnostic: r=|q1-q2| against the outer-step index within each sampled
    window, for a random subset of training/validation windows -- shows how close each
    window actually gets to periapsis, how well the sampled instants resolve the passage
    (few points near the minimum means a coarse view of it), and, same as the trajectory
    plot above, immediately flags any run-away magnitude in the generated data (a genuine
    singularity approach shows up as a curve plunging toward the log-scale floor; a data bug
    inflating orbit size shows up as curves reaching unexpectedly large r).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, data, title in ((axes[0], train_data, "training windows"), (axes[1], val_data, "validation windows")):
        trajectories, indices = _sample_windows(data, n_sample)
        r = (trajectories[:, :, 4:6] - trajectories[:, :, 6:8]).norm(dim=-1).cpu().numpy()  # [N, n_sample]
        step_idx = indices.cpu().numpy().T.astype(float)  # [N, n_sample]
        for b in range(r.shape[1]):
            ax.plot(step_idx[:, b], r[:, b], marker="o", ms=3, linewidth=1, alpha=0.6)
        ax.set(xlabel="outer step index within window", ylabel="$r=|q_1-q_2|$", title=title)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    fig.savefig(output_dir / "training_data_radial.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "training_data_radial.pdf", bbox_inches="tight")
    plt.close(fig)


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
    print("Generating sparse two-body trajectories...")
    train_data, validation_data = make_dataset(cfg, device, dtype)
    plot_training_data_trajectories(train_data, validation_data, output_dir)
    plot_training_data_radial(train_data, validation_data, output_dir)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}")

    residuals_fn = lambda *args, **kwargs: generic_residuals(*args, **kwargs, p_dim=4)  # noqa: E731
    history, checkpoint_states = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=validation_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay, max_grad_norm=cfg.max_grad_norm,
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
    plot_attribution_scatter(all_results, parameter_slices, output_dir)
    plot_magnitude_diagnostics(all_results, parameter_slices, output_dir)
    plot_module_symmetry_comparison(all_results, parameter_slices, output_dir)
    plot_symmetry_attribution_scatter(all_results, parameter_slices, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))

    model.load_state_dict(checkpoint_states[max(checkpoint_states.keys())])
    plot_learned_force_field(model, cfg, device, dtype, output_dir)
    plot_learned_potential_radial(model, cfg, device, dtype, output_dir)
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
