"""POOLED-DIRECTION variant of asrnn_two_body_symmetry_sensitivity.py, for direct comparison.

The only difference from the un-suffixed sibling script: for each symmetry (rotation,
translation), c_i (generator attribution) and b_i (equivariance-defect attribution) are each
solved ONCE per checkpoint from a single combined least-squares system that stacks
Jacobian/target rows across every alpha in cfg.training_alphas union cfg.analysis_alphas,
instead of being solved independently at each analysis alpha and then RMS-aggregated into a
magnitude-only "score" afterward. See analyse_checkpoint's "Pooled c_i/b_i" block for the full
rationale (in short: a per-alpha c_i, if ever used as an actual parameter-space step, gives a
different perturbed model at each alpha, which isn't one coherent direction; pooling first
avoids that -- same reasoning as the mexican-hat pooled script). Run with the SAME config.json
as an existing un-suffixed run (same seed => same training trajectory, since training itself
never touches c_i/b_i) to get a directly comparable set of plots differing only in this one
definitional choice.

Functional-sensitivity experiment for the planar two-body problem, kept in the full 8D lab frame.

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
    plot_both_signed_quantity,
    plot_ei_initial_vs_final,
    plot_magnitude_vs_quantity,
    plot_signed_initial_vs_final,
    plot_training_history,
    prettify_parameter_name,
    robust_linthresh,
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
    parameter_magnitude_ci_correlation,
    participation_ratio,
    per_parameter_equivariance_error,
    relative_energy_drift,
    sensitivity_transform_residual,
    tangent_projection,
)
from two_body_dynamics import (
    F,
    InvariantTwoBodyHamiltonianMLP,
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
    output_dir: str = "outputs/asrnn_two_body_symmetry_pooled"

    architecture: str = "equivariant"  # hamiltonian, direct_mlp, or equivariant

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
    trajectory_window: int = 12
    trajectory_splits: int = 14
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
    r_apo_max: float = 3.0
    # Fraction of windows per orbit deliberately time-shifted so periapsis passage falls near
    # the window's middle (guaranteeing the network actually trains on that orbit's closest
    # approach); the rest use a uniformly random phase across the full orbital period.
    periapsis_centered_fraction: float = 0.2
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
    training_steps: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    l1_weight: float = 0.0
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
    # Number of angles used to numerically approximate the SO(2) orbit-average
    # Pi_rot F = (1/2pi) int R_phi^{-1} F(R_phi q) dphi (the g-vs-delta decomposition, mirroring
    # the mexican-hat script's compute_orbit_average). translation's orbit average has no
    # analogous "number of samples" knob -- see compute_translation_orbit_average's docstring
    # for why (translation is non-compact, so it uses the probe grid's own finite CM extent
    # instead of a tunable discretisation of a convergent integral).
    orbit_average_n_phi: int = 64

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
    parser.add_argument("--architecture", choices=["hamiltonian", "direct_mlp", "equivariant"], help="Override Config.architecture.")
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
    if cfg.architecture == "equivariant":
        model = InvariantTwoBodyHamiltonianMLP(
            kin_hidden_dim=cfg.kinetic_hidden_dim, kin_n_hidden=cfg.kinetic_hidden_layers,
            pot_hidden_dim=cfg.potential_hidden_dim, pot_n_hidden=cfg.potential_hidden_layers,
            device=device,
        ).to(device=device, dtype=dtype)
        integrator = SubsteppedVerletIntegrator(model=model, dt=cfg.integration_dt, substeps=cfg.integrator_substeps)
        return model, integrator
    if cfg.architecture != "hamiltonian":
        raise ValueError(f"Unknown architecture: {cfg.architecture!r} (must be hamiltonian, direct_mlp, or equivariant).")
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


def _evaluate_learned_force_batched(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype,
    alpha: float, q1x, q1y, q2x, q2y,
) -> torch.Tensor:
    """Batched (single forward+backward call, no per-point python loop, no per-parameter
    autograd) learned-force evaluator at arbitrary (q1,q2) points, shape [N, 4]. The
    forward-only building block compute_rotation_orbit_average needs repeatedly --
    evaluate_at_points is far more expensive for this since it also computes the full
    per-parameter Jacobian (via a per-point python loop) that orbit-averaging has no use for."""
    q = torch.stack([q1x, q1y, q2x, q2y], dim=1)
    alpha_col = torch.full((q.shape[0], 1), float(alpha), device=device, dtype=dtype)
    if cfg.architecture == "direct_mlp":
        p_zero = torch.zeros_like(q)
        with torch.no_grad():
            dpdt, _ = model(p_zero, q, alpha_col)
        return dpdt
    q_grad = q.clone().requires_grad_(True)
    with torch.enable_grad():
        potential = model.V_net(torch.cat((q_grad, alpha_col), dim=1))
        force = -torch.autograd.grad(potential.sum(), q_grad)[0].detach()
    return force


def compute_rotation_orbit_average(
    model: torch.nn.Module, cfg: Config, device: torch.device, dtype: torch.dtype,
    alpha: float, q1x, q1y, q2x, q2y,
) -> torch.Tensor:
    r"""SO(2) orbit average Pi_rot F(q) = (1/2pi) int R_phi^{-1} F(R_phi q) dphi, both bodies
    rotated together by the same angle -- the two-body analogue of the mexican-hat script's
    compute_orbit_average, generalised to the 4D joint-rotation representation
    (rotation_rep_matrix_4d). Since Pi_rot projects onto ker(X_rot) *by construction*
    (averaging over the whole compact group annihilates any phi-dependence), Pi_rot F is
    exactly rotationally invariant regardless of how equivariant F currently is; delta_rot =
    F - Pi_rot F (computed at the call site) is then the genuine "distance from the
    rotation-invariant subspace" that g_rot = X_rot F is provably orthogonal to (identical
    argument to the mexican-hat script, since X_rot is skew-adjoint under the same L2 inner
    product Pi_rot is built from).
    """
    n_phi = cfg.orbit_average_n_phi
    total = torch.zeros(q1x.shape[0], 4, device=device, dtype=dtype)
    for k in range(n_phi):
        phi_degrees = 360.0 * k / n_phi
        rot_mat = rotation_matrix_2d(phi_degrees, device, dtype)
        rep = rotation_rep_matrix_4d(rot_mat)
        q1x_rot, q1y_rot, q2x_rot, q2y_rot = rotate_points(q1x, q1y, q2x, q2y, rot_mat)
        force_rot = _evaluate_learned_force_batched(model, cfg, device, dtype, alpha, q1x_rot, q1y_rot, q2x_rot, q2y_rot)
        # Row-vector convention (established by rotate_points/transform_points throughout this
        # project): `v @ M.T` implements `M @ v`, so `v @ M` implements `M.T @ v = M^{-1} @ v`
        # for orthogonal M -- i.e. plain `@ rep` (not `@ rep.T`) is what applies rep^{-1} here.
        # Verified: `@ rep.T` (the wrong sign) gives Pi_rot F NOT equivariant, relative error
        # ~1.3; `@ rep` gives relative error ~1e-15.
        total = total + force_rot @ rep
    return total / n_phi


def compute_translation_orbit_average(f_x: torch.Tensor, cfg: Config) -> torch.Tensor:
    r"""Discrete translation orbit-average, restricted to the probe grid's own finite CM
    sub-grid: Pi_trans F(r) = average of F over the cm_x, cm_y grid points sharing that r.

    Unlike rotation (a compact group -- SO(2) has a finite, normalisable Haar measure, so
    "average over the whole group" is well-defined), translation is non-compact: there is no
    analogous "average over every possible shift" that converges. Pi_trans instead
    marginalises out exactly the CM-direction dependence a translation-invariant force must
    lack, using the finite CM extent/resolution build_probe_grid already samples (cm_extent,
    cm_grid_points) -- the direct discrete analogue of compute_rotation_orbit_average's finite
    n_phi discretisation, except here the restriction to a finite domain is unavoidable (not
    just a tunable approximation to an otherwise-exact infinite average). No extra model
    evaluations needed: f_x is already evaluated at every point of the same probe grid used
    for c_i, so this only reshapes/averages values already in hand.

    Verify before trusting (matches this project's standing convention): (a) Pi_trans F is
    exactly constant in cm at fixed r by construction (trivial, since it's literally a cm-axis
    mean, no floating-point check needed); (b) applied to the true analytic two-body force
    (exactly r-only, i.e. already exactly translation-invariant), Pi_trans must recover it
    exactly with zero discretisation error (unlike rotation's Pi, which only recovers the true
    equivariant force in the n_phi -> infinity limit) -- checked in the smoke test before use.
    """
    n_r = cfg.q_grid_points_per_axis
    n_cm = cfg.cm_grid_points
    d = f_x.shape[-1]
    grid = f_x.reshape(n_r, n_r, n_cm, n_cm, d)
    pi_f = grid.mean(dim=(2, 3), keepdim=True).expand(n_r, n_r, n_cm, n_cm, d)
    return pi_f.reshape(f_x.shape[0], d)


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
    parameter_value = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().tolist()
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
        # Signed companion to the RMS-magnitude "sensitivity" above: mean (not RMS) of the raw
        # signed per-point, per-component Jacobian entries (mirrors the mexican-hat script).
        signed_sensitivity = torch.mean(j_x.detach().cpu().double(), dim=(0, 1))
        jac_flat = j_x.reshape(n_points * 4, -1)

        # xrot_*/xtrans_*/delta_rot_*/delta_trans_* (c_i/b_i for each symmetry) are NOT solved
        # per-alpha here -- this script pools rows across alpha into one combined solve instead
        # (see the "Pooled c_i/b_i" block after this loop) rather than solving independently at
        # each alpha and RMS-aggregating afterward (that original definition lives in the
        # un-suffixed sibling script, asrnn_two_body_symmetry_sensitivity.py).

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
                "signed_sensitivity": signed_sensitivity.cpu().tolist(),
                "rotation_equivariance_error_by_parameter": rotation_equivariance_error.cpu().tolist(),
                "translation_equivariance_error_by_parameter": translation_equivariance_error.cpu().tolist(),
                "module_sensitivity": aggregate_by_module(sensitivity, parameter_slices),
                "module_rotation_equivariance_error": aggregate_by_module_mean(
                    rotation_equivariance_error, parameter_slices
                ),
                "module_translation_equivariance_error": aggregate_by_module_mean(
                    translation_equivariance_error, parameter_slices
                ),
                # xrot_*/xtrans_*/delta_rot_*/delta_trans_* fields are filled in uniformly
                # below, after the pooled solve -- see the "Pooled c_i/b_i" block.
            }
        )

    # Pooled c_i/b_i, one combined least-squares solve per (symmetry, g-vs-delta) pair: stacks
    # Jacobian/target rows across every alpha in cfg.training_alphas union cfg.analysis_alphas
    # (alpha is just a network input feature here, not something requiring "data" at that
    # value, so both sets are fair game -- see the mexican-hat pooled script's identical
    # construction and full rationale). This is the direct generalisation of how
    # tangent_projection already pools rows across every probe point q within one alpha to also
    # pool across alpha itself, giving one c/b vector per symmetry valid across every probed
    # regime at once, rather than a separate vector per alpha (RMS-aggregated only for
    # magnitude-ranking purposes in the un-suffixed sibling script). The identical pooled
    # vector is written into every alpha_results row below, so every downstream consumer (RMS
    # "score" aggregation, CSV export, signed plots) picks it up unchanged -- RMS of an
    # alpha-constant vector is just its own magnitude.
    direction_alphas = sorted(set(round(a, 6) for a in cfg.training_alphas) | set(round(a, 6) for a in cfg.analysis_alphas))
    jac_blocks, xrot_target_blocks, xtrans_target_blocks = [], [], []
    delta_rot_target_blocks, delta_trans_target_blocks = [], []
    for alpha in direction_alphas:
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1x_grid, q1y_grid, q2x_grid, q2y_grid, alpha, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        jac_blocks.append(j_x.reshape(n_points * 4, -1))
        xrot_target_blocks.append(rotation_generator_target(q1x_grid, q1y_grid, q2x_grid, q2y_grid, f_x, spatial_jac_x).reshape(-1))
        xtrans_target_blocks.append(translation_generator_target(spatial_jac_x, translation_direction).reshape(-1))
        pi_rot_f_x = compute_rotation_orbit_average(model, cfg, device, dtype, alpha, q1x_grid, q1y_grid, q2x_grid, q2y_grid)
        delta_rot_target_blocks.append((f_x - pi_rot_f_x).reshape(-1))
        pi_trans_f_x = compute_translation_orbit_average(f_x, cfg)
        delta_trans_target_blocks.append((f_x - pi_trans_f_x).reshape(-1))

    jac_pooled = torch.cat(jac_blocks, dim=0)
    (
        xrot_projection_error, xrot_principal_angle,
        xrot_coefficients, xrot_resolved_rank, xrot_singular_values,
    ) = tangent_projection(jac_pooled, torch.cat(xrot_target_blocks, dim=0), cfg.tangent_svd_relative_cutoff)
    (
        xtrans_projection_error, xtrans_principal_angle,
        xtrans_coefficients, xtrans_resolved_rank, xtrans_singular_values,
    ) = tangent_projection(jac_pooled, torch.cat(xtrans_target_blocks, dim=0), cfg.tangent_svd_relative_cutoff)
    (
        delta_rot_projection_error, delta_rot_principal_angle,
        delta_rot_coefficients, delta_rot_resolved_rank, delta_rot_singular_values,
    ) = tangent_projection(jac_pooled, torch.cat(delta_rot_target_blocks, dim=0), cfg.tangent_svd_relative_cutoff)
    (
        delta_trans_projection_error, delta_trans_principal_angle,
        delta_trans_coefficients, delta_trans_resolved_rank, delta_trans_singular_values,
    ) = tangent_projection(jac_pooled, torch.cat(delta_trans_target_blocks, dim=0), cfg.tangent_svd_relative_cutoff)

    for row in alpha_results:
        row["xrot_projection_error"] = xrot_projection_error
        row["xrot_principal_angle_degrees"] = xrot_principal_angle
        row["xrot_resolved_rank"] = xrot_resolved_rank
        row["xrot_singular_values"] = xrot_singular_values
        row["xrot_attribution_coefficients"] = xrot_coefficients.cpu().tolist()
        row["module_xrot_attribution"] = aggregate_by_module(xrot_coefficients.abs(), parameter_slices)
        row["xtrans_projection_error"] = xtrans_projection_error
        row["xtrans_principal_angle_degrees"] = xtrans_principal_angle
        row["xtrans_resolved_rank"] = xtrans_resolved_rank
        row["xtrans_singular_values"] = xtrans_singular_values
        row["xtrans_attribution_coefficients"] = xtrans_coefficients.cpu().tolist()
        row["module_xtrans_attribution"] = aggregate_by_module(xtrans_coefficients.abs(), parameter_slices)
        row["delta_rot_projection_error"] = delta_rot_projection_error
        row["delta_rot_principal_angle_degrees"] = delta_rot_principal_angle
        row["delta_rot_resolved_rank"] = delta_rot_resolved_rank
        row["delta_rot_singular_values"] = delta_rot_singular_values
        row["delta_rot_attribution_coefficients"] = delta_rot_coefficients.cpu().tolist()
        row["module_delta_rot_attribution"] = aggregate_by_module(delta_rot_coefficients.abs(), parameter_slices)
        row["delta_trans_projection_error"] = delta_trans_projection_error
        row["delta_trans_principal_angle_degrees"] = delta_trans_principal_angle
        row["delta_trans_resolved_rank"] = delta_trans_resolved_rank
        row["delta_trans_singular_values"] = delta_trans_singular_values
        row["delta_trans_attribution_coefficients"] = delta_trans_coefficients.cpu().tolist()
        row["module_delta_trans_attribution"] = aggregate_by_module(delta_trans_coefficients.abs(), parameter_slices)

    xrot_coefficient_matrix = np.asarray([r["xrot_attribution_coefficients"] for r in alpha_results])
    xrot_score = np.sqrt(np.mean(xrot_coefficient_matrix**2, axis=0))
    top_xrot = np.argsort(xrot_score)[::-1][: cfg.top_parameters_to_report]

    xtrans_coefficient_matrix = np.asarray([r["xtrans_attribution_coefficients"] for r in alpha_results])
    xtrans_score = np.sqrt(np.mean(xtrans_coefficient_matrix**2, axis=0))
    top_xtrans = np.argsort(xtrans_score)[::-1][: cfg.top_parameters_to_report]

    # b_rot_i/b_trans_i: RMS-over-alpha magnitude of each delta attribution (c_i's analogue
    # for g_rot/g_trans -- see compute_rotation_orbit_average/compute_translation_orbit_average
    # docstrings for why these are genuinely distinct questions, not the same one recomputed).
    delta_rot_coefficient_matrix = np.asarray([r["delta_rot_attribution_coefficients"] for r in alpha_results])
    b_rot_score = np.sqrt(np.mean(delta_rot_coefficient_matrix**2, axis=0))
    top_b_rot = np.argsort(b_rot_score)[::-1][: cfg.top_parameters_to_report]

    delta_trans_coefficient_matrix = np.asarray([r["delta_trans_attribution_coefficients"] for r in alpha_results])
    b_trans_score = np.sqrt(np.mean(delta_trans_coefficient_matrix**2, axis=0))
    top_b_trans = np.argsort(b_trans_score)[::-1][: cfg.top_parameters_to_report]

    rotation_equivariance_matrix = np.asarray(
        [r["rotation_equivariance_error_by_parameter"] for r in alpha_results]
    )
    rotation_equivariance_score = np.sqrt(np.mean(rotation_equivariance_matrix**2, axis=0))

    translation_equivariance_matrix = np.asarray(
        [r["translation_equivariance_error_by_parameter"] for r in alpha_results]
    )
    translation_equivariance_score = np.sqrt(np.mean(translation_equivariance_matrix**2, axis=0))

    vnet_mask = np.array([name.startswith("V_net.") for name in flat_names])
    corr_mask = vnet_mask if vnet_mask.any() else np.ones_like(vnet_mask, dtype=bool)

    # Pearson correlations between attributions themselves (not either one against
    # |theta_i|) -- the central "do these share support" questions, as reported numbers
    # rather than something only ever eyeballed off a scatter plot. V_net-only: K_net is
    # architecturally zero for every one of these four quantities (the rotation/translation
    # diagnostics only ever differentiate V_net), so including it would just add a cluster of
    # trivial (0,0) pairs, not real correlation signal.
    xrot_brot_correlation = parameter_magnitude_ci_correlation(xrot_score, b_rot_score, mask=corr_mask)
    xtrans_btrans_correlation = parameter_magnitude_ci_correlation(xtrans_score, b_trans_score, mask=corr_mask)
    xrot_xtrans_correlation = parameter_magnitude_ci_correlation(xrot_score, xtrans_score, mask=corr_mask)
    brot_btrans_correlation = parameter_magnitude_ci_correlation(b_rot_score, b_trans_score, mask=corr_mask)

    return {
        "step": step,
        "parameter_magnitude": parameter_magnitude,
        "parameter_value": parameter_value,
        "alpha_results": alpha_results,
        "xrot_score": xrot_score.tolist(),
        "xtrans_score": xtrans_score.tolist(),
        "b_rot_score": b_rot_score.tolist(),
        "b_trans_score": b_trans_score.tolist(),
        "rotation_equivariance_score": rotation_equivariance_score.tolist(),
        "translation_equivariance_score": translation_equivariance_score.tolist(),
        "xrot_participation_ratio": participation_ratio(xrot_score[vnet_mask]) if vnet_mask.any() else participation_ratio(xrot_score),
        "xtrans_participation_ratio": participation_ratio(xtrans_score[vnet_mask]) if vnet_mask.any() else participation_ratio(xtrans_score),
        "b_rot_participation_ratio": participation_ratio(b_rot_score[vnet_mask]) if vnet_mask.any() else participation_ratio(b_rot_score),
        "b_trans_participation_ratio": participation_ratio(b_trans_score[vnet_mask]) if vnet_mask.any() else participation_ratio(b_trans_score),
        "xrot_brot_correlation": xrot_brot_correlation,
        "xtrans_btrans_correlation": xtrans_btrans_correlation,
        "xrot_xtrans_correlation": xrot_xtrans_correlation,
        "brot_btrans_correlation": brot_btrans_correlation,
        "top_xrot_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xrot_score[i])}
            for i in top_xrot
        ],
        "top_xtrans_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(xtrans_score[i])}
            for i in top_xtrans
        ],
        "top_b_rot_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(b_rot_score[i])}
            for i in top_b_rot
        ],
        "top_b_trans_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(b_trans_score[i])}
            for i in top_b_trans
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
    write_csv_rows(
        result["top_b_rot_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_b_rot_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_b_trans_parameters"], ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_b_trans_parameters_step_{step:06d}.csv",
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


def _signed_like_score(checkpoint: dict[str, Any], raw_key: str, unsigned_score: np.ndarray) -> np.ndarray:
    """Signed companion to an alpha-aggregated unsigned score (RMS or mean-of-RMS), with
    matching magnitude -- |signed_i| == unsigned_score_i exactly, always. Sign taken from the
    mean (across all analysis_alphas) of the raw per-alpha signed values, not an arbitrary
    single alpha's sign (see the mexican-hat script's identical helper for the full rationale:
    pairing an RMS-aggregated unsigned magnitude with a single alpha's raw sign is an
    inconsistency, since the two numbers come from different underlying statistics)."""
    raw_matrix = np.asarray([r[raw_key] for r in checkpoint["alpha_results"]])
    sign = np.sign(raw_matrix.mean(axis=0))
    sign[sign == 0] = 1.0  # deterministic tie-break for exact zeros (e.g. structurally-zero params)
    return sign * np.asarray(unsigned_score)


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


def plot_module_b_attribution(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """b_i analogue of plot_module_symmetry_comparison: rotation-defect vs. translation-defect
    attribution, side by side by module (b_rot_i/b_trans_i = the delta = F - Pi F attributions
    -- see compute_rotation_orbit_average/compute_translation_orbit_average docstrings)."""
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
    modules = list(plotting_slices.keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    for ax, checkpoint, subtitle in ((axes[0], first, "random init"), (axes[1], last, "trained")):
        b_rot = np.asarray(checkpoint["b_rot_score"])
        b_trans = np.asarray(checkpoint["b_trans_score"])
        b_rot_by_module = [np.linalg.norm(b_rot[plotting_slices[m]]) for m in modules]
        b_trans_by_module = [np.linalg.norm(b_trans[plotting_slices[m]]) for m in modules]
        ax.bar(x - width / 2, b_rot_by_module, width, label=r"rotation defect ($\delta_{\rm rot}$)")
        ax.bar(x + width / 2, b_trans_by_module, width, label=r"translation defect ($\delta_{\rm trans}$)")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
        ax.set(ylabel="attribution norm within module")
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "module_b_attribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "module_b_attribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_symmetry_b_attribution_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """b_i analogue of plot_symmetry_attribution_scatter: per-parameter rotation-defect
    attribution vs. translation-defect attribution."""
    plotting_slices = _plotting_slices(parameter_slices)
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
        b_rot = np.asarray(checkpoint["b_rot_score"])
        b_trans = np.asarray(checkpoint["b_trans_score"])
        for name in modules:
            sl = plotting_slices[name]
            ax.scatter(
                np.maximum(b_rot[sl], 1e-12), np.maximum(b_trans[sl], 1e-12),
                s=14, color=colours[name], label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=r"rotation-defect attribution $|b_i|$", ylabel=r"translation-defect attribution $|b_i|$")
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_dir / "symmetry_b_attribution_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "symmetry_b_attribution_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_module_c_vs_b_comparison(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Primary c_i-vs-b_i comparison, module level, one figure per symmetry: g = X F (tangent
    to that symmetry's orbit) vs. delta = F - Pi F (the actual equivariance defect for that
    symmetry). Both are provably orthogonal as functions (identical skew-adjoint argument to
    the mexican-hat script), so a priori there is no reason their per-parameter attributions
    have to agree -- this is the module-aggregated first look at whether they nonetheless
    share support. K_net is excluded (unlike plot_module_symmetry_comparison/
    plot_module_b_attribution): c_i and b_i are architecturally zero there for the identical
    reason within each symmetry, so unlike the rotation-vs-translation comparisons there's no
    split to see, only a redundant floor cluster."""
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
    modules = list(plotting_slices.keys())

    for c_key, b_key, stem, gen in (
        ("xrot_score", "b_rot_score", "module_c_vs_b_comparison_rotation", "rotation"),
        ("xtrans_score", "b_trans_score", "module_c_vs_b_comparison_translation", "translation"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(modules)), 5.5), constrained_layout=True)
        x = np.arange(len(modules))
        width = 0.38
        for ax, checkpoint, subtitle in ((axes[0], first, "random init"), (axes[1], last, "trained")):
            c = np.asarray(checkpoint[c_key])
            b = np.asarray(checkpoint[b_key])
            c_by_module = [np.linalg.norm(c[plotting_slices[m]]) for m in modules]
            b_by_module = [np.linalg.norm(b[plotting_slices[m]]) for m in modules]
            ax.bar(x - width / 2, c_by_module, width, label=r"$c_i$ (orbit-tangent $g$)")
            ax.bar(x + width / 2, b_by_module, width, label=r"$b_i$ (defect $\delta$)")
            ax.set_xticks(x)
            ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right", fontsize=8)
            ax.set(ylabel="attribution norm within module")
            ax.set_title(subtitle, fontsize=10)
            ax.grid(alpha=0.25, axis="y")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{gen.capitalize()}: $c_i$ vs. $b_i$ by module")
        fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_c_vs_b_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Primary c_i-vs-b_i comparison, per parameter, one figure per symmetry: |c_i| (RMS over
    alpha) against |b_i| (RMS over alpha). Since g and delta are provably orthogonal as
    functions, agreement here reflects a genuine (if a priori unnecessary) alignment of which
    parameters move each quantity, not a restatement of the same computation. K_net excluded,
    same reasoning as plot_module_c_vs_b_comparison."""
    plotting_slices = _plotting_slices(parameter_slices)
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    for c_key, b_key, stem, gen in (
        ("xrot_score", "b_rot_score", "c_vs_b_scatter_rotation", "rotation"),
        ("xtrans_score", "b_trans_score", "c_vs_b_scatter_translation", "translation"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
        for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
            c = np.asarray(checkpoint[c_key])
            b = np.asarray(checkpoint[b_key])
            for name in modules:
                sl = plotting_slices[name]
                ax.scatter(
                    np.maximum(c[sl], 1e-12), np.maximum(b[sl], 1e-12),
                    s=14, color=colours[name], label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set(xlabel=r"$|c_i|$ (orbit-tangent $g$ attribution)", ylabel=r"$|b_i|$ (defect $\delta$ attribution)")
            ax.set_title(subtitle, fontsize=10)
            ax.grid(alpha=0.25, which="both")
        axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"{gen.capitalize()}: $c_i$ vs. $b_i$, per parameter")
        fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_c_vs_b_scatter_signed(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Signed companion of plot_c_vs_b_scatter: same magnitude as the unsigned |c_i|/|b_i| RMS
    scores there, signed by the mean (across all analysis_alphas) of the raw per-alpha
    coefficients. Sign here indicates the direction a parameter would move to realise g /
    delta respectively -- agreement in sign is a stronger, more specific form of alignment
    than agreement in magnitude alone. K_net excluded, same reasoning as plot_c_vs_b_scatter."""
    plotting_slices = _plotting_slices(parameter_slices)
    modules = list(plotting_slices.keys())
    colours = module_colours(modules)

    for c_key, c_raw, b_key, b_raw, stem, gen in (
        ("xrot_score", "xrot_attribution_coefficients", "b_rot_score", "delta_rot_attribution_coefficients",
         "c_vs_b_scatter_rotation_signed", "rotation"),
        ("xtrans_score", "xtrans_attribution_coefficients", "b_trans_score", "delta_trans_attribution_coefficients",
         "c_vs_b_scatter_translation_signed", "translation"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
        for ax, checkpoint, subtitle in ((axes[0], all_results[0], "random init"), (axes[1], all_results[-1], "trained")):
            c = _signed_like_score(checkpoint, c_raw, checkpoint[c_key])
            b = _signed_like_score(checkpoint, b_raw, checkpoint[b_key])
            linthresh = robust_linthresh(np.concatenate([c, b]))
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
            ax.set(xlabel=r"$c_i$ (orbit-tangent $g$ attribution)", ylabel=r"$b_i$ (defect $\delta$ attribution)")
            ax.set_title(subtitle, fontsize=10)
            ax.grid(alpha=0.25, which="both")
        axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"{gen.capitalize()}: $c_i$ vs. $b_i$, per parameter (signed)")
        fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """One plot per symmetry (rotation, translation), each with init-vs-trained bars side by
    side on the same axes -- matches the mexican-hat script's equivariance_by_module style,
    just split per generator instead of one figure with an init/trained panel pair, since here
    there are two symmetries to compare rather than one."""
    first, last = all_results[0], all_results[-1]

    def central_row(checkpoint):
        rows = checkpoint["alpha_results"]
        return rows[len(rows) // 2]

    first_row, last_row = central_row(first), central_row(last)
    modules = _exclude_structurally_zero(list(first_row["module_rotation_equivariance_error"].keys()))

    for key, stem, label in (
        ("module_rotation_equivariance_error", "equivariance_by_module_rotation", "rotation"),
        ("module_translation_equivariance_error", "equivariance_by_module_translation", "translation"),
    ):
        before = [first_row[key][m] for m in modules]
        after = [last_row[key][m] for m in modules]

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
        x = np.arange(len(modules))
        width = 0.38
        ax.bar(x - width / 2, before, width, label="random init")
        ax.bar(x + width / 2, after, width, label="after training")
        ax.set_xticks(x)
        ax.set_xticklabels([prettify_parameter_name(m, modules) for m in modules], rotation=60, ha="right")
        ax.set(ylabel=f"mean $E_i$ within module ({label})")
        ax.grid(alpha=0.25, axis="y")
        ax.legend()
        fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
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
    """Per-parameter generator attribution |c_i|/|b_i|: random init vs. trained, one plot per
    (generator, g-vs-delta) pair, plus a signed companion for each.

    Same story as equivariance_scatter -- does training concentrate each generator's
    attribution onto fewer parameters, or leave the pattern essentially unchanged? b_rot_i/
    b_trans_i are the delta = F - Pi F attributions -- see compute_rotation_orbit_average/
    compute_translation_orbit_average docstrings for why these are genuinely distinct
    questions from c_i, not the same one recomputed.
    """
    first, last = all_results[0], all_results[-1]
    plotting_slices = _plotting_slices(parameter_slices)
    for score_key, raw_key, stem, symbol, label in (
        ("xrot_score", "xrot_attribution_coefficients", "attribution_scatter_rotation", "c",
         r"Rotation-generator attribution $|c_i|$"),
        ("xtrans_score", "xtrans_attribution_coefficients", "attribution_scatter_translation", "c",
         r"Translation-generator attribution $|c_i|$"),
        ("b_rot_score", "delta_rot_attribution_coefficients", "b_attribution_scatter_rotation", "b",
         r"Rotation-defect attribution $|b_i|$"),
        ("b_trans_score", "delta_trans_attribution_coefficients", "b_attribution_scatter_translation", "b",
         r"Translation-defect attribution $|b_i|$"),
    ):
        plot_ei_initial_vs_final(
            np.asarray(first[score_key]), np.asarray(last[score_key]), plotting_slices,
            title=f"{label}: init vs. trained",
            output_stem=output_dir / stem,
            quantity_label=f"$|{symbol}_i|$",
            log_scale=True,
        )
        signed_initial = _signed_like_score(first, raw_key, first[score_key])
        signed_final = _signed_like_score(last, raw_key, last[score_key])
        plot_signed_initial_vs_final(
            signed_initial, signed_final, plotting_slices,
            output_stem=output_dir / f"{stem}_signed",
            quantity_label=f"${symbol}_i$",
        )


def plot_magnitude_diagnostics(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Check whether small S_i/E_i/c_i/b_i values at specific parameters are a genuine effect
    or simply an artefact of those parameters having small |theta_i| (e.g. from L1), plus
    signed companions of every pair where neither axis is a genuine non-negative magnitude."""
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
    # (score_key, raw_key, stem, symbol, label) -- c_rot, c_trans, then their b_i analogues
    # (delta = F - Pi F attributions; see compute_rotation_orbit_average/
    # compute_translation_orbit_average docstrings for why these are distinct from c_i).
    attribution_specs = (
        ("xrot_score", "xrot_attribution_coefficients", "rotation", "c", "rotation-generator"),
        ("xtrans_score", "xtrans_attribution_coefficients", "translation", "c", "translation-generator"),
        ("b_rot_score", "delta_rot_attribution_coefficients", "rotation", "b", "rotation-defect"),
        ("b_trans_score", "delta_trans_attribution_coefficients", "translation", "b", "translation-defect"),
    )
    equivariance_specs = (
        ("xrot_score", "rotation_equivariance_score", "rotation", "c", "rotation-generator"),
        ("xtrans_score", "translation_equivariance_score", "translation", "c", "translation-generator"),
        ("b_rot_score", "rotation_equivariance_score", "rotation", "b", "rotation-defect"),
        ("b_trans_score", "translation_equivariance_score", "translation", "b", "translation-defect"),
    )

    for score_key, raw_key, gen, symbol, label in attribution_specs:
        stem = f"magnitude_vs_{'attribution' if symbol == 'c' else 'b_attribution'}_{gen}"
        plot_magnitude_vs_quantity(
            magnitude_initial, magnitude_final,
            np.asarray(first[score_key]), np.asarray(last[score_key]),
            plotting_slices,
            quantity_label=rf"$|{symbol}_i|$ ({label} attribution)",
            title=rf"Parameter magnitude vs. {label} attribution $|{symbol}_i|$ ($V_{{\rm net}}$ only)",
            output_stem=output_dir / stem,
        )
    for score_key, error_key, gen, symbol, label in equivariance_specs:
        stem = f"attribution_vs_equivariance_{gen}" if symbol == "c" else f"b_attribution_vs_equivariance_{gen}"
        plot_magnitude_vs_quantity(
            np.asarray(first[score_key]), np.asarray(last[score_key]),
            np.asarray(first[error_key]), np.asarray(last[error_key]),
            plotting_slices,
            quantity_label=rf"$E_i$ ({gen} sensitivity-equivariance error)",
            title=rf"{label.capitalize()} attribution $|{symbol}_i|$ vs. equivariance error $E_i$",
            output_stem=output_dir / stem,
            x_label=rf"$|{symbol}_i|$ ({label} attribution)",
        )
    for score_key, raw_key, gen, symbol, label in attribution_specs:
        stem = f"attribution_vs_sensitivity_{gen}" if symbol == "c" else f"b_attribution_vs_sensitivity_{gen}"
        plot_magnitude_vs_quantity(
            np.asarray(first[score_key]), np.asarray(last[score_key]),
            _alpha_averaged(first, "sensitivity"), _alpha_averaged(last, "sensitivity"),
            plotting_slices,
            quantity_label=r"$S_i = |\partial f_\theta/\partial\theta_i|$ (mean over $\alpha$)",
            title=rf"{label.capitalize()} attribution $|{symbol}_i|$ vs. sensitivity $S_i$",
            output_stem=output_dir / stem,
            x_label=rf"$|{symbol}_i|$ ({label} attribution)",
        )

    # --- Signed companions. Same magnitude as the unsigned scores above (whatever
    # alpha-aggregation they already use), sign from the mean (across all analysis_alphas) of
    # the raw per-alpha signed values. Not |theta_i| but the raw signed theta_i, since neither
    # axis in any of these pairs is a genuine non-negative magnitude. Equivariance error E_i
    # has no sign (a defect norm), so attribution-vs-equivariance has no signed companion. ---
    theta_signed_initial = np.asarray(first["parameter_value"])
    theta_signed_final = np.asarray(last["parameter_value"])
    s_signed_initial = _signed_like_score(first, "signed_sensitivity", _alpha_averaged(first, "sensitivity"))
    s_signed_final = _signed_like_score(last, "signed_sensitivity", _alpha_averaged(last, "sensitivity"))
    plot_both_signed_quantity(
        theta_signed_initial, theta_signed_final, s_signed_initial, s_signed_final,
        plotting_slices,
        x_label=r"$\theta_i$ (signed parameter value)",
        y_label=r"$S_i$ (signed sensitivity, mean over probe grid)",
        output_stem=output_dir / "magnitude_vs_sensitivity_signed",
    )
    for score_key, raw_key, gen, symbol, label in attribution_specs:
        signed_initial = _signed_like_score(first, raw_key, first[score_key])
        signed_final = _signed_like_score(last, raw_key, last[score_key])
        mag_stem = f"magnitude_vs_{'attribution' if symbol == 'c' else 'b_attribution'}_{gen}_signed"
        plot_both_signed_quantity(
            theta_signed_initial, theta_signed_final, signed_initial, signed_final,
            plotting_slices,
            x_label=r"$\theta_i$ (signed parameter value)",
            y_label=rf"${symbol}_i$ ({label} attribution)",
            output_stem=output_dir / mag_stem,
        )
        sens_stem = f"attribution_vs_sensitivity_{gen}_signed" if symbol == "c" else f"b_attribution_vs_sensitivity_{gen}_signed"
        plot_both_signed_quantity(
            signed_initial, signed_final, s_signed_initial, s_signed_final,
            plotting_slices,
            x_label=rf"${symbol}_i$ ({label} attribution)",
            y_label=r"$S_i$ (signed sensitivity)",
            output_stem=output_dir / sens_stem,
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
    plot_module_b_attribution(all_results, parameter_slices, output_dir)
    plot_symmetry_b_attribution_scatter(all_results, parameter_slices, output_dir)
    plot_module_c_vs_b_comparison(all_results, parameter_slices, output_dir)
    plot_c_vs_b_scatter(all_results, parameter_slices, output_dir)
    plot_c_vs_b_scatter_signed(all_results, parameter_slices, output_dir)
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
    print(
        f"Final c-b correlations: rotation={all_results[-1]['xrot_brot_correlation']:.3f}, "
        f"translation={all_results[-1]['xtrans_btrans_correlation']:.3f}; "
        f"cross-symmetry: c_rot-c_trans={all_results[-1]['xrot_xtrans_correlation']:.3f}, "
        f"b_rot-b_trans={all_results[-1]['brot_btrans_correlation']:.3f}"
    )
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()
