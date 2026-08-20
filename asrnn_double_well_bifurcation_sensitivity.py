"""Functional-sensitivity experiment for the ASRNN double-well model.

The physical system is

    H(p, q; alpha) = p**2 / 2 + alpha*q**2 / 2 + q**4 / 4,
    dq/dt = p,
    dp/dt = F(q, alpha) = -alpha*q - q**3.

Thus alpha > 0 gives one well, and alpha < 0
gives two wells. 
At selected training checkpoints we compute:

1. The functional Jacobian J_alpha(q, i) = d F_theta(q, alpha) / d theta_i.
2. Per-parameter RMS sensitivities as a function of alpha.
3. How well the physical bifurcation direction dF/dalpha = -q lies in the
   functional tangent space Im(J_alpha): relative projection error and angle.
4. The minimum-norm parameter vector c_alpha such that J_alpha c_alpha ~= -q.
5. The learned curvature d^2 V_theta/dq^2 at q=0, whose sign distinguishes
   the single-well and double-well regimes.
6. The genuine discrete parity symmetry (q, p) -> (-q, -p), including
   evenness of V, oddness of F, and odd equivariance of parameter sensitivities.

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
    domain_parity_energy_fraction,
    finite_transform_residual,
    per_parameter_equivariance_error,
    relative_energy_drift,
    sensitivity_transform_residual,
)

ROOT = Path(__file__).resolve().parent
ASRNN_DOUBLE_WELL = ROOT / "ASRNN_Sparse_Data" / "Double_Well_Code"
if not ASRNN_DOUBLE_WELL.exists():
    raise FileNotFoundError(
        f"Expected the cloned ASRNN repository at {ASRNN_DOUBLE_WELL}"
    )
sys.path.insert(0, str(ASRNN_DOUBLE_WELL))

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
    output_dir: str = "outputs/asrnn_double_well_bifurcation"

    # "hamiltonian": the ASRNN V_net/K_net split (architecturally guarantees
    # conservation of the network's own energy via the symplectic Verlet
    # step). "direct_mlp": one plain MLP outputting (dp/dt, dq/dt) directly,
    # integrated with the same kick-drift-kick step shape but with no
    # conservation structure at all (Experiment 2's architecture comparison).
    architecture: str = "hamiltonian"  # hamiltonian or direct_mlp

    # ASRNN Hamiltonian network. V_net receives (q, alpha); K_net receives p.
    kinetic_hidden_dim: int = 50
    kinetic_hidden_layers: int = 2
    potential_hidden_dim: int = 50
    potential_hidden_layers: int = 2

    # direct_mlp architecture only.
    direct_mlp_hidden_dim: int = 50
    direct_mlp_hidden_layers: int = 2

    # Sparse/noisy trajectory generation. Do not include exactly alpha=0 in
    # training_alphas: the repository's initial-condition sampler is singular
    # there. Analysis may and should include alpha=0.
    training_alphas: list[float] = field(
        default_factory=lambda: [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9]
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
    # Doubles the training set with a parity-flipped copy of every trajectory --
    # since V(-q)=V(q) exactly for this system, (p,q)->(-p,-q) is an exact symmetry
    # of the true dynamics for every alpha, so this is free, exactly-valid additional
    # data teaching the Z2 parity implicitly through data diversity.
    augment_dataset: bool = False

    # Training. Adam is convenient for resolving the evolution in time.
    # Set optimizer="lbfgs" to mirror the original model_generator.py.
    optimizer: str = "adam"  # adam or lbfgs
    training_steps: int = 10000
    learning_rate: float = 1e-3  # Adam default; use 1.0 for LBFGS
    weight_decay: float = 0.0
    # Checkpoints are specified as fractions of training_steps (each in
    # [0, 1]) so they scale automatically if training_steps changes, e.g.
    # under --quick. Resolved to absolute step indices in validate_config.
    checkpoint_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
    )
    checkpoint_steps: list[int] = field(default_factory=list)
    lbfgs_history_size: int = 10

    # Functional sensitivity probe. A denser alpha grid gives a clearer view
    # of the transition but requires more second-order autodiff calls.
    analysis_alphas: list[float] = field(
        default_factory=lambda: np.linspace(-1.0, 1.0, 21).tolist()
    )
    q_min: float = -2.0
    q_max: float = 2.0
    q_probe_points: int = 41
    potential_probe_alphas: list[float] = field(
        default_factory=lambda: [-0.8, -0.3, 0.0, 0.3, 0.8]
    )
    potential_plot_points: int = 301
    # Only singular directions above this fraction of sigma_max define the
    # numerically resolved tangent space. Without truncation an overparameterised
    # network can trivially span every point on a finite q grid at initialisation.
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
        "--config",
        type=Path,
        help="Optional JSON file overriding fields in Config.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a very small end-to-end smoke test.",
    )
    parser.add_argument("--device", help="Override Config.device.")
    parser.add_argument("--output-dir", help="Override Config.output_dir.")
    parser.add_argument(
        "--architecture", choices=["hamiltonian", "direct_mlp"], help="Override Config.architecture."
    )
    parser.add_argument("--augment-dataset", dest="augment_dataset", action="store_true", default=None, help="Override Config.augment_dataset to True.")
    parser.add_argument("--no-augment-dataset", dest="augment_dataset", action="store_false", help="Override Config.augment_dataset to False.")
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
    if args.architecture:
        cfg.architecture = args.architecture
    if args.augment_dataset is not None:
        cfg.augment_dataset = args.augment_dataset
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
        cfg.analysis_alphas = [-0.5, 0.0, 0.5]
        cfg.q_probe_points = 9
        cfg.top_parameters_to_report = 5
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if any(abs(alpha) < 1e-14 for alpha in cfg.training_alphas):
        raise ValueError("training_alphas must not contain exactly 0; use nearby values.")
    if cfg.optimizer.lower() not in {"adam", "lbfgs"}:
        raise ValueError("optimizer must be 'adam' or 'lbfgs'.")
    if cfg.sampled_instants > cfg.trajectory_window:
        raise ValueError("sampled_instants cannot exceed trajectory_window.")
    if cfg.trajectory_splits < 1:
        raise ValueError("trajectory_splits must be at least 1.")
    if cfg.q_probe_points < 2:
        raise ValueError("q_probe_points must be at least 2.")
    if cfg.attribution_method not in {"l2", "l1", "elastic_net"}:
        raise ValueError("attribution_method must be 'l2', 'l1', or 'elastic_net'.")
    if not math.isclose(cfg.q_min, -cfg.q_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Parity analysis requires a symmetric q interval: q_min=-q_max.")
    if cfg.potential_plot_points < 2:
        raise ValueError("potential_plot_points must be at least 2.")
    if not 0.0 < cfg.tangent_svd_relative_cutoff < 1.0:
        raise ValueError("tangent_svd_relative_cutoff must lie strictly between 0 and 1.")
    if not all(0.0 <= f <= 1.0 for f in cfg.checkpoint_fractions):
        raise ValueError("checkpoint_fractions must all lie in [0, 1].")
    cfg.checkpoint_steps = sorted(
        {int(round(f * cfg.training_steps)) for f in cfg.checkpoint_fractions}
        | {0, cfg.training_steps}
    )


def augment_with_parity(trajectories: torch.Tensor) -> torch.Tensor:
    """Double a trajectory batch with a parity-flipped copy: (p, q) -> (-p, -q).

    ``trajectories`` has layout ``[T, batch, 2]`` = ``[p, q]``. Since
    V(-q)=V(q) exactly (F(p,q,alpha) = -alpha*q - q**3, odd in q), negating
    the whole trajectory gives another exactly valid trajectory of the same
    system for every alpha -- not applied per-timestep independently, since
    the parity flip is a symmetry of the whole trajectory, not a single
    point (dq/dt=p is preserved: d(-q)/dt = -p).
    """
    return -trajectories


def make_dataset(cfg: Config, device: torch.device, dtype: torch.dtype):
    alphas = torch.tensor(cfg.training_alphas, device=device, dtype=dtype)
    total_length = cfg.trajectory_splits + cfg.trajectory_window - 1
    noise_variance_fraction = cfg.noise_standard_deviation**2
    if cfg.noise_standard_deviation > 0:
        if cfg.noise_correlation_time <= 0:
            raise ValueError(
                "noise_correlation_time must be positive when noise is enabled."
            )
        theta = 1.0 / cfg.noise_correlation_time
    else:
        # generate_data ignores theta when apply_ou_noise=False.
        theta = 0.0
    trajectories, params, indices = generate_data(
        alphas,
        F,
        total_length,
        cfg.trajectory_window,
        cfg.sampled_instants,
        dt=cfg.integration_dt,
        in_conds=cfg.initial_conditions_per_alpha,
        coarsening_factor=cfg.coarsening_factor,
        nsr=noise_variance_fraction,
        theta=theta,
        burn_in_var=0,
        apply_ou_noise=cfg.noise_standard_deviation > 0,
        device=device,
        dtype=dtype,
        seed_ou=cfg.seed,
    )
    if cfg.augment_dataset:
        trajectories = torch.cat((trajectories, augment_with_parity(trajectories)), dim=1)
        params = torch.cat((params, params), dim=0)
        indices = torch.cat((indices, indices), dim=0)
        perm = torch.randperm(trajectories.size(1), device=trajectories.device)
        trajectories, params, indices = trajectories[:, perm, :], params[perm], indices[perm]
    return train_test_split(
        trajectories, params, indices, val_size=cfg.validation_fraction
    )


def build_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    if cfg.architecture == "direct_mlp":
        model = DirectDynamicsMLP(
            p_dim=1, q_dim=1, param_dim=1,
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
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.learning_rate,
        history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )


def learned_force(model: torch.nn.Module, q: torch.Tensor, alpha: torch.Tensor, architecture: str):
    """Return the model's dp/dt ("the force") with a graph retained for parameter derivatives.

    For ``architecture="direct_mlp"`` there is no V_net to differentiate --
    dp/dt is read directly off the model's output at a fixed reference
    momentum p=0 (the same architectural fact that makes the Hamiltonian
    net's force independent of p at all).
    """
    if architecture == "direct_mlp":
        q2 = q.reshape(1, 1)
        alpha2 = alpha.reshape(1, 1)
        p = torch.zeros_like(q2)
        dpdt, _ = model(p, q2, alpha2)
        return dpdt.squeeze(), q2
    q = q.reshape(1, 1).requires_grad_(True)
    alpha = alpha.reshape(1, 1)
    potential = model.V_net(torch.cat((q, alpha), dim=1))
    force = -torch.autograd.grad(potential.sum(), q, create_graph=True)[0]
    return force.squeeze(), q


def learned_dqdt(model: torch.nn.Module, p_values: torch.Tensor, alpha_value: float, architecture: str) -> torch.Tensor:
    """Return the model's own dq/dt at each probe momentum, at a fixed reference q=0.

    For the Hamiltonian net, dq/dt = dK_theta/dp is architecturally
    independent of both q and alpha. For direct_mlp there is no such
    guarantee, so alpha must be supplied and q is held at a reference value.
    """
    if architecture == "direct_mlp":
        p = p_values.reshape(-1, 1)
        q = torch.zeros_like(p)
        alpha = torch.full_like(p, float(alpha_value))
        with torch.no_grad():
            _, dqdt = model(p, q, alpha)
        return dqdt.squeeze(-1)
    p = p_values.reshape(-1, 1).clone().requires_grad_(True)
    kinetic = model.K_net(p)
    dqdt = torch.autograd.grad(kinetic.sum(), p)[0].squeeze(-1)
    return dqdt.detach()


def force_parameter_row(
    model: torch.nn.Module, q_value: float, alpha_value: float, architecture: str, *, device, dtype
) -> tuple[torch.Tensor, float]:
    q = torch.tensor(q_value, device=device, dtype=dtype)
    alpha = torch.tensor(alpha_value, device=device, dtype=dtype)
    force, _ = learned_force(model, q, alpha, architecture)
    params = tuple(model.parameters())
    grads = torch.autograd.grad(force, params, allow_unused=True)
    row = torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, grads)
        ]
    )
    return row.detach(), float(force.detach().cpu())


def curvature_and_parameter_gradient(
    model: torch.nn.Module, alpha_value: float, architecture: str, *, device, dtype
) -> tuple[float, torch.Tensor]:
    """Learned d^2V/dq^2 at q=0 and its parameter gradient. Undefined for direct_mlp (no V_net):
    returns NaN and an all-zero gradient so downstream aggregation stays well-formed."""
    params = tuple(model.parameters())
    if architecture == "direct_mlp":
        zeros = torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
        return float("nan"), zeros
    q = torch.zeros(1, 1, device=device, dtype=dtype, requires_grad=True)
    alpha = torch.full_like(q, alpha_value)
    potential = model.V_net(torch.cat((q, alpha), dim=1))
    first = torch.autograd.grad(potential.sum(), q, create_graph=True)[0]
    curvature = torch.autograd.grad(first.sum(), q, create_graph=True)[0].squeeze()
    grads = torch.autograd.grad(curvature, params, allow_unused=True)
    flat = torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, grads)
        ]
    )
    return float(curvature.detach().cpu()), flat.detach()


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
    q_values = torch.linspace(
        cfg.q_min, cfg.q_max, cfg.q_probe_points, device=device, dtype=dtype
    )
    # Reuse the same range/count for a paired p probe (not a full p-q mesh,
    # to keep the energy-conservation diagnostic's cost the same order as the
    # existing force probe).
    p_values = q_values.clone()
    alpha_results = []
    for alpha in tqdm(cfg.analysis_alphas, desc=f"analysing step {step}", leave=False):
        rows, predicted_force = [], []
        for q in q_values.detach().cpu().tolist():
            row, force = force_parameter_row(
                model, q, alpha, cfg.architecture, device=device, dtype=dtype
            )
            rows.append(row)
            predicted_force.append(force)
        jacobian = torch.stack(rows)
        predicted_force_tensor = torch.tensor(
            predicted_force, device=device, dtype=dtype
        )
        learned_dqdt_values = learned_dqdt(model, p_values, alpha, cfg.architecture)

        # The Z2 parity generator (q, p) -> (-q, -p): V is even, F is odd, and an
        # exactly parity-equivariant network's Jacobian must itself be odd in q.
        # There is no V_net for direct_mlp, so the potential-invariance check
        # does not apply there (only force-level parity does).
        parity_rep = torch.tensor([[-1.0]])
        if cfg.architecture == "direct_mlp":
            potential_even_residual = float("nan")
        else:
            alpha_column = torch.full_like(q_values, float(alpha))
            with torch.no_grad():
                predicted_potential = model.V_net(
                    torch.stack((q_values, alpha_column), dim=1)
                ).squeeze()
                # Remove the unidentifiable additive V(0, alpha) before normalising.
                predicted_potential = (
                    predicted_potential
                    - predicted_potential[torch.argmin(q_values.abs())]
                )
            potential_even_residual = finite_transform_residual(
                predicted_potential.unsqueeze(-1),
                torch.flip(predicted_potential, dims=[0]).unsqueeze(-1),
            )
        force_odd_residual = finite_transform_residual(
            predicted_force_tensor.unsqueeze(-1),
            torch.flip(predicted_force_tensor, dims=[0]).unsqueeze(-1),
            parity_rep,
        )
        reflected_jacobian = torch.flip(jacobian, dims=[0])
        jacobian_3d = jacobian.unsqueeze(1)
        reflected_jacobian_3d = reflected_jacobian.unsqueeze(1)
        sensitivity_odd_residual = sensitivity_transform_residual(
            jacobian_3d, reflected_jacobian_3d, parity_rep
        )
        tangent_odd_energy_fraction = domain_parity_energy_fraction(
            jacobian, reflected_jacobian
        )
        # Per-parameter sensitivity-equivariance error E_i (PDF sec. 0.6): how far each
        # parameter's own sensitivity S_i is from transforming correctly under parity,
        # localising the aggregate sensitivity_odd_residual to individual parameters.
        equivariance_error_by_parameter = per_parameter_equivariance_error(
            jacobian_3d, reflected_jacobian_3d, parity_rep
        )
        bifurcation_direction = -q_values.detach()
        (
            projection_error,
            principal_angle,
            coefficients,
            resolved_rank,
            singular_values,
        ) = resolve_attribution_coefficients(
            cfg, attribution_l1_ratio_cache, "bifurcation", jacobian, bifurcation_direction
        )
        sensitivity = torch.sqrt(torch.mean(jacobian.detach().cpu().double().square(), dim=0))
        curvature, curvature_gradient = curvature_and_parameter_gradient(
            model, alpha, cfg.architecture, device=device, dtype=dtype
        )
        # Noether energy-conservation diagnostic: does the learned dynamics
        # conserve the *true* energy? The leapfrog integrator only guarantees
        # conservation of the network's *own* H_theta, not the true energy --
        # so this is expected to start large (confirmed ~1 at random init)
        # and fall only insofar as training makes H_theta approach the truth.
        grad_e_q = float(alpha) * q_values + q_values**3
        energy_conservation_violation = relative_energy_drift(
            p_values.detach().cpu().unsqueeze(-1).double(),
            grad_e_q.detach().cpu().unsqueeze(-1).double(),
            predicted_force_tensor.detach().cpu().unsqueeze(-1).double(),
            learned_dqdt_values.detach().cpu().unsqueeze(-1).double(),
        )
        alpha_results.append(
            {
                "alpha": float(alpha),
                "predicted_force": predicted_force,
                "energy_conservation_violation": energy_conservation_violation,
                "true_force": (
                    -float(alpha) * q_values - q_values**3
                ).detach().cpu().tolist(),
                "sensitivity": sensitivity.cpu().tolist(),
                "potential_even_residual": potential_even_residual,
                "force_odd_residual": force_odd_residual,
                "sensitivity_odd_residual": sensitivity_odd_residual,
                "tangent_odd_energy_fraction": tangent_odd_energy_fraction,
                "equivariance_error_by_parameter": equivariance_error_by_parameter.cpu().tolist(),
                "projection_error": projection_error,
                "principal_angle_degrees": principal_angle,
                "resolved_tangent_rank": resolved_rank,
                "jacobian_singular_values": singular_values,
                "attribution_coefficients": coefficients.cpu().tolist(),
                "curvature": curvature,
                "true_curvature": float(alpha),
                "curvature_sensitivity": curvature_gradient.detach().cpu().double().abs().tolist(),
                "module_sensitivity": aggregate_by_module(sensitivity, parameter_slices),
                "module_equivariance_error": aggregate_by_module_mean(
                    equivariance_error_by_parameter, parameter_slices
                ),
                "module_attribution": aggregate_by_module(
                    coefficients.abs(), parameter_slices
                ),
            }
        )

    # Rank parameters by their strongest bifurcation attribution over alpha.
    coefficient_matrix = np.asarray(
        [r["attribution_coefficients"] for r in alpha_results]
    )
    score = np.sqrt(np.mean(coefficient_matrix**2, axis=0))
    top = np.argsort(score)[::-1][: cfg.top_parameters_to_report]
    equivariance_error_matrix = np.asarray(
        [r["equivariance_error_by_parameter"] for r in alpha_results]
    )
    equivariance_score = np.sqrt(np.mean(equivariance_error_matrix**2, axis=0))
    top_equivariance_violators = np.argsort(equivariance_score)[::-1][: cfg.top_parameters_to_report]
    return {
        "step": step,
        "q_values": q_values.detach().cpu().tolist(),
        "alpha_results": alpha_results,
        "top_bifurcation_parameters": [
            {"flat_index": int(i), "name": flat_names[i], "rms_coefficient": float(score[i])}
            for i in top
        ],
        "top_equivariance_violating_parameters": [
            {
                "flat_index": int(i),
                "name": flat_names[i],
                "rms_equivariance_error": float(equivariance_score[i]),
            }
            for i in top_equivariance_violators
        ],
    }


def write_checkpoint_outputs(result: dict[str, Any], output_dir: Path) -> None:
    step = result["step"]
    write_result_json(result, output_dir, step)
    write_csv_rows(
        result["top_bifurcation_parameters"],
        ["flat_index", "name", "rms_coefficient"],
        output_dir / f"top_parameters_step_{step:06d}.csv",
    )
    write_csv_rows(
        result["top_equivariance_violating_parameters"],
        ["flat_index", "name", "rms_equivariance_error"],
        output_dir / f"top_equivariance_violations_step_{step:06d}.csv",
    )


def plot_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colours = cmap(np.linspace(0.05, 0.95, len(all_results)))

    for colour, checkpoint in zip(colours, all_results):
        rows = checkpoint["alpha_results"]
        alphas = np.asarray([r["alpha"] for r in rows])
        label = f"step {checkpoint['step']}"
        axes[0, 0].plot(
            alphas, [r["curvature"] for r in rows], marker="o", ms=3,
            color=colour, label=label,
        )
        axes[0, 1].plot(
            alphas, [r["projection_error"] for r in rows], marker="o", ms=3,
            color=colour, label=label,
        )
        axes[1, 0].plot(
            alphas, [r["principal_angle_degrees"] for r in rows], marker="o", ms=3,
            color=colour, label=label,
        )
        total_sensitivity = [
            float(np.linalg.norm(np.asarray(r["sensitivity"]))) for r in rows
        ]
        axes[1, 1].plot(
            alphas, total_sensitivity, marker="o", ms=3, color=colour, label=label
        )

    alpha_reference = np.linspace(
        min(all_results[0]["alpha_results"], key=lambda x: x["alpha"])["alpha"],
        max(all_results[0]["alpha_results"], key=lambda x: x["alpha"])["alpha"],
        200,
    )
    axes[0, 0].plot(alpha_reference, alpha_reference, "k--", lw=1.5, label="true")
    axes[0, 0].set(title="Curvature at q=0", ylabel=r"$\partial_q^2 V_\theta(0,\alpha)$")
    axes[0, 1].set(title=r"Representation of $\partial_\alpha F=-q$", ylabel="relative projection error")
    axes[1, 0].set(title="Tangent-space angle", ylabel="degrees")
    axes[1, 1].set(title="Total force sensitivity", ylabel=r"$\|\mathrm{RMS}_q(J)\|_2$")
    for ax in axes.flat:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "bifurcation_sensitivity_summary.png", dpi=200)
    fig.savefig(output_dir / "bifurcation_sensitivity_summary.pdf")
    plt.close(fig)


def plot_parity_summary(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot the learned Z2 symmetry and sensitivity-equivariance diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(all_results)))
    residual_keys = (
        "potential_even_residual",
        "force_odd_residual",
        "sensitivity_odd_residual",
    )

    for colour, checkpoint in zip(colours, all_results):
        rows = checkpoint["alpha_results"]
        alphas = np.asarray([r["alpha"] for r in rows])
        label = f"step {checkpoint['step']}"
        for ax, key in zip(axes.flat[:3], residual_keys):
            values = np.maximum([r[key] for r in rows], 1e-12)
            ax.plot(alphas, values, marker="o", ms=3, color=colour, label=label)
        axes[1, 1].plot(
            alphas,
            [r["tangent_odd_energy_fraction"] for r in rows],
            marker="o",
            ms=3,
            color=colour,
            label=label,
        )

    axes[0, 0].set(
        title=r"Potential parity: $V(-q)=V(q)$",
        ylabel="relative evenness residual",
    )
    axes[0, 1].set(
        title=r"Force parity: $F(-q)=-F(q)$",
        ylabel="relative oddness residual",
    )
    axes[1, 0].set(
        title=r"Sensitivity equivariance: $J(-q)=-J(q)$",
        ylabel="relative Frobenius residual",
    )
    axes[1, 1].set(
        title="Odd content of the functional Jacobian",
        ylabel="odd Jacobian energy fraction",
        ylim=(0.0, 1.02),
    )
    for ax in axes.flat[:3]:
        ax.set_yscale("log")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    for ax in axes.flat:
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "parity_symmetry_summary.png", dpi=200)
    fig.savefig(output_dir / "parity_symmetry_summary.pdf")
    plt.close(fig)


def plot_energy_conservation(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Noether energy-conservation violation vs. training step (mean over analysis alpha).

    Note what this architecture actually guarantees: the symplectic leapfrog
    integrator exactly conserves the network's *own* learned energy
    H_theta=V_theta+K_theta, not the true physical energy. So this metric
    (drift of the *true* energy under the learned flow) is NOT
    architecturally pinned near zero -- at random init V_theta/K_theta bear
    no relation to the true system, so drift starts large (~1, confirmed
    empirically) and should fall as training makes H_theta approach the true
    Hamiltonian. The informative comparison is against the direct-MLP
    architecture (added separately), which has no conserved quantity at all,
    built-in or learned -- so its drift has no structural reason to fall with
    training the way this one does.
    """
    steps = [c["step"] for c in all_results]
    mean_drift = [
        float(np.mean([r["energy_conservation_violation"] for r in c["alpha_results"]]))
        for c in all_results
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(steps, np.maximum(mean_drift, 1e-12), marker="o", ms=4)
    ax.set(
        title="Noether energy-conservation violation vs. training step",
        xlabel="training step", ylabel="relative energy drift (mean over alpha)",
        yscale="log",
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "energy_conservation.png", dpi=200)
    fig.savefig(output_dir / "energy_conservation.pdf")
    plt.close(fig)


def plot_equivariance_by_module(all_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Localise the sensitivity-equivariance defect (sec. 0.6's E_i) to individual layers.

    Compares the mean per-parameter E_i within each named module (V_net/K_net
    layers) at random initialisation versus the final trained checkpoint, at
    the bifurcation point alpha=0's nearest analysis value. A high, flat bar
    pattern that barely changes between the two checkpoints is direct
    evidence that training is not concentrating equivariant sensitivities
    anywhere in particular -- i.e. that functional symmetry (which the
    parity_symmetry_summary plot shows training does learn) is not the same
    thing as sensitivity equivariance.
    """
    first, last = all_results[0], all_results[-1]
    alpha_idx = min(
        range(len(first["alpha_results"])),
        key=lambda i: abs(first["alpha_results"][i]["alpha"]),
    )
    modules = list(first["alpha_results"][alpha_idx]["module_equivariance_error"].keys())
    before = [first["alpha_results"][alpha_idx]["module_equivariance_error"][m] for m in modules]
    after = [last["alpha_results"][alpha_idx]["module_equivariance_error"][m] for m in modules]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(modules)), 5), constrained_layout=True)
    x = np.arange(len(modules))
    width = 0.38
    ax.bar(x - width / 2, before, width, label=f"step {first['step']} (random init)")
    ax.bar(x + width / 2, after, width, label=f"step {last['step']} (trained)")
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=60, ha="right", fontsize=8)
    ax.set(
        title=r"Sensitivity-equivariance error $E_i$ by module (near $\alpha=0$)",
        ylabel="mean $E_i$ within module",
    )
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.savefig(output_dir / "equivariance_by_module.png", dpi=200)
    fig.savefig(output_dir / "equivariance_by_module.pdf")
    plt.close(fig)


def plot_equivariance_scatter(
    all_results: list[dict[str, Any]], parameter_slices: dict[str, slice], output_dir: Path
) -> None:
    """Per-parameter E_i at random init vs. final checkpoint, one point per parameter."""
    first, last = all_results[0], all_results[-1]
    alpha_idx = min(
        range(len(first["alpha_results"])),
        key=lambda i: abs(first["alpha_results"][i]["alpha"]),
    )
    ei_initial = np.asarray(first["alpha_results"][alpha_idx]["equivariance_error_by_parameter"])
    ei_final = np.asarray(last["alpha_results"][alpha_idx]["equivariance_error_by_parameter"])
    plot_ei_initial_vs_final(
        ei_initial, ei_final, parameter_slices,
        title=r"Parity sensitivity-equivariance $E_i$: init vs. trained (near $\alpha=0$)",
        output_stem=output_dir / "equivariance_scatter",
    )


def potential_and_force_curves(
    model: torch.nn.Module,
    q_values: torch.Tensor,
    alpha_value: float,
    architecture: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate learned/true potential and force on one alpha slice.

    Potentials are aligned at q=0 because force/trajectory supervision cannot
    identify an additive function C(alpha) in V(q, alpha). There is no V_net
    for direct_mlp, so the potential curve is NaN there (skipped by the caller).
    """
    true_v = 0.5 * alpha_value * q_values.squeeze() ** 2 + 0.25 * q_values.squeeze() ** 4
    if architecture == "direct_mlp":
        learned_v = torch.full_like(true_v, float("nan"))
    else:
        alpha_column = torch.full_like(q_values, alpha_value)
        with torch.no_grad():
            learned_v = model.V_net(torch.cat((q_values, alpha_column), dim=1)).squeeze()
            q_zero = torch.zeros(1, 1, device=q_values.device, dtype=q_values.dtype)
            alpha_zero = torch.full_like(q_zero, alpha_value)
            learned_offset = model.V_net(torch.cat((q_zero, alpha_zero), dim=1)).squeeze()
            learned_v = learned_v - learned_offset

    learned_f = []
    for q in q_values.squeeze().detach().cpu().tolist():
        q_tensor = torch.tensor(q, device=q_values.device, dtype=q_values.dtype)
        alpha_tensor = torch.tensor(
            alpha_value, device=q_values.device, dtype=q_values.dtype
        )
        force, _ = learned_force(model, q_tensor, alpha_tensor, architecture)
        learned_f.append(float(force.detach().cpu()))
    true_f = -alpha_value * q_values.squeeze() - q_values.squeeze() ** 3
    return (
        learned_v.detach().cpu().numpy(),
        true_v.detach().cpu().numpy(),
        np.asarray(learned_f),
        true_f.detach().cpu().numpy(),
    )


def plot_learned_physics(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> None:
    """Compare the final learned potential and force with the analytic system."""
    model.eval()
    q_values = torch.linspace(
        cfg.q_min,
        cfg.q_max,
        cfg.potential_plot_points,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)
    n_alpha = len(cfg.potential_probe_alphas)
    n_cols = min(3, n_alpha)
    n_rows = math.ceil(n_alpha / n_cols)

    # No V_net for direct_mlp, so the standalone potential comparison doesn't apply.
    quantities = ("force",) if cfg.architecture == "direct_mlp" else ("potential", "force")
    for quantity in quantities:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.6 * n_cols, 3.7 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, alpha in zip(axes.flat, cfg.potential_probe_alphas):
            learned_v, true_v, learned_f, true_f = potential_and_force_curves(
                model, q_values, alpha, cfg.architecture
            )
            if quantity == "potential":
                learned, true = learned_v, true_v
                ylabel = r"$V(q,\alpha)-V(0,\alpha)$"
            else:
                learned, true = learned_f, true_f
                ylabel = r"$F(q,\alpha)$"
            rmse = float(np.sqrt(np.mean((learned - true) ** 2)))
            q_numpy = q_values.squeeze().detach().cpu().numpy()
            ax.plot(q_numpy, true, "k--", linewidth=2, label="true")
            ax.plot(q_numpy, learned, color="tab:blue", linewidth=1.8, label="learned")
            ax.set(
                title=rf"$\alpha={alpha:g}$; RMSE={rmse:.3e}",
                xlabel=r"$q$",
                ylabel=ylabel,
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[n_alpha:]:
            ax.set_visible(False)
        stem = "learned_potential" if quantity == "potential" else "learned_force"
        fig.savefig(output_dir / f"{stem}_final.png", dpi=200)
        fig.savefig(output_dir / f"{stem}_final.pdf")
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
    print("Generating sparse double-well trajectories...")
    train_data, validation_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}")

    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *args, **kwargs: generic_residuals(*args, **kwargs, p_dim=1)
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
            model,
            step,
            cfg,
            device,
            dtype,
            flat_names,
            parameter_slices,
            attribution_l1_ratio_cache,
        )
        write_checkpoint_outputs(result, output_dir)
        all_results.append(result)

    plot_summary(all_results, output_dir)
    plot_parity_summary(all_results, output_dir)
    plot_energy_conservation(all_results, output_dir)
    plot_equivariance_by_module(all_results, output_dir)
    plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    model.load_state_dict(checkpoint_states[max(checkpoint_states.keys())])
    plot_learned_physics(model, cfg, device, dtype, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(
        json.dumps(all_results, indent=2)
    )
    print(f"Finished. Results written to {output_dir.resolve()}")


def main() -> None:
    train_and_analyse(load_config(parse_args()))


if __name__ == "__main__":
    main()
