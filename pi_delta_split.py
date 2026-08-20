"""g vs. delta split: symmetry representation vs. symmetry-breaking attribution.

Per the review notes (Prop. 1): g = X f_theta is tangent to the group orbit
of f_theta at *constant* distance from the equivariant subspace E = ker X,
so <delta, g> = 0 and g does not measure -- and its attribution does not
license conclusions about -- symmetry *breaking*. delta = Q f_theta (the
actual departure from E, via the group-averaging projector Pi = I - Q) is
the object whose attribution licenses that language.

This script:
1. Trains a model, computes g (the existing xrot generator target) and
   delta (via group_average_and_defect), and verifies Prop. 1 numerically:
   <delta, g> should be ~0 relative to ||delta|| ||g||.
2. Computes c^rep = EN-attribution(J, g) ("which parameters represent the
   symmetry") and c^def = EN-attribution(J_delta, delta) ("which parameters
   control the defect"), using the SAME elastic-net pipeline
   (tangent_projection_auto) for both, as requested.
3. Tests the review notes' prediction: do c^rep and c^def share support /
   agree at module level, but differ in within-module ranking? Reports
   support overlap (Jaccard), Spearman rank correlation, and per-module
   ||c|| comparison.
4. Reports ||g||, ||delta|| (order parameters) and representation quality
   for both.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config,
    build_model,
    evaluate_at_points,
    group_average_and_defect,
    make_dataset,
    make_optimizer,
    residuals,
    rotation_generator_target,
    validate_config,
)
from experiment_common import parameter_layout, run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    participation_ratio_l1l2,
    tangent_projection_auto,
)

ARCHITECTURE = os.environ.get("ARCHITECTURE", "hamiltonian")
DEVICE = os.environ.get("DEVICE", "cpu")
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "400"))
SMALL_ARCH = os.environ.get("SMALL_ARCH", "1") == "1"
SEED = 0
ALPHA = 0.3
N_QUADRATURE = 24  # must match the angle count used for the probe grid below, for exact grid invariance
OUTPUT_DIR = Path("outputs/pi_delta_split")


def build_polar_probe_grid(
    n_radii: int, n_angles: int, r_max: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probe points on concentric circles, evenly spaced in angle -- exactly rotation-invariant
    as a SET under any rotation by a multiple of 360/n_angles degrees.

    Prop. 1 (<delta, g> = 0) requires a rotation-invariant probe measure --
    the standard square Cartesian grid (build_probe_grid) is NOT invariant
    under rotation (a rotated square grid doesn't map back onto itself), so
    the discrete inner product only approximately satisfies the theorem's
    premise on that grid. Using this polar grid with n_angles == the
    quadrature's own angle count (N_QUADRATURE) makes the probe set exactly
    invariant under every rotation used inside group_average_and_defect,
    which should make the numerical Prop. 1 check hold far more tightly.
    """
    radii = torch.linspace(r_max / n_radii, r_max, n_radii, device=device, dtype=dtype)
    angles = torch.linspace(0, 2 * math.pi, n_angles + 1, device=device, dtype=dtype)[:-1]
    r_grid, a_grid = torch.meshgrid(radii, angles, indexing="ij")
    q1 = (r_grid * torch.cos(a_grid)).reshape(-1)
    q2 = (r_grid * torch.sin(a_grid)).reshape(-1)
    return q1, q2


def make_config() -> Config:
    cfg = Config()
    cfg.seed = SEED
    cfg.architecture = ARCHITECTURE
    cfg.device = DEVICE
    cfg.training_steps = TRAINING_STEPS
    cfg.checkpoint_fractions = [0.0, 1.0]
    if SMALL_ARCH:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.coarsening_factor = 2
        cfg.q_grid_points_per_axis = 4
    validate_config(cfg)
    return cfg


def top_support(c: torch.Tensor, k: int) -> frozenset[int]:
    return frozenset(torch.argsort(c.abs(), descending=True)[:k].tolist())


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    print(f"architecture={cfg.architecture}  device={device}  training_steps={cfg.training_steps}  small_arch={SMALL_ARCH}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    optimizer = make_optimizer(cfg, model)
    run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    model.eval()

    q1_grid, q2_grid = build_polar_probe_grid(
        n_radii=cfg.q_grid_points_per_axis, n_angles=N_QUADRATURE, r_max=cfg.q_extent, device=device, dtype=dtype
    )
    n_points = q1_grid.shape[0]
    f_x, j_x, pi_f, j_pi, delta, j_delta = group_average_and_defect(
        model, q1_grid, q2_grid, ALPHA, cfg.architecture, device=device, dtype=dtype, n_quadrature=N_QUADRATURE
    )
    _, _, _, spatial_jac_x = evaluate_at_points(
        model, q1_grid, q2_grid, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
    )
    xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
    g_flat = xrot_target.reshape(n_points * 2)
    jac_flat = j_x.reshape(n_points * 2, -1)
    delta_flat = delta.reshape(n_points * 2)
    jac_delta_flat = j_delta.reshape(n_points * 2, -1)

    # --- Prop. 1 check: <delta, g> should be ~0 relative to ||delta|| ||g|| ---
    g_norm = float(torch.linalg.vector_norm(g_flat))
    delta_norm = float(torch.linalg.vector_norm(delta_flat))
    inner = float(torch.dot(delta_flat.double(), g_flat.double()))
    cos_angle = inner / max(g_norm * delta_norm, 1e-30)
    print(f"\n||g|| = {g_norm:.6g}   ||delta|| = {delta_norm:.6g}")
    print(f"<delta, g> = {inner:.6g}   cos(angle) = {cos_angle:.6g}  (Prop. 1 predicts ~0)")

    cutoff = cfg.tangent_svd_relative_cutoff

    ratio_rep = choose_l1_ratio_for_sparsity(jac_flat, g_flat, cutoff)
    err_rep, ang_rep, c_rep, rank_rep, _ = tangent_projection_auto(
        jac_flat, g_flat, cutoff, method="elastic_net", l1_ratio=ratio_rep
    )
    ratio_def = choose_l1_ratio_for_sparsity(jac_delta_flat, delta_flat, cutoff)
    err_def, ang_def, c_def, rank_def, _ = tangent_projection_auto(
        jac_delta_flat, delta_flat, cutoff, method="elastic_net", l1_ratio=ratio_def
    )

    print(f"\nc^rep (attribution of g):     rank={rank_rep}  rel_error={err_rep:.4g}  angle_deg={ang_rep:.4g}  "
          f"n_eff={participation_ratio_l1l2(c_rep.numpy()):.2f}  l1_ratio={ratio_rep:.3f}")
    print(f"c^def (attribution of delta): rank={rank_def}  rel_error={err_def:.4g}  angle_deg={ang_def:.4g}  "
          f"n_eff={participation_ratio_l1l2(c_def.numpy()):.2f}  l1_ratio={ratio_def:.3f}")

    k = min(rank_rep, rank_def) or 1
    sup_rep, sup_def = top_support(c_rep, k), top_support(c_def, k)
    overlap = jaccard(sup_rep, sup_def)
    rho, pval = spearmanr(c_rep.abs().numpy(), c_def.abs().numpy())
    print(f"\nTop-{k} support overlap (Jaccard): {overlap:.3f}")
    print(f"Spearman rho(|c^rep|, |c^def|) across ALL parameters: {rho:.3f} (p={pval:.4g})")

    mod_rep = {name: float(c_rep[sl].norm()) for name, sl in parameter_slices.items()}
    mod_def = {name: float(c_def[sl].norm()) for name, sl in parameter_slices.items()}
    mod_names = list(mod_rep.keys())
    mod_rep_vals = np.array([mod_rep[n] for n in mod_names])
    mod_def_vals = np.array([mod_def[n] for n in mod_names])
    mod_rho, mod_p = spearmanr(mod_rep_vals, mod_def_vals)
    print(f"Module-level Spearman rho(||c^rep||, ||c^def||): {mod_rho:.3f} (p={mod_p:.4g})")
    print("\nPer-module ||c^rep|| vs ||c^def||:")
    for n in mod_names:
        print(f"  {n:20s} rep={mod_rep[n]:.4f}  def={mod_def[n]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(c_rep.abs().numpy() + 1e-30, c_def.abs().numpy() + 1e-30, s=10, alpha=0.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("|c^rep_i|  (attribution of g)")
    axes[0].set_ylabel("|c^def_i|  (attribution of delta)")
    axes[0].set_title(f"Per-parameter: rho={rho:.2f}")
    axes[0].grid(alpha=0.3)

    x = np.arange(len(mod_names))
    width = 0.35
    axes[1].bar(x - width / 2, mod_rep_vals, width, label="||c^rep|| (g)")
    axes[1].bar(x + width / 2, mod_def_vals, width, label="||c^def|| (delta)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mod_names, rotation=75, ha="right", fontsize=7)
    axes[1].set_title(f"Per-module: rho={mod_rho:.2f}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "g_vs_delta_attribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
