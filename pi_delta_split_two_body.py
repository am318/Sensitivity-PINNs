"""g vs. delta split, extended to the two-body benchmark (rotation generator only).

See pi_delta_split.py (Mexican-hat) for the full derivation. Only the
rotation generator is used here: translation is a noncompact group (R^2),
so the group-average integral Pi f = (1/|G|) integral rho(h)^{-1} f(h.x) dh
does not converge for it -- this is a genuine mathematical limitation of the
group-averaging construction, not an implementation gap, so translation's
g/delta split is out of scope by design.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from asrnn_two_body_symmetry_sensitivity import (
    Config,
    build_model,
    build_polar_probe_grid_rotation_invariant,
    evaluate_at_points,
    generic_residuals,
    group_average_and_defect,
    make_dataset,
    make_optimizer,
    rotation_generator_target,
    validate_config,
)
from experiment_common import parameter_layout, run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    participation_ratio_l1l2,
    tangent_projection_auto,
)

DEVICE = os.environ.get("DEVICE", "cpu")
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "400"))
SMALL_ARCH = os.environ.get("SMALL_ARCH", "1") == "1"
SEED = 0
ALPHA = 1.6
N_QUADRATURE = 48  # 24 leaves a ~1e-3 Prop.1 residual (genuine discretization error, confirmed by
# convergence to ~1e-8 at 48); the two-body force's r^-3 nonlinearity needs more quadrature
# points than the Mexican-hat's polynomial force for the same tightness.
OUTPUT_DIR = Path("outputs/pi_delta_split_two_body")


def make_config() -> Config:
    cfg = Config()
    cfg.seed = SEED
    cfg.device = DEVICE
    cfg.training_steps = TRAINING_STEPS
    cfg.checkpoint_steps = [0, TRAINING_STEPS]
    if SMALL_ARCH:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.direct_mlp_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.q_grid_points_per_axis = 2
        cfg.cm_grid_points = 2
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
    print(f"device={device}  training_steps={cfg.training_steps}  small_arch={SMALL_ARCH}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = lambda *a, **kw: generic_residuals(*a, **kw, p_dim=4)  # noqa: E731
    run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    model.eval()

    q1x, q1y, q2x, q2y = build_polar_probe_grid_rotation_invariant(
        n_radii=max(cfg.q_grid_points_per_axis, 2), n_angles=N_QUADRATURE,
        r_max=cfg.q_extent, device=device, dtype=dtype,
    )
    n_points = q1x.shape[0]
    f_x, j_x, pi_f, j_pi, delta, j_delta = group_average_and_defect(
        model, q1x, q1y, q2x, q2y, ALPHA, cfg.architecture, device=device, dtype=dtype, n_quadrature=N_QUADRATURE
    )
    _, _, _, spatial_jac_x = evaluate_at_points(
        model, q1x, q1y, q2x, q2y, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
    )
    xrot_target = rotation_generator_target(q1x, q1y, q2x, q2y, f_x, spatial_jac_x)
    g_flat = xrot_target.reshape(n_points * 4)
    jac_flat = j_x.reshape(n_points * 4, -1)
    delta_flat = delta.reshape(n_points * 4)
    jac_delta_flat = j_delta.reshape(n_points * 4, -1)

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
