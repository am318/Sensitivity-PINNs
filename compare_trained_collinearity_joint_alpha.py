"""Same cross-seed / within-model decomposition, but with a joint multi-alpha Jacobian.

The previous run (compare_trained_collinearity.py) solved tangent_projection
at a single alpha, which made every hidden unit's alpha-input weight exactly
collinear with that unit's bias (weight_alpha * alpha_fixed is a constant,
indistinguishable from the bias term at one alpha). This script instead
stacks several alpha values into one joint Jacobian/target before solving,
which should break that specific degeneracy (the alpha-weight's contribution
now varies with alpha while the bias's doesn't) and is compared against the
single-alpha baseline for both the collinearity diagnostic and the
cross-seed vs. within-model stability decomposition.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config,
    build_model,
    build_probe_grid,
    evaluate_at_points,
    generic_residuals,
    make_dataset,
    make_optimizer,
    residuals,
    rotation_generator_target,
    validate_config,
)
from experiment_common import parameter_layout, run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    _resolve_tangent_svd,
    tangent_projection,
    tangent_projection_elastic_net,
    tangent_projection_l1,
)

OUTPUT_DIR = Path("outputs/trained_collinearity_joint_alpha")
SEEDS = [0, 1, 2, 3, 4]
TRAINING_STEPS = 400
SINGLE_ALPHA = 0.3
JOINT_ALPHAS = [-0.6, -0.3, 0.3, 0.6]
N_BOOTSTRAP = 25


def make_quick_config(seed: int) -> Config:
    cfg = Config()
    cfg.seed = seed
    cfg.kinetic_hidden_dim = 8
    cfg.potential_hidden_dim = 8
    cfg.initial_conditions_per_alpha = 2
    cfg.trajectory_splits = 2
    cfg.coarsening_factor = 2
    cfg.q_grid_points_per_axis = 4
    cfg.training_steps = TRAINING_STEPS
    cfg.checkpoint_fractions = [0.0, 1.0]
    validate_config(cfg)
    return cfg


def train_one_seed(seed: int):
    cfg = make_quick_config(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *a, **kw: generic_residuals(*a, **kw, p_dim=2)
    )
    run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    print(f"seed={seed} trained")
    return cfg, model, device, dtype


def single_alpha_jacobian_and_target(model, cfg, device, dtype, q1_pts, q2_pts, alpha=SINGLE_ALPHA):
    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_pts, q2_pts, alpha, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    n_points = q1_pts.shape[0]
    jac_flat = j_x.reshape(n_points * 2, -1)
    xrot_target = rotation_generator_target(q1_pts, q2_pts, f_x, spatial_jac_x).reshape(n_points * 2)
    return jac_flat, xrot_target


def joint_alpha_jacobian_and_target(model, cfg, device, dtype, q1_pts, q2_pts, alphas=JOINT_ALPHAS):
    jac_parts, target_parts = [], []
    n_points = q1_pts.shape[0]
    for alpha in alphas:
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_pts, q2_pts, alpha, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        jac_parts.append(j_x.reshape(n_points * 2, -1))
        target_parts.append(rotation_generator_target(q1_pts, q2_pts, f_x, spatial_jac_x).reshape(n_points * 2))
    return torch.cat(jac_parts, dim=0), torch.cat(target_parts, dim=0)


def top_support(c: torch.Tensor, k: int) -> frozenset[int]:
    return frozenset(torch.argsort(c.abs(), descending=True)[:k].tolist())


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(supports: list[frozenset]) -> float:
    pairs = combinations(range(len(supports)), 2)
    return float(np.mean([jaccard(supports[i], supports[j]) for i, j in pairs]))


def analyse(jac_fn, label: str, per_seed_models):
    per_seed = []
    for entry in per_seed_models:
        cfg, model, device, dtype, q1_grid, q2_grid = entry
        jac_flat, target = jac_fn(model, cfg, device, dtype, q1_grid, q2_grid)
        _, _, c_l2, rank, _ = tangent_projection(jac_flat, target, cfg.tangent_svd_relative_cutoff)
        _, _, c_l1, _, _ = tangent_projection_l1(jac_flat, target, cfg.tangent_svd_relative_cutoff)
        _, _, c_en, _, _ = tangent_projection_elastic_net(jac_flat, target, cfg.tangent_svd_relative_cutoff, l1_ratio=0.5)
        per_seed.append({"rank": rank, "c_l2": c_l2, "c_l1": c_l1, "c_en": c_en})

    ranks = {d["rank"] for d in per_seed}
    k = min(ranks)
    print(f"\n[{label}] resolved ranks across seeds: {sorted(ranks)}  (using top-{k})")

    cross_seed_stability = {
        "L2": mean_pairwise_jaccard([top_support(d["c_l2"], k) for d in per_seed]),
        "L1": mean_pairwise_jaccard([top_support(d["c_l1"], k) for d in per_seed]),
        "EN(0.5)": mean_pairwise_jaccard([top_support(d["c_en"], k) for d in per_seed]),
    }
    print(f"[{label}] cross-seed support stability ({len(per_seed)} seeds):")
    for name, val in cross_seed_stability.items():
        print(f"  {name}: {val:.3f}")

    # within-model bootstrap on the first seed's model
    cfg, model, device, dtype, q1_grid, q2_grid = per_seed_models[0]
    n_probe = q1_grid.shape[0]
    rng = np.random.default_rng(12345)
    boot_supports = {"L2": [], "L1": [], "EN(0.5)": []}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_probe, size=n_probe)
        idx_t = torch.as_tensor(idx, device=q1_grid.device)
        q1_bs, q2_bs = q1_grid[idx_t], q2_grid[idx_t]
        jac_bs, target_bs = jac_fn(model, cfg, device, dtype, q1_bs, q2_bs)
        _, _, c_l2_bs, _, _ = tangent_projection(jac_bs, target_bs, cfg.tangent_svd_relative_cutoff)
        _, _, c_l1_bs, _, _ = tangent_projection_l1(jac_bs, target_bs, cfg.tangent_svd_relative_cutoff)
        _, _, c_en_bs, _, _ = tangent_projection_elastic_net(jac_bs, target_bs, cfg.tangent_svd_relative_cutoff, l1_ratio=0.5)
        boot_supports["L2"].append(top_support(c_l2_bs, k))
        boot_supports["L1"].append(top_support(c_l1_bs, k))
        boot_supports["EN(0.5)"].append(top_support(c_en_bs, k))
    within_model_stability = {name: mean_pairwise_jaccard(s) for name, s in boot_supports.items()}
    print(f"[{label}] within-model support stability ({N_BOOTSTRAP} bootstraps, fixed model):")
    for name, val in within_model_stability.items():
        print(f"  {name}: {val:.3f}")

    return cross_seed_stability, within_model_stability, k, per_seed[0]


def collinearity_check(jac_fn, label: str, cfg, model, device, dtype, q1_grid, q2_grid, flat_names, coeffs):
    jac_flat, target = jac_fn(model, cfg, device, dtype, q1_grid, q2_grid)
    j = jac_flat.detach().cpu().to(torch.float64)
    g = target.detach().cpu().to(torch.float64)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, sv, col_scale = _resolve_tangent_svd(
        j, g, cfg.tangent_svd_relative_cutoff
    )
    a_eq = (vh_kept / col_scale).numpy()
    norms = np.linalg.norm(a_eq, axis=0)
    nz = np.where(norms > 1e-12)[0]
    a_unit = a_eq[:, nz] / norms[nz]
    cos = a_unit.T @ a_unit
    np.fill_diagonal(cos, 0)
    max_abs_cos = float(np.abs(cos).max()) if cos.size else 0.0
    n_near_exact = int((np.abs(cos) > 0.999).sum() // 2)
    print(f"[{label}] max |cosine| between distinct parameter columns in resolved subspace: {max_abs_cos:.5f}"
          f"  ({n_near_exact} pairs with |cos|>0.999)")
    if n_near_exact:
        flat_idx = np.argsort(-np.abs(cos), axis=None)[:10]
        seen = set()
        c_l2, c_l1, c_en = coeffs
        for fi in flat_idx:
            i, jx = np.unravel_index(fi, cos.shape)
            if i >= jx or (i, jx) in seen:
                continue
            seen.add((i, jx))
            gi, gj = nz[i], nz[jx]
            print(f"    ({flat_names[gi]}, {flat_names[gj]})  cos={cos[i, jx]:.5f}  "
                  f"L2=({c_l2[gi]:.4f},{c_l2[gj]:.4f})  L1=({c_l1[gi]:.4f},{c_l1[gj]:.4f})  "
                  f"EN0.5=({c_en[gi]:.4f},{c_en[gj]:.4f})")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_models = []
    for seed in SEEDS:
        cfg, model, device, dtype = train_one_seed(seed)
        q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
        per_seed_models.append((cfg, model, device, dtype, q1_grid, q2_grid))

    results = {}
    for label, jac_fn in [("single-alpha", single_alpha_jacobian_and_target), ("joint-alpha", joint_alpha_jacobian_and_target)]:
        cross, within, k, ref = analyse(jac_fn, label, per_seed_models)
        results[label] = {"cross": cross, "within": within, "k": k}

        cfg, model, device, dtype, q1_grid, q2_grid = per_seed_models[0]
        flat_names, _ = parameter_layout(model)
        collinearity_check(
            jac_fn, label, cfg, model, device, dtype, q1_grid, q2_grid, flat_names,
            (ref["c_l2"], ref["c_l1"], ref["c_en"]),
        )

    print("\n=== Summary: single-alpha vs joint-alpha ===")
    for method in ("L2", "L1", "EN(0.5)"):
        s_cross = results["single-alpha"]["cross"][method]
        s_within = results["single-alpha"]["within"][method]
        j_cross = results["joint-alpha"]["cross"][method]
        j_within = results["joint-alpha"]["within"][method]
        print(f"  {method}: cross-seed {s_cross:.3f} -> {j_cross:.3f}   within-model {s_within:.3f} -> {j_within:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    methods = ["L2", "L1", "EN(0.5)"]
    x = np.arange(len(methods))
    width = 0.35
    for ax, kind, title in [(axes[0], "cross", "Cross-seed stability"), (axes[1], "within", "Within-model stability")]:
        single_vals = [results["single-alpha"][kind][m] for m in methods]
        joint_vals = [results["joint-alpha"][kind][m] for m in methods]
        ax.bar(x - width / 2, single_vals, width, label="single-alpha")
        ax.bar(x + width / 2, joint_vals, width, label="joint multi-alpha")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("mean pairwise Jaccard overlap of top support")
    fig.suptitle("Single-alpha vs. joint multi-alpha Jacobian: attribution-support stability")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_vs_joint_alpha_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
