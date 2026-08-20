"""Isolate collinearity-driven attribution instability from genuine seed variability.

Two experiments on *trained* (not random-init) Mexican Hat models:

1. Cross-seed instability: train several independent models (different random
   seeds -> different init AND, since data generation also depends on the
   seed, slightly different training data draw), and measure how much the
   top-resolved-rank attribution support (L1 and elastic net) overlaps across
   these genuinely different trained solutions. This combines two effects:
   real differences in which parameters each independently-trained solution
   relies on, plus any solver-level arbitrariness from collinear parameter
   columns.

2. Fixed-model probe-bootstrap instability: hold ONE trained model fixed and
   bootstrap-resample (with replacement) the probe-grid points used to build
   the Jacobian, re-solving L1/elastic-net attribution each time. Since the
   model never changes here, any instability in the selected support is
   attributable only to estimation noise / collinear-column tie-breaking --
   not to genuine model-to-model differences.

Comparing (1) and (2) decomposes how much of the cross-seed instability is
"real" (different trained solutions genuinely differ) versus a solver
artifact that elastic net's tie-breaking would fix regardless of seed.
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

OUTPUT_DIR = Path("outputs/trained_collinearity")
SEEDS = [0, 1, 2, 3, 4]
TRAINING_STEPS = 400
ALPHA = 0.3
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
    history, _ = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    final_train_loss = history["train_loss"][-1] if history.get("train_loss") else None
    print(f"seed={seed}  final_train_loss={final_train_loss}")
    return cfg, model, device, dtype


def jacobian_and_target(model, cfg, device, dtype, q1_pts, q2_pts, alpha=ALPHA):
    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_pts, q2_pts, alpha, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    n_points = q1_pts.shape[0]
    jac_flat = j_x.reshape(n_points * 2, -1)
    xrot_target = rotation_generator_target(q1_pts, q2_pts, f_x, spatial_jac_x).reshape(n_points * 2)
    return jac_flat, xrot_target


def top_support(c: torch.Tensor, k: int) -> frozenset[int]:
    return frozenset(torch.argsort(c.abs(), descending=True)[:k].tolist())


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(supports: list[frozenset]) -> float:
    pairs = combinations(range(len(supports)), 2)
    return float(np.mean([jaccard(supports[i], supports[j]) for i, j in pairs]))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Experiment 1: cross-seed instability on TRAINED models.
    # ---------------------------------------------------------------
    per_seed = []
    for seed in SEEDS:
        cfg, model, device, dtype = train_one_seed(seed)
        q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
        jac_flat, xrot_target = jacobian_and_target(model, cfg, device, dtype, q1_grid, q2_grid)
        _, _, c_l2, rank, _ = tangent_projection(jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff)
        _, _, c_l1, _, _ = tangent_projection_l1(jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff)
        _, _, c_en, _, _ = tangent_projection_elastic_net(
            jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff, l1_ratio=0.5
        )
        per_seed.append({
            "seed": seed, "cfg": cfg, "model": model, "device": device, "dtype": dtype,
            "rank": rank, "c_l2": c_l2, "c_l1": c_l1, "c_en": c_en,
            "q1_grid": q1_grid, "q2_grid": q2_grid,
        })

    ranks = {d["rank"] for d in per_seed}
    print("\nresolved ranks across trained seeds:", sorted(ranks))
    k = min(ranks)

    cross_seed_stability = {
        "L2": mean_pairwise_jaccard([top_support(d["c_l2"], k) for d in per_seed]),
        "L1": mean_pairwise_jaccard([top_support(d["c_l1"], k) for d in per_seed]),
        "EN(0.5)": mean_pairwise_jaccard([top_support(d["c_en"], k) for d in per_seed]),
    }
    print(f"\nCross-seed support stability (top-{k}, {len(SEEDS)} trained seeds):")
    for label, val in cross_seed_stability.items():
        print(f"  {label}: {val:.3f}")

    # ---------------------------------------------------------------
    # Experiment 2: fixed-model probe-bootstrap instability.
    # ---------------------------------------------------------------
    ref = per_seed[0]
    cfg, model, device, dtype = ref["cfg"], ref["model"], ref["device"], ref["dtype"]
    q1_grid, q2_grid = ref["q1_grid"], ref["q2_grid"]
    n_probe = q1_grid.shape[0]
    rng = np.random.default_rng(12345)

    boot_supports = {"L2": [], "L1": [], "EN(0.5)": []}
    boot_ranks = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_probe, size=n_probe)
        idx_t = torch.as_tensor(idx, device=q1_grid.device)
        q1_bs, q2_bs = q1_grid[idx_t], q2_grid[idx_t]
        jac_flat_bs, xrot_target_bs = jacobian_and_target(model, cfg, device, dtype, q1_bs, q2_bs)
        _, _, c_l2_bs, rank_bs, _ = tangent_projection(jac_flat_bs, xrot_target_bs, cfg.tangent_svd_relative_cutoff)
        _, _, c_l1_bs, _, _ = tangent_projection_l1(jac_flat_bs, xrot_target_bs, cfg.tangent_svd_relative_cutoff)
        _, _, c_en_bs, _, _ = tangent_projection_elastic_net(
            jac_flat_bs, xrot_target_bs, cfg.tangent_svd_relative_cutoff, l1_ratio=0.5
        )
        boot_ranks.append(rank_bs)
        boot_supports["L2"].append(top_support(c_l2_bs, k))
        boot_supports["L1"].append(top_support(c_l1_bs, k))
        boot_supports["EN(0.5)"].append(top_support(c_en_bs, k))

    print(f"\nBootstrap resolved ranks (fixed model, seed={ref['seed']}): "
          f"min={min(boot_ranks)} max={max(boot_ranks)} (top-{k} support used throughout)")
    within_model_stability = {
        label: mean_pairwise_jaccard(supports) for label, supports in boot_supports.items()
    }
    print(f"\nWithin-model (probe-bootstrap, {N_BOOTSTRAP} resamples, seed={ref['seed']} fixed) support stability:")
    for label, val in within_model_stability.items():
        print(f"  {label}: {val:.3f}")

    print("\n=== Decomposition ===")
    for label in ("L2", "L1", "EN(0.5)"):
        cs, wm = cross_seed_stability[label], within_model_stability[label]
        print(f"  {label}: cross-seed={cs:.3f}  within-model={wm:.3f}  "
              f"gap (genuine-variability signal)={cs - wm:+.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    labels = ["L2", "L1", "EN(0.5)"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [cross_seed_stability[l] for l in labels], width, label=f"cross-seed ({len(SEEDS)} trained models)")
    ax.bar(x + width / 2, [within_model_stability[l] for l in labels], width, label=f"within-model ({N_BOOTSTRAP} probe bootstraps, fixed model)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"mean pairwise Jaccard overlap of top-{k} support")
    ax.set_title("Cross-seed vs. within-model attribution-support stability (trained models)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_seed_vs_within_model_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Near-collinear-pair diagnostic on the fixed trained model.
    # ---------------------------------------------------------------
    flat_names, _ = parameter_layout(model)
    jac_flat, xrot_target = jacobian_and_target(model, cfg, device, dtype, q1_grid, q2_grid)
    j = jac_flat.detach().cpu().to(torch.float64)
    g = xrot_target.detach().cpu().to(torch.float64)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, sv, col_scale = _resolve_tangent_svd(
        j, g, cfg.tangent_svd_relative_cutoff
    )
    a_eq = (vh_kept / col_scale).numpy()
    norms = np.linalg.norm(a_eq, axis=0)
    nz = np.where(norms > 1e-12)[0]
    a_unit = a_eq[:, nz] / norms[nz]
    cos = a_unit.T @ a_unit
    np.fill_diagonal(cos, 0)
    flat_idx = np.argsort(-np.abs(cos), axis=None)[:40]
    seen = set()
    c_l2, c_l1, c_en = ref["c_l2"], ref["c_l1"], ref["c_en"]
    print(f"\nTop near-collinear parameter pairs on trained seed={ref['seed']} model (cosine in resolved subspace):")
    printed = 0
    for fi in flat_idx:
        i, jx = np.unravel_index(fi, cos.shape)
        if i >= jx or (i, jx) in seen:
            continue
        seen.add((i, jx))
        gi, gj = nz[i], nz[jx]
        print(
            f"  ({flat_names[gi]}, {flat_names[gj]})  cos={cos[i, jx]:.5f}  "
            f"L2=({c_l2[gi]:.4f},{c_l2[gj]:.4f})  L1=({c_l1[gi]:.4f},{c_l1[gj]:.4f})  "
            f"EN0.5=({c_en[gi]:.4f},{c_en[gj]:.4f})"
        )
        printed += 1
        if printed >= 10:
            break

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
