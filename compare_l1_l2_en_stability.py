"""Multi-seed comparison of min-2-norm, min-1-norm, and elastic-net c_i.

Answers two methodological questions about tangent_projection_l1's basis
dependence:

1. Given multiple random seeds, is elastic net's collinearity-aware tie
   breaking actually needed, or does plain L1 already give a stable support
   across seeds? Measured via mean pairwise Jaccard overlap of the top
   resolved-rank support across seeds, as a function of l1_ratio (0 = pure
   min-2-norm, 1 = pure min-1-norm).
2. If elastic net helps, how should l1_ratio be chosen in a principled way?
   Because tangent_projection's equality constraint forces an exact
   reconstruction of the target projection (no fit-quality/sparsity
   tradeoff), l1_ratio only controls tie-breaking among (near-)collinear
   columns -- so it is chosen here via a stability-selection criterion
   (Meinshausen & Buhlmann, 2010): the smallest l1_ratio whose support
   stability is already within 5% of the best achievable stability, i.e.
   the least amount of ridge regularization needed for a reproducible
   answer.

Also plots the requested 3-way distribution comparison (L1 vs L2 vs the
chosen elastic net) for one representative seed.
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
    rotation_generator_target,
)
from experiment_common import parameter_layout, select_device, select_dtype
from sensitivity_tools import tangent_projection, tangent_projection_elastic_net, tangent_projection_l1

OUTPUT_DIR = Path("outputs/l1_l2_en_stability")
SEEDS = list(range(8))
L1_RATIOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]


def build_jacobian_and_target(seed: int, cfg: Config, device, dtype, alpha: float = 0.3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, _ = build_model(cfg, device, dtype)
    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    n_points = q1_grid.shape[0]
    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_grid, q2_grid, alpha, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    jac_flat = j_x.reshape(n_points * 2, -1)
    xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x).reshape(n_points * 2)
    return jac_flat, xrot_target, model


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
    cfg = Config()
    cfg.kinetic_hidden_dim = 8
    cfg.potential_hidden_dim = 8
    cfg.q_grid_points_per_axis = 4
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)

    per_seed = []
    for seed in SEEDS:
        jac_flat, xrot_target, model = build_jacobian_and_target(seed, cfg, device, dtype)
        _, _, c_l2, rank, _ = tangent_projection(jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff)
        _, _, c_l1, _, _ = tangent_projection_l1(jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff)
        en_by_ratio = {0.0: c_l2, 1.0: c_l1}
        for ratio in L1_RATIOS:
            if ratio in en_by_ratio:
                continue
            _, _, c_en, _, _ = tangent_projection_elastic_net(
                jac_flat, xrot_target, cfg.tangent_svd_relative_cutoff, l1_ratio=ratio,
            )
            en_by_ratio[ratio] = c_en
        per_seed.append({"seed": seed, "rank": rank, "en": en_by_ratio, "model": model})
        print(f"seed={seed}  resolved_rank={rank}")

    ranks = {d["rank"] for d in per_seed}
    print("resolved ranks across seeds:", sorted(ranks))
    k = min(ranks)

    stability_by_ratio = [
        mean_pairwise_jaccard([top_support(d["en"][ratio], k) for d in per_seed])
        for ratio in L1_RATIOS
    ]
    print(f"\nStability (mean pairwise Jaccard of top-{k} support across {len(SEEDS)} seeds):")
    for ratio, stab in zip(L1_RATIOS, stability_by_ratio):
        print(f"  l1_ratio={ratio:.2f}  stability={stab:.3f}")

    best_stability = max(stability_by_ratio)
    target = 0.95 * best_stability
    chosen_ratio = next(
        r for r, s in zip(L1_RATIOS, stability_by_ratio) if s >= target
    )
    print(f"\nBest achievable stability: {best_stability:.3f} (at l1_ratio={L1_RATIOS[int(np.argmax(stability_by_ratio))]})")
    print(f"Principled choice (smallest ratio within 95% of best): l1_ratio={chosen_ratio}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(L1_RATIOS, stability_by_ratio, marker="o")
    ax.axvline(chosen_ratio, color="gray", linestyle="--", linewidth=1, label=f"chosen l1_ratio={chosen_ratio}")
    ax.set_xlabel("l1_ratio  (0 = min 2-norm, 1 = min 1-norm)")
    ax.set_ylabel(f"mean pairwise Jaccard overlap\nof top-{k} support across {len(SEEDS)} seeds")
    ax.set_title("Support stability across random seeds vs. elastic-net mixing ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "stability_vs_l1_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 3-way distribution comparison for one representative seed ---
    ref = per_seed[0]
    _, parameter_slices = parameter_layout(ref["model"])
    c_l2, c_l1, c_en = ref["en"][0.0], ref["en"][1.0], ref["en"][chosen_ratio]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, label in [(c_l2, "min 2-norm"), (c_l1, "min 1-norm"), (c_en, f"elastic net (l1_ratio={chosen_ratio})")]:
        sorted_c = np.sort(np.abs(c.numpy()))[::-1]
        axes[0].semilogy(np.arange(1, len(sorted_c) + 1), sorted_c + 1e-30, label=label)
    axes[0].axvline(ref["rank"], color="gray", linestyle="--", linewidth=1, label=f"resolved rank ({ref['rank']})")
    axes[0].set_xlabel("parameter rank (sorted by |c_i|)")
    axes[0].set_ylabel("|c_i| (log scale)")
    axes[0].set_title(f"seed={ref['seed']}: sorted attribution magnitude")
    axes[0].legend(fontsize=8)

    names = list(parameter_slices.keys())
    x = np.arange(len(names))
    width = 0.27
    for offset, (c, label) in zip(
        (-width, 0, width),
        [(c_l2, "min 2-norm"), (c_l1, "min 1-norm"), (c_en, f"elastic net (l1_ratio={chosen_ratio})")],
    ):
        c_np = c.numpy()
        agg = [float(np.linalg.norm(c_np[sl])) for sl in parameter_slices.values()]
        axes[1].bar(x + offset, agg, width, label=label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=75, ha="right", fontsize=6)
    axes[1].set_ylabel("||c_i|| over module")
    axes[1].set_title(f"seed={ref['seed']}: per-module attribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l1_l2_en_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for label, c in [("L2", c_l2), ("L1", c_l1), ("EN", c_en)]:
        c_np = c.numpy()
        n_eff = (np.abs(c_np).sum() ** 2) / (c_np**2).sum()
        print(f"{label}: n_eff={n_eff:.2f}  max|c_i|={np.abs(c_np).max():.4f}")

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
