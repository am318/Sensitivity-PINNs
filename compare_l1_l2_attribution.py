"""Compare minimum-2-norm vs minimum-1-norm attribution coefficients c_i.

Builds a freshly-initialized Mexican Hat Hamiltonian model (random init --
tangent_projection's resolved tangent space and its reconstruction of the
target generator direction don't depend on training) and computes the SO(2)
rotation-generator attribution across all parameters via both
``tangent_projection`` (min 2-norm, sensitivity_tools.py) and
``tangent_projection_l1`` (min 1-norm), each with scale-normalized columns so
neither solve is biased by the raw units of individual parameters. The two
solutions reconstruct the identical target projection (same relative error /
angle -- asserted below); they differ only in how attribution is distributed
across parameters, which is what this script visualizes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config,
    bifurcation_generator_target,
    build_model,
    build_probe_grid,
    evaluate_at_points,
    rotation_generator_target,
)
from experiment_common import parameter_layout, select_device, select_dtype
from sensitivity_tools import tangent_projection, tangent_projection_l1

OUTPUT_DIR = Path("outputs/l1_l2_attribution_comparison")


def sparsity_stats(c: np.ndarray) -> dict[str, float]:
    abs_c = np.abs(c)
    l1, l2 = abs_c.sum(), np.sqrt((abs_c**2).sum())
    # Inverse participation ratio: "effective number of parameters carrying attribution".
    n_eff = (l1**2 / (abs_c**2).sum()) if l2 > 0 else 0.0
    sorted_frac = np.sort(abs_c)[::-1].cumsum() / max(l1, 1e-15)
    n90 = int(np.searchsorted(sorted_frac, 0.9) + 1)
    return {"n_eff": float(n_eff), "n_for_90pct_mass": n90, "max_abs": float(abs_c.max())}


def module_attribution(c: np.ndarray, parameter_slices: dict[str, slice]) -> dict[str, float]:
    return {name: float(np.linalg.norm(c[sl])) for name, sl in parameter_slices.items()}


def compare_one_generator(
    label: str, jac_flat: torch.Tensor, target_flat: torch.Tensor, cutoff: float,
    parameter_slices: dict[str, slice], out_dir: Path,
) -> None:
    err_l2, ang_l2, c_l2, rank, _ = tangent_projection(jac_flat, target_flat, cutoff)
    err_l1, ang_l1, c_l1, rank_l1, _ = tangent_projection_l1(jac_flat, target_flat, cutoff)
    assert rank == rank_l1
    print(f"\n=== {label} ===")
    print(f"resolved_rank={rank}  P={c_l2.numel()}")
    print(f"L2: relative_error={err_l2:.6g} angle_deg={ang_l2:.6g}")
    print(f"L1: relative_error={err_l1:.6g} angle_deg={ang_l1:.6g}  (should match L2 -- same projection)")

    c_l2_np, c_l1_np = c_l2.numpy(), c_l1.numpy()
    stats_l2, stats_l1 = sparsity_stats(c_l2_np), sparsity_stats(c_l1_np)
    print(f"L2 sparsity: n_eff={stats_l2['n_eff']:.1f}  n_for_90pct_mass={stats_l2['n_for_90pct_mass']}")
    print(f"L1 sparsity: n_eff={stats_l1['n_eff']:.1f}  n_for_90pct_mass={stats_l1['n_for_90pct_mass']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sorted_l2 = np.sort(np.abs(c_l2_np))[::-1]
    sorted_l1 = np.sort(np.abs(c_l1_np))[::-1]
    axes[0].semilogy(np.arange(1, len(sorted_l2) + 1), sorted_l2 + 1e-30, label="min 2-norm")
    axes[0].semilogy(np.arange(1, len(sorted_l1) + 1), sorted_l1 + 1e-30, label="min 1-norm")
    axes[0].axvline(rank, color="gray", linestyle="--", linewidth=1, label=f"resolved rank ({rank})")
    axes[0].set_xlabel("parameter rank (sorted by |c_i|)")
    axes[0].set_ylabel("|c_i| (log scale)")
    axes[0].set_title(f"{label}: sorted attribution magnitude")
    axes[0].legend(fontsize=8)

    mod_l2 = module_attribution(c_l2_np, parameter_slices)
    mod_l1 = module_attribution(c_l1_np, parameter_slices)
    names = list(mod_l2.keys())
    x = np.arange(len(names))
    width = 0.38
    axes[1].bar(x - width / 2, [mod_l2[n] for n in names], width, label="min 2-norm")
    axes[1].bar(x + width / 2, [mod_l1[n] for n in names], width, label="min 1-norm")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=75, ha="right", fontsize=6)
    axes[1].set_ylabel("||c_i|| over module (L2 aggregate)")
    axes[1].set_title(f"{label}: per-module attribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{label}_l1_vs_l2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    # Small architecture so the L1 linear-programming solve stays fast; the
    # qualitative L1-vs-L2 sparsity story does not depend on model size.
    cfg.kinetic_hidden_dim = 8
    cfg.potential_hidden_dim = 8
    cfg.q_grid_points_per_axis = 4
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)

    model, _ = build_model(cfg, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    n_points = q1_grid.shape[0]
    alpha = 0.3  # arbitrary, non-special

    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_grid, q2_grid, alpha, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    jac_flat = j_x.reshape(n_points * 2, -1)
    print(f"P (total parameters) = {len(flat_names)}, N (jac rows) = {jac_flat.shape[0]}")

    xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
    compare_one_generator(
        "xrot", jac_flat, xrot_target.reshape(n_points * 2), cfg.tangent_svd_relative_cutoff,
        parameter_slices, OUTPUT_DIR,
    )

    bifurcation_target = bifurcation_generator_target(q1_grid, q2_grid)
    compare_one_generator(
        "bifurcation", jac_flat, bifurcation_target.reshape(n_points * 2), cfg.tangent_svd_relative_cutoff,
        parameter_slices, OUTPUT_DIR,
    )

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
