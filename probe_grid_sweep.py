"""Does the identified symmetry subnetwork converge as the probe grid is refined?

The obvious objection to a probe-resampling instability result is that the grid
was simply too small: with enough probe points the Jacobian would be better
determined and the top-k parameter set would settle down. That is a testable
claim, and this tests it -- refining the grid (more radii, more angles) and
asking whether each score's top-k set converges toward its own value at the
finest grid.

Two outcomes, both informative:
  - converges -> the subnetwork is a real property of the network, and any
    instability at the working grid is a resolution artefact to be fixed by
    using more probe points (and the required resolution is measured here);
  - does not converge -> the instability is intrinsic, because the quantity is
    underdetermined rather than merely undersampled.

Reads a model.pt written by causal_symmetry_control.py, so no retraining.

Usage: python probe_grid_sweep.py outputs/<run_dir> [--topk 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config,
    build_model,
    evaluate_at_points,
    rotation_generator_target,
    validate_config,
)
from attribution_stability import jaccard, scores_for, unit_index_map, unit_vector
from causal_symmetry_control import build_polar_probe_grid
from experiment_common import select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    linear_generator_target,
    random_symmetry_test_matrix,
)

ALPHA = 0.3
N_NULLS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = Config()
    cfg.architecture, cfg.device, cfg.training_steps = blob["architecture"], "cpu", 1
    validate_config(cfg)
    device, dtype = select_device("cpu"), select_dtype(cfg.dtype)
    model, _ = build_model(cfg, device, dtype)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    units = unit_index_map(model)
    tgen = torch.Generator().manual_seed(0)
    cutoff = cfg.tangent_svd_relative_cutoff

    grids = [(4, 12), (5, 16), (7, 24), (9, 32), (11, 48), (14, 64)]
    print(f"{run_dir.name}: refining the probe grid, {len(grids)} resolutions")

    per_grid, sizes = [], []
    l1_ratio = None
    for n_radii, n_angles in grids:
        q1, q2 = build_polar_probe_grid(n_radii, n_angles, cfg.q_extent, device, dtype)
        n_points = q1.shape[0]
        _, f_x, j_x, spatial = evaluate_at_points(
            model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True)
        g = rotation_generator_target(q1, q2, f_x, spatial).reshape(n_points * 2)
        j_flat = j_x.reshape(n_points * 2, -1)
        positions = torch.stack([q1, q2], dim=1)
        nulls = [linear_generator_target(
            positions, f_x, spatial,
            random_symmetry_test_matrix(device=device, dtype=dtype, generator=tgen)).reshape(n_points * 2)
            for _ in range(N_NULLS)]
        if l1_ratio is None:
            l1_ratio = float(choose_l1_ratio_for_sparsity(j_flat, g, cutoff))
        j_np = j_flat.detach().cpu().to(torch.float64).numpy()
        sc, _ = scores_for(j_np, g.detach().cpu().to(torch.float64).numpy(),
                           [t.detach().cpu().to(torch.float64).numpy() for t in nulls], l1_ratio)
        per_grid.append(sc)
        sizes.append(n_points * 2)
        rank = int(np.linalg.matrix_rank(j_np, tol=cutoff * np.linalg.norm(j_np, 2)))
        print(f"  grid {n_radii:2d}x{n_angles:2d} -> {n_points:4d} points, {n_points * 2:4d} rows, rank {rank}")

    finest = per_grid[-1]
    names = list(finest)
    n_top_u = max(3, len(units) // 10)
    print(f"\n-- top-{args.topk} parameter overlap with the finest grid ({sizes[-1]} rows) --")
    header = "  " + " " * 34 + "".join(f"{s:>7d}" for s in sizes)
    print(header)
    curves = {}
    for name in names:
        ref = np.argsort(-finest[name])[: args.topk]
        vals = [jaccard(np.argsort(-per_grid[i][name])[: args.topk], ref) for i in range(len(grids))]
        curves[name] = vals
        print(f"  {name:34s}" + "".join(f"{v:7.2f}" for v in vals))

    print(f"\n-- top-{n_top_u} HIDDEN-UNIT overlap with the finest grid --")
    print(header)
    unit_curves = {}
    for name in names:
        ref = np.argsort(-unit_vector(finest[name], units))[:n_top_u]
        vals = [jaccard(np.argsort(-unit_vector(per_grid[i][name], units))[:n_top_u], ref)
                for i in range(len(grids))]
        unit_curves[name] = vals
        print(f"  {name:34s}" + "".join(f"{v:7.2f}" for v in vals))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, curveset, label in ((axes[0], curves, f"top-{args.topk} parameters"),
                                (axes[1], unit_curves, f"top-{n_top_u} hidden units")):
        for name, vals in curveset.items():
            style = "--" if "CONTROL" in name or "reference" in name else "-"
            ax.plot(sizes, vals, style, marker="o", markersize=4, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("probe rows (2 x probe points)")
        ax.set_ylabel(f"overlap of {label} with finest grid")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        ax.set_title(f"Convergence under grid refinement: {label}", fontsize=9)
    fig.tight_layout()
    out = run_dir / "probe_grid_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
