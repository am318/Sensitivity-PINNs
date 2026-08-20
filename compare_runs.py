"""Side-by-side headline table across runs -- the matched-convergence contrast.

The point of running an augmented (rotation-consistent) and a plain model at the
same architecture, optimiser and step budget is to vary equivariance quality
while holding convergence and conditioning fixed. Any statistic that moves with
equivariance is about the symmetry; any statistic that does not is about
training. This prints both models' headline numbers next to each other so that
comparison can actually be read off.

Usage: python compare_runs.py outputs/<run_a> outputs/<run_b> ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from attribution_stability import jaccard, scores_for
from gauge_dependence import top_set, weighted_min_norm
from sensitivity_tools import choose_l1_ratio_for_sparsity, tangent_projection

CUTOFF = 1e-3


def headline(run_dir: Path) -> dict:
    d = np.load(run_dir / "analysis_inputs.npz")
    summary = np.load(run_dir / "causal_symmetry_control.npz", allow_pickle=True)
    j_np, g_np = d["j"], d["g"]
    nulls = [d[k] for k in d.files if k.startswith("null_target_")]
    j, g = torch.from_numpy(j_np), torch.from_numpy(g_np)
    colnorm = j.norm(dim=0)
    live = (colnorm > colnorm.max() * 1e-8).numpy()

    out = {
        "equivariance residual": float(summary["equivariance_residual"]),
        "||delta||/||f||": float(summary["defect_ratio"]),
        "trajectory loss": float(summary["final_loss"]),
    }

    # conditioning share of the |c_i| ranking
    _, _, c_cn, rank, _ = tangent_projection(j, g, CUTOFF, normalize_columns=True)
    a = c_cn.numpy() * colnorm.numpy() ** 2
    m = (np.abs(c_cn.numpy()) > 0) & live
    lc, ls = np.log(np.abs(c_cn.numpy()[m])), np.log(colnorm.numpy()[m])
    des = np.stack([ls, np.ones_like(ls)], 1)
    out["resolved rank"] = float(rank)
    out["R2(log|c| ~ log||S_i||)"] = 1.0 - float(
        np.var(lc - des @ np.linalg.lstsq(des, lc, rcond=None)[0]) / max(np.var(lc), 1e-30))

    # gauge disagreement
    c1, _, _ = weighted_min_norm(j, g, torch.ones_like(colnorm))
    c2, _, _ = weighted_min_norm(j, g, colnorm.clamp_min(colnorm.max() * 1e-8))
    out["top-20 Jaccard, J^+g vs code default"] = jaccard(top_set(c1.numpy(), 20), top_set(c2.numpy(), 20))
    out["Spearman(|J^+g|, ||S_i||)"] = float(spearmanr(np.abs(c1.numpy()[live]), colnorm.numpy()[live]).statistic)
    out["Spearman(|c_colnorm|, ||S_i||)"] = float(spearmanr(np.abs(c2.numpy()[live]), colnorm.numpy()[live]).statistic)

    # identifiability under probe resampling
    l1r = float(choose_l1_ratio_for_sparsity(j, g, CUTOFF))
    rng = np.random.default_rng(0)
    n_rows = j_np.shape[0]
    keep = max(8, int(n_rows * 0.8))
    per: dict[str, list[np.ndarray]] = {}
    for _ in range(24):
        rows = rng.choice(n_rows, size=keep, replace=False)
        sc, _ = scores_for(j_np[rows], g_np[rows], [t[rows] for t in nulls], l1r)
        for k, v in sc.items():
            per.setdefault(k, []).append(v)
    for k, vs in per.items():
        tops = [np.argsort(-v)[:20] for v in vs]
        out[f"stability(top-20): {k}"] = float(np.mean(
            [jaccard(tops[a], tops[b]) for a in range(len(tops)) for b in range(a + 1, len(tops))]))
    return out


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        raise SystemExit("usage: python compare_runs.py <run_dir> [<run_dir> ...]")
    results = {d.name: headline(d) for d in dirs}
    keys = list(next(iter(results.values())))
    width = max(len(k) for k in keys) + 2
    print(f"{'':{width}}" + "".join(f"{n[-34:]:>36s}" for n in results))
    for k in keys:
        print(f"{k:{width}}" + "".join(f"{results[n][k]:>36.4f}" for n in results))


if __name__ == "__main__":
    main()
