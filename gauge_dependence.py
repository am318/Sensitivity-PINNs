"""Is "which parameters realise the symmetry" determined by the geometry, or by a norm choice?

J_theta is heavily rank-deficient, so J_theta c = g has a solution set of
dimension P - r, with r the resolved rank and P the parameter count -- in these
models an affine subspace of dimension in the thousands. Every point in it
reproduces the *same* function-space direction P_T g exactly. Picking one
requires minimising something over that set, and which point you land on
depends entirely on what you minimise:

    c(w) = argmin { ||w . c||_2 : V_r c = alpha }        (a diagonal metric w)
    c_L1 = argmin { ||c||_1    : V_r c = alpha }         (a different geometry entirely)

The draft's c* = J^+ g is the choice w = 1; this project's code default is
w = ||S_i|| (column normalisation), motivated by the fact that w = 1 makes a
parameter with naturally small gradients look "cheap" purely because of units.
Both are defensible. They are also different vectors, and there is nothing in
the data that selects between them.

That makes the identity of the top-|c_i| parameters gauge-dependent, while the
function-space content P_T g is gauge-invariant. This script measures the size
of that gap: it verifies P_T g agrees across gauges to solver tolerance, then
reports how much the top-k parameter set, the ranking, and the module-level
profile move between gauges -- including under random diagonal metrics, which
are pure changes of parameter units and should not change a claim about which
parameters matter, if that claim is about the network rather than the chart.

Usage: python gauge_dependence.py <analysis_inputs.npz> [--topk 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from asrnn_mexican_hat_symmetry_sensitivity import Config, build_model, validate_config
from attribution_stability import unit_index_map, unit_vector
from experiment_common import parameter_layout, select_device, select_dtype
from sensitivity_tools import (
    _resolve_tangent_svd,
    _minimum_l1_norm_solution,
    choose_l1_ratio_for_sparsity,
    tangent_projection_auto,
)

CUTOFF = 1e-3


def weighted_min_norm(j: torch.Tensor, g: torch.Tensor, w: torch.Tensor, cutoff: float = CUTOFF):
    """argmin ||w . c||_2 subject to reproducing P_T g -- the whole family of gauges.

    w = 1 is the textbook Moore-Penrose c* = J^+ g; w = ||S_i|| is column
    normalisation; a random positive w is a pure change of parameter units.
    """
    _, s_kept, vh_kept, coordinates, projection, rank, _, _, dead = _resolve_tangent_svd(j, g, cutoff)
    if rank == 0:
        return torch.zeros(j.shape[1], dtype=torch.float64), projection, 0
    alpha = coordinates / s_kept
    a = vh_kept / w
    if dead.any():
        a = a.clone()
        a[:, dead] = 0.0
    c_tilde = a.T @ torch.linalg.solve(a @ a.T, alpha)
    return c_tilde / w, projection, rank


def top_set(c: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-np.abs(c))[:k].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3, help="random diagonal metrics to sample")
    args = ap.parse_args()

    d = np.load(Path(args.npz))
    j = torch.from_numpy(d["j"])
    g = torch.from_numpy(d["g"])
    colnorm = j.norm(dim=0)
    live = (colnorm > colnorm.max() * 1e-8)
    ones = torch.ones_like(colnorm)

    gauges: dict[str, torch.Tensor] = {}
    projections: dict[str, torch.Tensor] = {}

    c, proj, rank = weighted_min_norm(j, g, ones)
    gauges["L2, w=1  (the draft's J^+ g)"], projections["L2, w=1  (the draft's J^+ g)"] = c, proj
    c, proj, _ = weighted_min_norm(j, g, colnorm.clamp_min(colnorm.max() * 1e-8))
    gauges["L2, w=||S_i||  (code default)"], projections["L2, w=||S_i||  (code default)"] = c, proj
    c, proj, _ = weighted_min_norm(j, g, colnorm.clamp_min(colnorm.max() * 1e-8) ** 0.5)
    gauges["L2, w=||S_i||^{1/2}"], projections["L2, w=||S_i||^{1/2}"] = c, proj

    gen = torch.Generator().manual_seed(0)
    for si in range(args.seeds):
        # a pure change of parameter units: log-uniform over two decades
        w = torch.exp((torch.rand(j.shape[1], generator=gen, dtype=torch.float64) - 0.5) * 2 * np.log(10))
        c, proj, _ = weighted_min_norm(j, g, w)
        gauges[f"L2, random units #{si + 1}"], projections[f"L2, random units #{si + 1}"] = c, proj

    err, _, c_en, _, _ = tangent_projection_auto(
        j, g, CUTOFF, method="elastic_net",
        l1_ratio=choose_l1_ratio_for_sparsity(j, g, CUTOFF))
    gauges["elastic net (code default method)"] = c_en.to(torch.float64)
    err, _, c_l1, _, _ = tangent_projection_auto(j, g, CUTOFF, method="l1")
    gauges["L1"] = c_l1.to(torch.float64)

    print(f"J {tuple(j.shape)}, resolved rank {rank}, "
          f"solution set dimension {j.shape[1] - rank} (every point reproduces the SAME P_T g)")

    ref = projections["L2, w=1  (the draft's J^+ g)"]
    print("\n-- function-space content is gauge-invariant --")
    for name, c in gauges.items():
        recon = (j @ c.to(j.dtype))
        rel = float((recon - ref).norm() / ref.norm().clamp_min(1e-30))
        print(f"   {name:38s} ||J c - P_T g|| / ||P_T g|| = {rel:.3e}")

    # The review notes propose atilde_i = c_i <S_i,g> / ||P_T g||^2 as a scale-invariant
    # fix. It IS invariant under a diagonal reparametrisation theta_i -> lambda_i theta_i
    # (c_i -> lambda_i c_i and <S_i,g> -> <S_i,g>/lambda_i cancel), which is what the notes
    # claim for it. But that is a different question from gauge choice: different gauges
    # select genuinely different points of the solution set, not reparametrisations of one
    # point, so atilde inherits the ambiguity. Measured here rather than assumed.
    jg = (j.to(torch.float64).T @ g.to(torch.float64))
    atilde = {}
    for name, c in gauges.items():
        pt = j.to(torch.float64) @ c
        atilde[name] = (c * jg / max(float(pt @ pt), 1e-30)).numpy()

    # The dilemma made explicit. Whichever gauge is defended, it commits to a
    # systematic preference along the conditioning axis: w = 1 charges nothing for
    # moving a high-sensitivity parameter and so favours them; w = ||S_i|| gives
    # c_i = a_i/||S_i||^2 and so favours the LEAST sensitive parameters -- an odd
    # thing to call "the parameters that realise the symmetry". There is no gauge
    # that is neutral, because the ranking has to be taken in some units.
    print("\n-- every gauge commits to a preference along the conditioning axis --")
    cn_live = colnorm.numpy()[live.numpy()]
    for name, c in gauges.items():
        rho_s = float(spearmanr(np.abs(c.numpy()[live.numpy()]), cn_live).statistic)
        print(f"   {name:38s} Spearman(|c_i|, ||S_i||) = {rho_s:+.3f}")

    names = list(gauges)
    n = len(names)
    jac = np.zeros((n, n))
    rho = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            ca, cb = gauges[names[a]].numpy(), gauges[names[b]].numpy()
            jac[a, b] = jaccard(top_set(ca, args.topk), top_set(cb, args.topk))
            rho[a, b] = float(spearmanr(np.abs(ca[live.numpy()]), np.abs(cb[live.numpy()])).statistic)

    print(f"\n-- but the parameter-level answer is NOT: top-{args.topk} overlap between gauges --")
    print("   " + " " * 38 + "".join(f"{i + 1:>7d}" for i in range(n)))
    for a in range(n):
        print(f"   {a + 1:2d} {names[a]:35s}" + "".join(f"{jac[a, b]:7.2f}" for b in range(n)))
    off = jac[~np.eye(n, dtype=bool)]
    print(f"\n   mean off-diagonal top-{args.topk} Jaccard = {off.mean():.3f}  "
          f"(1.0 would mean the gauge choice does not matter)")
    off_rho = rho[~np.eye(n, dtype=bool)]
    print(f"   mean off-diagonal Spearman            = {off_rho.mean():.3f}")

    jac_a = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            jac_a[a, b] = jaccard(top_set(atilde[names[a]], args.topk),
                                  top_set(atilde[names[b]], args.topk))
    off_a = jac_a[~np.eye(n, dtype=bool)]
    print(f"\n-- and the proposed scale-invariant fix does not repair this --")
    print(f"   mean off-diagonal top-{args.topk} Jaccard for atilde_i = {off_a.mean():.3f}")
    print(f"   (atilde_i is invariant under theta_i -> lambda_i theta_i, but that is "
          f"reparametrisation,\n    not gauge: it does not pick a point out of the "
          f"{j.shape[1] - rank}-dimensional solution set.)")

    # ---- the constructive half: which granularity, if any, IS gauge-invariant? ----
    cfg = Config()
    cfg.architecture, cfg.device, cfg.training_steps = "hamiltonian", "cpu", 1
    validate_config(cfg)
    model, _ = build_model(cfg, select_device("cpu"), select_dtype(cfg.dtype))
    _, slices = parameter_layout(model)
    units = unit_index_map(model)
    n_params = j.shape[1]
    coarse_rho = {}
    if sum(sl.stop - sl.start for sl in slices.values()) == n_params:
        mod = {nm: np.array([np.linalg.norm(np.abs(gauges[nm].numpy())[sl]) for sl in slices.values()])
               for nm in names}
        uni = {nm: unit_vector(np.abs(gauges[nm].numpy()), units) for nm in names}
        for label, table in (("module", mod), ("hidden unit", uni)):
            vals = [float(spearmanr(table[names[a]], table[names[b]]).statistic)
                    for a in range(n) for b in range(a + 1, n)]
            coarse_rho[label] = float(np.nanmean(vals))
        n_top_u = max(3, len(units) // 10)
        ujac = float(np.mean([jaccard(set(np.argsort(-uni[names[a]])[:n_top_u].tolist()),
                                      set(np.argsort(-uni[names[b]])[:n_top_u].tolist()))
                              for a in range(n) for b in range(a + 1, n)]))
        print("\n-- does a coarser, permutation-safe granularity survive the gauge choice? --")
        print(f"   mean cross-gauge Spearman, module level      = {coarse_rho['module']:+.3f}")
        print(f"   mean cross-gauge Spearman, hidden-unit level = {coarse_rho['hidden unit']:+.3f}")
        print(f"   mean cross-gauge top-{n_top_u} hidden-unit Jaccard   = {ujac:.3f}")
        print(f"   (compare parameter level: Spearman {off_rho.mean():+.3f}, "
              f"top-{args.topk} Jaccard {off.mean():.3f})")
    else:
        print("\n   (module/unit layout does not match this J; skipping the granularity comparison)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(jac, vmin=0, vmax=1, cmap="magma")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(range(1, n + 1))
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{i + 1}. {nm}" for i, nm in enumerate(names)], fontsize=7)
    axes[0].set_title(f"top-{args.topk} parameter-set overlap between gauges\n"
                      f"(all reproduce the identical function-space direction)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    for name in names:
        c = np.abs(gauges[name].numpy())
        c = np.sort(c[live.numpy()])[::-1]
        axes[1].plot(np.arange(1, len(c) + 1), np.clip(c, 1e-300, None) / max(c.max(), 1e-300),
                     linewidth=1.3, label=name)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("|c_i| / max |c_i|")
    axes[1].set_title("Attribution profile under different gauges", fontsize=9)
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.npz).parent / "gauge_dependence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
