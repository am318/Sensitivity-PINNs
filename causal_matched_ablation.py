"""Does a score's selection control the symmetry beyond what its sensitivity profile explains?

Test B in causal_symmetry_control.py perturbs the top-k parameters of each score
and compares the induced change in the symmetry defect D = ||delta||^2/||f||^2
against the change in task loss. Two things that run cannot separate:

1. A large selectivity ratio dlogD/dlogL is not by itself evidence: if a score
   happens to select near-dead parameters, both changes are ~0 and their ratio
   is noise. (Observed: random selection scored a selectivity of 5.7 while
   moving D less than the conditioning baseline did.)

2. Scores differ enormously in the sensitivity of the parameters they pick --
   that is the whole content of the conditioning confound -- so a score that
   moves D a lot may be doing nothing except selecting high-||S_i|| parameters.

The fix is a **sensitivity-matched control**: for each selected parameter, draw
a replacement from the same decile of log||S_i|| among the live parameters. The
control therefore has, by construction, the same conditioning profile and the
same number of parameters, and differs only in *which* parameters within that
profile were chosen -- which is exactly the symmetry-specific content the claim
is about. The quantity of interest is the excess

    dlogD(score)  -  dlogD(sensitivity-matched control),

the causal analogue of the partial correlation controlling for log||S_i||.

Perturbation sizes are swept, because a global norm of 0.05||theta|| concentrated
on 20 of ~4500 parameters is far outside any linear regime (it drove dlogD to
+7, i.e. the defect up by a factor of ~1000).

Runs off the saved model.pt -- no retraining.

Usage: python causal_matched_ablation.py outputs/<run_dir> [--topk 20]
"""

from __future__ import annotations

import argparse
import os
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
    make_dataset,
    residuals,
    rotation_generator_target,
    validate_config,
)
from causal_symmetry_control import build_polar_probe_grid, defect_order_parameter
from experiment_common import select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    linear_generator_target,
    random_symmetry_test_matrix,
    tangent_projection,
    tangent_projection_auto,
)

ALPHA = 0.3
N_NULLS = 5
N_ANGLES = 24
EPSILONS = tuple(float(x) for x in os.environ.get("EPSILONS", "0.002,0.01").split(","))
N_DRAWS = 16
N_MATCHED = 16
N_NEIGHBOURS = 4


def matched_indices(support: np.ndarray, log_s: np.ndarray, live: np.ndarray,
                    rng: np.random.Generator, n_neighbours: int = N_NEIGHBOURS) -> np.ndarray:
    """Replace each selected parameter by a live one of nearly identical log||S_i||.

    Decile matching is not good enough here. Scores like ||S_i|| itself select the extreme
    tail of the distribution, and within the top decile ||S_i|| still varies by orders of
    magnitude -- so a decile-matched control is systematically *less* sensitive than the
    selection, and reports a large spurious "excess" for a score that by construction
    carries no symmetry-specific information at all. Matching to the nearest neighbours in
    log||S_i|| (excluding the selection itself, drawing among the closest few so the control
    still has variance) removes that bias.
    """
    live_idx = np.flatnonzero(live)
    order = live_idx[np.argsort(log_s[live_idx])]
    pos = {int(j): k for k, j in enumerate(order)}
    chosen = set(int(i) for i in support)
    out = []
    for i in support:
        k = pos.get(int(i))
        if k is None:
            out.append(i)
            continue
        lo, hi = max(0, k - n_neighbours), min(len(order), k + n_neighbours + 1)
        pool = [int(j) for j in order[lo:hi] if int(j) not in chosen]
        out.append(rng.choice(pool) if pool else int(i))
    return np.array(out, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = Config()
    cfg.architecture, cfg.device, cfg.training_steps = blob["architecture"], "cpu", 1
    cfg.augment_dataset = bool(blob.get("augment", False))
    validate_config(cfg)
    device, dtype = select_device("cpu"), select_dtype(cfg.dtype)
    torch.manual_seed(blob.get("seed", 0))
    np.random.seed(blob.get("seed", 0))
    train_data, _ = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    n_points = q1.shape[0]
    cutoff = cfg.tangent_svd_relative_cutoff
    _, f_x, j_x, spatial = evaluate_at_points(
        model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True)
    g = rotation_generator_target(q1, q2, f_x, spatial).reshape(n_points * 2)
    j_flat = j_x.reshape(n_points * 2, -1)
    positions = torch.stack([q1, q2], dim=1)
    tgen = torch.Generator().manual_seed(0)
    nulls = [linear_generator_target(
        positions, f_x, spatial,
        random_symmetry_test_matrix(device=device, dtype=dtype, generator=tgen)).reshape(n_points * 2)
        for _ in range(N_NULLS)]

    colnorm = j_flat.detach().cpu().to(torch.float64).norm(dim=0).numpy()
    live = colnorm > colnorm.max() * 1e-8
    log_s = np.log(np.clip(colnorm, 1e-300, None))

    def l2(t):
        _, _, c, _, _ = tangent_projection(j_flat, t, cutoff, normalize_columns=True)
        return c.numpy()

    c_l2 = l2(g)
    c_nulls = np.stack([l2(t) for t in nulls])
    calibrated = np.where(live, np.abs(c_l2) / np.clip(
        np.exp(np.log(np.clip(np.abs(c_nulls), 1e-300, None)).mean(0)), 1e-300, None), 0.0)
    ratio = choose_l1_ratio_for_sparsity(j_flat, g, cutoff)
    _, _, c_en, _, _ = tangent_projection_auto(j_flat, g, cutoff, method="elastic_net", l1_ratio=ratio)

    scores = {
        "c^L2 = |J^+ g| (paper's score)": np.abs(c_l2),
        "r_i (null-calibrated)": calibrated,
        "c^EN (gauge-free selection)": np.abs(c_en.numpy()),
        "||S_i|| (conditioning)": colnorm,
        "|J^T g| (naive alignment)": np.abs((j_flat.detach().cpu().to(torch.float64).T
                                             @ g.detach().cpu().to(torch.float64)).numpy()),
    }

    params = list(model.parameters())
    sizes = [p.numel() for p in params]
    flat0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    theta_norm = float(flat0.norm())

    def set_flat(v):
        off = 0
        with torch.no_grad():
            for p, n in zip(params, sizes):
                p.copy_(v[off:off + n].view_as(p))
                off += n

    def measure(support, eps, rng_t):
        d_l, l_l = [], []
        for _ in range(N_DRAWS):
            direction = torch.zeros_like(flat0)
            noise = torch.randn(len(support), generator=rng_t, dtype=torch.float32).to(flat0.dtype)
            direction[torch.from_numpy(np.asarray(support))] = noise
            direction = direction / direction.norm().clamp_min(1e-30) * (eps * theta_norm)
            set_flat(flat0 + direction)
            d_l.append(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
            l_l.append(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
        set_flat(flat0)
        return float(np.mean(np.log(np.clip(d_l, 1e-300, None)))), float(np.mean(np.log(np.clip(l_l, 1e-300, None))))

    set_flat(flat0)
    base_d = np.log(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
    base_l = np.log(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
    print(f"{run_dir.name}: baseline D={np.exp(base_d):.4e}  loss={np.exp(base_l):.4e}  "
          f"live params {int(live.sum())}/{len(colnorm)}  top-k={args.topk}")

    rng = np.random.default_rng(0)
    rng_t = torch.Generator().manual_seed(0)
    rows = []
    for eps in EPSILONS:
        print(f"\n=== perturbation norm {eps:.3g} * ||theta|| ===")
        print(f"  {'score':32s}{'dlogD':>9s}{'dlogL':>9s}{'matched dlogD':>15s}{'EXCESS':>9s}{'sel':>7s}")
        for name, sc in scores.items():
            support = np.argsort(-sc)[: args.topk]
            d_s, l_s = measure(support, eps, rng_t)
            m_d, m_gap = [], []
            for _ in range(N_MATCHED):
                m_sup = matched_indices(support, log_s, live, rng)
                m_gap.append(float(np.mean(np.abs(log_s[m_sup] - log_s[support]))))
                md, _ = measure(m_sup, eps, rng_t)
                m_d.append(md)
            matched = float(np.mean(m_d))
            matched_se = float(np.std(m_d) / np.sqrt(N_MATCHED))
            dd, dl = d_s - base_d, l_s - base_l
            excess = dd - (matched - base_d)
            rows.append({"eps": eps, "score": name, "dlogD": dd, "dlogL": dl,
                         "matched": matched - base_d, "excess": excess, "matched_se": matched_se})
            sel = dd / dl if abs(dl) > 1e-9 else float("nan")
            print(f"  {name:32s}{dd:>9.3f}{dl:>9.3f}{matched - base_d:>15.3f}"
                  f"{excess:>9.3f}{sel:>7.2f}   (se {matched_se:.3f}, "
                  f"mean |dlog||S_i||| between selection and control {np.mean(m_gap):.3f})")
    print("\n  EXCESS = dlogD(score) - dlogD(sensitivity-matched control): the symmetry-specific")
    print("  effect at fixed conditioning. Positive and larger than the matched s.e. is the")
    print("  only pattern that supports 'these parameters control the symmetry'.")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(scores)
    x = np.arange(len(names))
    for i, eps in enumerate(EPSILONS):
        vals = [r["excess"] for r in rows if r["eps"] == eps]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=f"eps={eps}")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("excess dlogD over sensitivity-matched control")
    ax.set_title(f"{run_dir.name}\nsymmetry-specific causal effect at fixed conditioning", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = run_dir / "causal_matched_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(run_dir / "causal_matched_ablation.npz",
             **{k: np.array([r[k] for r in rows]) for k in ("eps", "dlogD", "dlogL", "matched", "excess")},
             score=np.array([r["score"] for r in rows]))
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
