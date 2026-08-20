"""Is control of the learned symmetry concentrated in a few parameters -- and in how few?

The draft's concentration claim is about |c_i|, which §2-3 show is not a
property of the network. But "is symmetry control concentrated?" is a good
question independently of how one scores it, and it can be asked directly of
the causal quantity rather than of an attribution.

Two parts.

1. **Concentration of causal influence itself.** grad_theta D is the exact
   per-parameter causal influence on the symmetry defect D = ||delta||^2/||f||^2,
   and grad_theta L is the same for the task loss. Comparing their Lorenz
   curves answers whether symmetry control is *more* concentrated than task
   control, or whether both simply inherit the concentration of ||S_i||. The
   task gradient is the honest null here: if symmetry influence is no more
   concentrated than task influence, concentration is not a symmetry finding.

2. **How few parameters suffice.** A sweep over k of the sensitivity-matched
   ablation: perturb the top-k parameters by |<S_i, g>| and measure the excess
   over a control matched on log||S_i||, so the effect is symmetry-specific by
   construction. If the excess saturates at small k, a small subset genuinely
   controls the symmetry; if it grows with k, control is distributed.

Usage: python symmetry_concentration.py outputs/<run_dir>
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
    Config, build_model, evaluate_at_points, make_dataset, residuals,
    rotation_generator_target, validate_config,
)
from causal_matched_ablation import matched_indices
from causal_symmetry_control import build_polar_probe_grid, defect_order_parameter
from experiment_common import select_device, select_dtype
from sensitivity_tools import participation_ratio_l1l2

ALPHA = 0.3
N_ANGLES = 24
EPS = float(os.environ.get("EPS", "0.002"))
K_VALUES = [3, 5, 10, 20, 50, 100, 200]
N_DRAWS = 16
N_MATCHED = 12


def lorenz(v: np.ndarray) -> np.ndarray:
    v = np.sort(np.abs(v))[::-1]
    t = v.sum()
    return np.cumsum(v) / t if t > 0 else np.zeros_like(v)


def share_at(curve: np.ndarray, frac: float) -> float:
    return float(curve[max(0, int(len(curve) * frac) - 1)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = Config()
    cfg.architecture, cfg.device, cfg.training_steps = blob["architecture"], "cpu", 1
    cfg.augment_dataset = bool(blob.get("augment", False))
    validate_config(cfg)
    device, dtype = select_device("cpu"), select_dtype(cfg.dtype)
    torch.manual_seed(blob.get("seed", 0)); np.random.seed(blob.get("seed", 0))
    train_data, _ = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    model.load_state_dict(blob["state_dict"]); model.eval()
    params = tuple(model.parameters())

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    n_points = q1.shape[0]
    _, f_x, j_x, spatial = evaluate_at_points(
        model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True)
    g = rotation_generator_target(q1, q2, f_x, spatial).reshape(n_points * 2)
    j_flat = j_x.reshape(n_points * 2, -1)
    j64 = j_flat.detach().cpu().to(torch.float64)
    colnorm = j64.norm(dim=0).numpy()
    live = colnorm > colnorm.max() * 1e-8
    log_s = np.log(np.clip(colnorm, 1e-300, None))
    align = np.abs((j64.T @ g.detach().cpu().to(torch.float64)).numpy())

    def flat_grad(scalar):
        gr = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=False)
        return torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1)
                          for p, q in zip(params, gr)]).detach().cpu().to(torch.float64).abs().numpy()

    grad_d = flat_grad(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES))
    grad_l = flat_grad(residuals(*train_data, cfg.trajectory_window, integrator))

    print(f"{run_dir.name}: {int(live.sum())} live parameters of {len(colnorm)}")
    print("\n=== 1. Is causal influence on the SYMMETRY more concentrated than on the TASK? ===")
    curves = {
        "|grad D| (symmetry)": lorenz(grad_d[live]),
        "|grad L| (task loss)": lorenz(grad_l[live]),
        "||S_i|| (conditioning)": lorenz(colnorm[live]),
        "|<S_i, g>| (alignment)": lorenz(align[live]),
    }
    print(f"  {'quantity':26s}{'top 1%':>9s}{'top 5%':>9s}{'top 10%':>9s}{'n_eff':>10s}")
    vals = {"|grad D| (symmetry)": grad_d, "|grad L| (task loss)": grad_l,
            "||S_i|| (conditioning)": colnorm, "|<S_i, g>| (alignment)": align}
    for name, c in curves.items():
        print(f"  {name:26s}{share_at(c,.01)*100:>8.1f}%{share_at(c,.05)*100:>8.1f}%"
              f"{share_at(c,.10)*100:>8.1f}%{participation_ratio_l1l2(vals[name][live]):>10.1f}")
    print("  n_eff = (sum|v|)^2 / sum v^2, the effective number of parameters carrying the quantity.")
    print("  |grad L| is the null: concentration shared with the task is not a symmetry finding.")

    print(f"\n=== 2. How few parameters suffice? top-k by |<S_i,g>|, eps={EPS} ===")
    sizes = [p.numel() for p in params]
    flat0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    theta_norm = float(flat0.norm())

    def set_flat(v):
        off = 0
        with torch.no_grad():
            for p, n in zip(params, sizes):
                p.copy_(v[off:off + n].view_as(p)); off += n

    rng_t = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)

    def measure(support):
        dl, ll = [], []
        for _ in range(N_DRAWS):
            direction = torch.zeros_like(flat0)
            noise = torch.randn(len(support), generator=rng_t, dtype=torch.float32).to(flat0.dtype)
            direction[torch.from_numpy(np.asarray(support))] = noise
            direction = direction / direction.norm().clamp_min(1e-30) * (EPS * theta_norm)
            set_flat(flat0 + direction)
            dl.append(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
            ll.append(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
        set_flat(flat0)
        return float(np.mean(np.log(dl))), float(np.mean(np.log(ll)))

    set_flat(flat0)
    base_d = np.log(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
    base_l = np.log(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
    order = np.argsort(-align)
    rows = []
    print(f"  {'k':>5}{'dlogD':>9}{'dlogL':>9}{'matched':>10}{'EXCESS':>9}{'se':>7}{'sel':>7}")
    for k in K_VALUES:
        sup = order[:k]
        d_s, l_s = measure(sup)
        md = [measure(matched_indices(sup, log_s, live, rng))[0] for _ in range(N_MATCHED)]
        matched, se = float(np.mean(md)), float(np.std(md) / np.sqrt(N_MATCHED))
        dd, dl_ = d_s - base_d, l_s - base_l
        ex = dd - (matched - base_d)
        rows.append({"k": k, "dlogD": dd, "dlogL": dl_, "matched": matched - base_d, "excess": ex, "se": se})
        print(f"  {k:>5}{dd:>9.3f}{dl_:>9.3f}{matched-base_d:>10.3f}{ex:>9.3f}{se:>7.3f}"
              f"{dd/dl_ if abs(dl_)>1e-9 else float('nan'):>7.2f}")
    print("  EXCESS = effect beyond a control matched on log||S_i||: symmetry-specific by construction.")
    print("  sel = dlogD/dlogL: >1 means the perturbation moves the symmetry more than the task.")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for name, c in curves.items():
        frac = np.arange(1, len(c) + 1) / len(c)
        axes[0].plot(frac, c, "--" if "task" in name or "conditioning" in name else "-", label=name)
    axes[0].set_xscale("log"); axes[0].set_xlabel("fraction of live parameters (sorted)")
    axes[0].set_ylabel("cumulative share"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    axes[0].set_title("Concentration of causal influence", fontsize=10)
    ks = [r["k"] for r in rows]
    axes[1].errorbar(ks, [r["excess"] for r in rows], yerr=[r["se"] for r in rows], marker="o", capsize=3)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("k (parameters perturbed)")
    axes[1].set_ylabel("excess dlog D over matched control")
    axes[1].set_title(f"Symmetry-specific causal effect vs k (eps={EPS})", fontsize=10)
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(run_dir / "symmetry_concentration.png", dpi=200, bbox_inches="tight")
    print(f"\nplot -> {run_dir / 'symmetry_concentration.png'}")


if __name__ == "__main__":
    main()
