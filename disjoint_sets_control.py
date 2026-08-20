"""Is the controlling set unique, or is it one representative of many?

"These 20 parameters control the symmetry" is a claim about a particular set.
But the resolved tangent space is 22-dimensional while ~2200 parameters are
live, so roughly a hundred parameters share each independent function-space
direction and are mutually substitutable. If that is what is going on, then
*disjoint* sets of comparably-aligned parameters should each control the
symmetry to a comparable degree -- the set would be real but not unique, and
naming its members would be a convention rather than a finding.

This tests it directly: rank live parameters by |<S_i, g>|, cut into disjoint
blocks of equal size, and measure each block's causal effect on the symmetry
defect against its own sensitivity-matched control, so every block is judged
symmetry-specifically. A random live block is included as the floor.

Usage: python disjoint_sets_control.py outputs/<run_dir>
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

ALPHA = 0.3
N_ANGLES = 24
EPS = float(os.environ.get("EPS", "0.002"))
BLOCK = int(os.environ.get("BLOCK", "20"))
BLOCK_STARTS = [0, 20, 40, 60, 80, 100, 200, 500, 1000]
N_DRAWS = 16
N_MATCHED = 10


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
    j64 = j_x.reshape(n_points * 2, -1).detach().cpu().to(torch.float64)
    colnorm = j64.norm(dim=0).numpy()
    live = colnorm > colnorm.max() * 1e-8
    log_s = np.log(np.clip(colnorm, 1e-300, None))
    align = np.abs((j64.T @ g.detach().cpu().to(torch.float64)).numpy())
    align_live_order = np.flatnonzero(live)[np.argsort(-align[live])]

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
            d = torch.zeros_like(flat0)
            noise = torch.randn(len(support), generator=rng_t, dtype=torch.float32).to(flat0.dtype)
            d[torch.from_numpy(np.asarray(support))] = noise
            d = d / d.norm().clamp_min(1e-30) * (EPS * theta_norm)
            set_flat(flat0 + d)
            dl.append(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
            ll.append(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
        set_flat(flat0)
        return float(np.mean(np.log(dl))), float(np.mean(np.log(ll)))

    set_flat(flat0)
    base_d = np.log(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_ANGLES).detach()))
    base_l = np.log(float(residuals(*train_data, cfg.trajectory_window, integrator).detach()))
    print(f"{run_dir.name}: {int(live.sum())} live parameters, disjoint blocks of {BLOCK} "
          f"by |<S_i,g>| rank, eps={EPS}")
    print(f"  {'block (rank)':>16}{'dlogD':>9}{'dlogL':>9}{'matched':>10}{'EXCESS':>9}{'se':>7}{'sel':>7}")

    rows = []
    for start in BLOCK_STARTS:
        if start + BLOCK > len(align_live_order):
            continue
        sup = align_live_order[start:start + BLOCK]
        d_s, l_s = measure(sup)
        md = [measure(matched_indices(sup, log_s, live, rng))[0] for _ in range(N_MATCHED)]
        matched, se = float(np.mean(md)), float(np.std(md) / np.sqrt(N_MATCHED))
        dd, dl_ = d_s - base_d, l_s - base_l
        ex = dd - (matched - base_d)
        rows.append({"start": start, "dlogD": dd, "dlogL": dl_, "excess": ex, "se": se})
        print(f"  {f'{start+1}-{start+BLOCK}':>16}{dd:>9.3f}{dl_:>9.3f}{matched-base_d:>10.3f}"
              f"{ex:>9.3f}{se:>7.3f}{dd/dl_ if abs(dl_)>1e-9 else float('nan'):>7.2f}")

    rand = rng.choice(np.flatnonzero(live), size=BLOCK, replace=False)
    d_s, l_s = measure(rand)
    md = [measure(matched_indices(rand, log_s, live, rng))[0] for _ in range(N_MATCHED)]
    matched = float(np.mean(md))
    print(f"  {'random live':>16}{d_s-base_d:>9.3f}{l_s-base_l:>9.3f}{matched-base_d:>10.3f}"
          f"{d_s-base_d-(matched-base_d):>9.3f}{'':>7}"
          f"{(d_s-base_d)/(l_s-base_l) if abs(l_s-base_l)>1e-9 else float('nan'):>7.2f}")
    print("\n  Several disjoint blocks each showing a clear positive excess would mean the")
    print("  controlling set is real but NOT unique -- one representative of many.")

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [r["start"] + 1 for r in rows]
    ax.errorbar(xs, [r["excess"] for r in rows], yerr=[r["se"] for r in rows], marker="o", capsize=3)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xscale("log"); ax.set_xlabel(f"rank of first parameter in the disjoint block of {BLOCK}")
    ax.set_ylabel("excess dlog D over matched control")
    ax.set_title(f"{run_dir.name}\nDo disjoint sets each control the symmetry?", fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(run_dir / "disjoint_sets_control.png", dpi=200, bbox_inches="tight")
    print(f"plot -> {run_dir / 'disjoint_sets_control.png'}")


if __name__ == "__main__":
    main()
