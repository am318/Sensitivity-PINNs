"""Is symmetry control concentrated -- against a null that is not the task loss?

Comparing the concentration of grad_theta D (symmetry) against grad_theta L
(task) is not a fair test on a symmetric benchmark. The true dynamics IS
rotationally equivariant, so a network that fits the task must become
equivariant: the two gradients are not independent quantities, and finding them
equally concentrated may only mean the symmetry has been absorbed into the task.

The matched null is a non-symmetry transformation of the same construction:

    E(M) = ||f(Mx) - M f(x)||^2 / ||f(x)||^2

which is defined for ANY linear M (no group average needed, so M need not
generate a compact group), reduces to the equivariance error when M is the true
rotation, and is differentiable in theta. grad_theta E(M) is then the exact
per-parameter causal influence on "M-equivariance". Concentration of
grad E(R_true) is symmetry-specific only insofar as it exceeds the distribution
of concentrations of grad E(M) over random non-orthogonal M.

The script also evaluates an untrained model, because the reconciliation
question is directional: if symmetry and task concentration are distinct when
equivariance error is high and converge as it falls, then their agreement in a
converged network is a finding about what training did, not an absence of
signal.

Usage: python symmetry_concentration_null.py outputs/<run_dir>
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
    Config, build_model, make_dataset, residuals, rotation_matrix, transform_points, validate_config,
)
from causal_symmetry_control import batched_force, build_polar_probe_grid
from experiment_common import select_device, select_dtype
from sensitivity_tools import participation_ratio_l1l2, random_symmetry_test_matrix

ALPHA = 0.3
N_ANGLES = 24
N_NULLS = int(os.environ.get("N_NULLS", "8"))


def transform_defect_scalar(model, q1, q2, arch, m):
    """E(M) = ||f(Mx) - M f(x)||^2 / ||f(x)||^2, differentiable in theta."""
    f_x = batched_force(model, q1, q2, ALPHA, arch)
    q1m, q2m = transform_points(q1, q2, m)
    f_mx = batched_force(model, q1m, q2m, ALPHA, arch)
    return (f_mx - f_x @ m.T).pow(2).sum() / f_x.pow(2).sum().clamp_min(1e-30)


def flat_grad(scalar, params):
    gr = torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=False)
    return torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1)
                      for p, q in zip(params, gr)]).detach().cpu().to(torch.float64).abs().numpy()


def summarise(v, live):
    vv = np.abs(v[live])
    s = np.sort(vv)[::-1]
    tot = s.sum()
    cum = np.cumsum(s) / tot if tot > 0 else np.zeros_like(s)
    top = lambda f: float(cum[max(0, int(len(cum) * f) - 1)])
    return participation_ratio_l1l2(vv), top(0.01), top(0.10)


def analyse(model, tag, cfg, device, dtype, q1, q2, train_data, integrator, gen):
    params = tuple(model.parameters())
    j_cols = None
    rot = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    e_true = float(transform_defect_scalar(model, q1, q2, cfg.architecture, rot).detach())
    g_true = flat_grad(transform_defect_scalar(model, q1, q2, cfg.architecture, rot), params)
    g_task = flat_grad(residuals(*train_data, cfg.trajectory_window, integrator), params)
    live = g_true + g_task > 0
    # live set from sensitivity, not from the gradients themselves
    f_probe = batched_force(model, q1, q2, ALPHA, cfg.architecture)
    live = np.zeros(len(g_true), dtype=bool)
    off = 0
    for p in params:
        n = p.numel()
        live[off:off + n] = p.requires_grad
        off += n
    live &= (g_true > 0) | (g_task > 0)

    nulls = []
    for _ in range(N_NULLS):
        m = random_symmetry_test_matrix(device=device, dtype=dtype, generator=gen)
        nulls.append((float(transform_defect_scalar(model, q1, q2, cfg.architecture, m).detach()),
                      flat_grad(transform_defect_scalar(model, q1, q2, cfg.architecture, m), params)))

    n_true, t1_true, t10_true = summarise(g_true, live)
    n_task, t1_task, t10_task = summarise(g_task, live)
    ns = np.array([summarise(gn, live) for _, gn in nulls])
    print(f"\n===== {tag} =====")
    print(f"  equivariance error E(R_true) = {e_true:.4e}   "
          f"E(M) over {N_NULLS} random non-orthogonal M: {np.mean([e for e, _ in nulls]):.4e}")
    print(f"  {'quantity':34s}{'n_eff':>9s}{'top 1%':>9s}{'top 10%':>9s}")
    print(f"  {'grad E(R_true)  [symmetry]':34s}{n_true:>9.1f}{t1_true*100:>8.1f}%{t10_true*100:>8.1f}%")
    print(f"  {'grad E(M)       [matched null]':34s}{ns[:,0].mean():>9.1f}"
          f"{ns[:,1].mean()*100:>8.1f}%{ns[:,2].mean()*100:>8.1f}%"
          f"   (sd {ns[:,0].std():.1f})")
    print(f"  {'grad L          [task]':34s}{n_task:>9.1f}{t1_task*100:>8.1f}%{t10_task*100:>8.1f}%")
    z = (ns[:, 0].mean() - n_true) / max(ns[:, 0].std(), 1e-12)
    print(f"  concentration z-score of the true rotation vs the matched nulls: {z:+.2f}"
          f"   (positive = symmetry MORE concentrated than a non-symmetry transform)")
    from scipy.stats import spearmanr
    print(f"  Spearman(|grad E(R_true)|, |grad L|) over live params = "
          f"{float(spearmanr(g_true[live], g_task[live]).statistic):+.3f}"
          f"   <- how coextensive symmetry control and task control are")
    return {"tag": tag, "e_true": e_true, "n_true": n_true, "n_task": n_task,
            "n_null_mean": float(ns[:, 0].mean()), "n_null_sd": float(ns[:, 0].std()), "z": z,
            "rho_task": float(spearmanr(g_true[live], g_task[live]).statistic)}


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
    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    gen = torch.Generator().manual_seed(0)

    rows = []
    untrained, integrator = build_model(cfg, device, dtype)
    untrained.eval()
    rows.append(analyse(untrained, "UNTRAINED (high equivariance error)", cfg, device, dtype,
                        q1, q2, train_data, integrator, gen))

    trained, integrator2 = build_model(cfg, device, dtype)
    trained.load_state_dict(blob["state_dict"]); trained.eval()
    rows.append(analyse(trained, "TRAINED (low equivariance error)", cfg, device, dtype,
                        q1, q2, train_data, integrator2, gen))

    print("\n===== reconciliation =====")
    for r in rows:
        print(f"  {r['tag']:38s} E={r['e_true']:.2e}  n_eff(symm)={r['n_true']:7.1f}  "
              f"n_eff(task)={r['n_task']:7.1f}  ratio={r['n_true']/max(r['n_task'],1e-9):.3f}  "
              f"rho(symm,task)={r['rho_task']:+.3f}")
    print("  If the ratio and rho rise towards 1 as equivariance error falls, then symmetry")
    print("  control becoming coextensive with task control is a RESULT of training on a")
    print("  symmetric system, not evidence that symmetry concentration is unremarkable.")
    np.savez(run_dir / "concentration_null.npz", **{k: np.array([r[k] for r in rows])
                                                    for k in rows[0] if k != "tag"})


if __name__ == "__main__":
    main()
