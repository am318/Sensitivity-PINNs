"""Does symmetry control delocalise as the network learns the symmetry?

The endpoint comparison (symmetry_concentration_null.py) found that an untrained
network shows symmetry-specific concentration of causal influence (z = +3.0
against matched non-symmetry nulls, and control decoupled from the task,
rho = -0.06), while the same network trained to equivariance error 8e-5 shows
neither (z = +0.24, rho = +0.30). Two points do not make a trend, and the claim
that training *absorbs* symmetry control into task control is the paper's main
positive result, so it needs the intermediate checkpoints.

At every checkpoint this records, all against the same probe grid:

  - the equivariance error under the true rotation, and E(M) for random
    non-orthogonal M (the matched null: same construction, not a symmetry);
  - n_eff of |grad E(R_true)|, of |grad E(M)| over several M, and of |grad L|;
  - the concentration z-score of the true rotation against the null distribution
    -- the symmetry-specific concentration signal;
  - Spearman(|grad E(R_true)|, |grad L|) -- how coextensive symmetry control and
    task control have become.

E(M) = ||f(Mx) - M f(x)||^2 / ||f||^2 is used rather than the group-averaged
defect because it is defined for any linear M (no compact group needed), so the
true rotation and the nulls are measured by literally the same functional.

Env: TRAINING_STEPS, AUGMENT, SEED, N_NULLS.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config, build_model, make_dataset, make_optimizer, residuals,
    rotation_matrix, transform_points, validate_config,
)
from causal_symmetry_control import batched_force, build_polar_probe_grid
from experiment_common import run_training_loop, select_device, select_dtype
from sensitivity_tools import participation_ratio_l1l2, random_symmetry_test_matrix

ALPHA = 0.3
N_ANGLES = 24
SEED = int(os.environ.get("SEED", "0"))
AUGMENT = os.environ.get("AUGMENT", "1") == "1"
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "8000"))
N_NULLS = int(os.environ.get("N_NULLS", "8"))
CHECKPOINT_FRACTIONS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
OUTPUT_DIR = Path("outputs/delocalisation" + ("_augmented" if AUGMENT else ""))


def e_of_m(model, q1, q2, arch, m):
    """E(M) = ||f(Mx) - M f(x)||^2 / ||f(x)||^2, differentiable in theta."""
    f_x = batched_force(model, q1, q2, ALPHA, arch)
    q1m, q2m = transform_points(q1, q2, m)
    f_mx = batched_force(model, q1m, q2m, ALPHA, arch)
    return (f_mx - f_x @ m.T).pow(2).sum() / f_x.pow(2).sum().clamp_min(1e-30)


def flat_grad(scalar, params):
    gr = torch.autograd.grad(scalar, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1)
                      for p, q in zip(params, gr)]).detach().cpu().to(torch.float64).abs().numpy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.seed, cfg.architecture, cfg.device = SEED, "hamiltonian", "cpu"
    cfg.training_steps, cfg.augment_dataset = TRAINING_STEPS, AUGMENT
    cfg.checkpoint_fractions = CHECKPOINT_FRACTIONS
    validate_config(cfg)
    torch.manual_seed(SEED); np.random.seed(SEED)
    device, dtype = select_device(cfg.device), select_dtype(cfg.dtype)
    gen = torch.Generator().manual_seed(SEED)
    print(f"augment={AUGMENT} steps={TRAINING_STEPS} checkpoints={len(CHECKPOINT_FRACTIONS)} nulls={N_NULLS}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    history, checkpoints = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data, trajectory_window=cfg.trajectory_window,
        residuals_fn=residuals, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay)

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    rot = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    ms = [random_symmetry_test_matrix(device=device, dtype=dtype, generator=gen) for _ in range(N_NULLS)]
    step_to_loss = dict(zip(history["step"], history["trajectory_loss"]))

    rows = []
    for step in sorted(checkpoints):
        model.load_state_dict(checkpoints[step]); model.eval()
        params = tuple(model.parameters())
        e_true = float(e_of_m(model, q1, q2, cfg.architecture, rot).detach())
        g_true = flat_grad(e_of_m(model, q1, q2, cfg.architecture, rot), params)
        g_task = flat_grad(residuals(*train_data, cfg.trajectory_window, integrator), params)
        live = (g_true > 0) | (g_task > 0)
        nulls = [(float(e_of_m(model, q1, q2, cfg.architecture, m).detach()),
                  flat_grad(e_of_m(model, q1, q2, cfg.architecture, m), params)) for m in ms]
        n_true = participation_ratio_l1l2(g_true[live])
        n_task = participation_ratio_l1l2(g_task[live])
        n_null = np.array([participation_ratio_l1l2(gn[live]) for _, gn in nulls])
        z = float((n_null.mean() - n_true) / max(n_null.std(), 1e-12))
        rho = float(spearmanr(g_true[live], g_task[live]).statistic)
        row = {"step": step, "loss": step_to_loss.get(step, float("nan")), "e_true": e_true,
               "e_null": float(np.mean([e for e, _ in nulls])), "n_true": n_true,
               "n_task": n_task, "n_null": float(n_null.mean()), "n_null_sd": float(n_null.std()),
               "z": z, "rho_task": rho}
        rows.append(row)
        print(f"step={step:6d} loss={row['loss']:.3e} E(R)={e_true:.3e} E(M)={row['e_null']:.3e} "
              f"n_eff: symm={n_true:7.1f} null={n_null.mean():7.1f}+-{n_null.std():5.1f} task={n_task:7.1f} "
              f"| z={z:+.2f} rho(symm,task)={rho:+.3f}")

    np.savez(OUTPUT_DIR / "delocalisation.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    e = np.array([r["e_true"] for r in rows]); z = np.array([r["z"] for r in rows])
    rho = np.array([r["rho_task"] for r in rows])
    from scipy.stats import pearsonr
    le = np.log(np.clip(e, 1e-300, None))
    print(f"\ncorr(log equivariance error, concentration z) = {pearsonr(le, z)[0]:+.3f} (p={pearsonr(le, z)[1]:.4f})")
    print(f"corr(log equivariance error, rho(symm,task))  = {pearsonr(le, rho)[0]:+.3f} (p={pearsonr(le, rho)[1]:.4f})")
    print("positive first / negative second = as equivariance improves, symmetry-specific")
    print("concentration falls and symmetry control merges into task control.")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    axes[0].errorbar(e, z, yerr=1.0, marker="o", capsize=3)
    axes[0].axhline(0, color="gray", lw=1); axes[0].set_xscale("log"); axes[0].invert_xaxis()
    axes[0].set_xlabel("equivariance error E(R)  (better ->)")
    axes[0].set_ylabel("concentration z vs matched non-symmetry nulls")
    axes[0].set_title("Symmetry-specific concentration", fontsize=10); axes[0].grid(alpha=.3)
    axes[1].plot(e, rho, "-o")
    axes[1].set_xscale("log"); axes[1].invert_xaxis(); axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xlabel("equivariance error E(R)  (better ->)")
    axes[1].set_ylabel(r"Spearman($|\nabla E(R)|$, $|\nabla L|$)")
    axes[1].set_title("Symmetry control vs task control", fontsize=10); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "delocalisation.png", dpi=200, bbox_inches="tight")
    print(f"\nWritten to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
