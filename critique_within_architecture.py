"""Within-architecture check: does the c_i concentration-gap track E_i, or just training loss?

Trains ONE model (architecture fixed, via ARCHITECTURE env var) with many
checkpoints spread across training. At each checkpoint computes:

- training loss (to control for raw convergence level)
- E_i(real generator): mean per-parameter equivariance error under the true
  rotation, and E_i(random M): the same under many draws of a random
  non-symmetric matrix (random_symmetry_test_matrix) as a null baseline --
  the equivariance-error analogue of the c_i null control.
- c_i (elastic net) concentration (n_eff, PR(c)) for the real xrot
  generator, and for many draws of a *matched-construction* null target
  (linear_generator_target with a random non-symmetric matrix M in place of
  the true rotation generator Omega, rather than isotropic noise -- a toy-
  scale check found the isotropic null gives a spuriously large apparent
  real-vs-null gap (z~1.6) that mostly evaporates (z~0.24) once the null is
  properly matched in construction/smoothness to the real target, so this
  is not optional -- an isotropic null is not a fair comparison).
- z-scores for both gaps (concentration z-score, equivariance z-score), so
  they are comparable in units across checkpoints regardless of the raw
  scale of E_i or n_eff at that point in training.

Then reports the correlation between the concentration z-score and the
equivariance z-score across checkpoints, and separately between each and
training loss -- the test proposed to separate "the model learned the
symmetry" (concentration gap tracks E_i specifically) from "the model just
converged" (concentration gap tracks loss regardless of E_i).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config,
    build_model,
    build_probe_grid,
    evaluate_at_points,
    generic_residuals,
    make_dataset,
    make_optimizer,
    residuals,
    rotation_generator_target,
    rotation_matrix,
    transform_points,
    validate_config,
)
from experiment_common import run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    linear_generator_target,
    participation_ratio_l1l2,
    per_parameter_equivariance_error,
    random_symmetry_test_matrix,
    tangent_projection_auto,
)

ARCHITECTURE = os.environ.get("ARCHITECTURE", "hamiltonian")
DEVICE = os.environ.get("DEVICE", "cpu")
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "8000"))
SEED = 0
ALPHA = 0.3
N_NULL_DRAWS = 15
CHECKPOINT_FRACTIONS = [0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0]
OUTPUT_DIR = Path(f"outputs/within_architecture_{ARCHITECTURE}")


def make_full_config() -> Config:
    cfg = Config()
    cfg.seed = SEED
    cfg.architecture = ARCHITECTURE
    cfg.device = DEVICE
    cfg.training_steps = TRAINING_STEPS
    cfg.checkpoint_fractions = CHECKPOINT_FRACTIONS
    validate_config(cfg)
    return cfg


def zscore(real_value: float, null_values: np.ndarray, higher_is_more_concentrated: bool) -> float:
    """Standardized gap: positive means the real value is "more extreme" (per the given direction)."""
    mean, std = float(null_values.mean()), float(null_values.std())
    std = max(std, 1e-12)
    raw = (mean - real_value) if higher_is_more_concentrated else (real_value - mean)
    return raw / std


def analyse_checkpoint(model, cfg, device, dtype, q1_grid, q2_grid, rot_mat, rng, l1_ratio_state):
    n_points = q1_grid.shape[0]
    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_grid, q2_grid, ALPHA, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    jac_flat = j_x.reshape(n_points * 2, -1)
    xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x).reshape(n_points * 2)
    positions = torch.stack([q1_grid, q2_grid], dim=1)
    cutoff = cfg.tangent_svd_relative_cutoff

    # --- E_i: real rotation vs. random-M null baseline ---
    q1_rot, q2_rot = transform_points(q1_grid, q2_grid, rot_mat)
    _, _, j_rot = evaluate_at_points(model, q1_rot, q2_rot, ALPHA, cfg.architecture, device=device, dtype=dtype)
    e_i_real = float(per_parameter_equivariance_error(j_x, j_rot, rot_mat).mean())

    e_i_null_draws = []
    for _ in range(N_NULL_DRAWS):
        m = random_symmetry_test_matrix(device=device, dtype=dtype, generator=rng)
        q1_m, q2_m = transform_points(q1_grid, q2_grid, m)
        _, _, j_m = evaluate_at_points(model, q1_m, q2_m, ALPHA, cfg.architecture, device=device, dtype=dtype)
        e_i_null_draws.append(float(per_parameter_equivariance_error(j_x, j_m, m).mean()))
    e_i_null_draws = np.array(e_i_null_draws)
    e_i_z = zscore(e_i_real, e_i_null_draws, higher_is_more_concentrated=True)  # lower E_i is "better"

    # --- c_i: real generator vs. random-target null baseline ---
    if l1_ratio_state["ratio"] is None:
        l1_ratio_state["ratio"] = choose_l1_ratio_for_sparsity(jac_flat, xrot_target, cutoff)
    ratio = l1_ratio_state["ratio"]
    _, _, c_real, resolved_rank, _ = tangent_projection_auto(
        jac_flat, xrot_target, cutoff, method="elastic_net", l1_ratio=ratio
    )
    n_eff_real = participation_ratio_l1l2(c_real.numpy())

    n_eff_null_draws = []
    for _ in range(N_NULL_DRAWS):
        m_null = random_symmetry_test_matrix(device=device, dtype=dtype, generator=rng)
        null_target = linear_generator_target(positions, f_x, spatial_jac_x, m_null).reshape(n_points * 2)
        _, _, c_null, _, _ = tangent_projection_auto(
            jac_flat, null_target, cutoff, method="elastic_net", l1_ratio=ratio
        )
        n_eff_null_draws.append(participation_ratio_l1l2(c_null.numpy()))
    n_eff_null_draws = np.array(n_eff_null_draws)
    c_z = zscore(n_eff_real, n_eff_null_draws, higher_is_more_concentrated=True)  # lower n_eff is "more concentrated"

    return {
        "resolved_rank": resolved_rank,
        "e_i_real": e_i_real, "e_i_null_mean": float(e_i_null_draws.mean()), "e_i_z": e_i_z,
        "n_eff_real": n_eff_real, "n_eff_null_mean": float(n_eff_null_draws.mean()), "c_z": c_z,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(SEED)
    cfg = make_full_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    print(f"architecture={cfg.architecture}  device={device}  training_steps={cfg.training_steps}  "
          f"n_checkpoints={len(CHECKPOINT_FRACTIONS)}  n_null_draws={N_NULL_DRAWS}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *a, **kw: generic_residuals(*a, **kw, p_dim=2)
    )
    history, checkpoint_states = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )

    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    rot_mat = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    l1_ratio_state = {"ratio": None}

    step_to_loss = dict(zip(history["step"], history["training_loss"]))
    rows = []
    for step in sorted(checkpoint_states.keys()):
        model.load_state_dict(checkpoint_states[step])
        model.eval()
        result = analyse_checkpoint(model, cfg, device, dtype, q1_grid, q2_grid, rot_mat, rng, l1_ratio_state)
        result["step"] = step
        result["loss"] = step_to_loss.get(step, step_to_loss.get(min(step_to_loss.keys(), default=step)))
        rows.append(result)
        print(
            f"step={step:6d} loss={result['loss']:.4e}  rank={result['resolved_rank']:3d}  "
            f"E_i(real)={result['e_i_real']:.4f} E_i(null_mean)={result['e_i_null_mean']:.4f} e_i_z={result['e_i_z']:+.3f}  "
            f"n_eff(real)={result['n_eff_real']:.2f} n_eff(null_mean)={result['n_eff_null_mean']:.2f} c_z={result['c_z']:+.3f}"
        )

    steps = np.array([r["step"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    e_i_z = np.array([r["e_i_z"] for r in rows])
    c_z = np.array([r["c_z"] for r in rows])
    log_loss = np.log(np.clip(losses, 1e-12, None))

    r_ec, p_ec = pearsonr(e_i_z, c_z)
    r_lc, p_lc = pearsonr(log_loss, c_z)
    r_le, p_le = pearsonr(log_loss, e_i_z)
    print("\n=== Correlations across checkpoints ===")
    print(f"corr(E_i z-score, concentration z-score) = {r_ec:+.3f}  (p={p_ec:.4f})")
    print(f"corr(log loss, concentration z-score)     = {r_lc:+.3f}  (p={p_lc:.4f})")
    print(f"corr(log loss, E_i z-score)                = {r_le:+.3f}  (p={p_le:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(steps, losses, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("Convergence")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, e_i_z, marker="o", label="E_i z-score (higher = more equivariant than null)")
    axes[1].plot(steps, c_z, marker="s", label="concentration z-score (higher = more concentrated than null)")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("z-score vs. null baseline")
    axes[1].set_title(f"{ARCHITECTURE}: gaps over training")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    axes[2].scatter(e_i_z, c_z, c=np.log(steps + 1), cmap="viridis")
    axes[2].set_xlabel("E_i z-score")
    axes[2].set_ylabel("concentration z-score")
    axes[2].set_title(f"corr={r_ec:+.3f} (p={p_ec:.4f}); color=log(step)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "within_architecture_check.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
