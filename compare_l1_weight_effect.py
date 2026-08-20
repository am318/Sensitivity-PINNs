"""Does the training-time L1 penalty (cfg.l1_weight) affect post-hoc c_i attribution?

These are two different, easily-conflated things:
- cfg.l1_weight: an L1 penalty on the raw network parameters added to the
  TRAINING loss -- it directly shrinks/sparsifies the trained weights
  themselves, changing the model.
- The attribution method (l2/l1/elastic_net): a POST-HOC choice of which c_i
  reconstructs a fixed target direction within a fixed trained model's
  tangent space -- mathematically independent of how that model was trained.

They can't interact directly (the attribution solve doesn't see the training
loss), but training-time L1 changes the actual Jacobian J being attributed
over -- so it can affect the attribution distributions *indirectly*, through
the model. This script trains several models (same seed, same architecture)
at different l1_weight values and compares: network-level weight sparsity,
resolved tangent-space rank, and the L2/L1/elastic-net attribution n_eff for
each.
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
    build_model,
    build_probe_grid,
    evaluate_at_points,
    generic_residuals,
    make_dataset,
    make_optimizer,
    residuals,
    rotation_generator_target,
    validate_config,
)
from experiment_common import run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    tangent_projection_auto,
)

OUTPUT_DIR = Path("outputs/l1_weight_effect")
SEED = 0
TRAINING_STEPS = 400
ALPHA = 0.3
L1_WEIGHTS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]


def make_quick_config(l1_weight: float) -> Config:
    cfg = Config()
    cfg.seed = SEED
    cfg.kinetic_hidden_dim = 8
    cfg.potential_hidden_dim = 8
    cfg.initial_conditions_per_alpha = 2
    cfg.trajectory_splits = 2
    cfg.coarsening_factor = 2
    cfg.q_grid_points_per_axis = 4
    cfg.training_steps = TRAINING_STEPS
    cfg.checkpoint_fractions = [0.0, 1.0]
    cfg.l1_weight = l1_weight
    validate_config(cfg)
    return cfg


def train(cfg: Config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)
    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *a, **kw: generic_residuals(*a, **kw, p_dim=2)
    )
    history, _ = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    print(f"l1_weight={cfg.l1_weight:g}  final_train_loss={history['training_loss'][-1]:.4e}")
    return model, device, dtype


def n_eff(c: np.ndarray) -> float:
    sq = float((c**2).sum())
    return 0.0 if sq <= 0 else float(np.abs(c).sum() ** 2) / sq


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for l1_weight in L1_WEIGHTS:
        cfg = make_quick_config(l1_weight)
        model, device, dtype = train(cfg)

        all_params = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()])
        weight_sparsity = float((all_params < 1e-4).float().mean())
        weight_scale = float(all_params.mean())

        q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
        n_points = q1_grid.shape[0]
        v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
            model, q1_grid, q2_grid, ALPHA, cfg.architecture,
            device=device, dtype=dtype, need_spatial_jacobian=True,
        )
        jac_flat = j_x.reshape(n_points * 2, -1)
        xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x).reshape(n_points * 2)
        cutoff = cfg.tangent_svd_relative_cutoff

        _, _, c_l2, rank, _ = tangent_projection_auto(jac_flat, xrot_target, cutoff, method="l2")
        _, _, c_l1, _, _ = tangent_projection_auto(jac_flat, xrot_target, cutoff, method="l1")
        chosen_ratio = choose_l1_ratio_for_sparsity(jac_flat, xrot_target, cutoff)
        _, _, c_en, _, _ = tangent_projection_auto(
            jac_flat, xrot_target, cutoff, method="elastic_net", l1_ratio=chosen_ratio
        )

        row = {
            "l1_weight": l1_weight,
            "weight_sparsity": weight_sparsity,
            "weight_scale": weight_scale,
            "resolved_rank": rank,
            "n_eff_l2": n_eff(c_l2.numpy()),
            "n_eff_l1": n_eff(c_l1.numpy()),
            "n_eff_en": n_eff(c_en.numpy()),
            "chosen_l1_ratio": chosen_ratio,
        }
        rows.append(row)
        print(
            f"  weight_sparsity(<1e-4)={weight_sparsity:.3f}  weight_scale={weight_scale:.4f}  "
            f"resolved_rank={rank}  n_eff: L2={row['n_eff_l2']:.2f} L1={row['n_eff_l1']:.2f} "
            f"EN={row['n_eff_en']:.2f} (ratio={chosen_ratio:.2f})"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    l1_weights_plot = [max(r["l1_weight"], 1e-7) for r in rows]  # avoid log(0)

    axes[0].plot(l1_weights_plot, [r["weight_sparsity"] for r in rows], marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training l1_weight")
    axes[0].set_ylabel("fraction of raw weights with |w| < 1e-4")
    axes[0].set_title("Network-level weight sparsity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(l1_weights_plot, [r["resolved_rank"] for r in rows], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training l1_weight")
    axes[1].set_ylabel("resolved tangent-space rank")
    axes[1].set_title("Resolved rank vs. training l1_weight")
    axes[1].grid(alpha=0.3)

    for key, label in [("n_eff_l2", "min 2-norm"), ("n_eff_l1", "min 1-norm"), ("n_eff_en", "elastic net")]:
        axes[2].plot(l1_weights_plot, [r[key] for r in rows], marker="o", label=label)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("training l1_weight")
    axes[2].set_ylabel("attribution n_eff (xrot)")
    axes[2].set_title("Attribution sparsity vs. training l1_weight")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l1_weight_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
