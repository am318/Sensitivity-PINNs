"""Cheap diagnostics requested by the attribution-methodology review notes.

Implements, for the Mexican-hat xrot generator at random init and after a
quick training run:

1. Null control: attribute a random matched-norm target through the same
   pipeline as the real symmetry generator. If the resulting concentration
   (participation ratio, Lorenz curve) looks the same, the "few dominant
   parameters" story is the conditioning of J, not evidence about symmetry.
2. Order parameter / representation quality: ||g||, ||P_T g||/||g||,
   resolved_rank, and n_eff(c)/resolved_rank for L2/L1/elastic-net -- see
   the theoretical note that n_eff(L1) <= resolved_rank always (LP vertex
   theory) while n_eff(L2) has no such ceiling.
3. Global-scale confound check: the classic init-vs-trained |c_i| scatter
   (as in the paper's Fig. 1a) reproduced both in raw units and as
   normalized shares |c_i| / sum|c_j|, to see whether the "two-cluster"
   structure survives once the global ||g||-shrinkage across training is
   divided out.
4. Per-parameter scale confound: Spearman rho(|c_i|, ||S_i||) and
   rho(|c_i|, |theta_i|) -- do the attribution coefficients just recover
   "this parameter is already big/sensitive," rather than anything specific
   to the symmetry-generator target?
5. Scale-invariant score a~_i = c_i<S_i,g> / ||P_T g||^2 (sums to 1),
   reported alongside raw |c_i| for comparison.
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
from experiment_common import parameter_layout, run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    attribution_scale_diagnostics,
    participation_ratio_l1l2,
    random_matched_norm_target,
    scale_invariant_attribution_score,
    tangent_projection_auto,
)

OUTPUT_DIR = Path("outputs/critique_diagnostics")
SEED = 0
TRAINING_STEPS = 400
ALPHA = 0.3


def make_quick_config() -> Config:
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
    validate_config(cfg)
    return cfg


def build_jacobian_and_target(model, cfg, device, dtype):
    q1_grid, q2_grid = build_probe_grid(cfg, device, dtype)
    n_points = q1_grid.shape[0]
    v_x, f_x, j_x, spatial_jac_x = evaluate_at_points(
        model, q1_grid, q2_grid, ALPHA, cfg.architecture,
        device=device, dtype=dtype, need_spatial_jacobian=True,
    )
    jac_flat = j_x.reshape(n_points * 2, -1)
    xrot_target = rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x).reshape(n_points * 2)
    sensitivity = torch.sqrt(torch.mean(j_x.detach().cpu().double().square(), dim=(0, 1)))
    return jac_flat, xrot_target, sensitivity


def analyse(label, model, cfg, device, dtype, parameter_magnitude, rng):
    jac_flat, target, sensitivity = build_jacobian_and_target(model, cfg, device, dtype)
    target_norm = float(torch.linalg.vector_norm(target))
    cutoff = cfg.tangent_svd_relative_cutoff
    result = {"label": label, "target_norm": target_norm}

    null_target = random_matched_norm_target(target, generator=rng)

    for method in ("l2", "l1", "elastic_net"):
        err, angle, c, rank, _ = tangent_projection_auto(jac_flat, target, cutoff, method=method)
        repr_quality = float(np.sqrt(max(0.0, 1.0 - err**2)))
        n_eff = participation_ratio_l1l2(c.numpy())
        diag = attribution_scale_diagnostics(c, sensitivity, parameter_magnitude)
        a_tilde = scale_invariant_attribution_score(jac_flat, target, c)

        _, _, c_null, rank_null, _ = tangent_projection_auto(jac_flat, null_target, cutoff, method=method)
        n_eff_null = participation_ratio_l1l2(c_null.numpy())

        result[method] = {
            "resolved_rank": rank,
            "relative_error": err,
            "angle_degrees": angle,
            "representation_quality": repr_quality,
            "n_eff": n_eff,
            "n_eff_over_rank": n_eff / rank if rank else float("nan"),
            "n_eff_null_control": n_eff_null,
            "rho_sensitivity": diag["rho_sensitivity"],
            "rho_magnitude": diag["rho_magnitude"],
            "a_tilde_sum": float(a_tilde.sum()),
            "coefficients": c,
            "a_tilde": a_tilde,
            "null_coefficients": c_null,
        }
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(SEED)
    cfg = make_quick_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)

    model, integrator = build_model(cfg, device, dtype)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    init_param_magnitude = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()])

    train_data, val_data = make_dataset(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *a, **kw: generic_residuals(*a, **kw, p_dim=2)
    )
    run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data,
        trajectory_window=cfg.trajectory_window, residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    trained_param_magnitude = torch.cat([p.detach().abs().reshape(-1) for p in model.parameters()])

    results = {}
    results["trained"] = analyse("trained", model, cfg, device, dtype, trained_param_magnitude, rng)
    model.load_state_dict(init_state)
    results["init"] = analyse("init", model, cfg, device, dtype, init_param_magnitude, rng)

    print("\n=== Order parameter / representation quality ===")
    for label in ("init", "trained"):
        r = results[label]
        print(f"{label}: ||g|| = {r['target_norm']:.6g}")
        for method in ("l2", "l1", "elastic_net"):
            m = r[method]
            print(
                f"  {method:12s} rank={m['resolved_rank']:3d}  repr_quality={m['representation_quality']:.4f}  "
                f"n_eff={m['n_eff']:7.2f}  n_eff/rank={m['n_eff_over_rank']:.3f}  "
                f"n_eff(null_control)={m['n_eff_null_control']:7.2f}  "
                f"rho(|c|,||S||)={m['rho_sensitivity']:+.3f}  rho(|c|,|theta|)={m['rho_magnitude']:+.3f}  "
                f"sum(a~_i)={m['a_tilde_sum']:.4f}"
            )

    # --- global-scale confound: raw vs normalized-share init-vs-trained scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for method, color in zip(("l2", "l1", "elastic_net"), ("tab:blue", "tab:orange", "tab:green")):
        c_init = results["init"][method]["coefficients"].abs().numpy()
        c_trained = results["trained"][method]["coefficients"].abs().numpy()
        axes[0].scatter(c_init + 1e-30, c_trained + 1e-30, s=8, alpha=0.5, label=method, color=color)

        share_init = c_init / max(c_init.sum(), 1e-30)
        share_trained = c_trained / max(c_trained.sum(), 1e-30)
        axes[1].scatter(share_init + 1e-30, share_trained + 1e-30, s=8, alpha=0.5, label=method, color=color)

    for ax, title in zip(axes, ["raw |c_i|: init vs trained", "share |c_i|/sum|c_j|: init vs trained"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        lims = [1e-12, 1e1]
        ax.plot(lims, lims, "k--", linewidth=1, label="y=x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("init")
        ax.set_ylabel("trained")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scale_confound_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- null control Lorenz curve (trained model, elastic net) ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, label, style in [
        ("coefficients", "real xrot generator", "-"),
        ("null_coefficients", "random matched-norm target (null control)", "--"),
    ]:
        c = results["trained"]["elastic_net"][key].abs().numpy()
        sorted_share = np.sort(c)[::-1].cumsum() / max(c.sum(), 1e-30)
        frac_params = np.arange(1, len(sorted_share) + 1) / len(sorted_share)
        ax.plot(frac_params, sorted_share, style, label=label)
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="uniform (no concentration)")
    ax.set_xlabel("fraction of parameters (sorted by |c_i|, descending)")
    ax.set_ylabel("cumulative share of |c_i| mass")
    ax.set_title("Lorenz curve: real generator vs. null control (trained, elastic net)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "null_control_lorenz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
