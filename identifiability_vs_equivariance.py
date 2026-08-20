"""Does symmetry attribution become LESS identifiable as the network becomes MORE equivariant?

The claim under examination is conditioned on "a network with low equivariance
error". But for an exactly equivariant force field the rotational Lie
derivative vanishes identically, so the attribution target itself goes to zero:

    g = X f_theta = (dF/dq)(Omega q) - Omega F(q)  ->  0   as f_theta -> E = ker X.

So does delta = Q f_theta. Both are order parameters for symmetry breaking, and
Prop. 1 says they are orthogonal -- but they vanish *together*. The attribution
c = J^+ g is therefore a direction estimated from a quantity that is
disappearing exactly in the regime where the paper wants to talk about it, and
the natural worry is that what remains is dominated by whatever residual is
left over: optimisation noise, finite-probe-grid error, float precision.

That makes a sharp, falsifiable prediction the draft does not test:
identifiability of the symmetry-specific ranking should DEGRADE as equivariance
improves, even while |c_i| continues to look clean and sparse (because its
apparent structure is carried by the target-blind conditioning factor
1/||S_i||^2, which does not vanish and does not care about the symmetry).

This trains one model and, at each checkpoint, records
  - the equivariance error and the order parameter ||g||,
  - representation quality ||P_T g|| / ||g||,
  - and the probe-resampling identifiability of |c_i|, of the conditioning-free
    r_i, and of the null-vs-null control r'_i, at parameter and hidden-unit
    granularity,
so identifiability can be plotted directly against equivariance error.

Env vars: ARCHITECTURE, DEVICE, TRAINING_STEPS, AUGMENT, SEED, RESAMPLES.
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
    Config,
    build_model,
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
from attribution_stability import jaccard, scores_for, unit_index_map, unit_vector
from causal_symmetry_control import build_polar_probe_grid
from experiment_common import run_training_loop, select_device, select_dtype
from sensitivity_tools import (
    finite_transform_residual,
    linear_generator_target,
    random_symmetry_test_matrix,
    tangent_projection,
)

ARCHITECTURE = os.environ.get("ARCHITECTURE", "hamiltonian")
DEVICE = os.environ.get("DEVICE", "cpu")
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "8000"))
AUGMENT = os.environ.get("AUGMENT", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
RESAMPLES = int(os.environ.get("RESAMPLES", "24"))
TOY = os.environ.get("TOY", "0") == "1"
ALPHA = 0.3
N_NULLS = 5
N_ANGLES = 24
TOPK = 20
CHECKPOINT_FRACTIONS = [0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
OUTPUT_DIR = Path(f"outputs/identifiability_{ARCHITECTURE}" + ("_augmented" if AUGMENT else "")
                  + ("_toy" if TOY else ""))


def make_config() -> Config:
    cfg = Config()
    cfg.seed, cfg.architecture, cfg.device = SEED, ARCHITECTURE, DEVICE
    cfg.training_steps = TRAINING_STEPS
    cfg.augment_dataset = AUGMENT
    cfg.checkpoint_fractions = CHECKPOINT_FRACTIONS
    if TOY:
        cfg.kinetic_hidden_dim = cfg.potential_hidden_dim = 8
        cfg.initial_conditions_per_alpha = cfg.trajectory_splits = cfg.coarsening_factor = 2
        cfg.q_grid_points_per_axis = 4
    validate_config(cfg)
    return cfg


def stability_at(j: np.ndarray, g: np.ndarray, nulls: list[np.ndarray], units, rng, n_draws: int):
    """Probe-resampling identifiability of each score, at parameter and unit granularity."""
    n_rows = j.shape[0]
    keep = max(8, int(n_rows * 0.8))
    per_draw: dict[str, list[np.ndarray]] = {}
    for _ in range(n_draws):
        rows = rng.choice(n_rows, size=keep, replace=False)
        sc, _ = scores_for(j[rows], g[rows], [t[rows] for t in nulls])
        for name, vec in sc.items():
            per_draw.setdefault(name, []).append(vec)
    out = {}
    for name, vecs in per_draw.items():
        mat = np.stack(vecs)
        tops = [np.argsort(-m)[:TOPK] for m in mat]
        out[f"{name}|param_jaccard"] = float(np.mean(
            [jaccard(tops[a], tops[b]) for a in range(len(tops)) for b in range(a + 1, len(tops))]))
        if units:
            uu = np.stack([unit_vector(v, units) for v in vecs])
            n_top_u = max(3, uu.shape[1] // 10)
            utops = [np.argsort(-u)[:n_top_u] for u in uu]
            out[f"{name}|unit_jaccard"] = float(np.mean(
                [jaccard(utops[a], utops[b]) for a in range(len(utops)) for b in range(a + 1, len(utops))]))
            out[f"{name}|unit_rho"] = float(np.nanmean(
                [float(spearmanr(uu[a], uu[b]).statistic) for a in range(len(uu)) for b in range(a + 1, len(uu))]))
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tgen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device, dtype = select_device(cfg.device), select_dtype(cfg.dtype)
    print(f"architecture={cfg.architecture} augment={AUGMENT} steps={cfg.training_steps} "
          f"checkpoints={len(CHECKPOINT_FRACTIONS)} resamples={RESAMPLES}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    optimizer = make_optimizer(cfg, model)
    residuals_fn = residuals if cfg.architecture == "hamiltonian" else (
        lambda *a, **kw: generic_residuals(*a, **kw, p_dim=2))
    history, checkpoints = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data, trajectory_window=cfg.trajectory_window,
        residuals_fn=residuals_fn, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    n_points = q1.shape[0]
    rot = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    q1r, q2r = transform_points(q1, q2, rot)
    units = unit_index_map(model)
    step_to_loss = dict(zip(history["step"], history["trajectory_loss"]))
    cutoff = cfg.tangent_svd_relative_cutoff

    rows = []
    for step in sorted(checkpoints):
        model.load_state_dict(checkpoints[step])
        model.eval()
        _, f_x, j_x, spatial = evaluate_at_points(
            model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True)
        _, f_rot, _ = evaluate_at_points(model, q1r, q2r, ALPHA, cfg.architecture, device=device, dtype=dtype)
        equi = finite_transform_residual(f_x, f_rot, rot)

        g_t = rotation_generator_target(q1, q2, f_x, spatial).reshape(n_points * 2)
        j_flat = j_x.reshape(n_points * 2, -1)
        positions = torch.stack([q1, q2], dim=1)
        nulls_t = [linear_generator_target(
            positions, f_x, spatial,
            random_symmetry_test_matrix(device=device, dtype=dtype, generator=tgen)).reshape(n_points * 2)
            for _ in range(N_NULLS)]

        err, _, _, rank, _ = tangent_projection(j_flat, g_t, cutoff)
        g_norm = float(torch.linalg.vector_norm(g_t))
        f_norm = float(torch.linalg.vector_norm(f_x))
        j_np = j_flat.detach().cpu().to(torch.float64).numpy()
        g_np = g_t.detach().cpu().to(torch.float64).numpy()
        nulls_np = [t.detach().cpu().to(torch.float64).numpy() for t in nulls_t]
        stab = stability_at(j_np, g_np, nulls_np, units, rng, RESAMPLES)

        row = {"step": step, "loss": step_to_loss.get(step, float("nan")),
               "equivariance": equi, "g_norm": g_norm, "g_rel": g_norm / max(f_norm, 1e-30),
               "rep_error": err, "rank": rank, **stab}
        rows.append(row)
        keys = ('|c_i| = |J^+ g|', 'r_i (null-calibrated)', "r'_i (null-vs-null CONTROL)")
        pj = [stab[f"{k}|param_jaccard"] for k in keys]
        uj = [stab.get(f"{k}|unit_jaccard", float("nan")) for k in keys]
        print(f"step={step:6d} loss={row['loss']:.3e} equi={equi:.4e} "
              f"||g||/||f||={row['g_rel']:.4e} rep_err={err:.3f} rank={rank:3d}"
              f" | param Jacc |c|={pj[0]:.3f} r={pj[1]:.3f} r'={pj[2]:.3f}"
              f" | unit Jacc |c|={uj[0]:.3f} r={uj[1]:.3f} r'={uj[2]:.3f}")

    def npz_key(k: str) -> str:  # score labels carry |, ^, spaces; keep npz keys plain
        return "".join(ch if ch.isalnum() else "_" for ch in k).strip("_")

    np.savez(OUTPUT_DIR / "identifiability.npz",
             **{npz_key(k): np.array([r[k] for r in rows]) for k in rows[0]})

    equi = np.array([r["equivariance"] for r in rows])
    steps = np.array([r["step"] for r in rows])

    # The prediction is directional, so state it as a correlation rather than leaving it to
    # the eye: does the symmetry-specific margin (calibrated score minus its null-vs-null
    # control) shrink as the equivariance residual falls, and does representation quality
    # degrade with it?
    from scipy.stats import pearsonr, spearmanr

    key_c, key_r, key_n = "|c_i| = |J^+ g|", "r_i (null-calibrated)", "r'_i (null-vs-null CONTROL)"
    log_equi = np.log(np.clip(equi, 1e-300, None))
    rep_err = np.array([r["rep_error"] for r in rows])
    print("\n=== does identifiability degrade as equivariance improves? ===")
    for gran in ("param", "unit"):
        margin = np.array([r[f"{key_r}|{gran}_jaccard"] - r[f"{key_n}|{gran}_jaccard"] for r in rows])
        pr_, pp = pearsonr(log_equi, margin)
        print(f"  {gran:5s}: mean margin (r - r') = {margin.mean():+.3f}   "
              f"corr(log equivariance, margin) r={pr_:+.3f} (p={pp:.3f}) "
              f"rho={spearmanr(log_equi, margin).statistic:+.3f}")
    pr_, pp = pearsonr(log_equi, rep_err)
    print(f"  corr(log equivariance residual, representation error) = {pr_:+.3f} (p={pp:.4f})")
    uc = np.array([r[f"{key_c}|unit_jaccard"] for r in rows])
    pr_, pp = pearsonr(log_equi, uc)
    print(f"  corr(log equivariance residual, |c_i| unit stability) = {pr_:+.3f} (p={pp:.3f})")
    print("  positive margin correlation = the symmetry-specific signal shrinks as the model")
    print("  becomes more equivariant; a flat |c_i| correlation = |c_i| is not tracking symmetry.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(steps, equi, marker="o", label="equivariance residual")
    axes[0].plot(steps, [r["g_rel"] for r in rows], marker="s", label="||g||/||f|| (order parameter)")
    axes[0].set_xlabel("training step")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("The attribution target vanishes as equivariance improves")

    for key, style in (("|c_i| = |J^+ g|", "-o"), ("r_i (null-calibrated)", "-s"),
                       ("r'_i (null-vs-null CONTROL)", "--^")):
        for ax, gran in zip(axes[1:], ("param", "unit")):
            ax.plot(equi, [r[f"{key}|{gran}_jaccard"] for r in rows], style, markersize=4, label=key)
    for ax, gran in zip(axes[1:], ("parameter", "hidden unit")):
        ax.set_xscale("log")
        ax.set_xlabel("equivariance residual  (better ->)")
        ax.set_ylabel(f"top-k {gran} reproducibility (Jaccard)")
        ax.invert_xaxis()
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"Identifiability at {gran} level vs equivariance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "identifiability_vs_equivariance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWritten to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
