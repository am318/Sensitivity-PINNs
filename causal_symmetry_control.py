"""Does c_i identify which parameters *control* the symmetry, or just read off J's conditioning?

The claim under test is causal and identity-level: "the attribution
coefficients indicate *which parameters control the realisation of the
symmetry* in a network with low equivariance error". Every diagnostic in the
project so far tests *concentration* (n_eff / PR(c) against a null), which is
neither identity-level nor causal, and is exactly the quantity a
conditioning-driven artefact would also produce.

The confound is not hypothetical -- it is algebraic. For the L2 attribution,

    c_i = (V_{:,i} . w) / ||J_{:,i}||^2,     w = (A A^T)^{-1} U_k^T g / s_k,

(verified to 4e-16 in ``_closed_form_residual`` below). The factor
``1/||J_{:,i}||^2`` is *completely target-independent*: it is the same for the
true symmetry generator, for a random target, for anything. Only ``V_{:,i}.w``
knows about the symmetry. If the dynamic range of the conditioning factor
dominates that of the alignment factor -- which is what a converged, badly
conditioned J gives you -- then the |c_i| *ranking* is a readout of training
convergence and says nothing about symmetry. Elastic net inherits the same
``1/colscale`` weighting through its normalized-basis solve.

So this script tests the claim the way the claim is worded, with the
conditioning confound as an explicit, matched baseline throughout:

  Test D (decomposition). How much of Var(log|c_i|) is the pure conditioning
    factor? Reports R^2 of log|c_i| on log||J_{:,i}||, and the closed-form
    identity check.

  Test A (sufficiency, cross-selection). Select the top-k parameters by each
    score, then refit *by unconstrained least squares on those k columns
    only* (so no method is penalised for shrinkage) and measure how well the
    k-parameter subset reproduces each target. Run as a cross-matrix over
    (selection target) x (evaluation target). Diagonal dominance = the
    attribution picks parameters specific to the target it was computed
    from; a flat matrix = every selection is picking the same
    well-conditioned columns, i.e. the confound.

  Test B (necessity, causal ablation). Perturb *only* the selected k
    parameters, with the perturbation norm matched across methods, and
    measure the induced change in the symmetry defect D = ||delta||^2/||f||^2
    against the change in task loss. The claim needs c-selected parameters to
    move the symmetry defect *disproportionately* -- a conditioning-selected
    set moves everything (loss included), which is what "control" must be
    separated from. Reported as selectivity dlogD / dlogL.

  Test C (dense, per-parameter causal ground truth). grad_theta D is the exact
    local causal influence of every parameter on the symmetry defect. Reports
    Spearman(|c_i|, |grad_i D|) and, crucially, the *partial* Spearman
    controlling for log||J_{:,i}|| -- i.e. does the attribution predict causal
    influence on the symmetry beyond what conditioning alone already predicts?

Baselines everywhere: conditioning-only (column norm, resolved-subspace
leverage), naive target alignment |J^T g|, task-gradient magnitude, matched-
construction null attribution, and random.

Env vars: ARCHITECTURE, DEVICE, TRAINING_STEPS, TOY, SEED, N_QUADRATURE.
"""

from __future__ import annotations

import math
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
    group_average_and_defect,
    make_dataset,
    make_optimizer,
    residuals,
    rotation_generator_target,
    rotation_matrix,
    transform_points,
    validate_config,
)
from experiment_common import parameter_layout, run_training_loop, select_device, select_dtype
from mexican_hat_dynamics import VerletIntegrator  # noqa: F401  (integrator built by build_model)
from sensitivity_tools import (
    choose_l1_ratio_for_sparsity,
    finite_transform_residual,
    linear_generator_target,
    participation_ratio_l1l2,
    scale_invariant_attribution_score,
    random_symmetry_test_matrix,
    tangent_projection,
    tangent_projection_auto,
)

ARCHITECTURE = os.environ.get("ARCHITECTURE", "hamiltonian")
DEVICE = os.environ.get("DEVICE", "cpu")
TOY = os.environ.get("TOY", "0") == "1"
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "400" if TOY else "8000"))
SEED = int(os.environ.get("SEED", "0"))
N_QUADRATURE = int(os.environ.get("N_QUADRATURE", "24"))
# The claim is conditioned on "a network with low equivariance error", so the
# regime has to be produced deliberately, and contrasted against an equally
# converged network that did NOT learn the symmetry as well. Rotation
# augmentation is exactly valid extra data here (the dynamics is exactly
# SO(2)-equivariant at every alpha), so AUGMENT=1 vs AUGMENT=0 at matched
# training steps varies equivariance quality while holding architecture,
# optimiser and convergence level fixed -- which is what separates "the
# attribution tracks the symmetry" from "the attribution tracks convergence".
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
ALPHA = 0.3
N_NULL_DRAWS = 5
N_ABLATION_DRAWS = 24
ABLATION_EPSILON = 0.05
OUTPUT_DIR = Path(f"outputs/causal_symmetry_control_{ARCHITECTURE}"
                  + ("_augmented" if AUGMENT else "") + ("_toy" if TOY else ""))


# ---------------------------------------------------------------- setup ----


def make_config() -> Config:
    cfg = Config()
    cfg.seed = SEED
    cfg.architecture = ARCHITECTURE
    cfg.device = DEVICE
    cfg.training_steps = TRAINING_STEPS
    cfg.augment_dataset = AUGMENT
    cfg.checkpoint_fractions = [0.0, 1.0]
    if TOY:
        cfg.kinetic_hidden_dim = 8
        cfg.potential_hidden_dim = 8
        cfg.initial_conditions_per_alpha = 2
        cfg.trajectory_splits = 2
        cfg.coarsening_factor = 2
        cfg.q_grid_points_per_axis = 4
    validate_config(cfg)
    return cfg


def build_polar_probe_grid(n_radii, n_angles, r_max, device, dtype):
    """Concentric circles evenly spaced in angle: exactly invariant as a SET under
    rotation by any multiple of 360/n_angles, which the group average needs."""
    radii = torch.linspace(r_max / n_radii, r_max, n_radii, device=device, dtype=dtype)
    angles = torch.linspace(0, 2 * math.pi, n_angles + 1, device=device, dtype=dtype)[:-1]
    r_grid, a_grid = torch.meshgrid(radii, angles, indexing="ij")
    return (r_grid * torch.cos(a_grid)).reshape(-1), (r_grid * torch.sin(a_grid)).reshape(-1)


def batched_force(model, q1, q2, alpha, architecture, *, create_graph=False):
    """F(q) for all probe points at once, differentiable w.r.t. theta.

    ``evaluate_at_points`` loops point-by-point because it also builds the
    per-parameter Jacobian rows; the causal tests below only ever need the
    force itself, thousands of times (24-48 rotations x many perturbation
    draws), so they need the batched path instead.
    """
    q = torch.stack([q1, q2], dim=1).clone().requires_grad_(True)
    alpha_t = torch.full((q.shape[0], 1), float(alpha), device=q.device, dtype=q.dtype)
    if architecture == "direct_mlp":
        dpdt, _ = model(torch.zeros_like(q), q, alpha_t)
        return dpdt
    potential = model.V_net(torch.cat((q, alpha_t), dim=1))
    return -torch.autograd.grad(potential.sum(), q, create_graph=True)[0]


def defect_order_parameter(model, q1, q2, alpha, architecture, n_quadrature):
    """D = ||f - Pi f||^2 / ||f||^2 on the probe grid: 0 iff exactly SO(2)-equivariant.

    Scale-free (invariant to an overall rescaling of the force) and fully
    differentiable in theta, so it serves both as the intervention read-out
    in Test B and as the dense causal ground truth grad_theta D in Test C.
    """
    f_x = batched_force(model, q1, q2, alpha, architecture)
    pi_f = torch.zeros_like(f_x)
    for k in range(n_quadrature):
        r_phi = rotation_matrix(360.0 * k / n_quadrature, f_x.device, f_x.dtype)
        q1_rot, q2_rot = transform_points(q1, q2, r_phi)
        f_rot = batched_force(model, q1_rot, q2_rot, alpha, architecture)
        pi_f = pi_f + f_rot @ r_phi  # rotate back by R_phi^{-1} = R_phi^T, applied on the right as R_phi
    pi_f = pi_f / n_quadrature
    return (f_x - pi_f).pow(2).sum() / f_x.pow(2).sum().clamp_min(1e-30)


def task_loss(model, train_data, window, integrator, residuals_fn) -> torch.Tensor:
    trajectories, params, instants = train_data
    return residuals_fn(trajectories, params, instants, window, integrator)


# ------------------------------------------------------- score building ----


def resolved_subspace_leverage(j: torch.Tensor, cutoff: float) -> torch.Tensor:
    """||V_k[:, i]||^2 -- how much of parameter i lives in the resolved tangent space.

    A purely target-independent conditioning score: it is a property of J
    alone. Together with the column norm it spans "everything the attribution
    could know without ever looking at the symmetry".
    """
    _, s, vh = torch.linalg.svd(j.to(torch.float64), full_matrices=False)
    keep = s >= cutoff * s[0]
    return vh[keep, :].pow(2).sum(dim=0)


def _closed_form_residual(j: torch.Tensor, g: torch.Tensor, cutoff: float) -> tuple[float, torch.Tensor]:
    """Check c_i == (V_{:,i}.w)/colscale_i^2 for the L2 solve, and return the closed form."""
    jj, gg = j.to(torch.float64), g.to(torch.float64)
    _, _, c_l2, _, _ = tangent_projection(jj, gg, cutoff, normalize_columns=True)
    u, s, vh = torch.linalg.svd(jj, full_matrices=False)
    keep = s >= cutoff * s[0]
    vk, sk, uk = vh[keep, :], s[keep], u[:, keep]
    colscale = jj.norm(dim=0).clamp_min(jj.norm(dim=0).max() * 1e-8)
    # The solve zeroes dead columns (a parameter with no effect on the output must get
    # exactly zero attribution, not SVD noise divided by a floor), so the closed form has
    # to do the same or the comparison is against a different estimator.
    dead = jj.norm(dim=0) <= jj.norm(dim=0).max() * 1e-8
    a = vk / colscale
    a[:, dead] = 0.0
    w = torch.linalg.solve(a @ a.T, (uk.T @ gg) / sk)
    c_closed = (vk.T @ w) / colscale**2
    c_closed[dead] = 0.0
    denom = c_l2.abs().max().clamp_min(1e-300)
    return float((c_l2 - c_closed).abs().max() / denom), c_l2


def _l2_attribution(j_flat, target, cutoff) -> np.ndarray:
    """|J^+ g| with column normalisation -- the exact object the decomposition applies to."""
    _, _, c, _, _ = tangent_projection(j_flat, target, cutoff, normalize_columns=True)
    return c.numpy()


def attribution(j_flat, target, cutoff, l1_ratio=None):
    ratio = l1_ratio if l1_ratio is not None else choose_l1_ratio_for_sparsity(j_flat, target, cutoff)
    err, ang, c, rank, _ = tangent_projection_auto(
        j_flat, target, cutoff, method="elastic_net", l1_ratio=ratio
    )
    return c, err, rank, ratio


# ----------------------------------------------------------- the tests ----


def subset_refit_error(j: np.ndarray, target: np.ndarray, support: np.ndarray) -> float:
    """Relative error of the BEST fit of ``target`` using only columns ``support``.

    Unconstrained least squares on the selected columns, so a method is judged
    purely on *which* parameters it picked -- never on the magnitude the
    (shrunk, regularised) attribution happened to assign them.
    """
    if support.size == 0:
        return 1.0
    a = j[:, support]
    coef, *_ = np.linalg.lstsq(a, target, rcond=None)
    return float(np.linalg.norm(target - a @ coef) / max(np.linalg.norm(target), 1e-30))


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(score))[:k]


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman(x, y) with the (rank-)linear effect of z removed from both."""
    from scipy.stats import pearsonr, rankdata

    rx, ry, rz = (rankdata(v).astype(np.float64) for v in (x, y, z))
    design = np.stack([rz, np.ones_like(rz)], axis=1)
    rx_res = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry_res = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    r, p = pearsonr(rx_res, ry_res)
    return float(r), float(p)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    device, dtype = select_device(cfg.device), select_dtype(cfg.dtype)
    print(f"architecture={cfg.architecture} device={device} steps={cfg.training_steps} "
          f"toy={TOY} augment={AUGMENT} n_quadrature={N_QUADRATURE}")

    train_data, val_data = make_dataset(cfg, device, dtype)
    model, integrator = build_model(cfg, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    optimizer = make_optimizer(cfg, model)
    history, _ = run_training_loop(
        training_steps=cfg.training_steps, checkpoint_steps=cfg.checkpoint_steps,
        optimizer=optimizer, optimizer_name=cfg.optimizer, model=model,
        train_data=train_data, val_data=val_data, trajectory_window=cfg.trajectory_window,
        residuals_fn=residuals, integrator=integrator,
        l1_weight=cfg.l1_weight, weight_decay=cfg.weight_decay,
    )
    model.eval()

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_QUADRATURE, cfg.q_extent, device, dtype)
    n_points = q1.shape[0]
    cutoff = cfg.tangent_svd_relative_cutoff

    # ---- regime check: is this actually a "network with low equivariance error"? ----
    f_x, j_x, pi_f, j_pi, delta, j_delta = group_average_and_defect(
        model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, n_quadrature=N_QUADRATURE
    )
    rot = rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    q1r, q2r = transform_points(q1, q2, rot)
    _, f_rot, j_rot = evaluate_at_points(model, q1r, q2r, ALPHA, cfg.architecture, device=device, dtype=dtype)
    equivariance_residual = finite_transform_residual(f_x, f_rot, rot)
    defect_ratio = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(f_x))
    d_value = float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_QUADRATURE).detach())
    final_loss = history["trajectory_loss"][-1]
    print(f"\n=== regime ===\ntrajectory loss = {final_loss:.4e}")
    print(f"equivariance residual ||f(gx)-rho f(x)||/||f|| at {cfg.rotation_angle_degrees} deg = {equivariance_residual:.4e}")
    print(f"||delta||/||f|| = {defect_ratio:.4e}   D = ||delta||^2/||f||^2 = {d_value:.4e}")

    _, _, _, spatial_jac = evaluate_at_points(
        model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
    )
    g = rotation_generator_target(q1, q2, f_x, spatial_jac).reshape(n_points * 2)
    j_flat = j_x.reshape(n_points * 2, -1)
    delta_flat = delta.reshape(n_points * 2)
    j_delta_flat = j_delta.reshape(n_points * 2, -1)
    positions = torch.stack([q1, q2], dim=1)
    n_params = j_flat.shape[1]

    # ---- targets: the true generator plus matched-construction nulls ----
    null_targets = []
    for _ in range(N_NULL_DRAWS):
        m = random_symmetry_test_matrix(device=device, dtype=dtype, generator=rng)
        null_targets.append(linear_generator_target(positions, f_x, spatial_jac, m).reshape(n_points * 2))

    c_rep, err_rep, rank_rep, ratio_rep = attribution(j_flat, g, cutoff)
    c_def, err_def, rank_def, _ = attribution(j_delta_flat, delta_flat, cutoff)
    c_nulls = [attribution(j_flat, t, cutoff, l1_ratio=ratio_rep)[0] for t in null_targets]
    print(f"\nc^rep: rank={rank_rep} rel_err={err_rep:.4g} n_eff={participation_ratio_l1l2(c_rep.numpy()):.2f}")
    print(f"c^def: rank={rank_def} rel_err={err_def:.4g} n_eff={participation_ratio_l1l2(c_def.numpy()):.2f}")

    # ---- conditioning-only and naive scores ----
    j64 = j_flat.detach().cpu().to(torch.float64)
    colnorm = j64.norm(dim=0)
    leverage = resolved_subspace_leverage(j64, cutoff)
    align = (j64.T @ g.detach().cpu().to(torch.float64)).abs()
    loss_for_grad = task_loss(model, train_data, cfg.trajectory_window, integrator, residuals)
    task_grad = torch.cat([
        (torch.zeros_like(p) if gr is None else gr).reshape(-1)
        for p, gr in zip(model.parameters(),
                         torch.autograd.grad(loss_for_grad, tuple(model.parameters()), allow_unused=True))
    ]).detach().cpu().to(torch.float64).abs()
    rand_score = torch.rand(n_params, generator=rng, dtype=torch.float64)

    # The algebra above says |c_i| = |V_:,i . w| / ||J_:,i||^2 carries a purely
    # target-independent conditioning factor. Multiplying it back out leaves the
    # *only* part of the attribution that knows about the symmetry -- so if the
    # claim is true of anything, it should be true of this, and more cleanly.
    alignment_only = (c_rep.abs().to(torch.float64) * colnorm**2).numpy()
    a_share = scale_invariant_attribution_score(j_flat, g, c_rep).abs().numpy()

    # The conditioning factor 1/||S_i||^2 in c_i = a_i/||S_i||^2 is *identical* for
    # every target on the same J, so it cancels EXACTLY in a ratio of attributions
    # against two targets -- no regression, no modelling assumption. That makes
    #     r_i = |c_i(g)| / geomean_k |c_i(null_k)|
    # a provably conditioning-free attribution: it isolates the alignment factor
    # a_i = V_{:,i}.w, the only part of c_i that ever knew about the symmetry.
    # Computed from the L2/pseudo-inverse solve (which is what the identity is about,
    # and what the paper's J^+ g is); elastic net's sparsity would make the ratio 0/0.
    live = (colnorm > colnorm.max() * 1e-8).numpy()
    c_l2_g = _l2_attribution(j_flat, g, cutoff)
    c_l2_nulls = np.stack([_l2_attribution(j_flat, t, cutoff) for t in null_targets])
    log_null = np.log(np.clip(np.abs(c_l2_nulls), 1e-300, None)).mean(axis=0)
    calibrated = np.where(live, np.abs(c_l2_g) / np.clip(np.exp(log_null), 1e-300, None), 0.0)
    print(f"\nlive (force-affecting) parameters: {int(live.sum())} of {n_params}")

    scores = {
        "c^rep (attribution of g)": c_rep.abs().numpy(),
        "c^L2 = |J^+ g| (the paper's score)": np.abs(c_l2_g),
        "r_i (null-calibrated, conditioning-free)": calibrated,
        "c^rep * ||J_:,i||^2 (alignment only)": alignment_only,
        "a_i (scale-invariant share)": a_share,
        "c^def (attribution of delta)": c_def.abs().numpy(),
        "c^null (matched null target)": c_nulls[0].abs().numpy(),
        "||J_:,i|| (conditioning)": colnorm.numpy(),
        "leverage ||V_k[:,i]||^2 (conditioning)": leverage.numpy(),
        "|J^T g| (naive alignment)": align.numpy(),
        "|task gradient|": task_grad.numpy(),
        "random": rand_score.numpy(),
    }

    # ================================================== Test D: decomposition ----
    print("\n=== Test D: how much of the attribution ranking is pure conditioning? ===")
    closed_res, c_l2 = _closed_form_residual(j_flat, g, cutoff)
    print(f"closed-form identity c_i = (V_:,i.w)/||J_:,i||^2 residual = {closed_res:.3e}  (L2 solve)")
    log_col = np.log(np.clip(colnorm.numpy(), 1e-300, None))
    decomposition = {}
    for label, vec in (("c^rep", c_rep.abs().numpy()), ("c^def", c_def.abs().numpy()),
                       ("c^null", c_nulls[0].abs().numpy()), ("c (L2, g)", c_l2.abs().numpy())):
        live = vec > 0
        if live.sum() < 3:
            continue
        lc = np.log(vec[live])
        design = np.stack([log_col[live], np.ones(int(live.sum()))], axis=1)
        fit = design @ np.linalg.lstsq(design, lc, rcond=None)[0]
        r2 = 1.0 - float(np.var(lc - fit) / max(np.var(lc), 1e-30))
        rho = float(spearmanr(vec[live], colnorm.numpy()[live]).statistic)
        decomposition[label] = {"r2_on_log_colnorm": r2, "spearman_colnorm": rho, "n_live": int(live.sum())}
        print(f"  {label:12s} R^2(log|c| ~ log||J_:,i||) = {r2:.3f}   Spearman(|c|, ||J_:,i||) = {rho:+.3f}  "
              f"(n_live={int(live.sum())})")
    # Correlating a sparse c against anything over all P coordinates is dominated by
    # the thousands of coordinates both vectors set to exactly zero, which inflates
    # agreement for reasons that have nothing to do with the symmetry. Every rank
    # statistic below is therefore evaluated on ONE common, method-neutral index set:
    # the union of the live supports of all the attributions being compared.
    live_union = np.zeros(n_params, dtype=bool)
    for vec in [c_rep, c_def, *c_nulls]:
        live_union |= vec.abs().numpy() > 0
    print(f"  common live support (union of all attribution supports): {int(live_union.sum())} of {n_params} parameters")
    rho_rep_null_all = float(spearmanr(c_rep.abs().numpy(), c_nulls[0].abs().numpy()).statistic)
    rho_rep_null = float(spearmanr(c_rep.abs().numpy()[live_union], c_nulls[0].abs().numpy()[live_union]).statistic)
    print(f"  Spearman(|c^rep|, |c^null|) on live support = {rho_rep_null:+.3f}  (over all P: {rho_rep_null_all:+.3f})")
    print(f"     ~1 would mean the attribution ranking barely depends on the target at all")

    # ============================== Test A: sufficiency / cross-selection matrix ----
    print("\n=== Test A: does a k-parameter subset chosen for target T reproduce T better than other targets? ===")
    j_np = j64.numpy()
    eval_targets = {"g (true generator)": g.detach().cpu().to(torch.float64).numpy()}
    for i, t in enumerate(null_targets):
        eval_targets[f"null_{i+1}"] = t.detach().cpu().to(torch.float64).numpy()
    select_scores = {"c(g)": c_rep.abs().numpy()}
    for i, c_n in enumerate(c_nulls):
        select_scores[f"c(null_{i+1})"] = c_n.abs().numpy()
    select_scores["c(g)*||J||^2"] = alignment_only
    select_scores["r_i (calibrated)"] = calibrated
    select_scores["||J_:,i||"] = colnorm.numpy()
    select_scores["leverage"] = leverage.numpy()
    select_scores["random"] = rand_score.numpy()

    # k must stay well below the row count N = 2*n_points: with k comparable to N an
    # unconstrained least-squares refit reproduces the target from *any* support, and
    # the matrix goes flat for reasons unrelated to how good the selection was.
    n_rows = n_points * 2
    k_cross = int(max(4, min(rank_rep // 4, n_rows // 8, 40)))
    cross_by_k = {}
    for k_here in sorted({max(2, k_cross // 4), max(3, k_cross // 2), k_cross}):
        mat = np.zeros((len(select_scores), len(eval_targets)))
        for si, (sname, sc) in enumerate(select_scores.items()):
            support = top_k(sc, k_here)
            for ti, (tname, tv) in enumerate(eval_targets.items()):
                mat[si, ti] = subset_refit_error(j_np, tv, support)
        cross_by_k[k_here] = mat
        print(f"\n  relative refit error using only the top-k={k_here} parameters of N={n_rows} rows "
              f"(lower = better); rows = selection, cols = evaluated target")
        print("  " + " " * 16 + "".join(f"{t:>16s}" for t in eval_targets))
        for si, sname in enumerate(select_scores):
            print("  " + f"{sname:16s}" + "".join(f"{mat[si, ti]:16.4f}" for ti in range(len(eval_targets))))
        diag_adv = float(np.mean([mat[si, si] - np.mean(np.delete(mat[si], si))
                                  for si in range(min(len(select_scores), len(eval_targets)))]))
        print(f"  mean (own-target - other-target) refit error over the attribution rows: {diag_adv:+.4f}  "
              f"(negative = target-specific selection)")
    cross = cross_by_k[k_cross]

    # sweep k for the headline curve (true generator only)
    k_values = sorted({max(2, int(v)) for v in np.geomspace(4, min(4 * rank_rep, n_params, n_points * 2 - 1), 8)})
    sufficiency = {name: [subset_refit_error(j_np, eval_targets["g (true generator)"], top_k(sc, k)) for k in k_values]
                   for name, sc in scores.items()}

    # ===================================== Test B: causal ablation / selectivity ----
    print("\n=== Test B: perturb ONLY the selected k parameters (matched norm); "
          "does the symmetry defect move more than the task loss? ===")
    base_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    flat_theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    theta_norm = float(flat_theta.norm())
    base_d = float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_QUADRATURE).detach())
    base_l = float(task_loss(model, train_data, cfg.trajectory_window, integrator, residuals).detach())
    print(f"  baseline D = {base_d:.6e}   baseline task loss = {base_l:.6e}   "
          f"perturbation norm = {ABLATION_EPSILON:.3g} * ||theta|| = {ABLATION_EPSILON * theta_norm:.4g}")

    k_ablate = k_cross
    param_list = list(model.parameters())
    sizes = [p.numel() for p in param_list]

    def set_flat(vec: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p, n in zip(param_list, sizes):
                p.copy_(vec[offset:offset + n].view_as(p))
                offset += n

    ablation_rows = {}
    for name, sc in scores.items():
        support = torch.from_numpy(top_k(sc, k_ablate)).to(device=flat_theta.device)
        d_ratios, l_ratios = [], []
        for _ in range(N_ABLATION_DRAWS):
            direction = torch.zeros_like(flat_theta)
            noise = torch.randn(support.numel(), generator=rng, dtype=torch.float32).to(
                device=flat_theta.device, dtype=flat_theta.dtype
            )
            direction[support] = noise
            direction = direction / direction.norm().clamp_min(1e-30) * (ABLATION_EPSILON * theta_norm)
            set_flat(flat_theta + direction)
            d_ratios.append(float(defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_QUADRATURE).detach()) / max(base_d, 1e-30))
            l_ratios.append(float(task_loss(model, train_data, cfg.trajectory_window, integrator, residuals).detach()) / max(base_l, 1e-30))
        model.load_state_dict(base_state)
        d_log = float(np.mean(np.log(np.clip(d_ratios, 1e-30, None))))
        l_log = float(np.mean(np.log(np.clip(l_ratios, 1e-30, None))))
        d_se = float(np.std(np.log(np.clip(d_ratios, 1e-30, None))) / math.sqrt(N_ABLATION_DRAWS))
        ablation_rows[name] = {
            "dlogD": d_log, "dlogD_se": d_se, "dlogL": l_log,
            "selectivity": d_log / l_log if abs(l_log) > 1e-12 else float("nan"),
        }
        print(f"  {name:40s} dlogD={d_log:+.4f} (se {d_se:.4f})  dlogL={l_log:+.4f}  "
              f"selectivity={ablation_rows[name]['selectivity']:+.3f}")

    # =================== Test C: dense causal ground truth grad_theta D ----
    print("\n=== Test C: does |c_i| predict per-parameter causal influence on the defect, "
          "beyond what conditioning already predicts? ===")
    model.load_state_dict(base_state)
    d_scalar = defect_order_parameter(model, q1, q2, ALPHA, cfg.architecture, N_QUADRATURE)
    grad_d = torch.cat([
        (torch.zeros_like(p) if gr is None else gr).reshape(-1)
        for p, gr in zip(model.parameters(),
                         torch.autograd.grad(d_scalar, tuple(model.parameters()), allow_unused=True))
    ]).detach().cpu().to(torch.float64).abs().numpy()

    log_col_all = np.log(np.clip(colnorm.numpy(), 1e-300, None))
    print(f"  (evaluated on the {int(live_union.sum())}-parameter common live support)")
    causal_rows = {}
    for name, sc in scores.items():
        sub_sc, sub_gd, sub_col = sc[live_union], grad_d[live_union], log_col_all[live_union]
        rho = float(spearmanr(sub_sc, sub_gd).statistic)
        prho, ppval = partial_spearman(sub_sc, sub_gd, sub_col)
        rho_all = float(spearmanr(sc, grad_d).statistic)
        causal_rows[name] = {"spearman": rho, "partial_spearman": prho, "partial_p": ppval,
                             "spearman_allP": rho_all}
        print(f"  {name:40s} Spearman(score, |grad D|) = {rho:+.3f}   "
              f"partial (control ||J_:,i||) = {prho:+.3f} (p={ppval:.3g})   [all-P rho {rho_all:+.3f}]")

    # ------------------------------------------------------------- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for name, curve in sufficiency.items():
        style = "-" if name.startswith("c^") else "--"
        axes[0].plot(k_values, curve, style, marker="o", markersize=3, label=name)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("k (parameters retained)")
    axes[0].set_ylabel("relative refit error of g")
    axes[0].set_title("Test A: sufficiency of the top-k parameters")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(cross, cmap="viridis")
    axes[1].set_xticks(range(len(eval_targets)))
    axes[1].set_xticklabels(list(eval_targets), rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(select_scores)))
    axes[1].set_yticklabels(list(select_scores), fontsize=7)
    axes[1].set_title(f"Test A: cross-selection refit error (k={k_cross}, N={n_points * 2})\ndiagonal dip = target-specific")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    names = list(ablation_rows)
    dlog_d = [ablation_rows[n]["dlogD"] for n in names]
    dlog_l = [ablation_rows[n]["dlogL"] for n in names]
    y = np.arange(len(names))
    axes[2].barh(y - 0.2, dlog_d, 0.4, label="dlog D (symmetry defect)")
    axes[2].barh(y + 0.2, dlog_l, 0.4, label="dlog L (task loss)")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=6)
    axes[2].set_title(f"Test B: matched-norm perturbation of top-{k_ablate}")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "causal_symmetry_control.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Persist J, the targets and the per-parameter scores: training is the expensive
    # part, and every observational test above (A, C, D) is a pure function of these,
    # so any later re-analysis (different k, different cutoff, extra baselines) costs
    # seconds instead of hours.
    torch.save({"state_dict": base_state, "architecture": cfg.architecture,
                "augment": AUGMENT, "training_steps": cfg.training_steps, "seed": SEED},
               OUTPUT_DIR / "model.pt")
    np.savez_compressed(
        OUTPUT_DIR / "analysis_inputs.npz",
        j=j_np, j_delta=j_delta_flat.detach().cpu().to(torch.float64).numpy(),
        g=eval_targets["g (true generator)"], delta=delta_flat.detach().cpu().to(torch.float64).numpy(),
        theta=flat_theta.detach().cpu().numpy(), live=live,
        c_l2_g=c_l2_g, c_l2_nulls=c_l2_nulls,
        **{f"null_target_{i+1}": t.detach().cpu().to(torch.float64).numpy() for i, t in enumerate(null_targets)},
        **{f"score__{name}": vec for name, vec in scores.items()},
    )
    np.savez(
        OUTPUT_DIR / "causal_symmetry_control.npz",
        k_values=np.array(k_values), cross=cross, k_cross=k_cross,
        **{f'cross_k{k}': m for k, m in cross_by_k.items()},
        cross_rows=np.array(list(select_scores)), cross_cols=np.array(list(eval_targets)),
        grad_d=grad_d, colnorm=colnorm.numpy(), leverage=leverage.numpy(),
        c_rep=c_rep.numpy(), c_def=c_def.numpy(), c_null=c_nulls[0].numpy(),
        equivariance_residual=equivariance_residual, defect_ratio=defect_ratio,
        final_loss=final_loss,
        **{f"suff_{i}": np.array(v) for i, v in enumerate(sufficiency.values())},
    )
    print(f"\nWritten to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
