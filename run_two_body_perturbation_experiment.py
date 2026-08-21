"""Two-body c_i/b_i intervention experiment: does stepping the trained model's weights along
c_rot/c_trans (each symmetry's own orbit-tangent generator) leave that symmetry's finite-
transform residual roughly flat, while stepping along -b_rot/-b_trans (each symmetry's own
equivariance-defect direction) shrinks it -- the same prediction verified for the mexican-hat
script, now for two independent symmetries at once, plus two "common" directions that try to
satisfy BOTH symmetries jointly with one shared step.

Six directions, each a SINGLE pooled least-squares solve (not per-alpha then RMS-aggregated --
see the pooled sibling scripts' rationale: a per-alpha direction, if walked, gives a different
perturbed model at every alpha, not one coherent direction), iteratively recomputed at every
step (c_i/b_i are only exact to first order at the CURRENT weights):

  - c_rot / b_rot:    pools rows across alpha only, target = g_rot / delta_rot.
  - c_trans / b_trans: pools rows across alpha only, target = g_trans / delta_trans.
  - c_common / b_common: pools rows across BOTH alpha AND generator -- i.e. finds the single
    theta_dot that best satisfies J theta_dot ~= g_rot AND J theta_dot ~= g_trans at once (resp.
    delta_rot and delta_trans for b_common). Tests whether the two symmetries' generators (or
    defects) admit one shared realising direction, not just two separate ones.

For every direction, at every step, BOTH the rotation and translation finite-transform residuals
are measured (not just the "own" symmetry's) -- this directly probes the cross-symmetry
entanglement found in the pooled analysis (e.g. b_rot/b_trans attribution correlation ~0.5): if
walking along c_rot also moves the translation residual, that is itself informative.

Produces, for each residual type (rotation, translation):
  - perturbation_<residual>_by_alpha.png: one subplot per cfg.plotting_alphas, all six
    directions overlaid (colour = symmetry, linestyle = c vs b), solid=forward/dashed=backward.
  - perturbation_<residual>_averaged.png: the same, averaged over cfg.plotting_alphas.

Usage:
    python3 run_two_body_perturbation_experiment.py outputs/asrnn_two_body_symmetry_l1_0
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import asrnn_two_body_symmetry_sensitivity as m
from sensitivity_tools import finite_transform_residual, tangent_projection


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-30)


def step_state(model: torch.nn.Module, state: dict, direction: torch.Tensor, step_size: float) -> dict:
    new_state = dict(state)
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        d = direction[offset:offset + n].reshape(p.shape).to(dtype=p.dtype)
        new_state[name] = state[name] + step_size * d
        offset += n
    return new_state


KINDS = ["c_rot", "b_rot", "c_trans", "b_trans", "c_common", "b_common"]


def combined_direction(
    model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind: str,
) -> torch.Tensor:
    """One pooled least-squares solve at the model's CURRENT weights, stacking rows across
    every alpha in direction_alphas -- and, for the '*_common' kinds, across both generators
    (rotation and translation) at once. Returns the unit c-direction, or the unit DEFECT-
    REDUCING (-b) direction for the b_*/*_common-with-delta kinds.

    rot_grid and trans_grid are DIFFERENT probe geometries (build_rotation_probe_grid's
    rotation-uniform-density polar grid vs build_probe_grid's Cartesian box) -- see
    build_rotation_probe_grid's docstring for why rotation-only quantities must not be fit on
    the Cartesian box."""
    is_b = kind.startswith("b_")
    want_rot = kind in ("c_rot", "b_rot", "c_common", "b_common")
    want_trans = kind in ("c_trans", "b_trans", "c_common", "b_common")

    jac_blocks, target_blocks = [], []
    for alpha in direction_alphas:
        if want_rot:
            q1x, q1y, q2x, q2y = rot_grid
            n_points = q1x.shape[0]
            v_x, f_x, j_x, spatial_jac_x = m.evaluate_at_points(
                model, q1x, q1y, q2x, q2y, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
            )
            jac_blocks.append(j_x.reshape(n_points * 4, -1))
            if is_b:
                pi_rot = m.compute_rotation_orbit_average(model, cfg, device, dtype, alpha, q1x, q1y, q2x, q2y)
                target_blocks.append((f_x - pi_rot).reshape(-1))
            else:
                target_blocks.append(m.rotation_generator_target(q1x, q1y, q2x, q2y, f_x, spatial_jac_x).reshape(-1))
        if want_trans:
            q1x, q1y, q2x, q2y = trans_grid
            n_points = q1x.shape[0]
            v_x, f_x, j_x, spatial_jac_x = m.evaluate_at_points(
                model, q1x, q1y, q2x, q2y, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
            )
            jac_blocks.append(j_x.reshape(n_points * 4, -1))
            if is_b:
                pi_trans = m.compute_translation_orbit_average(f_x, cfg)
                target_blocks.append((f_x - pi_trans).reshape(-1))
            else:
                target_blocks.append(m.translation_generator_target(spatial_jac_x, translation_direction).reshape(-1))

    jac_pooled = torch.cat(jac_blocks, dim=0)
    target_pooled = torch.cat(target_blocks, dim=0)
    _, _, vec, _, _ = tangent_projection(jac_pooled, target_pooled, cfg.tangent_svd_relative_cutoff)
    return unit(vec) if not is_b else unit(-vec)


def walk(model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind: str,
         step_size: float, n_steps: int, base_state: dict) -> list[dict]:
    model.load_state_dict(base_state)
    states = [{k: v.clone() for k, v in base_state.items()}]
    for _ in range(n_steps):
        d_hat = combined_direction(model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind)
        new_state = step_state(model, model.state_dict(), d_hat, step_size)
        model.load_state_dict(new_state)
        states.append({k: v.clone() for k, v in new_state.items()})
    return states


def residuals_at(model, cfg, device, dtype, alpha,
                  rot_q1x, rot_q1y, rot_q2x, rot_q2y, rot_q1x_rot, rot_q1y_rot, rot_q2x_rot, rot_q2y_rot, rot_rep,
                  trans_q1x, trans_q1y, trans_q2x, trans_q2y,
                  trans_q1x_shift, trans_q1y_shift, trans_q2x_shift, trans_q2y_shift) -> tuple[float, float]:
    """Rotation residual measured on the SAME rotation-uniform-density grid the rotation
    direction is fit on (consistent with the mexican-hat convention); translation residual on
    the Cartesian box, likewise matching what the translation direction is fit on."""
    _, f_x, _ = m.evaluate_at_points(model, rot_q1x, rot_q1y, rot_q2x, rot_q2y, alpha, cfg.architecture, device=device, dtype=dtype)
    _, f_rot, _ = m.evaluate_at_points(model, rot_q1x_rot, rot_q1y_rot, rot_q2x_rot, rot_q2y_rot, alpha, cfg.architecture, device=device, dtype=dtype)
    rot_res = finite_transform_residual(f_x, f_rot, rot_rep)
    _, f_x2, _ = m.evaluate_at_points(model, trans_q1x, trans_q1y, trans_q2x, trans_q2y, alpha, cfg.architecture, device=device, dtype=dtype)
    _, f_shift, _ = m.evaluate_at_points(model, trans_q1x_shift, trans_q1y_shift, trans_q2x_shift, trans_q2y_shift, alpha, cfg.architecture, device=device, dtype=dtype)
    trans_res = finite_transform_residual(f_x2, f_shift, None)
    return rot_res, trans_res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="Alphas to evaluate the walked models at; defaults to cfg.plotting_alphas.")
    parser.add_argument("--direction-alphas", type=float, nargs="+", default=None, help="Alphas pooled into the direction-defining solve; defaults to cfg.training_alphas union cfg.analysis_alphas.")
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--n-forward", type=int, default=4)
    parser.add_argument("--n-backward", type=int, default=2)
    args = parser.parse_args()

    saved_fields = json.loads((args.source_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})

    device = torch.device("cpu")
    dtype = torch.float64
    model, _ = m.build_model(cfg, device, dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in torch.load(args.source_dir / "final_model.pt", map_location=device).items()})
    model.eval()

    output_dir = Path(str(args.source_dir).rstrip("/") + "_perturbation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Two DIFFERENT probe geometries: rotation-only quantities (g_rot/delta_rot fitting, and the
    # rotation residual) use the rotation-uniform-density polar grid; translation-only
    # quantities keep the Cartesian box (translation_generator_target/compute_translation_orbit_average
    # need its regular cm_x/cm_y axis structure, and it has no angular-bias concern to begin
    # with). See build_rotation_probe_grid's docstring.
    rot_q1x, rot_q1y, rot_q2x, rot_q2y = m.build_rotation_probe_grid(cfg, device, dtype)
    rot_grid = (rot_q1x, rot_q1y, rot_q2x, rot_q2y)
    rot_mat = m.rotation_matrix_2d(cfg.rotation_angle_degrees, device, dtype)
    rot_rep = m.rotation_rep_matrix_4d(rot_mat)
    rot_q1x_rot, rot_q1y_rot, rot_q2x_rot, rot_q2y_rot = m.rotate_points(rot_q1x, rot_q1y, rot_q2x, rot_q2y, rot_mat)

    trans_q1x, trans_q1y, trans_q2x, trans_q2y = m.build_probe_grid(cfg, device, dtype)
    trans_grid = (trans_q1x, trans_q1y, trans_q2x, trans_q2y)
    trans_q1x_shift, trans_q1y_shift, trans_q2x_shift, trans_q2y_shift = m.translate_points(
        trans_q1x, trans_q1y, trans_q2x, trans_q2y, cfg.translation_shift, cfg.translation_angle_degrees
    )
    psi = math.radians(cfg.translation_angle_degrees)
    translation_direction = torch.tensor([math.cos(psi), math.sin(psi), math.cos(psi), math.sin(psi)], device=device, dtype=dtype)

    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    alphas = args.alphas if args.alphas is not None else cfg.plotting_alphas
    direction_alphas = args.direction_alphas if args.direction_alphas is not None else sorted(
        set(round(a, 6) for a in cfg.training_alphas) | set(round(a, 6) for a in cfg.analysis_alphas)
    )
    step_indices = list(range(-args.n_backward, args.n_forward + 1))

    print(f"direction alphas ({len(direction_alphas)}) = {direction_alphas}")
    print(f"evaluation alphas = {alphas}")
    print(f"step_size={args.step_size}, n_forward={args.n_forward}, n_backward={args.n_backward}\n")

    walks: dict[str, list[dict]] = {}
    for kind in KINDS:
        print(f"walking {kind} (forward + backward)...")
        fwd = walk(model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind, args.step_size, args.n_forward, base_state)
        bwd = walk(model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind, -args.step_size, args.n_backward, base_state)
        walks[kind] = list(reversed(bwd[1:])) + fwd
    model.load_state_dict(base_state)

    # results[kind][alpha] = (rot_residuals[step], trans_residuals[step])
    results: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for kind in KINDS:
        results[kind] = {}
        for alpha in alphas:
            rot_res = np.empty(len(step_indices))
            trans_res = np.empty(len(step_indices))
            for i, state in enumerate(walks[kind]):
                model.load_state_dict(state)
                rot_res[i], trans_res[i] = residuals_at(
                    model, cfg, device, dtype, alpha,
                    rot_q1x, rot_q1y, rot_q2x, rot_q2y, rot_q1x_rot, rot_q1y_rot, rot_q2x_rot, rot_q2y_rot, rot_rep,
                    trans_q1x, trans_q1y, trans_q2x, trans_q2y, trans_q1x_shift, trans_q1y_shift, trans_q2x_shift, trans_q2y_shift,
                )
            results[kind][alpha] = (rot_res, trans_res)
    model.load_state_dict(base_state)

    (output_dir / "two_body_perturbation_results.json").write_text(json.dumps(
        {
            "step_indices": step_indices,
            "step_size": args.step_size,
            "alphas": alphas,
            "results": {
                kind: {str(a): {"rotation": r[0].tolist(), "translation": r[1].tolist()} for a, r in per_alpha.items()}
                for kind, per_alpha in results.items()
            },
        },
        indent=2,
    ))

    # Consistent colour scheme throughout: c = tab:blue, b = tab:orange (same convention as the
    # mexican-hat perturbation figures) -- colour never encodes symmetry or step here, only
    # c-vs-b, so it can't be misread against the potential-walk plots' step-gradient colouring.
    colours = {"c": "tab:blue", "b": "tab:orange"}
    labels = {
        "c_rot": "along $c_{rot}$", "b_rot": "along $-b_{rot}$",
        "c_trans": "along $c_{trans}$", "b_trans": "along $-b_{trans}$",
        "c_common": "along $c_{common}$", "b_common": "along $-b_{common}$",
    }

    def plot_pair(residual_index: int, residual_name: str, c_kind: str, b_kind: str, tag: str) -> None:
        """One figure per (residual, direction-pair): c and b as separate subplot rows, one
        column per alpha, plus a mean-over-alphas summary figure with c/b overlaid."""
        kinds = [c_kind, b_kind]

        # --- one subplot per alpha, c/b as separate rows. sharey="col" is load-bearing: without
        # it each subplot autoscales its OWN y-range, so a tiny 0.2%-magnitude c wiggle fills the
        # same visual height as a real 5%-magnitude b change and the two look equally "active"
        # even though they aren't (verified via the raw residual arrays: c_rot's total change is
        # 3-30x smaller than b_rot's at every alpha). Sharing y within each alpha column makes
        # relative magnitude between the c and b rows honestly comparable. ---
        fig, axes = plt.subplots(2, len(alphas), figsize=(4.4 * len(alphas), 6.4), constrained_layout=True, squeeze=False, sharey="col")
        for row, kind in enumerate(kinds):
            colour = colours[kind[0]]
            for col, alpha in enumerate(alphas):
                ax = axes[row, col]
                series = results[kind][alpha][residual_index]
                ax.plot(step_indices, np.maximum(series, 1e-12), color=colour, marker="o", markersize=3)
                ax.axvline(0, color="0.7", linewidth=0.8, zorder=0)
                ax.set_yscale("log")
                ax.grid(alpha=0.25, which="both")
                if row == 0:
                    ax.set_title(f"$\\alpha={alpha:+.2f}$")
                if col == 0:
                    ax.set_ylabel(f"{labels[kind]}\n{residual_name} residual", fontsize=9)
                if row == len(kinds) - 1:
                    ax.set_xlabel("walk step")
        fig.suptitle(f"{residual_name.capitalize()} residual walking {labels[c_kind]} / {labels[b_kind]}")
        path = output_dir / f"perturbation_{residual_name}_{tag}_by_alpha.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        # --- averaged over alphas, c and b overlaid on one panel ---
        fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
        for kind in kinds:
            colour = colours[kind[0]]
            stacked = np.stack([results[kind][a][residual_index] for a in alphas])
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
            ax.plot(step_indices, np.maximum(mean, 1e-12), color=colour, label=labels[kind])
            ax.fill_between(step_indices, np.maximum(mean - std, 1e-12), mean + std, color=colour, alpha=0.2)
        ax.axvline(0, color="0.7", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set(xlabel="walk step", ylabel=f"{residual_name} finite-transform residual",
               title=f"Mean $\\pm$ std over {len(alphas)} evaluation alphas")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        path = output_dir / f"perturbation_{residual_name}_{tag}_averaged.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    plot_pair(0, "rotation", "c_rot", "b_rot", "own")
    plot_pair(1, "translation", "c_trans", "b_trans", "own")
    plot_pair(0, "rotation", "c_common", "b_common", "common")
    plot_pair(1, "translation", "c_common", "b_common", "common")

    print(f"\nDone. Results written to {output_dir}")


if __name__ == "__main__":
    main()
