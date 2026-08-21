"""Two-body visual c_i/b_i "walk" experiment, mirroring run_potential_walk_experiment.py
(mexican hat) but for two independent symmetries plus the two "common" (jointly-pooled)
directions from run_two_body_perturbation_experiment.py.

For each of the six walk directions (c_rot, b_rot, c_trans, b_trans, c_common, b_common -- see
run_two_body_perturbation_experiment.py for exactly how each is solved: one pooled
least-squares system per direction, recomputed at every iterative step), two profiles of
delta_V = V_theta - <V_theta> (the non-equivariant part, isolated because the large invariant
part of V otherwise swamps it) are shown:

  - ROTATION profile: V read off around a circle of fixed radius |q| centred at the origin
    (both bodies -- i.e. the whole (q1,q2) configuration -- swept through a full joint
    rotation), one curve per walk step. A rotation of the *shape* shows up as a phase shift; a
    change in how rotationally-equivariant the network is shows up as an amplitude change with
    no phase shift.
  - TRANSLATION profile: V read off along a line of centre-of-mass positions in the
    cfg.translation_angle_degrees direction, at fixed relative separation r -- the direct
    analogue for the non-compact symmetry (a linear sweep instead of an angular one, since
    there is no periodic "phase" for translation). A translation of the shape shows up as the
    profile shifting sideways along this line; a change in translation-equivariance shows up
    as an amplitude change with no lateral shift.

Both figures are laid out as 6 rows (one per direction) x one column per cfg.plotting_alphas,
so cross-symmetry effects are directly visible (e.g. does walking along c_trans/c_common show
up as a phase shift in the ROTATION profile too, or only an amplitude change?).

Usage:
    python3 run_two_body_potential_walk_experiment.py outputs/asrnn_two_body_symmetry_l1_0
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
from sensitivity_tools import tangent_projection

KINDS = ["c_rot", "b_rot", "c_trans", "b_trans", "c_common", "b_common"]
LABELS = {
    "c_rot": "along $c_{rot}$", "b_rot": "along $-b_{rot}$",
    "c_trans": "along $c_{trans}$", "b_trans": "along $-b_{trans}$",
    "c_common": "along $c_{common}$", "b_common": "along $-b_{common}$",
}


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


def combined_direction(model, cfg, device, dtype, direction_alphas, rot_grid, trans_grid, translation_direction, kind: str) -> torch.Tensor:
    """Identical construction to run_two_body_perturbation_experiment.py's combined_direction
    -- including the rot_grid/trans_grid split (rotation-only quantities are fit on the
    rotation-uniform-density polar grid, not the Cartesian box; see build_rotation_probe_grid's
    docstring for why)."""
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


def evaluate_v(model, cfg, device, dtype, alpha, q1x, q1y, q2x, q2y) -> torch.Tensor:
    """Forward-only learned potential at arbitrary (q1,q2) points -- no autograd needed since
    we're just visualising V, not force."""
    if cfg.architecture == "direct_mlp":
        raise ValueError("direct_mlp has no potential to visualise")
    q = torch.stack([q1x, q1y, q2x, q2y], dim=1)
    alpha_col = torch.full((q.shape[0], 1), float(alpha), device=device, dtype=dtype)
    with torch.no_grad():
        potential = model.V_net(torch.cat((q, alpha_col), dim=1))
    return potential.squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="Alphas to evaluate at (one column each); defaults to cfg.plotting_alphas.")
    parser.add_argument("--direction-alphas", type=float, nargs="+", default=None, help="Alphas pooled into the direction-defining solve; defaults to cfg.training_alphas union cfg.analysis_alphas.")
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--n-forward", type=int, default=4)
    parser.add_argument("--n-backward", type=int, default=2)
    parser.add_argument("--r-magnitude", type=float, default=None, help="Fixed |r| for the rotation profile; defaults to 0.7*q_extent.")
    parser.add_argument("--cm-magnitude", type=float, default=None, help="Fixed |cm| (at 45 degrees) for the rotation profile; defaults to 0.5*cm_extent.")
    parser.add_argument("--translation-sweep-extent", type=float, default=None, help="Half-length of the CM sweep line for the translation profile; defaults to cm_extent.")
    args = parser.parse_args()

    saved_fields = json.loads((args.source_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})
    if cfg.architecture == "direct_mlp":
        raise SystemExit("direct_mlp has no potential to visualise; use run_two_body_perturbation_experiment.py instead.")

    device = torch.device("cpu")
    dtype = torch.float64
    model, _ = m.build_model(cfg, device, dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in torch.load(args.source_dir / "final_model.pt", map_location=device).items()})
    model.eval()

    output_dir = Path(str(args.source_dir).rstrip("/") + "_perturbation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rotation-only quantities are fit on the rotation-uniform-density polar grid, translation-
    # only quantities on the Cartesian box -- see build_rotation_probe_grid's docstring.
    rot_grid = m.build_rotation_probe_grid(cfg, device, dtype)
    trans_grid = m.build_probe_grid(cfg, device, dtype)
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

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-args.n_backward, vmax=args.n_forward)

    # --- Figure 1: rotation profile. V read off around a circle of fixed |q| through the
    # origin (both bodies rotated jointly), theta = angle of the whole configuration. ---
    r_mag = args.r_magnitude if args.r_magnitude is not None else 0.7 * cfg.q_extent
    cm_mag = args.cm_magnitude if args.cm_magnitude is not None else 0.5 * cfg.cm_extent
    rx0, ry0 = r_mag, 0.0
    cmx0, cmy0 = cm_mag / math.sqrt(2), cm_mag / math.sqrt(2)
    q1x0, q1y0 = cmx0 + 0.5 * rx0, cmy0 + 0.5 * ry0
    q2x0, q2y0 = cmx0 - 0.5 * rx0, cmy0 - 0.5 * ry0

    n_theta = 240
    theta_np = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    theta = torch.tensor(theta_np, device=device, dtype=dtype)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # Vectorised joint rotation of both fixed points (q1x0,q1y0)/(q2x0,q2y0) by every theta at
    # once: standard CCW 2D rotation [[cos,-sin],[sin,cos]] applied per point.
    q1x_circ = q1x0 * cos_t - q1y0 * sin_t
    q1y_circ = q1x0 * sin_t + q1y0 * cos_t
    q2x_circ = q2x0 * cos_t - q2y0 * sin_t
    q2y_circ = q2x0 * sin_t + q2y0 * cos_t

    fig, axes = plt.subplots(len(KINDS), len(alphas), figsize=(4.6 * len(alphas), 3.4 * len(KINDS)), constrained_layout=True, squeeze=False)
    for row, kind in enumerate(KINDS):
        for col, alpha in enumerate(alphas):
            ax = axes[row, col]
            for step_idx, state in zip(step_indices, walks[kind]):
                model.load_state_dict(state)
                v_circ = evaluate_v(model, cfg, device, dtype, alpha, q1x_circ, q1y_circ, q2x_circ, q2y_circ)
                delta_v = (v_circ - v_circ.mean()).cpu().numpy()
                colour = cmap(norm(step_idx))
                lw = 2.2 if step_idx == 0 else 1.2
                ls = "-" if step_idx >= 0 else "--"
                ax.plot(np.degrees(theta_np), delta_v, color=colour, linewidth=lw, linestyle=ls)
            ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"$\\alpha={alpha:+.2f}$")
            if col == 0:
                ax.set_ylabel(f"{LABELS[kind]}\n" + r"$\delta_V(\theta)$", fontsize=9)
            if row == len(KINDS) - 1:
                ax.set_xlabel(r"joint-rotation angle $\theta$ (deg)")
    model.load_state_dict(base_state)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, ticks=step_indices)
    cbar.set_label("walk step (solid = forward, dashed = backward)")
    fig.suptitle(f"Rotation profile of the non-equivariant part of $V$ at $|q|={r_mag:.2f}$, along each walk direction")
    path = output_dir / "potential_walk_rotation_profile.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")

    # --- Figure 2: translation profile. V read off along a line of CM positions in the
    # translation direction, at fixed relative separation r. ---
    t_extent = args.translation_sweep_extent if args.translation_sweep_extent is not None else cfg.cm_extent
    n_t = 240
    t_values = np.linspace(-t_extent, t_extent, n_t)
    rx1, ry1 = r_mag, 0.0
    dirx, diry = math.cos(psi), math.sin(psi)
    q1x_line = torch.tensor(t_values * dirx + 0.5 * rx1, dtype=dtype)
    q1y_line = torch.tensor(t_values * diry + 0.5 * ry1, dtype=dtype)
    q2x_line = torch.tensor(t_values * dirx - 0.5 * rx1, dtype=dtype)
    q2y_line = torch.tensor(t_values * diry - 0.5 * ry1, dtype=dtype)

    fig, axes = plt.subplots(len(KINDS), len(alphas), figsize=(4.6 * len(alphas), 3.4 * len(KINDS)), constrained_layout=True, squeeze=False)
    for row, kind in enumerate(KINDS):
        for col, alpha in enumerate(alphas):
            ax = axes[row, col]
            for step_idx, state in zip(step_indices, walks[kind]):
                model.load_state_dict(state)
                v_line = evaluate_v(model, cfg, device, dtype, alpha, q1x_line, q1y_line, q2x_line, q2y_line)
                delta_v = (v_line - v_line.mean()).cpu().numpy()
                colour = cmap(norm(step_idx))
                lw = 2.2 if step_idx == 0 else 1.2
                ls = "-" if step_idx >= 0 else "--"
                ax.plot(t_values, delta_v, color=colour, linewidth=lw, linestyle=ls)
            ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"$\\alpha={alpha:+.2f}$")
            if col == 0:
                ax.set_ylabel(f"{LABELS[kind]}\n" + r"$\delta_V(t)$", fontsize=9)
            if row == len(KINDS) - 1:
                ax.set_xlabel("CM position $t$ along translation direction")
    model.load_state_dict(base_state)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, ticks=step_indices)
    cbar.set_label("walk step (solid = forward, dashed = backward)")
    fig.suptitle(f"Translation profile of the non-equivariant part of $V$ at $|r|={r_mag:.2f}$, along each walk direction")
    path = output_dir / "potential_walk_translation_profile.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


if __name__ == "__main__":
    main()
