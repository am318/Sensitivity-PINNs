"""Visual c_i/b_i "walk" experiment, analogous to a decision-boundary walk: show the learned
potential V_theta(q)'s non-equivariant part as we ITERATIVELY step the trained model's weights
along ONE shared direction c (the g = X_rot F attribution) and ONE shared direction -b (the
delta = F - Pi F attribution), then evaluate the resulting model across every plotting alpha.

Important: this is a genuine walk, not one big jump along a direction computed once. c_i and
b_i are only exact to first order at the CURRENT weights -- so each step here is: take one
small step of fixed size along the current c (or -b) direction, then RECOMPUTE c (or b) at the
new weights before taking the next step. This follows the (possibly curving) integral curve of
the c-vector-field / b-vector-field through parameter space, rather than linearly extrapolating
from the start point.

The direction is SHARED across alpha, not recomputed per alpha: alpha is a genuine input
feature of this network (the force field F_theta(q; alpha) differs by alpha), so a naive
per-alpha c/b would give a different walked model at every alpha, which isn't one coherent
walk -- it's several unrelated ones that happen to share a start point. Instead, at every step,
c (or b) is solved from ONE combined least-squares system that stacks rows from every alpha in
cfg.training_alphas union cfg.analysis_alphas (alpha is just a network input here, not
something requiring "data" at that value, so both sets are fair game) -- the direct
generalisation of how the existing methodology already pools rows across every probe point q
within a single alpha. The resulting single c_hat/b_hat is then walked once, and the SAME
walked model is evaluated at each of cfg.plotting_alphas for the figures below.

Two figures, both laid out as 2 rows (c-walk on top, -b-walk on bottom) x one column per
plotting alpha (default cfg.plotting_alphas, override with --alphas):

  - potential_walk_angular_profile_all_alphas.png (the primary, most legible figure): delta_V =
    V_theta - Pi(V_theta) read off at a fixed radius as a function of angle, one curve per walk
    step (solid = forward, dashed = backward, colour = step index). A rotation shows up as a
    clean phase shift of the whole curve; a change in how equivariant the network is shows up as
    a clean amplitude change (shrink/grow) with no phase shift.
  - potential_walk_V_contours_all_alphas.png: contours of the raw potential V_theta itself (the
    direct analogue of a decision boundary), all steps overlaid on one panel. Kept as a
    documented null result: on this domain (the true ring of minima sits almost exactly at
    q_extent), the contours overlap almost perfectly across every step, showing the large
    radially-symmetric part of V dominates its level sets by orders of magnitude -- which is
    exactly why delta_V (isolating the non-equivariant part) is needed for the angular profile
    above, rather than plotting raw V there too.

Usage:
    python3 run_potential_walk_experiment.py outputs/asrnn_mexican_hat_symmetry
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import asrnn_mexican_hat_symmetry_sensitivity as m
from sensitivity_tools import tangent_projection


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


def combined_direction(model, cfg, device, dtype, direction_alphas: list[float], q1_probe, q2_probe, kind: str) -> torch.Tensor:
    """Recompute c (kind='c') or the defect-reducing -b (kind='b') at the model's CURRENT
    weights, from ONE combined least-squares system stacking rows across every alpha in
    direction_alphas (not a separate solve per alpha)."""
    n_probe = q1_probe.shape[0]
    jac_blocks, target_blocks = [], []
    for alpha in direction_alphas:
        v_x, f_x, j_x, spatial_jac_x = m.evaluate_at_points(
            model, q1_probe, q2_probe, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
        )
        jac_blocks.append(j_x.reshape(n_probe * 2, -1))
        if kind == "c":
            g = m.rotation_generator_target(q1_probe, q2_probe, f_x, spatial_jac_x)
            target_blocks.append(g.reshape(-1))
        else:
            pi_f = m.compute_orbit_average(model, cfg, device, dtype, alpha, q1_probe, q2_probe)
            target_blocks.append((f_x - pi_f).reshape(-1))
    jac_combined = torch.cat(jac_blocks, dim=0)
    target_combined = torch.cat(target_blocks, dim=0)
    _, _, vec, _, _ = tangent_projection(jac_combined, target_combined, cfg.tangent_svd_relative_cutoff)
    return unit(vec) if kind == "c" else unit(-vec)  # -b: the defect-REDUCING direction


def walk(model, cfg, device, dtype, direction_alphas: list[float], q1_probe, q2_probe, kind: str,
         step_size: float, n_steps: int, base_state: dict) -> list[dict]:
    """Iteratively step-then-recompute (the shared direction re-derived from current weights
    each step), following the vector field's actual integral curve rather than one linear
    extrapolation. Returns [base_state, state_after_1_step, ..., state_after_n_steps];
    step_size may be negative to walk backward."""
    model.load_state_dict(base_state)
    states = [{k: v.clone() for k, v in base_state.items()}]
    for _ in range(n_steps):
        d_hat = combined_direction(model, cfg, device, dtype, direction_alphas, q1_probe, q2_probe, kind)
        new_state = step_state(model, model.state_dict(), d_hat, step_size)
        model.load_state_dict(new_state)
        states.append({k: v.clone() for k, v in new_state.items()})
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="Alphas to EVALUATE the walked model at (one column each); defaults to the source run's cfg.plotting_alphas.")
    parser.add_argument("--direction-alphas", type=float, nargs="+", default=None, help="Alphas to pool into the combined least-squares system that DEFINES the walk direction; defaults to cfg.training_alphas union cfg.analysis_alphas.")
    parser.add_argument("--polar-n-radii", type=int, default=15)
    parser.add_argument("--polar-n-angles", type=int, default=32)
    parser.add_argument("--step-size", type=float, default=0.02, help="Fixed per-step size (L2 distance moved in parameter space each step).")
    parser.add_argument("--n-forward", type=int, default=5, help="Number of forward steps.")
    parser.add_argument("--n-backward", type=int, default=3, help="Number of backward steps (opposite direction).")
    parser.add_argument("--grid-n", type=int, default=80, help="Resolution of the fine (q1,q2) visualisation grid, for the raw-V contour figure.")
    parser.add_argument("--n-v-levels", type=int, default=5, help="Number of raw-V contour levels (evenly spaced from the baseline's shifted-to-min range).")
    args = parser.parse_args()

    saved_fields = json.loads((args.source_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})
    # c_i/b_i solved on the rotationally-symmetric polar probe grid (see run_perturbation_experiment.py
    # for why: the square grid leaks a genuine ~10% g/delta cross-talk that would contaminate this).
    cfg = replace(cfg, probe_grid_shape="polar", polar_n_radii=args.polar_n_radii, polar_n_angles=args.polar_n_angles)

    device = torch.device("cpu")
    dtype = torch.float64
    model, _ = m.build_model(cfg, device, dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in torch.load(args.source_dir / "final_model.pt", map_location=device).items()})
    model.eval()

    output_dir = Path(str(args.source_dir).rstrip("/") + "_perturbation")
    output_dir.mkdir(parents=True, exist_ok=True)

    q1_probe, q2_probe = m.build_probe_grid(cfg, device, dtype)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    alphas = args.alphas if args.alphas is not None else cfg.plotting_alphas
    direction_alphas = args.direction_alphas if args.direction_alphas is not None else sorted(
        set(round(a, 6) for a in cfg.training_alphas) | set(round(a, 6) for a in cfg.analysis_alphas)
    )
    kinds = [("c", "along $c$ (rotates the shape)"), ("b", "along $-b$ (shrinks the defect)")]
    step_indices = list(range(-args.n_backward, args.n_forward + 1))
    step_ticks = step_indices
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-args.n_backward, vmax=args.n_forward)

    print(f"evaluation alphas={alphas}")
    print(f"direction alphas ({len(direction_alphas)}, pooled into one combined solve)={direction_alphas}")
    print(f"step_size={args.step_size}, n_forward={args.n_forward}, n_backward={args.n_backward}, "
          f"probe grid: polar {args.polar_n_radii}x{args.polar_n_angles}\n")

    # One walk per kind (not per alpha): the SAME walked model is evaluated at every alpha below.
    walks: dict[str, list[dict]] = {}
    for kind, _ in kinds:
        print(f"walking {kind} (forward + backward)...")
        fwd = walk(model, cfg, device, dtype, direction_alphas, q1_probe, q2_probe, kind, args.step_size, args.n_forward, base_state)
        bwd = walk(model, cfg, device, dtype, direction_alphas, q1_probe, q2_probe, kind, -args.step_size, args.n_backward, base_state)
        walks[kind] = list(reversed(bwd[1:])) + fwd  # most-backward ... step 0 ... most-forward
    model.load_state_dict(base_state)

    # --- Figure 1 (primary): angular profile of delta_V at fixed radius. Rows = {c, -b}, columns = alpha. ---
    n_theta = 240
    theta = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)

    fig, axes = plt.subplots(2, len(alphas), figsize=(4.6 * len(alphas), 9.5), constrained_layout=True, squeeze=False)
    for row_idx, (kind, label) in enumerate(kinds):
        for col_idx, alpha in enumerate(alphas):
            ring_radius = math.sqrt(-alpha) if alpha < 0 else None
            radius = min(ring_radius, cfg.q_extent) * 0.85 if ring_radius is not None else 0.7 * cfg.q_extent
            q1_circle = torch.tensor(radius * np.cos(theta), device=device, dtype=dtype)
            q2_circle = torch.tensor(radius * np.sin(theta), device=device, dtype=dtype)

            ax = axes[row_idx, col_idx]
            for step_idx, state in zip(step_indices, walks[kind]):
                model.load_state_dict(state)
                _, _, v_circle, _ = m._evaluate_force_and_potential(model, cfg, device, dtype, alpha, q1_circle, q2_circle)
                delta_v_theta = (v_circle - v_circle.mean()).cpu().numpy()
                colour = cmap(norm(step_idx))
                lw = 2.2 if step_idx == 0 else 1.3
                ls = "-" if step_idx >= 0 else "--"
                ax.plot(np.degrees(theta), delta_v_theta, color=colour, linewidth=lw, linestyle=ls, label=f"step {step_idx}")
            ax.set(xlabel=r"angle $\theta$ (deg) at $r=%.2f$" % radius, title=f"$\\alpha={alpha:+.2f}$")
            ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
            ax.grid(alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n" + r"$\delta_V(\theta) = V_\theta(r,\theta) - \overline{V_\theta}(r)$")
    model.load_state_dict(base_state)
    axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="solid=fwd, dashed=bwd")
    fig.suptitle("Angular profile of the non-equivariant part along one shared iterative walk, evaluated across alphas")
    profile_path = output_dir / "potential_walk_angular_profile_all_alphas.png"
    fig.savefig(profile_path, dpi=200, bbox_inches="tight")
    fig.savefig(profile_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {profile_path}")

    # --- Figure 2 (null-result companion): raw V contours, all steps overlaid. Rows = {c, -b}, columns = alpha. ---
    q1_fine, q2_fine = m._make_grid(cfg, args.grid_n, device, dtype)
    n = int(math.isqrt(q1_fine.shape[0]))
    q1_np = q1_fine.cpu().numpy().reshape(n, n)
    q2_np = q2_fine.cpu().numpy().reshape(n, n)
    theta_circle = np.linspace(0, 2 * math.pi, 200)

    fig, axes = plt.subplots(2, len(alphas), figsize=(4.6 * len(alphas), 10.5), constrained_layout=True, squeeze=False)
    for row_idx, (kind, label) in enumerate(kinds):
        for col_idx, alpha in enumerate(alphas):
            ax = axes[row_idx, col_idx]
            v_fields = []
            for state in walks[kind]:
                model.load_state_dict(state)
                _, _, learned_v, _ = m._evaluate_force_and_potential(model, cfg, device, dtype, alpha, q1_fine, q2_fine)
                v_fields.append((learned_v - learned_v.min()).cpu().numpy().reshape(n, n))
            v0 = v_fields[step_indices.index(0)]
            v_levels = np.linspace(0.0, float(v0.max()), args.n_v_levels + 2)[1:-1]
            for step_idx, field in zip(step_indices, v_fields):
                colour = cmap(norm(step_idx))
                lw = 2.2 if step_idx == 0 else 1.1
                ls = "-" if step_idx >= 0 else "--"
                ax.contour(q1_np, q2_np, field, levels=v_levels, colors=[colour], linewidths=lw, linestyles=ls, alpha=0.85)
            ring_radius = math.sqrt(-alpha) if alpha < 0 else None
            if ring_radius is not None:
                ax.plot(ring_radius * np.cos(theta_circle), ring_radius * np.sin(theta_circle),
                        color="0.3", linewidth=1, linestyle=":", alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"$\\alpha={alpha:+.2f}$", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11)
    model.load_state_dict(base_state)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, ticks=step_ticks)
    cbar.set_label("walk step (solid = forward, dashed = backward)")
    fig.suptitle(r"Raw potential ($V_\theta$, shifted to min) contours along one shared iterative walk (null result -- see delta_V profile instead)")
    v_path = output_dir / "potential_walk_V_contours_all_alphas.png"
    fig.savefig(v_path, dpi=200, bbox_inches="tight")
    fig.savefig(v_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {v_path}")


if __name__ == "__main__":
    main()
