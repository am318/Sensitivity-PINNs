"""Proper version of the c_i/b_i intervention experiment: for every analysis alpha, sweep
parameter-space steps theta + eps*c_hat and theta + eps*b_hat (unit-normalized directions,
eps ranging over both signs) on the trained model, and measure the network's actual
finite-rotation equivariance error at each step -- the same relative residual
(``finite_transform_residual`` on the force field at a finite rotation of
``cfg.rotation_angle_degrees``) already plotted throughout this project (e.g. plot_summary's
"rotation_force_residual" curve), not a proxy.

Prediction being tested (see run_perturbation_experiment.py's original docstring / the earlier
tabular version of this experiment for the derivation): since g = X_rot F is exactly tangent
to the constant-equivariance-defect orbit through F_theta, stepping along c should leave this
error roughly flat (a first-order rotation of the function, not a change in how equivariant it
is); since -delta points from F_theta toward its orbit-average Pi(F_theta) (the nearest exactly
equivariant function), stepping along -b should reduce it, and stepping along +b should
increase it.

Produces:
  - perturbation_by_alpha.png: one subplot per analysis alpha, x=eps, y=equivariance error,
    one curve for the c-direction and one for the b-direction.
  - perturbation_averaged.png: the same curves averaged (mean +/- std across alphas) into one
    summary panel.

Usage:
    python3 run_perturbation_experiment.py outputs/asrnn_mexican_hat_symmetry
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import asrnn_mexican_hat_symmetry_sensitivity as m
from experiment_common import module_colours  # noqa: F401  (colours not needed here, kept for style parity)
from sensitivity_tools import finite_transform_residual, tangent_projection


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-30)


def scatter_into_model(model: torch.nn.Module, base_state: dict, direction: torch.Tensor, eps: float) -> dict:
    """theta_new = theta_base + eps * direction, direction flat in model.parameters() order."""
    full_state = dict(base_state)
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        step = direction[offset:offset + n].reshape(p.shape).to(dtype=p.dtype)
        full_state[name] = base_state[name] + eps * step
        offset += n
    return full_state


def rotation_residual_at(model, cfg, device, dtype, alpha, q1_grid, q2_grid, q1_rot, q2_rot, rot_mat) -> float:
    _, _, _, force = m._evaluate_force_and_potential(model, cfg, device, dtype, alpha, q1_grid, q2_grid)
    _, _, _, force_rot = m._evaluate_force_and_potential(model, cfg, device, dtype, alpha, q1_rot, q2_rot)
    return finite_transform_residual(force, force_rot, rot_mat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--polar-n-radii", type=int, default=15)
    parser.add_argument("--polar-n-angles", type=int, default=32)
    parser.add_argument("--eps-max", type=float, default=0.2)
    parser.add_argument("--n-eps", type=int, default=21)
    args = parser.parse_args()

    saved_fields = json.loads((args.source_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})
    # Compute c_i/b_i (via g, delta) on the rotationally-symmetric polar grid, not the
    # source's square grid -- the square grid was found to leak a genuine ~10% cross-talk
    # between g and delta (a domain-boundary artefact of continuous-rotation integration on a
    # non-rotation-invariant region), which would contaminate exactly the c-vs-b separation
    # this experiment is testing. The finite-rotation residual we measure as the outcome
    # doesn't have this issue (it's a direct function comparison at two points, not an
    # infinitesimal-generator inner product), so any probe set works for that part.
    cfg = replace(cfg, probe_grid_shape="polar", polar_n_radii=args.polar_n_radii, polar_n_angles=args.polar_n_angles)

    device = torch.device("cpu")
    dtype = torch.float64
    model, _ = m.build_model(cfg, device, dtype)
    state = torch.load(args.source_dir / "final_model.pt", map_location=device)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()})
    model.eval()

    output_dir = Path(str(args.source_dir).rstrip("/") + "_perturbation")
    output_dir.mkdir(parents=True, exist_ok=True)

    q1_grid, q2_grid = m.build_probe_grid(cfg, device, dtype)
    n_points = q1_grid.shape[0]
    rot_mat = m.rotation_matrix(cfg.rotation_angle_degrees, device, dtype)
    q1_rot, q2_rot = m.transform_points(q1_grid, q2_grid, rot_mat)
    eps_values = np.linspace(-args.eps_max, args.eps_max, args.n_eps)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"probe grid: polar {args.polar_n_radii}x{args.polar_n_angles} = {n_points} points; "
          f"eps in [{-args.eps_max}, {args.eps_max}], {args.n_eps} points; "
          f"{len(cfg.analysis_alphas)} alphas\n")

    per_alpha: dict[float, dict[str, np.ndarray]] = {}
    for alpha in cfg.analysis_alphas:
        model.load_state_dict(base_state)
        v_x, f_x, j_x, spatial_jac_x = m.evaluate_at_points(
            model, q1_grid, q2_grid, alpha, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True
        )
        g = m.rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
        pi_f = m.compute_orbit_average(model, cfg, device, dtype, alpha, q1_grid, q2_grid)
        delta = f_x - pi_f
        jac_flat = j_x.reshape(n_points * 2, -1)
        _, _, c_vec, _, _ = tangent_projection(jac_flat, g.reshape(-1), cfg.tangent_svd_relative_cutoff)
        _, _, b_vec, _, _ = tangent_projection(jac_flat, delta.reshape(-1), cfg.tangent_svd_relative_cutoff)
        c_hat, b_hat = unit(c_vec), unit(b_vec)

        baseline_residual = rotation_residual_at(model, cfg, device, dtype, alpha, q1_grid, q2_grid, q1_rot, q2_rot, rot_mat)

        residual_c = np.empty(len(eps_values))
        residual_b = np.empty(len(eps_values))
        for i, eps in enumerate(eps_values):
            model.load_state_dict(scatter_into_model(model, base_state, c_hat, float(eps)))
            residual_c[i] = rotation_residual_at(model, cfg, device, dtype, alpha, q1_grid, q2_grid, q1_rot, q2_rot, rot_mat)
            model.load_state_dict(scatter_into_model(model, base_state, b_hat, float(eps)))
            residual_b[i] = rotation_residual_at(model, cfg, device, dtype, alpha, q1_grid, q2_grid, q1_rot, q2_rot, rot_mat)
        model.load_state_dict(base_state)

        per_alpha[alpha] = {"residual_c": residual_c, "residual_b": residual_b, "baseline": baseline_residual}
        print(f"alpha={alpha:+.2f}  baseline residual={baseline_residual:.4e}  "
              f"residual_c range=[{residual_c.min():.3e},{residual_c.max():.3e}]  "
              f"residual_b range=[{residual_b.min():.3e},{residual_b.max():.3e}]")

    (output_dir / "perturbation_results.json").write_text(json.dumps(
        {
            "eps_values": eps_values.tolist(),
            "per_alpha": {
                str(a): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                for a, r in per_alpha.items()
            },
        },
        indent=2,
    ))

    # --- one subplot per alpha ---
    alphas = cfg.analysis_alphas
    ncols = 6
    nrows = int(np.ceil(len(alphas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), constrained_layout=True, squeeze=False)
    for idx, alpha in enumerate(alphas):
        ax = axes[idx // ncols, idx % ncols]
        r = per_alpha[alpha]
        ax.plot(eps_values, np.maximum(r["residual_c"], 1e-12), color="tab:blue", label="along $c$ (orbit-tangent)")
        ax.plot(eps_values, np.maximum(r["residual_b"], 1e-12), color="tab:orange", label="along $b$ (defect)")
        ax.axhline(r["baseline"], color="0.5", linewidth=1, linestyle=":", label="baseline ($\\epsilon=0$)")
        ax.axvline(0, color="0.7", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_title(f"$\\alpha={alpha:+.2f}$", fontsize=10)
        ax.grid(alpha=0.25, which="both")
    for idx in range(len(alphas), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    axes[0, 0].legend(fontsize=8, loc="upper center")
    fig.supxlabel(r"$\epsilon$ (signed step size along unit direction)")
    fig.supylabel("finite-rotation equivariance error (relative residual)")
    fig.savefig(output_dir / "perturbation_by_alpha.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "perturbation_by_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- averaged over alphas ---
    all_c = np.stack([per_alpha[a]["residual_c"] for a in alphas])
    all_b = np.stack([per_alpha[a]["residual_b"] for a in alphas])
    mean_c, std_c = all_c.mean(axis=0), all_c.std(axis=0)
    mean_b, std_b = all_b.mean(axis=0), all_b.std(axis=0)
    mean_baseline = np.mean([per_alpha[a]["baseline"] for a in alphas])

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.plot(eps_values, mean_c, color="tab:blue", label="along $c$ (orbit-tangent)")
    ax.fill_between(eps_values, np.maximum(mean_c - std_c, 1e-12), mean_c + std_c, color="tab:blue", alpha=0.2)
    ax.plot(eps_values, mean_b, color="tab:orange", label="along $b$ (defect)")
    ax.fill_between(eps_values, np.maximum(mean_b - std_b, 1e-12), mean_b + std_b, color="tab:orange", alpha=0.2)
    ax.axhline(mean_baseline, color="0.5", linewidth=1, linestyle=":", label="baseline ($\\epsilon=0$)")
    ax.axvline(0, color="0.7", linewidth=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set(xlabel=r"$\epsilon$ (signed step size along unit direction)",
           ylabel="finite-rotation equivariance error (relative residual)",
           title=f"Mean $\\pm$ std over {len(alphas)} analysis alphas")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(output_dir / "perturbation_averaged.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "perturbation_averaged.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone. Results written to {output_dir}")


if __name__ == "__main__":
    main()
