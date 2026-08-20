"""How few parameters SUFFICE to realise the symmetry generator -- and is that few for a reason?

This is the sufficiency question, which is different from the influence question
and is the one that supports "a small part of the network realises the symmetry".
Causal influence on the symmetry can be diffuse (spread over hundreds of
parameters) while the generator direction is still *reproducible* from a handful,
because the sensitivities are massively collinear: rank(J) ~ 22 against ~2200
live parameters.

Sufficiency is asked by greedy subset selection (orthogonal matching pursuit)
against the represented target P_T g: at each step take the parameter whose
sensitivity best reduces the residual, refit by least squares on the selected
support, and record the relative error. This references no metric on parameter
space -- there is no minimum-norm solve and so no gauge -- which is why it can
carry a claim that |c_i| cannot.

Two things make the answer meaningful rather than trivial:

- **The rank bound.** Any direction inside a rank-r tangent space is reproduced
  exactly by r well-chosen columns, so "22 parameters suffice" is true of the
  symmetry and of anything else. The claim has to be that the symmetry needs
  FEWER than a generic target does.
- **The matched null.** Matched-construction null targets (the same Lie-derivative
  machinery with a random non-symmetry matrix in place of the rotation generator)
  give the reference curve. k*(eps) for the true generator against the
  distribution of k*(eps) over nulls is the symmetry-specific statement.

Group OMP over hidden units is run alongside, since the hidden unit is the
finest granularity the network's permutation symmetry admits.

Usage: python symmetry_subspace_sparsity.py outputs/<run_dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from asrnn_mexican_hat_symmetry_sensitivity import (
    Config, build_model, evaluate_at_points, rotation_generator_target, validate_config,
)
from attribution_stability import unit_index_map
from causal_symmetry_control import build_polar_probe_grid
from experiment_common import select_device, select_dtype
from sensitivity_tools import _resolve_tangent_svd, linear_generator_target, random_symmetry_test_matrix

ALPHA = 0.3
N_ANGLES = 24
N_NULLS = int(os.environ.get("N_NULLS", "8"))
K_MAX = int(os.environ.get("K_MAX", "30"))
TOLERANCES = (0.20, 0.10, 0.05)
UNIT_TOLERANCES = (1e-2, 1e-3, 1e-4, 1e-5)


def omp(a: np.ndarray, target: np.ndarray, k_max: int, groups=None):
    """Greedy subset selection with least-squares refit; returns the relative-error curve.

    ``groups`` (a list of index arrays) selects whole hidden units at a time
    instead of single parameters.
    """
    residual = target.copy()
    chosen: list[int] = []
    cols: list[int] = []
    errs = []
    tnorm = max(np.linalg.norm(target), 1e-30)
    candidates = list(range(len(groups))) if groups is not None else list(range(a.shape[1]))
    for _ in range(min(k_max, len(candidates))):
        best, best_score = None, -np.inf
        for c in candidates:
            if c in chosen:
                continue
            idx = groups[c] if groups is not None else np.array([c])
            sub = a[:, idx]
            nrm = np.linalg.norm(sub, axis=0)
            keep = nrm > 0
            if not keep.any():
                continue
            proj = sub[:, keep] @ np.linalg.lstsq(sub[:, keep], residual, rcond=None)[0]
            score = float(proj @ proj)
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        chosen.append(best)
        cols.extend((groups[best] if groups is not None else np.array([best])).tolist())
        sub = a[:, cols]
        coef, *_ = np.linalg.lstsq(sub, target, rcond=None)
        residual = target - sub @ coef
        errs.append(float(np.linalg.norm(residual) / tnorm))
    return np.array(errs), chosen


def k_star(errs: np.ndarray, tol: float) -> float:
    hit = np.flatnonzero(errs <= tol)
    return float(hit[0] + 1) if hit.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = Config()
    cfg.architecture, cfg.device, cfg.training_steps = blob["architecture"], "cpu", 1
    validate_config(cfg)
    device, dtype = select_device("cpu"), select_dtype(cfg.dtype)
    model, _ = build_model(cfg, device, dtype)
    model.load_state_dict(blob["state_dict"]); model.eval()
    units = unit_index_map(model)

    q1, q2 = build_polar_probe_grid(cfg.q_grid_points_per_axis, N_ANGLES, cfg.q_extent, device, dtype)
    n_points = q1.shape[0]
    _, f_x, j_x, spatial = evaluate_at_points(
        model, q1, q2, ALPHA, cfg.architecture, device=device, dtype=dtype, need_spatial_jacobian=True)
    g = rotation_generator_target(q1, q2, f_x, spatial).reshape(n_points * 2)
    j_flat = j_x.reshape(n_points * 2, -1)
    positions = torch.stack([q1, q2], dim=1)
    gen = torch.Generator().manual_seed(0)
    nulls = [linear_generator_target(
        positions, f_x, spatial,
        random_symmetry_test_matrix(device=device, dtype=dtype, generator=gen)).reshape(n_points * 2)
        for _ in range(N_NULLS)]

    cutoff = cfg.tangent_svd_relative_cutoff
    j64 = j_flat.detach().cpu().to(torch.float64)
    a = j64.numpy()

    def represented(t):
        *_, proj, rank, _, _, _ = _resolve_tangent_svd(j64, t.detach().cpu().to(torch.float64), cutoff)
        return proj.numpy(), rank

    pg, rank = represented(g)
    print(f"{run_dir.name}: rank(J) = {rank}, {a.shape[1]} parameters, {len(units)} hidden units")
    print(f"  target is P_T g (the represented part), so any {rank} independent columns suffice "
          f"exactly -- the question is whether the symmetry needs FEWER than a generic target.")

    curves = {}
    errs_g, chosen_g = omp(a, pg, K_MAX)
    curves["true rotation generator"] = errs_g
    null_curves = []
    for i, t in enumerate(nulls):
        pn, _ = represented(t)
        e, _ = omp(a, pn, K_MAX)
        null_curves.append(e)
    n_len = min(len(x) for x in null_curves + [errs_g])
    null_mat = np.stack([x[:n_len] for x in null_curves])

    print(f"\n=== parameter-level sufficiency: k needed to reach a relative error ===")
    print(f"  {'tolerance':>11}{'true generator':>17}{'nulls (mean+-sd)':>22}{'z':>7}")
    for tol in TOLERANCES:
        kg = k_star(errs_g, tol)
        kn = np.array([k_star(x, tol) for x in null_curves])
        good = ~np.isnan(kn)
        m, s = (float(np.mean(kn[good])), float(np.std(kn[good]))) if good.any() else (np.nan, np.nan)
        z = (m - kg) / max(s, 1e-9) if not (np.isnan(kg) or np.isnan(m)) else float("nan")
        print(f"  {tol:>11.0%}{kg:>17.0f}{f'{m:.1f} +- {s:.1f}':>22}{z:>7.2f}")
    print("  positive z = the true generator is reproduced by FEWER parameters than a "
          "non-symmetry target")

    print(f"\n=== hidden-unit level (permutation-safe granularity) ===")
    unit_groups = [np.asarray(v) for v in units.values()]
    errs_gu, chosen_gu = omp(a, pg, min(K_MAX, 20), groups=unit_groups)
    null_u = []
    for t in nulls[:4]:
        pn, _ = represented(t)
        e, _ = omp(a, pn, min(K_MAX, 20), groups=unit_groups)
        null_u.append(e)
    print(f"  (one unit carries 36+ columns, more than rank {rank}, so the parameter-level"
          f" tolerances all saturate at k=1 -- finer ones are needed here)")
    print(f"  {'tolerance':>11}{'true generator':>17}{'nulls (mean+-sd)':>22}{'z':>7}")
    for tol in UNIT_TOLERANCES:
        kg = k_star(errs_gu, tol)
        kn = np.array([k_star(x, tol) for x in null_u])
        good = ~np.isnan(kn)
        m, s = (float(np.mean(kn[good])), float(np.std(kn[good]))) if good.any() else (np.nan, np.nan)
        z = (m - kg) / max(s, 1e-9) if not (np.isnan(kg) or np.isnan(m)) else float("nan")
        print(f"  {tol:>11.0e}{kg:>17.0f}{f'{m:.1f} +- {s:.1f}':>22}{z:>7.2f}")
    names = list(units)
    print(f"  units selected for the true generator, in order: "
          f"{', '.join(names[c] for c in chosen_gu[:8])}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, (eg, nl, lab, cap) in zip(axes, [
            (errs_g, null_curves, "parameters", K_MAX),
            (errs_gu, null_u, "hidden units", min(K_MAX, 24))]):
        ax.plot(np.arange(1, len(eg) + 1), eg, "-o", ms=4, label="true rotation generator")
        for i, e in enumerate(nl):
            ax.plot(np.arange(1, len(e) + 1), e, "--", lw=1, alpha=.6,
                    label="matched-construction nulls" if i == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel(f"{lab} selected (greedy, least-squares refit)")
        ax.set_ylabel("relative error reproducing $P_T g$")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title(f"Sufficiency at {lab} level", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "symmetry_subspace_sparsity.png", dpi=200, bbox_inches="tight")
    print(f"\nplot -> {run_dir / 'symmetry_subspace_sparsity.png'}")


if __name__ == "__main__":
    main()
