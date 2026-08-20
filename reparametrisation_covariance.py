"""Which reported quantities survive a change of parameter units, and which do not?

A diagonal reparametrisation theta_i -> lambda_i theta_i renames coordinates
without touching the learned function: the network, its outputs, its tangent
space and its symmetry are all unchanged. Sensitivities transform as
S_i -> S_i / lambda_i, so the *true* coefficients of any g in the tangent space
must transform covariantly, c_i -> lambda_i c_i. Any quantity offered as "which
parameters realise the symmetry" has to respect that, or it is reporting the
chart rather than the network.

Two things can break it, and this script separates them.

1. The norm minimised over the solution set. Minimising ||c||_2 (the draft's
   c* = J^+ g) is not a covariant objective: ||lambda . c||_2 is a different
   function. Minimising ||S_i c_i||_2 (column normalisation) IS covariant by
   construction, since ||S_i|| carries the compensating factor.

2. The truncation. The resolved tangent space is defined by discarding singular
   values below a *relative* cutoff, and column scaling changes J's singular
   spectrum -- so a different subspace is retained, and the projector P_T itself
   moves. This breaks covariance even for the covariant objective, and it is
   independent of any norm choice.

The two cannot be fixed at once by lowering the cutoff: covariance is restored
only when essentially nothing is truncated, which is precisely the regime the
truncation exists to avoid (tiny singular directions amplified by Sigma_r^{-1}).

E_i is included as the contrast. It is built from column norms of J(x) and
J(g.x) with no optimisation and no truncation anywhere, so it is exactly
invariant, and is the well-posed half of the framework.

Usage: python reparametrisation_covariance.py <analysis_inputs.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from sensitivity_tools import _resolve_tangent_svd, per_parameter_equivariance_error, tangent_projection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--decades", type=float, default=2.0, help="width of the random unit change")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    d = np.load(Path(args.npz))
    j, g = torch.from_numpy(d["j"]), torch.from_numpy(d["g"])
    gen = torch.Generator().manual_seed(args.seed)
    lam = torch.exp((torch.rand(j.shape[1], generator=gen, dtype=torch.float64) - 0.5)
                    * args.decades * np.log(10))
    colnorm = j.norm(dim=0)
    mx = colnorm.max()
    live = colnorm > mx * 1e-8
    # A third source, specific to the implementation: the dead-column rule uses a floor
    # relative to the LARGEST column, and that floor is itself not covariant. It only bites
    # when the sensitivity spectrum is wide enough for live columns to sit near it -- which
    # is exactly what a converged network gives (8 decades here). Columns clear of the
    # floor are used below so that the norm and truncation effects can be seen on their own.
    clear = colnorm > mx * 1e-4
    near_floor = int(((colnorm > mx * 1e-8) & (colnorm < mx * 1e-7)).sum())
    jl, laml = j[:, live], lam[live]
    jc, lamc = j[:, clear], lam[clear]
    print(f"{Path(args.npz).parent.name}: J {tuple(j.shape)}, {int(live.sum())} live columns "
          f"({int(clear.sum())} clear of the dead-column floor, {near_floor} within 10x of it), "
          f"||S_i|| spanning {float(torch.log10(colnorm[live].max() / colnorm[live].min())):.1f} decades, "
          f"units changed over {args.decades:.0f} decades")

    def departure(jj, ll, cut, **kw):
        _, _, a, _, _ = tangent_projection(jj, g, cut, **kw)
        _, _, b, _, _ = tangent_projection(jj / ll, g, cut, **kw)
        exp = ll * a
        return float((b - exp).norm() / exp.norm().clamp_min(1e-300)), a, b, exp

    print("\n-- source 3: the dead-column floor (covariant objective, no truncation) --")
    for tag, jj, ll in (("all live columns", jl, laml), ("columns clear of the floor", jc, lamc)):
        dep, *_ = departure(jj, ll, 1e-12, normalize_columns=True)
        print(f"   {tag:34s} departure from covariance = {dep:.3e}")
    print("   the floor is relative to the largest column, so it does not commute with a")
    print("   change of units; with a wide sensitivity spectrum this alone breaks covariance.")

    print(f"\n-- source 1: the minimised norm (at the project's cutoff 1e-3) --")
    for tag, kw in (("||c||_2      (draft's J^+ g)", dict(normalize_columns=False)),
                    ("|| ||S_i|| c ||_2 (code default)", dict(normalize_columns=True))):
        _, _, c1, _, _ = tangent_projection(jl, g, 1e-3, **kw)
        _, _, c2, _, _ = tangent_projection(jl / laml, g, 1e-3, **kw)
        expected = laml * c1
        dep = float((c2 - expected).norm() / expected.norm().clamp_min(1e-300))
        rho = float(spearmanr(c2.abs().numpy(), expected.abs().numpy()).statistic)
        print(f"   {tag:34s} departure from covariance = {dep:7.3f}   Spearman = {rho:+.3f}")

    print(f"\n-- source 2: the truncation (covariant objective, columns clear of the floor) --")
    print(f"   {'cutoff':>8} {'rank(J)':>8} {'rank(J/lam)':>12} {'||P_T g - P_T'' g||/||g||':>25} "
          f"{'departure':>11}")
    for cut in (1e-3, 1e-5, 1e-8, 1e-12):
        *_, p1, r1, _, _, _ = _resolve_tangent_svd(jc, g, cut)
        *_, p2, r2, _, _, _ = _resolve_tangent_svd(jc / lamc, g, cut)
        dep, *_ = departure(jc, lamc, cut, normalize_columns=True)
        print(f"   {cut:>8.0e} {r1:>8d} {r2:>12d} {float((p1 - p2).norm() / g.norm()):>25.3e} {dep:>11.3e}")
    print("   covariance returns only when essentially nothing is truncated -- which is")
    print("   exactly the regime the truncation exists to prevent.")

    n = j.shape[0] // 2
    jx = j.reshape(n, 2, -1)
    jgx = torch.from_numpy(d["j_delta"]).reshape(n, 2, -1)
    rep = torch.eye(2, dtype=torch.float64)
    e1 = per_parameter_equivariance_error(jx, jgx, rep)
    e2 = per_parameter_equivariance_error(jx / lam, jgx / lam, rep)
    mask = e1 > 0
    print(f"\n-- the contrast: E_i --")
    print(f"   max relative change of E_i under the same reparametrisation = "
          f"{float(((e2[mask] - e1[mask]).abs() / e1[mask]).max()):.3e}   (exactly invariant)")


if __name__ == "__main__":
    main()
