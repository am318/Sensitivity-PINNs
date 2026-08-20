"""How much of parameter-wise symmetry attribution is Jacobian conditioning?

The SCML draft's headline (Fig. 1a) is that |c_i| spans one to two orders of
magnitude above the bulk, and reads that spread as "this subset of parameters
realises the symmetry". The review notes raise the right worry -- a random
matched-norm target might give the same concentration, in which case "the
structure is the conditioning of J, not symmetry" -- but leave it as a
one-solve empirical check.

It is stronger than that: the confound is an exact algebraic identity. With
column normalisation (the project's default, so that the minimised norm is a
function-space quantity rather than a units artefact), the minimum-norm
attribution is

    c_i = a_i / ||S_i||^2,        a_i := (V_{:,i} . w),   w := (A A^T)^{-1} Sigma_r^{-1} U_r^T g,

with A = V_r / ||S_:||. Here ||S_i|| = ||J_{:,i}|| is parameter i's own
sensitivity magnitude, and it is *completely target-independent* -- identical
for the true generator, for a random target, for anything. Only a_i knows
about the symmetry. Therefore

    log|c_i| = log|a_i| - 2 log||S_i||,

and a spread in |c_i| is only evidence about symmetry to the extent that the
first term, not the second, produces it. This script measures which.

It also settles a subtlety the review notes get slightly wrong. The proposed
scale-invariant score

    atilde_i = c_i <S_i, g> / ||P_T g||^2

is indeed invariant under per-parameter rescaling theta_i -> lambda_i theta_i,
but it is NOT conditioning-free: <S_i, g> carries one power of ||S_i||, so
atilde_i ~ a_i / ||S_i||, i.e. it removes one of the two powers. The
prediction is a clean, falsifiable slope test:

    d log|c_i|      / d log||S_i|| = -2
    d log|atilde_i| / d log||S_i|| = -1
    d log|a_i|      / d log||S_i|| =  0

Reads a saved ``analysis_inputs.npz`` (J, targets, scores) so it re-runs in
seconds without retraining.

Usage: python conditioning_decomposition.py <analysis_inputs.npz> [more.npz ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from sensitivity_tools import choose_l1_ratio_for_sparsity, tangent_projection, tangent_projection_auto

CUTOFF = 1e-3


def alignment_part(j: np.ndarray, target: np.ndarray, cutoff: float = CUTOFF):
    """Return (c_l2, a, colnorm) with a_i = c_i * ||S_i||^2 the target-dependent factor."""
    jt = torch.from_numpy(j)
    gt = torch.from_numpy(target)
    _, _, c, rank, _ = tangent_projection(jt, gt, cutoff, normalize_columns=True)
    colnorm = jt.norm(dim=0).numpy()
    c = c.numpy()
    # Dead columns (here: K_net, which cannot affect the force at all) carry exactly zero
    # attribution by construction. Leaving them in every statistic would mean reporting
    # figures dominated by half the parameter vector being structurally absent.
    live = colnorm > colnorm.max() * 1e-8
    return c, c * colnorm**2, colnorm, rank, live


def log_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS slope of log|y| on log x, plus R^2, over entries where both are positive."""
    mask = (np.abs(y) > 0) & (x > 0)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    ly, lx = np.log(np.abs(y[mask])), np.log(x[mask])
    design = np.stack([lx, np.ones_like(lx)], axis=1)
    beta = np.linalg.lstsq(design, ly, rcond=None)[0]
    resid = ly - design @ beta
    r2 = 1.0 - float(np.var(resid) / max(np.var(ly), 1e-30))
    return float(beta[0]), r2


def variance_decomposition(a: np.ndarray, colnorm: np.ndarray) -> dict[str, float]:
    """Split Var(log|c|) into alignment, conditioning and cross terms.

    log|c| = log|a| - 2 log s  =>
    Var(log|c|) = Var(log|a|) + 4 Var(log s) - 4 Cov(log|a|, log s).
    """
    mask = (np.abs(a) > 0) & (colnorm > 0)
    la, ls = np.log(np.abs(a[mask])), np.log(colnorm[mask])
    v_total = float(np.var(la - 2 * ls))
    v_align, v_cond = float(np.var(la)), 4.0 * float(np.var(ls))
    cross = -4.0 * float(np.cov(la, ls)[0, 1])
    return {
        "var_log_c": v_total,
        "share_alignment": v_align / max(v_total, 1e-30),
        "share_conditioning": v_cond / max(v_total, 1e-30),
        "share_cross": cross / max(v_total, 1e-30),
        "sd_log10_c": float(np.std((la - 2 * ls) / np.log(10))),
        "sd_log10_conditioning": float(np.std(-2 * ls / np.log(10))),
        "sd_log10_alignment": float(np.std(la / np.log(10))),
        "n": int(mask.sum()),
    }


def lorenz(values: np.ndarray) -> np.ndarray:
    v = np.sort(np.abs(values))[::-1]
    total = v.sum()
    if total <= 0:
        return np.zeros_like(v)
    return np.cumsum(v) / total


def analyse(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    j, g = d["j"], d["g"]
    nulls = [d[k] for k in d.files if k.startswith("null_target_")]
    label = path.parent.name
    print(f"\n{'=' * 78}\n{label}   J shape {j.shape}\n{'=' * 78}")

    c_g, a_g, colnorm, rank, live = alignment_part(j, g)
    cl = colnorm[live]
    print(f"resolved rank = {rank};  {int(live.sum())} of {len(colnorm)} columns live;  "
          f"||S_i|| spans {cl.min():.3e} .. {cl.max():.3e} "
          f"({np.log10(cl.max() / max(cl.min(), 1e-300)):.1f} orders of magnitude over live columns)")

    # ---- the slope test: -2 for c, -1 for atilde, 0 for the alignment part ----
    pt_g = j @ np.linalg.lstsq(j, g, rcond=None)[0]
    atilde = c_g * (j.T @ g) / max(float(pt_g @ pt_g), 1e-30)
    print("\n-- slope of log(score) on log||S_i||  (theory: c -> -2, atilde -> -1, alignment -> 0) --")
    slopes = {}
    for name, vec in (("|c_i|", c_g), ("|atilde_i|", atilde), ("|a_i| (alignment)", a_g)):
        s, r2 = log_slope(vec, colnorm)
        slopes[name] = (s, r2)
        print(f"   {name:20s} slope = {s:+.3f}   R^2 = {r2:.3f}")

    # ---- how much of the |c_i| spread is conditioning? ----
    vd = variance_decomposition(a_g, colnorm)
    print("\n-- variance decomposition of log|c_i| --")
    print(f"   alignment (symmetry-specific) : {vd['share_alignment'] * 100:6.1f}%")
    print(f"   conditioning (target-blind)   : {vd['share_conditioning'] * 100:6.1f}%")
    print(f"   cross term                    : {vd['share_cross'] * 100:6.1f}%")
    print(f"   spread of |c_i|: {vd['sd_log10_c']:.2f} decades  "
          f"(conditioning alone would give {vd['sd_log10_conditioning']:.2f}; "
          f"alignment alone {vd['sd_log10_alignment']:.2f})")

    # ---- is the RANKING target-specific, once conditioning is removed? ----
    print("\n-- target-specificity of the ranking (vs matched-construction nulls) --")
    rows = []
    for i, t in enumerate(nulls):
        c_n, a_n, _, _, _ = alignment_part(j, t)
        both_c = (np.abs(c_g) > 0) & (np.abs(c_n) > 0)
        rho_c = float(spearmanr(np.abs(c_g[both_c]), np.abs(c_n[both_c])).statistic)
        rho_a = float(spearmanr(np.abs(a_g[both_c]), np.abs(a_n[both_c])).statistic)
        rows.append((rho_c, rho_a))
        print(f"   null {i + 1}: Spearman(|c(g)|,|c(null)|) = {rho_c:+.3f}   "
              f"Spearman(|a(g)|,|a(null)|) = {rho_a:+.3f}   "
              f"conditioning-induced agreement = {rho_c - rho_a:+.3f}")
    mean_rho_c = float(np.mean([r[0] for r in rows])) if rows else float("nan")
    mean_rho_a = float(np.mean([r[1] for r in rows])) if rows else float("nan")

    # ---- Lorenz: is the "small dominant subset" itself target-specific? ----
    curves = {"|c_i| (true generator)": lorenz(c_g), "|a_i| (alignment only)": lorenz(a_g)}
    for i, t in enumerate(nulls[:2]):
        c_n, a_n, _, _, _ = alignment_part(j, t)
        curves[f"|c_i| (null {i + 1})"] = lorenz(c_n)
        curves[f"|a_i| (null {i + 1})"] = lorenz(a_n)
    inv = np.zeros_like(colnorm)
    inv[live] = 1.0 / colnorm[live] ** 2
    curves["1/||S_i||^2 (conditioning only)"] = lorenz(inv)

    def top_share(curve, frac):
        idx = max(0, int(len(curve) * frac) - 1)
        return float(curve[idx])

    print("\n-- Lorenz: share of total attribution carried by the top 10% of parameters --")
    for name, curve in curves.items():
        print(f"   {name:34s} {top_share(curve, 0.10) * 100:5.1f}%")

    return {
        "label": label, "slopes": slopes, "vd": vd, "curves": curves,
        "colnorm": colnorm, "c_g": c_g, "a_g": a_g, "atilde": atilde,
        "mean_rho_c": mean_rho_c, "mean_rho_a": mean_rho_a,
    }


def plot(results: list[dict], out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.6 * n), squeeze=False)
    for r, row in zip(results, axes):
        cn = r["colnorm"]
        for ax, (name, vec, expected) in zip(
            row[:2],
            [("|c_i|", r["c_g"], -2.0), ("|a_i| = |c_i| ||S_i||^2", r["a_g"], 0.0)],
        ):
            mask = np.abs(vec) > 0
            ax.scatter(cn[mask], np.abs(vec[mask]), s=6, alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\|S_i\|$")
            ax.set_ylabel(name)
            slope, r2 = r["slopes"]["|c_i|" if expected == -2.0 else "|a_i| (alignment)"]
            ax.set_title(f"{r['label']}\n{name} vs conditioning: slope {slope:+.2f} "
                         f"(theory {expected:+.0f}), $R^2$={r2:.2f}", fontsize=8)
            ax.grid(alpha=0.3)
        ax = row[2]
        for name, curve in r["curves"].items():
            frac = np.arange(1, len(curve) + 1) / len(curve)
            ax.plot(frac, curve, "--" if "null" in name or "conditioning" in name else "-",
                    linewidth=1.4, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("fraction of parameters (sorted by |score|)")
        ax.set_ylabel("cumulative share")
        ax.set_title("Lorenz: does the true generator concentrate\nmore than a null or than conditioning alone?",
                     fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out}")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: python conditioning_decomposition.py <analysis_inputs.npz> [...]")
    results = [analyse(p) for p in paths]
    out = paths[0].parent / "conditioning_decomposition.png"
    plot(results, out)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for r in results:
        print(f"{r['label']}: conditioning explains {r['vd']['share_conditioning'] * 100:.0f}% of "
              f"Var(log|c_i|); slope(|c_i|) = {r['slopes']['|c_i|'][0]:+.2f}; "
              f"rank agreement with nulls: |c| {r['mean_rho_c']:+.2f} vs alignment {r['mean_rho_a']:+.2f}")


if __name__ == "__main__":
    main()
