"""Is the "small subset of dominant parameters" identifiable, or an artefact of the probe set?

The SCML draft's central empirical object is the identity of the top-|c_i|
parameters. But c = J^+ g is computed from a Jacobian evaluated on a *finite
probe grid*, and J is rank-deficient: the resolved rank is far below the
parameter count, so the coefficient vector is determined by a small number of
directions estimated from a modest number of probe points. Whether the
resulting parameter ranking is a property of the network or of the grid is an
empirical question that has not been asked.

This script asks it by resampling the probe rows (bootstrap over probe points)
and recomputing every score, then measuring how much the ranking and the
top-k set move. It needs no retraining: it works directly on the saved J.

Reference points, both essential for reading the numbers:

- ||S_i|| (column norm) is recomputed on each resample too. It is a simple,
  target-blind statistic, so it sets the "easy" stability level -- any score
  that is merely tracking conditioning should be about this stable.
- Module-level aggregation is also reported. Individual parameters are
  interchangeable under the network's own hidden-unit permutation symmetry,
  so a claim about *which parameters* carry a symmetry is on much weaker
  footing than a claim about which *modules* do. If parameter-level rankings
  are unstable while module-level ones are stable, that is the finding.

Usage: python attribution_stability.py <analysis_inputs.npz> [--draws N] [--frac F]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from asrnn_mexican_hat_symmetry_sensitivity import Config, build_model, validate_config
from experiment_common import parameter_layout, select_device, select_dtype
from sensitivity_tools import choose_l1_ratio_for_sparsity, tangent_projection, tangent_projection_auto

CUTOFF = 1e-3


def l2_attrib(j: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    _, _, c, _, _ = tangent_projection(j, t, CUTOFF, normalize_columns=True)
    return c.numpy()


def scores_for(j_rows: np.ndarray, g_rows: np.ndarray, null_rows: list[np.ndarray],
               l1_ratio: float | None = None) -> dict[str, np.ndarray]:
    jt = torch.from_numpy(j_rows)
    colnorm = jt.norm(dim=0).numpy()
    live = colnorm > max(colnorm.max() * 1e-8, 1e-300)
    c_g = l2_attrib(jt, torch.from_numpy(g_rows))
    c_n = np.stack([l2_attrib(jt, torch.from_numpy(t)) for t in null_rows])

    def calibrate(numerator: np.ndarray, denom_stack: np.ndarray) -> np.ndarray:
        log_denom = np.log(np.clip(np.abs(denom_stack), 1e-300, None)).mean(axis=0)
        return np.where(live, np.abs(numerator) / np.clip(np.exp(log_denom), 1e-300, None), 0.0)

    # A calibrated score divides by a geometric mean of a few noisy null solves, so some
    # of its instability is denominator noise rather than absence of signal. The
    # null-vs-null control has the SAME estimator and the same denominator noise but no
    # symmetry signal at all: whatever stability it shows is the noise floor, and only
    # the excess of r_i over it is evidence of a reproducible symmetry-specific ranking.
    out = {
        "|c_i| = |J^+ g|": np.abs(c_g),
        "r_i (null-calibrated)": calibrate(c_g, c_n),
        "r'_i (null-vs-null CONTROL)": calibrate(c_n[0], c_n[1:]),
        "|a_i| = |c_i| ||S_i||^2": np.abs(c_g) * colnorm**2,
        "||S_i|| (conditioning reference)": colnorm,
    }
    if l1_ratio is not None:
        # Elastic net approximates min ||c||_0 s.t. J c ~ g -- "which k parameters suffice
        # to reproduce the symmetry direction". Unlike a min-norm solution, that criterion
        # does not reference any metric on parameter space, so it is gauge-free by
        # construction: worth knowing whether it is also identifiable.
        _, _, c_en, _, _ = tangent_projection_auto(
            jt, torch.from_numpy(g_rows), CUTOFF, method="elastic_net", l1_ratio=l1_ratio)
        out["EN support (gauge-free selection)"] = np.abs(c_en.numpy())
    return out, live


def module_vector(vec: np.ndarray, slices: dict[str, slice]) -> np.ndarray:
    return np.array([np.linalg.norm(vec[sl]) for sl in slices.values()])


def unit_index_map(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Flat parameter indices owned by each hidden unit: its row of W_l, its bias, and its
    column of W_{l+1}.

    Individual weight indices are not well-posed targets for a "which parameters
    carry the symmetry" claim: the network's own hidden-unit permutation symmetry
    relabels them freely, so parameter 1729 means nothing across two runs, and
    nothing intrinsic within one. The permutation acts on *units* by relabelling,
    so a unit-level score is permutation-equivariant and its sorted profile /
    top-k multiset is permutation-invariant -- which makes the hidden unit the
    finest granularity at which localisation of a symmetry can be stated at all.
    """
    import re

    offsets, shapes, off = {}, {}, 0
    for name, p in model.named_parameters():
        offsets[name], shapes[name] = off, tuple(p.shape)
        off += p.numel()
    nets: dict[str, dict[int, dict[str, str]]] = {}
    for name in offsets:
        m = re.match(r"(.*)\.(\d+)\.(weight|bias)$", name)
        if m:
            nets.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = name
    units: dict[str, np.ndarray] = {}
    for prefix, layers in nets.items():
        ordered = sorted(layers)
        for li, layer in enumerate(ordered[:-1]):  # hidden layers only; the output layer has no units
            wname, bname = layers[layer].get("weight"), layers[layer].get("bias")
            if wname is None:
                continue
            out_dim, in_dim = shapes[wname]
            nxt = layers[ordered[li + 1]].get("weight")
            for u in range(out_dim):
                flat = list(range(offsets[wname] + u * in_dim, offsets[wname] + (u + 1) * in_dim))
                if bname is not None:
                    flat.append(offsets[bname] + u)
                if nxt is not None:
                    o2, (od2, id2) = offsets[nxt], shapes[nxt]
                    flat.extend(o2 + r * id2 + u for r in range(od2))
                units[f"{prefix}.{layer}.u{u}"] = np.array(flat, dtype=np.int64)
    return units


def unit_vector(vec: np.ndarray, units: dict[str, np.ndarray]) -> np.ndarray:
    return np.array([np.linalg.norm(vec[idx]) for idx in units.values()])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.8, help="fraction of probe rows kept per resample")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    path = Path(args.npz)
    d = np.load(path)
    j, g = d["j"], d["g"]
    nulls = [d[k] for k in d.files if k.startswith("null_target_")]
    n_rows, n_params = j.shape
    print(f"{path.parent.name}: J {j.shape}, {len(nulls)} nulls, "
          f"{args.draws} resamples keeping {args.frac:.0%} of probe rows")

    # module layout: fixed by architecture, so rebuild the model purely to read it off
    cfg = Config()
    cfg.architecture = "hamiltonian"
    cfg.device, cfg.training_steps = "cpu", 1
    validate_config(cfg)
    model, _ = build_model(cfg, select_device("cpu"), select_dtype(cfg.dtype))
    _, slices = parameter_layout(model)
    units = unit_index_map(model)
    if sum(sl.stop - sl.start for sl in slices.values()) != n_params:
        print("  (module layout does not match this J; skipping module/unit-level analysis)")
        slices, units = {}, {}
    else:
        print(f"  {len(slices)} modules, {len(units)} hidden units")

    l1_ratio = float(choose_l1_ratio_for_sparsity(torch.from_numpy(j), torch.from_numpy(g), CUTOFF))
    print(f"  elastic-net l1_ratio calibrated once on the full grid: {l1_ratio:.3f}")
    full_scores, live = scores_for(j, g, nulls, l1_ratio)
    rng = np.random.default_rng(0)
    keep_n = max(8, int(n_rows * args.frac))

    draws = {name: [] for name in full_scores}
    module_draws = {name: [] for name in full_scores}
    unit_draws = {name: [] for name in full_scores}
    for _ in range(args.draws):
        rows = rng.choice(n_rows, size=keep_n, replace=False)
        sc, _ = scores_for(j[rows], g[rows], [t[rows] for t in nulls], l1_ratio)
        for name, vec in sc.items():
            draws[name].append(vec)
            if slices:
                module_draws[name].append(module_vector(vec, slices))
            if units:
                unit_draws[name].append(unit_vector(vec, units))

    n_top_units = max(3, len(units) // 10) if units else 0
    unit_hdr = f"top-{n_top_units} unit" if units else "unit (n/a)"
    print(f"\n{'score':34s}{'rank rho':>11s}{'top-k Jacc':>12s}{'vs full':>10s}"
          f"{'UNIT Jacc':>12s}{'unit rho':>11s}{'module rho':>13s}")
    results = {}
    for name in full_scores:
        mat = np.stack(draws[name])
        sub = mat[:, live]
        rhos = [float(spearmanr(sub[a], sub[b]).statistic)
                for a in range(len(mat)) for b in range(a + 1, len(mat))]
        tops = [np.argsort(-m)[: args.topk] for m in mat]
        jac = [jaccard(tops[a], tops[b]) for a in range(len(tops)) for b in range(a + 1, len(tops))]
        full_top = np.argsort(-full_scores[name])[: args.topk]
        jac_full = [jaccard(t, full_top) for t in tops]
        if slices:
            mm = np.stack(module_draws[name])
            mrho = float(np.nanmean([float(spearmanr(mm[a], mm[b]).statistic)
                                     for a in range(len(mm)) for b in range(a + 1, len(mm))]))
        else:
            mrho = float("nan")
        if units:
            uu = np.stack(unit_draws[name])
            urho = float(np.nanmean([float(spearmanr(uu[a], uu[b]).statistic)
                                     for a in range(len(uu)) for b in range(a + 1, len(uu))]))
            utops = [np.argsort(-u)[:n_top_units] for u in uu]
            ujac = float(np.mean([jaccard(utops[a], utops[b])
                                  for a in range(len(utops)) for b in range(a + 1, len(utops))]))
        else:
            urho = ujac = float("nan")
        results[name] = {"rank_rho": float(np.nanmean(rhos)), "jaccard": float(np.mean(jac)),
                         "jaccard_full": float(np.mean(jac_full)), "module_rho": mrho,
                         "unit_rho": urho, "unit_jaccard": ujac}
        print(f"{name:34s}{results[name]['rank_rho']:>11.3f}{results[name]['jaccard']:>12.3f}"
              f"{results[name]['jaccard_full']:>10.3f}{ujac:>12.3f}{urho:>11.3f}{mrho:>13.3f}")
    print("\n  rank rho      : mean pairwise Spearman between resamples (live parameters only)")
    print("  top-k Jaccard : mean pairwise overlap of the top-k parameter sets between resamples")
    print(f"  top-k vs full : mean overlap of each resample's top-{args.topk} with the full-grid top-{args.topk}")
    print(f"  UNIT Jacc     : overlap of the {unit_hdr} sets between resamples")
    print("  unit rho      : rank agreement over hidden units (permutation-equivariant granularity)")
    print("  module rho    : same, after aggregating |score| to module level (coarsest, permutation-safe)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(results)
    x = np.arange(len(names))
    axes[0].bar(x - 0.25, [results[n]["jaccard"] for n in names], 0.25, label=f"parameter-level (top-{args.topk} Jaccard)")
    axes[0].bar(x, [results[n]["unit_jaccard"] for n in names], 0.25, label=f"unit-level (top-{n_top_units} Jaccard)")
    axes[0].bar(x + 0.25, [results[n]["module_rho"] for n in names], 0.25, label="module-level (rank rho)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("stability under probe-grid resampling")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("Is the identified subnetwork identifiable?")

    for name in names:
        mat = np.stack(draws[name])[:, live]
        med = np.median(mat, axis=0)
        order = np.argsort(-med)
        lo, hi = np.percentile(mat[:, order], [10, 90], axis=0)
        axes[1].fill_between(np.arange(len(order)), np.clip(lo, 1e-300, None),
                             np.clip(hi, 1e-300, None), alpha=0.25, label=name)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("parameter rank (by median score)")
    axes[1].set_ylabel("score (10-90% band over resamples)")
    axes[1].set_title("Spread of each score under probe resampling")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = path.parent / "attribution_stability.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
