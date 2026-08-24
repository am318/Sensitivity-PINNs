"""Re-render a handful of individual panels from the equivariance-flow figures as their
own standalone plots, for direct use in the paper. Reads back the saved
equivariance_flow_summary.json / angular_deviation_profiles.npz / equivariance_flow_parameters.npz
from both output directories -- no re-simulation, only forward passes for the decision-boundary
panels (the walk itself is not re-run).

Produces, under --output-dir (default outputs/paper_figures/equivariance_flow):

  angular_deviation_wedge_trained.{png,pdf}
      2x2 grid: rows = {c, -b}, columns = {resolved, fixed}, condition wedge/trained,
      pulled from outputs/aep_core_annulus_equivariance_flow and ..._fixed.

  decision_boundary_wedge_trained_{c,b}.{png,pdf}
      1x3 row from outputs/aep_core_annulus_equivariance_flow_fixed, condition wedge/trained,
      columns t=0, --mid-t (default 0.4), t=1.

Usage:
    python3 extract_equivariance_flow_subplots.py
    python3 extract_equivariance_flow_subplots.py --mid-t 0.6
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import aep_core_annulus_mlp as m
from aep_core_annulus_equivariance_flow import (
    KIND_TARGET,
    exact_profile,
    probe_grid,
    set_flat_parameters,
    task_metrics,
)
from aep_core_annulus_tangent import equivariance_defect, condition_style

RESOLVED_DIR = Path("outputs/aep_core_annulus_equivariance_flow")
FIXED_DIR = Path("outputs/aep_core_annulus_equivariance_flow_fixed")
CONDITION = "wedge/trained"
SOURCE_DIR = Path("outputs/aep_core_annulus_mlp")  # the run behind "wedge"
GROUP_ORDER = 64  # matches --group-order default used to produce these runs


def load_profile(output_dir: Path, kind: str, condition: str) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(output_dir / "angular_deviation_profiles.npz")
    return stored["angles"], stored[f"{kind}/{condition}"]


def snapshot_times(output_dir: Path) -> list[float]:
    payload = json.loads((output_dir / "equivariance_flow_summary.json").read_text())
    return [float(t) for t in ast.literal_eval(payload["args"]["snapshot_times"])]


def figure_angular_grid(args, output_dir: Path) -> None:
    """2x2: rows = {c, -b}, columns = {resolved, fixed}, one condition (wedge/trained)."""
    kinds = [("c", r"along $c$"), ("b", r"along $-b$")]
    modes = [("resolved", RESOLVED_DIR), ("fixed", FIXED_DIR)]
    times = snapshot_times(RESOLVED_DIR)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-times[-1], vmax=times[-1])

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.0), constrained_layout=True, squeeze=False)
    for row_idx, (kind, kind_label) in enumerate(kinds):
        for col_idx, (mode_name, mode_dir) in enumerate(modes):
            ax = axes[row_idx, col_idx]
            angles, profile = load_profile(mode_dir, kind, CONDITION)
            n_steps = profile.shape[0] - 1
            for t in times:
                step = min(max(int(round(t / (times[-1] / n_steps))), 0), n_steps)
                colour = cmap(norm(t))
                lw = 2.2 if step == 0 else 1.3
                ax.plot(np.degrees(angles), profile[step], color=colour, linewidth=lw,
                        label=f"$t = {t:g}$")
                ax.plot(np.degrees(angles), exact_profile(kind, profile[0], angles, t),
                        color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
            ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_title(f"{mode_name}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{kind_label}\n" + r"$d(\theta) = f(r,\theta) - \overline{f}(r)$")
            if row_idx == 1:
                ax.set_xlabel(r"angle $\theta$ (deg) at $r = 1.00$")
    axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       title="solid = walk, dotted = exact")
    fig.suptitle(f"Angular deviation profile, condition {CONDITION}: "
                 "re-solved (left) vs. one direction fixed at $t=0$ (right)")
    path = args.output_dir / "angular_deviation_wedge_trained.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def figure_decision_boundary_row(args, kind: str, mode_name: str, mode_dir: Path, device, dtype) -> None:
    """1x3 row from one run (resolved or fixed): t = 0, --mid-t, 1 -- condition wedge/trained."""
    checkpoint = torch.load(SOURCE_DIR / "checkpoints.pt", weights_only=False)
    cfg = checkpoint["config"]
    model = m.SimpleMLP(hidden_dim=cfg["hidden_dim"], depth=cfg["depth"]).to(device=device, dtype=dtype).eval()

    snapshots = np.load(mode_dir / "equivariance_flow_parameters.npz")
    times = snapshot_times(mode_dir)
    mid_t = min(times, key=lambda t: abs(t - args.mid_t))
    columns = [0.0, mid_t, 1.0]

    x = torch.tensor(checkpoint["x"], device=device, dtype=dtype)
    y = torch.tensor((checkpoint["y"] > 0).astype(np.float64), device=device, dtype=dtype)
    points = probe_grid(3.5, 24, 3.5, device, dtype)  # matches the --probe-* defaults used to produce these runs

    mesh = m.decision_grid(args.boundary_limit, args.boundary_grid, device, dtype)
    fig, axes = plt.subplots(1, 3, figsize=(3.4 * 3, 4.0), constrained_layout=True)
    for col_idx, t in enumerate(columns):
        ax = axes[col_idx]
        params = torch.tensor(snapshots[f"{kind}/{CONDITION}/{t:g}"], device=device, dtype=dtype)
        set_flat_parameters(model, params)
        m.draw_decision_panel(ax, model, mesh, checkpoint["x"], checkpoint["y"],
                              n_levels=args.boundary_levels)
        values, defect = equivariance_defect(model, points, GROUP_ORDER)
        relative = float(torch.linalg.vector_norm(defect)
                         / torch.linalg.vector_norm(values).clamp_min(1e-15))
        _, accuracy = task_metrics(model, x, y)
        ax.text(0.5, -0.02, f"$\\|(I-P)f\\|/\\|f\\|$ = {relative:.3f},  acc = {accuracy:.2f}",
                ha="center", va="top", fontsize=11, transform=ax.transAxes)
        label = "start" if t == 0.0 else ("$t=1$" if t == 1.0 else f"$t={t:g}$")
        ax.set_xlabel(label, fontsize=14, labelpad=22)
    axes[0].set_ylabel(CONDITION, fontsize=14, labelpad=10)
    mode_phrase = ("direction re-solved at every step" if mode_name == "resolved"
                   else "one direction fixed at $t=0$")
    fig.suptitle(f"Decision boundary, {CONDITION}, {mode_phrase}: walking {KIND_TARGET[kind]}",
                fontsize=13)
    path = args.output_dir / f"decision_boundary_wedge_trained_{kind}_{mode_name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures/equivariance_flow"))
    parser.add_argument("--mid-t", type=float, default=0.4,
                        help="Snapshot time closest to this value is used as the middle column.")
    parser.add_argument("--boundary-limit", type=float, default=4.0)
    parser.add_argument("--boundary-grid", type=int, default=220)
    parser.add_argument("--boundary-levels", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = torch.device("cpu"), torch.float64

    figure_angular_grid(args, args.output_dir)
    for kind in ("c", "b"):
        for mode_name, mode_dir in (("resolved", RESOLVED_DIR), ("fixed", FIXED_DIR)):
            figure_decision_boundary_row(args, kind, mode_name, mode_dir, device, dtype)


if __name__ == "__main__":
    main()
