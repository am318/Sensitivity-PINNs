"""Does ordinary training's own update point towards or away from ``Delta = f - P(f)``?

Companion to ``aep_core_annulus_mlp.py`` / ``aep_core_annulus_tangent.py``. The
tangent script answers a static question -- *can* the parameters realise the
non-equivariant component ``Delta = f - P(f)``, i.e. does it lie in
``T_theta = Im(J_theta)``. ``aep_core_annulus_equivariance_step.py`` then walks
deliberately along the min-norm solution ``c`` (``J c ~ Delta``, so ``-c`` is the
symmetrising direction). Neither says whether *unconstrained* training ever moves
that way on its own. That is what this script measures: at every epoch of the
exact training run saved in ``checkpoints.pt`` it records the cosine similarity
between ``Delta`` and the optimiser's update, for the wedge dataset (90-degree
outer arc, so the training set is not rotation-symmetric) and for the fully
invariant task (full annulus).

Sign convention, used everywhere below: ``Delta = f - P(f)``, so a cosine

    < 0   the update moves ``f`` *towards* ``P(f)``   (training symmetrises)
    > 0   the update moves ``f`` *away* from ``P(f)`` (training amplifies the defect)
    ~ 0   the update is (function-space) orthogonal to the defect

Two directions ``b`` are recorded per epoch, since "the gradient update" can
reasonably mean either:

``adam``
    ``b = theta_{t+1} - theta_t``, the step Adam actually took -- what training
    did. This is the one the walk in ``equivariance_step`` is comparable to.
``gradient``
    ``b = -grad_theta L``, the raw descent direction, before Adam's
    per-coordinate rescaling. Adam's preconditioner rotates the step away from
    this, sometimes a lot, so the two curves are genuinely different claims.

and the cosine itself is taken in both of the two spaces where the comparison
is well posed:

*function space* (``cos(Delta, J b)``, the panel to trust)
    The step's actual first-order effect on ``f`` on the probe grid. Needs no
    SVD cutoff, so it is the cutoff-free version of the question, and it is
    checked once per epoch against the realised finite change
    ``f_{theta+b} - f_theta`` (recorded as ``cos_realised_adam``).

*parameter space* (``cos(c, b)``)
    Compares the update directly with the min-norm coefficients ``c`` the
    equivariance-step walk uses. Two caveats, both reported rather than hidden:
    ``c`` depends on the SVD cutoff (as in ``aep_core_annulus_tangent.py``), and
    with ``P >> N`` most of ``b`` lives in ``ker(J)`` on this probe grid, which
    dilutes ``cos(c, b)`` towards zero for purely dimensional reasons. So the
    same cosine restricted to the resolved row space, ``cos(c, Pi_T b)``, is
    recorded alongside it, together with ``||Pi_T b|| / ||b||`` -- the fraction
    of the update that is visible on the probe grid at all. ``J`` is very badly
    conditioned here (its resolved rank is a few dozen out of ~19k parameters),
    so the two spaces can and do disagree in *sign*: a parameter-space cosine
    near zero, or of the opposite sign, does not contradict a large function-space
    one, it just means the alignment lives in directions ``J`` barely amplifies.
    When they disagree, the function-space panel is the one that describes what
    happened to ``f``.

Finally, the per-parameter scatter behind the parameter-space number: ``c_i``
against ``b_i`` at initialisation and at the final epoch, in the style of
``aep_core_annulus_ci_scatter.py`` (one point per parameter, coloured by
module). Perfect symmetrising alignment would put every point on ``y = -x``.

Training is replayed from the saved ``initial_state_dict`` with the saved
config, so the trajectory is the one already on disk rather than a fresh run
that merely shares a seed -- the max deviation of the replayed parameters from
the saved ``final_state_dict`` is printed with the summary (0.0, i.e. bit-exact,
for the runs in ``outputs/``) so that claim is checked rather than assumed.
Every curve and both coefficient vectors are written out, so ``--replot``
redraws the figures without recomputing a single Jacobian.

Usage:
    python aep_core_annulus_delta_gradient.py
    python aep_core_annulus_delta_gradient.py --probe-every 5
    python aep_core_annulus_delta_gradient.py --replot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import aep_core_annulus_mlp as mlp
from aep_core_annulus_mlp import SimpleMLP
from aep_core_annulus_tangent import (
    build_probe_grid,
    equivariance_defect,
    sensitivity_jacobian,
)
from experiment_common import (
    apply_config_overrides,
    module_colours,
    parameter_layout,
    plot_magnitude_vs_quantity,
    prettify_parameter_name,
    select_device,
)
from sensitivity_tools import tangent_projection

UPDATE_KINDS = ("adam", "gradient")
UPDATE_LABEL = {
    "adam": r"Adam step $\theta_{t+1}-\theta_t$",
    "gradient": r"raw descent $-\nabla_\theta L$",
}
RUN_LABEL = {"quarter": "wedge (90 deg arc)", "full": "full annulus"}
# One colour per dataset, one dash pattern per update kind, matching the
# quarter=red / full=blue convention of aep_core_annulus_tangent.CONDITION_STYLE.
RUN_COLOUR = {"quarter": "tab:red", "full": "tab:blue"}
UPDATE_STYLE = {"adam": "-", "gradient": "--"}


def run_label(run: str) -> str:
    """Readable name for a run, falling back to the ``--runs`` key for extra runs."""
    return RUN_LABEL.get(run, run)


def run_colour(run: str) -> str:
    return RUN_COLOUR.get(run, "tab:grey")


@dataclass
class Config:
    device: str = "auto"
    output_dir: str = "outputs/aep_core_annulus_delta_gradient"
    runs: dict[str, str] = field(
        default_factory=lambda: {
            "quarter": "outputs/aep_core_annulus_mlp",
            "full": "outputs/aep_core_annulus_mlp_full",
        }
    )

    # Probe geometry and projection group: identical to aep_core_annulus_tangent.py
    # so ``c`` here is the same vector that script reports and equivariance_step walks.
    probe_extent: float = 3.5
    probe_grid_points_per_axis: int = 24
    probe_max_radius: float = 3.5
    projection_group_order: int = 64
    tangent_svd_relative_cutoff: float = 1e-3

    # Record every ``probe_every``-th epoch. One record costs a Jacobian plus an
    # SVD (~0.3 s at the default grid), so every epoch of a 200-epoch run is
    # affordable; raise this for long runs.
    probe_every: int = 1

    # Rolling-mean window for the cosine curves, as a fraction of the run. The
    # raw per-epoch values are always drawn underneath at low alpha.
    smoothing_fraction: float = 0.02


# ------------------------------ flat vectors ------------------------------
def flat_parameters(model: nn.Module) -> torch.Tensor:
    """Current parameters as one flat vector, in ``model.parameters()`` order.

    That is the order the Jacobian columns and ``c`` use (see
    ``sensitivity_tools.parameter_gradient_row``), so index ``i`` means the same
    parameter in ``b`` and in ``c``.
    """
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_gradients(model: nn.Module) -> torch.Tensor:
    """Current ``.grad`` as one flat vector, in the same order; missing grads read as zero."""
    return torch.cat([
        torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1)
        for p in model.parameters()
    ])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """``<a, b> / (||a|| ||b||)``, or NaN if either vector vanishes.

    NaN rather than 0 on a degenerate input: a zero update has no direction, and
    a gap in the curve is the honest way to show that.
    """
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(torch.dot(a, b) / (norm_a * norm_b))


# --------------------------- the per-epoch probe ---------------------------
def probe_alignment(
    probe_model: nn.Module,
    points: torch.Tensor,
    directions: dict[str, torch.Tensor],
    cfg: Config,
    *,
    verify_against_tangent_projection: bool = False,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor]:
    """Cosines between ``Delta`` at the current parameters and each direction in ``directions``.

    One SVD of ``J`` serves all of it: the min-norm coefficients ``c``, the
    projector onto the resolved row space, and (via a matvec with the untruncated
    ``J``) the function-space image of every direction. Returns the record, ``c``,
    and ``Delta`` itself (the caller needs it for the finite realised-step check).
    """
    values, defect = equivariance_defect(probe_model, points, cfg.projection_group_order)
    jacobian = sensitivity_jacobian(probe_model, points)

    u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    keep = singular_values >= cfg.tangent_svd_relative_cutoff * singular_values[0]
    resolved_rank = int(keep.sum())
    row_basis = vh[keep, :]  # orthonormal rows spanning the resolved row space
    coordinates = u[:, keep].T @ defect
    coefficients = row_basis.T @ (coordinates / singular_values[keep])

    if verify_against_tangent_projection:
        _, _, reference, reference_rank, _ = tangent_projection(
            jacobian, defect, cfg.tangent_svd_relative_cutoff
        )
        assert reference_rank == resolved_rank, "resolved rank disagrees with tangent_projection"
        assert torch.allclose(coefficients.cpu(), reference, atol=1e-10), (
            "min-norm coefficients disagree with sensitivity_tools.tangent_projection"
        )

    function_norm = torch.linalg.vector_norm(values).clamp_min(1e-15)
    record: dict[str, float] = {
        "defect_norm": float(torch.linalg.vector_norm(defect)),
        "defect_relative": float(torch.linalg.vector_norm(defect) / function_norm),
        "coefficient_norm": float(torch.linalg.vector_norm(coefficients)),
        "resolved_rank": float(resolved_rank),
    }
    for kind, direction in directions.items():
        in_row_space = row_basis.T @ (row_basis @ direction)
        record[f"cos_function_{kind}"] = cosine(defect, jacobian @ direction)
        record[f"cos_parameter_{kind}"] = cosine(coefficients, direction)
        record[f"cos_parameter_rowspace_{kind}"] = cosine(coefficients, in_row_space)
        record[f"rowspace_fraction_{kind}"] = float(
            torch.linalg.vector_norm(in_row_space)
            / torch.linalg.vector_norm(direction).clamp_min(1e-300)
        )
        record[f"direction_norm_{kind}"] = float(torch.linalg.vector_norm(direction))
    return record, coefficients.cpu().numpy(), defect


@torch.no_grad()
def probe_values(model: nn.Module, points: torch.Tensor) -> torch.Tensor:
    return model(points).squeeze(-1)


def replay_run(
    run_dir: Path, points: torch.Tensor, cfg: Config, device: torch.device
) -> dict[str, Any]:
    """Re-run the saved training trajectory, probing the alignment along the way.

    Starts from the checkpoint's own ``initial_state_dict`` and its own config, so
    this is the trajectory already on disk rather than a fresh run that merely
    shares a seed; the deviation of the replayed final parameters from the saved
    ``final_state_dict`` is returned so that claim stays checkable.
    """
    checkpoint = torch.load(run_dir / "checkpoints.pt", weights_only=False)
    known = {f.name for f in dataclasses.fields(mlp.Config)}
    # Older runs predate some Config fields (e.g. l1_weight); those fall back to
    # the current defaults, which is exactly what the original run used.
    run_cfg = mlp.Config(**{k: v for k, v in checkpoint["config"].items() if k in known})
    dtype = getattr(torch, run_cfg.dtype)

    model = SimpleMLP(hidden_dim=run_cfg.hidden_dim, depth=run_cfg.depth)
    model.load_state_dict(checkpoint["initial_state_dict"])
    model = model.to(device=device, dtype=dtype)

    # The probe always runs in float64 on a copy: the alignment numbers involve an
    # SVD and a near-cancelling difference f - P(f), and float32 is not enough for
    # either. Training itself stays in the run's own dtype so the trajectory is
    # reproduced bit for bit.
    probe_model = SimpleMLP(hidden_dim=run_cfg.hidden_dim, depth=run_cfg.depth)
    probe_model = probe_model.to(device=device, dtype=torch.float64).eval()

    x = torch.tensor(checkpoint["x"], device=device, dtype=dtype)
    y = torch.tensor((checkpoint["y"] > 0).astype(np.float32), device=device, dtype=dtype)[:, None]
    optimiser = optim.Adam(model.parameters(), lr=run_cfg.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs: list[int] = []
    curves: dict[str, list[float]] = {}
    saved_vectors: dict[str, dict[str, np.ndarray]] = {}
    last_probed_epoch = max(
        e for e in range(run_cfg.epochs) if e % cfg.probe_every == 0
    )
    report_every = max(1, run_cfg.epochs // 10)

    for epoch in range(run_cfg.epochs):
        probing = epoch % cfg.probe_every == 0
        if probing:
            probe_model.load_state_dict(
                {k: v.detach().to(torch.float64) for k, v in model.state_dict().items()}
            )
            before = probe_values(probe_model, points)

        optimiser.zero_grad()
        bce = loss_fn(model(x), y)
        objective = bce
        if run_cfg.l1_weight > 0:
            objective = objective + run_cfg.l1_weight * sum(
                p.abs().sum() for p in model.parameters()
            )
        objective.backward()
        gradient = flat_gradients(model)
        theta_before = flat_parameters(model).clone()
        optimiser.step()
        update = flat_parameters(model) - theta_before

        if not probing:
            continue

        directions = {
            "adam": update.detach().to(torch.float64),
            "gradient": -gradient.detach().to(torch.float64),
        }
        record, coefficients, defect = probe_alignment(
            probe_model, points, directions, cfg,
            # The hand-rolled SVD above must agree with the shared helper the rest
            # of the project uses; check it once rather than trusting it.
            verify_against_tangent_projection=(epoch == 0),
        )

        # Finite check on the linearisation: probe_model still holds theta_before,
        # so load the post-step parameters and compare the function change the step
        # really produced against the J @ update version above. The two cosines
        # agreeing is what licenses reading cos(Delta, J b) as "what the step did".
        probe_model.load_state_dict(
            {k: v.detach().to(torch.float64) for k, v in model.state_dict().items()}
        )
        record["cos_realised_adam"] = cosine(defect, probe_values(probe_model, points) - before)

        record["bce"] = float(bce.detach())
        record["objective"] = float(objective.detach())

        epochs.append(epoch)
        for key, value in record.items():
            curves.setdefault(key, []).append(value)

        if epoch == 0 or epoch == last_probed_epoch:
            state = "initial" if epoch == 0 else "final"
            saved_vectors[state] = {
                "c": coefficients,
                **{f"b_{kind}": d.cpu().numpy() for kind, d in directions.items()},
            }

        if epoch % report_every == 0 or epoch == run_cfg.epochs - 1:
            print(
                f"  epoch {epoch:5d}  bce = {record['bce']:.3e}"
                f"  ||Delta||/||f|| = {record['defect_relative']:.3e}"
                f"  cos(Delta, J b_adam) = {record['cos_function_adam']:+.4f}"
                f"  cos(c, b_adam) = {record['cos_parameter_adam']:+.4f}"
            )

    reference = checkpoint["final_state_dict"]
    replayed = model.state_dict()
    deviation = max(
        float((replayed[k].detach().cpu() - v.detach().cpu()).abs().max()) for k, v in reference.items()
    )
    return {
        "run_config": asdict(run_cfg),
        "epochs": epochs,
        "curves": curves,
        "replay_max_deviation": deviation,
        "vectors": saved_vectors,
        "parameter_slices": {
            name: [sl.start, sl.stop] for name, sl in parameter_layout(model)[1].items()
        },
    }


# --------------------------------- plotting ---------------------------------
def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that ignores NaNs and keeps the input length."""
    if window <= 1:
        return values
    finite = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    pad = (window // 2, window - 1 - window // 2)
    kernel = np.ones(window)
    total = np.convolve(np.pad(filled, pad, mode="edge"), kernel, mode="valid")
    count = np.convolve(np.pad(finite, pad, mode="edge"), kernel, mode="valid")
    return np.where(count > 0, total / np.maximum(count, 1e-30), np.nan)


def draw_curve(ax, epochs, values, colour, style, label, window) -> None:
    values = np.asarray(values, dtype=float)
    ax.plot(epochs, values, color=colour, linestyle=style, linewidth=0.8, alpha=0.3)
    ax.plot(epochs, rolling_mean(values, window), color=colour, linestyle=style,
            linewidth=1.8, label=label)


def shade_symmetrising_half(ax) -> None:
    """Tint the ``cos < 0`` half-plane, where the update moves ``f`` towards ``P(f)``.

    Read off the axis limits and put them back afterwards so the band never
    stretches the y-range to include a sign the data never reaches -- on a panel
    whose curves stay positive throughout, no band appears at all, which is
    itself the finding.
    """
    low, high = ax.get_ylim()
    if low < 0:
        ax.axhspan(low, 0.0, color="#e9f2e9", zorder=0,
                   label=r"$\cos < 0$: step moves $f$ towards $P(f)$")
        ax.set_ylim(low, high)


def symlog_ticks(linear_threshold: float, limit: float) -> list[float]:
    """Sparse ticks for a symlog axis: 0 and every other decade out to ``limit``.

    Every decade would collide near zero, where the symlog linear region is
    narrower than the labels themselves (the same problem
    ``aep_core_annulus_tangent.plot_coefficient_distribution`` solves this way).
    """
    lowest = int(np.ceil(np.log10(linear_threshold)))
    highest = int(np.floor(np.log10(limit)))
    decades = [10.0**e for e in range(highest, lowest - 1, -2)][::-1]
    return [-t for t in reversed(decades)] + [0.0] + decades


def plot_alignment_history(records: dict[str, dict[str, Any]], cfg: Config, output_dir: Path) -> None:
    """The six alignment / context curves, all against the epoch axis."""
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.6), constrained_layout=True)
    axes = axes.ravel()

    windows = {
        run: max(1, int(cfg.smoothing_fraction * len(record["epochs"])))
        for run, record in records.items()
    }

    cosine_panels = [
        ("cos_function", r"$\cos(\Delta,\; J b)$", "function space (cutoff-free)"),
        ("cos_parameter", r"$\cos(c,\; b)$", "parameter space, full $b$"),
        ("cos_parameter_rowspace", r"$\cos(c,\; \Pi_T b)$",
         "parameter space, $b$ restricted to resolved row space"),
    ]
    for ax, (prefix, ylabel, title) in zip(axes, cosine_panels):
        for run, record in records.items():
            for kind in UPDATE_KINDS:
                key = f"{prefix}_{kind}"
                if key not in record["curves"]:
                    continue
                draw_curve(ax, record["epochs"], record["curves"][key], run_colour(run),
                           UPDATE_STYLE[kind], f"{run_label(run)}, {UPDATE_LABEL[kind]}",
                           windows[run])
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        # The sign is the whole point, so mark the symmetrising half-plane on the
        # panel itself rather than leaving it to the caption.
        shade_symmetrising_half(ax)
        ax.set(xlabel="epoch", ylabel=ylabel, title=title)
        ax.legend(fontsize=7.5)

    for run, record in records.items():
        draw_curve(axes[3], record["epochs"], record["curves"]["defect_relative"],
                   run_colour(run), "-", run_label(run), 1)
        draw_curve(axes[4], record["epochs"], record["curves"]["bce"],
                   run_colour(run), "-", run_label(run), 1)
        for kind in UPDATE_KINDS:
            draw_curve(axes[5], record["epochs"], record["curves"][f"rowspace_fraction_{kind}"],
                       run_colour(run), UPDATE_STYLE[kind],
                       f"{run_label(run)}, {UPDATE_LABEL[kind]}", windows[run])

    axes[3].set(xlabel="epoch", ylabel=r"$\|\Delta\| / \|f\|$", yscale="log",
                title="how non-equivariant the model is")
    axes[3].legend(fontsize=8)
    axes[4].set(xlabel="epoch", ylabel="BCE", yscale="log", title="task loss")
    axes[4].legend(fontsize=8)
    axes[5].set(xlabel="epoch", ylabel=r"$\|\Pi_T b\| / \|b\|$", ylim=(0.0, 1.05),
                title="fraction of the update visible on the probe grid")
    axes[5].legend(fontsize=7.5)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"delta_gradient_alignment.{suffix}")
    plt.close(fig)


def plot_cb_scatter(
    record: dict[str, Any], run: str, kind: str, output_dir: Path
) -> None:
    """``c_i`` against ``b_i``, one point per parameter, at init and at the final epoch.

    Both vectors are normalised to unit length first: ``||c||`` and ``||b||``
    differ by many orders of magnitude (and ``||b||`` shrinks as the loss
    converges), so the raw scales would leave the panels un-comparable and hide
    the only thing being asked about -- the *direction* relationship. On unit
    vectors the two dashed guides mean something definite: ``y = -x`` is a step
    that purely symmetrises, ``y = x`` one that purely amplifies ``Delta``.

    Expect the ``adam`` panel at ``initial`` to collapse onto two vertical
    stripes: with zero-initialised moment estimates Adam's very first step is
    ``+-lr`` in every coordinate regardless of gradient size, so ``b_i`` only
    takes two values there. That is Adam, not a plotting artefact -- the
    ``gradient`` panels show the underlying spread.
    """
    slices = {name: slice(*bounds) for name, bounds in record["parameter_slices"].items()}
    modules = list(slices)
    colours = module_colours(modules)
    # Two hidden weight matrices hold ~97% of the parameters, so drawing in
    # parameter order buries every bias and the output layer underneath them;
    # largest module first puts the small ones on top.
    draw_order = sorted(modules, key=lambda name: slices[name].stop - slices[name].start,
                        reverse=True)

    panels = [(state, record["vectors"][state]) for state in ("initial", "final")
              if state in record["vectors"]]
    unit = {
        state: (np.asarray(v["c"], float) / max(np.linalg.norm(v["c"]), 1e-300),
                np.asarray(v[f"b_{kind}"], float) / max(np.linalg.norm(v[f"b_{kind}"]), 1e-300))
        for state, v in panels
    }
    # One pair of axis limits for both panels: the interesting comparison is
    # init against final, which a per-panel autoscale would quietly undo.
    everything = np.abs(np.concatenate([a for pair in unit.values() for a in pair]))
    linear_threshold = max(float(np.percentile(everything[everything > 0], 5)), 1e-14)
    limit = float(everything.max()) * 1.6
    ticks = symlog_ticks(linear_threshold, limit)

    fig, axes = plt.subplots(1, len(panels), figsize=(7.6 * len(panels), 7.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (state, vectors) in zip(axes, panels):
        c = np.asarray(vectors["c"], dtype=float)
        b = np.asarray(vectors[f"b_{kind}"], dtype=float)
        c_norm, b_norm = np.linalg.norm(c), np.linalg.norm(b)
        c_hat, b_hat = unit[state]

        for name in draw_order:
            sl = slices[name]
            ax.scatter(b_hat[sl], c_hat[sl], s=10, color=colours[name],
                       label=prettify_parameter_name(name, modules), alpha=0.6,
                       edgecolors="none")
        ax.plot([-limit, limit], [limit, -limit], "k--", linewidth=1,
                label=r"$y=-x$ (pure symmetrising)")
        ax.plot([-limit, limit], [-limit, limit], color="0.5", linestyle=":", linewidth=1,
                label=r"$y=x$ (pure amplifying)")
        ax.set_xscale("symlog", linthresh=linear_threshold)
        ax.set_yscale("symlog", linthresh=linear_threshold)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
               xlabel=r"$b_i / \|b\|$", ylabel=r"$c_i / \|c\|$")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

        agreement = float(np.mean(np.sign(b) == -np.sign(c)))
        ax.set_title(
            f"{state}  ($\\|c\\| = {c_norm:.2e}$, $\\|b\\| = {b_norm:.2e}$)\n"
            f"$\\cos(c, b) = {float(np.dot(c_hat, b_hat)):+.4f}$, "
            f"sign$(b_i) = -$sign$(c_i)$ for {100 * agreement:.1f}% of parameters",
            fontsize=10,
        )
    axes[-1].legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    stem = output_dir / f"cb_scatter_{kind}_{run}"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_cb_magnitude(record: dict[str, Any], run: str, kind: str, output_dir: Path) -> None:
    """``|c_i|`` vs ``|b_i|`` on log-log axes, via the project's shared scatter helper."""
    slices = {name: slice(*bounds) for name, bounds in record["parameter_slices"].items()}
    initial, final = record["vectors"]["initial"], record["vectors"]["final"]
    plot_magnitude_vs_quantity(
        np.abs(initial[f"b_{kind}"]), np.abs(final[f"b_{kind}"]),
        np.abs(initial["c"]), np.abs(final["c"]), slices,
        quantity_label="$|c_i|$", title=f"{run_label(run)}, {UPDATE_LABEL[kind]}",
        output_stem=output_dir / f"cb_magnitude_{kind}_{run}",
        x_label=r"$|b_i|$ (update component)",
    )


# ----------------------------------- io -----------------------------------
def save_records(records: dict[str, dict[str, Any]], cfg: Config, output_dir: Path) -> None:
    history = {
        "config": asdict(cfg),
        "runs": {
            run: {k: v for k, v in record.items() if k != "vectors"}
            for run, record in records.items()
        },
    }
    (output_dir / "alignment_history.json").write_text(json.dumps(history, indent=2))
    for run, record in records.items():
        np.savez_compressed(
            output_dir / f"cb_vectors_{run}.npz",
            **{f"{state}/{name}": array
               for state, vectors in record["vectors"].items()
               for name, array in vectors.items()},
        )


def load_records(output_dir: Path) -> tuple[dict[str, dict[str, Any]], Config]:
    """Read back what a previous run wrote, so figures can be redrawn without recomputing."""
    history = json.loads((output_dir / "alignment_history.json").read_text())
    cfg = Config(**history["config"])
    records = history["runs"]
    for run, record in records.items():
        stored = np.load(output_dir / f"cb_vectors_{run}.npz")
        vectors: dict[str, dict[str, np.ndarray]] = {}
        for key in stored.files:
            state, name = key.split("/", 1)
            vectors.setdefault(state, {})[name] = stored[key]
        record["vectors"] = vectors
    return records, cfg


def summarise(records: dict[str, dict[str, Any]]) -> None:
    print("\nalignment summary (cosine < 0 means the update symmetrises)")
    for run, record in records.items():
        epochs = np.asarray(record["epochs"])
        early = epochs <= max(1, int(0.1 * epochs.max()))
        print(f"  {run_label(run)}:")
        for kind in UPDATE_KINDS:
            for prefix, name in (("cos_function", "cos(Delta, J b)"),
                                 ("cos_parameter", "cos(c, b)      "),
                                 ("cos_parameter_rowspace", "cos(c, Pi_T b) ")):
                values = np.asarray(record["curves"][f"{prefix}_{kind}"], dtype=float)
                print(
                    f"    {name} [{kind:>8s}]: first = {values[0]:+.4f}"
                    f"  mean(first 10% of epochs) = {np.nanmean(values[early]):+.4f}"
                    f"  mean(all) = {np.nanmean(values):+.4f}"
                    f"  final = {values[-1]:+.4f}"
                    f"  negative in {100 * np.nanmean(values < 0):5.1f}% of epochs"
                )
        defect = np.asarray(record["curves"]["defect_relative"], dtype=float)
        print(f"    ||Delta||/||f||: {defect[0]:.3e} -> {defect[-1]:.3e}"
              f"   (replay deviation from saved checkpoint: {record['replay_max_deviation']:.2e})")
        # The linearisation check: cos(Delta, J b) is only worth reading as "what the
        # step did to f" if it matches the cosine with the step's realised change.
        linearised = np.asarray(record["curves"]["cos_function_adam"], dtype=float)
        realised = np.asarray(record["curves"]["cos_realised_adam"], dtype=float)
        discrepancy = np.abs(linearised - realised)
        print(f"    |cos(Delta, J b) - cos(Delta, f(theta+b) - f(theta))| = "
              f"{np.nanmedian(discrepancy):.2e} (median), {np.nanmax(discrepancy):.2e} "
              f"(max, at epoch {record['epochs'][int(np.nanargmax(discrepancy))]}) "
              f"-- linearisation check")


def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="label=output_dir pairs pointing at aep_core_annulus_mlp runs")
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--probe-every", type=int)
    parser.add_argument("--tangent-svd-relative-cutoff", type=float)
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from a previous run's saved records")
    args = parser.parse_args()

    cfg = Config()
    apply_config_overrides(cfg, args.config)
    if args.runs:
        cfg.runs = dict(pair.split("=", 1) for pair in args.runs)
    for name in ("device", "output_dir", "probe_every", "tangent_svd_relative_cutoff"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    return cfg, args


def main() -> None:
    cfg, args = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        records, cfg = load_records(output_dir)
        print(f"replotting {', '.join(records)} from {output_dir}/alignment_history.json")
    else:
        device = select_device(cfg.device)
        points = build_probe_grid(cfg, device)
        print(f"probe grid: {points.shape[0]} points, |x| <= {cfg.probe_max_radius:g}; "
              f"probing every {cfg.probe_every} epoch(s) on {device}")
        records = {}
        for run, run_dir in cfg.runs.items():
            print(f"replaying {run} ({run_dir})")
            records[run] = replay_run(Path(run_dir), points, cfg, device)
        save_records(records, cfg, output_dir)

    plot_alignment_history(records, cfg, output_dir)
    for run, record in records.items():
        for kind in UPDATE_KINDS:
            plot_cb_scatter(record, run, kind, output_dir)
            plot_cb_magnitude(record, run, kind, output_dir)

    summarise(records)
    print(f"\nwrote {output_dir}/delta_gradient_alignment.png, "
          f"cb_scatter_{{{','.join(UPDATE_KINDS)}}}_{{{','.join(records)}}}.png, "
          f"cb_magnitude_*.png, alignment_history.json, cb_vectors_*.npz")


if __name__ == "__main__":
    main()
