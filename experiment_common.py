"""Shared scaffolding for ASRNN functional-sensitivity experiment scripts.

Shared functions for the different ASRNN experiments
"""

from __future__ import annotations

import copy
import csv
import json
import re
import signal
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def set_paper_style() -> None:
    """One shared, consistent look for every plot in this project (fonts, spines, grid).

    Called once at import time below, so every script that imports anything
    from this module picks it up automatically -- edit the values here to
    change font sizes/colours/etc. everywhere at once. Anything not covered
    by an rcParam (e.g. the categorical module colour palette) lives in
    ``module_colours`` just below.
    """
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    })


set_paper_style()


def _net_layer_index(name: str, prefix: str) -> int | None:
    """Extract the Sequential layer index from a raw parameter name sharing ``prefix``, or None."""
    if prefix:
        if not name.startswith(prefix):
            return None
        rest = name[len(prefix):]
    else:
        if name.startswith(("V_net.", "K_net.")):
            return None
        rest = name
    rest = rest.replace("inner.mlp.", "net.")
    match = re.match(r"net\.(\d+)\.(weight|weights|bias)$", rest)
    return int(match.group(1)) if match else None


def prettify_parameter_name(name: str, all_names: list[str] | None = None) -> str:
    """Turn a raw dotted parameter path (e.g. ``'V_net.net.2.weight'``) into a readable label.

    Every model in this project follows one of a few shapes: an ASRNN-style
    V_net/K_net split (``MLP.net`` is an ``nn.Sequential`` alternating
    ``Linear``/activation, so parameters sit at even indices 0, 2, 4, ...),
    a single unified dynamics net for direct_mlp (no V/K prefix), or the
    escnn-wrapped equivariant net (extra ``inner.mlp.`` wrapper layers to
    strip). This maps any of them to e.g. ``"layer 2 weights"``, falling
    back to the raw name if the pattern isn't recognised.

    ``MLP.__init__`` always appends one final output ``Linear`` after the
    hidden stack, so its index is the largest even index present for that
    net -- pass the full sibling list via ``all_names`` (e.g. the same
    ``modules`` list the caller is already iterating over) so this can spot
    that layer and label it ``"output weights"``/``"output biases"`` instead
    of e.g. ``"layer 4 weights"``, which otherwise reads as a 4th hidden
    layer on a network configured with only 3. Without ``all_names`` there's
    no way to know which index is last, so it falls back to plain
    "layer N" numbering.

    Deliberately drops the V_net/K_net distinction (the caller's context
    almost always makes it obvious, e.g. a plot that's already restricted to
    V_net-only parameters) -- if a single plot ever legitimately mixes V_net
    and K_net modules together, this will produce two identically-labelled
    legend entries distinguished only by colour; disambiguate at the call
    site in that case (e.g. by passing distinct labels directly) rather than
    reintroducing the prefix here.
    """
    prefix = ""
    if name.startswith("V_net."):
        prefix = "V_net."
    elif name.startswith("K_net."):
        prefix = "K_net."

    layer_index = _net_layer_index(name, prefix)
    if layer_index is None:
        return name
    kind = "biases" if name.endswith(".bias") else "weights"

    if all_names is not None:
        sibling_indices = [i for n in all_names if (i := _net_layer_index(n, prefix)) is not None]
        if sibling_indices and layer_index == max(sibling_indices):
            return f"output {kind}"

    layer_number = layer_index // 2 + 1
    return f"layer {layer_number} {kind}"


def module_colours(modules: list[str]) -> dict[str, Any]:
    """Consistent colour-per-module assignment shared by every per-parameter scatter plot."""
    cmap = plt.get_cmap("tab20" if len(modules) > 10 else "tab10")
    colours = cmap(np.linspace(0.0, 1.0, len(modules)))
    return dict(zip(modules, colours))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(name: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if name not in choices:
        raise ValueError(f"dtype must be one of {sorted(choices)}")
    return choices[name]


def apply_config_overrides(cfg: Any, config_path: Path | None) -> None:
    """Apply JSON field overrides from ``config_path`` onto a dataclass ``cfg`` in place."""
    if config_path is None:
        return
    overrides = json.loads(config_path.read_text())
    known = {f.name for f in fields(cfg)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    for key, value in overrides.items():
        setattr(cfg, key, value)


def parameter_layout(model: torch.nn.Module) -> tuple[list[str], dict[str, slice]]:
    names: list[str] = []
    slices: dict[str, slice] = {}
    offset = 0
    for name, parameter in model.named_parameters():
        names.extend(f"{name}[{i}]" for i in range(parameter.numel()))
        slices[name] = slice(offset, offset + parameter.numel())
        offset += parameter.numel()
    return names, slices


def run_training_loop(
    *,
    training_steps: int,
    checkpoint_steps: set[int] | list[int],
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    model: torch.nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    trajectory_window: int,
    residuals_fn: Callable[..., torch.Tensor],
    integrator: torch.nn.Module,
    l1_weight: float = 0.0,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
) -> tuple[dict[str, list[float]], dict[int, dict[str, torch.Tensor]]]:
    """Train with Adam or LBFGS, snapshotting the model's state at ``checkpoint_steps``.

    ``max_grad_norm``, if given, clips the total gradient norm (``torch.nn.utils.
    clip_grad_norm_``) right before every optimizer step. Off by default (``None``) to
    keep this a no-op for every script that doesn't ask for it. Added after a real
    instability found in the two-body script: Adam took one catastrophic step mid-training
    (trajectory loss 0.00009 -> 0.011 between two consecutive steps) and never fully
    recovered within the remaining budget -- occasional very large gradients are expected
    for a system whose true force genuinely blows up near collision (1/r**3), which a
    smooth, unconstrained MLP has no structural reason to avoid replicating locally during
    training. Clipping bounds the damage from any single such step without changing the
    overall optimization trajectory when gradients are already well-behaved.

    Both the double-well and Hénon-Heiles ASRNN helper modules expose the same
    ``(trajectories, params, instants)`` data layout and the same
    ``residuals(trajectories, params, instants, window, integrator)`` /
    ``VerletIntegrator`` signatures, so this loop is identical across systems.

    ``l1_weight > 0`` adds ``l1_weight * sum(|theta_i|)`` over every model
    parameter to the trajectory-fitting loss -- an explicit sparsity pressure,
    to test whether it makes an otherwise-unconstrained architecture's
    parameters align more with the equivariant directions (i.e. improve
    sensitivity equivariance) by cutting down the redundant/overparameterised
    capacity that has no reason to respect the symmetry. The pure trajectory
    loss (without the L1 term) is tracked separately in the returned history
    under ``"trajectory_loss"`` so the fit-quality/sparsity tradeoff can be
    read off directly, without the L1 term inflating what "training_loss"
    means relative to an ``l1_weight=0`` run.

    ``weight_decay`` is normally handled by passing it straight to the Adam
    optimizer (equivalent to adding ``0.5 * weight_decay * sum(theta_i**2)``
    to the loss -- that's exactly how ``torch.optim.Adam``'s own
    ``weight_decay`` is implemented, unlike AdamW's decoupled version).
    ``torch.optim.LBFGS`` has no ``weight_decay`` argument at all, so for
    that optimizer only, the same L2 penalty is added here directly into the
    closure's loss instead -- mathematically the same gradient contribution,
    and it additionally gets folded into the loss value LBFGS's strong-Wolfe
    line search sees, which is the correct/expected behaviour for a
    quasi-Newton method. When ``optimizer_name == "adam"`` this path is
    skipped so the penalty isn't applied twice (once here, once inside
    Adam's own step).

    A single Ctrl-C (SIGINT) during training does *not* abort the process: it finishes the
    current step, force-saves a checkpoint at that step (even if it wasn't one of the
    originally requested ``checkpoint_steps``), and returns normally so the caller proceeds
    straight to checkpoint analysis/plotting on whatever was trained so far -- useful for a
    long run whose loss has visibly plateaued, without losing the ability to inspect it. A
    second Ctrl-C while the first is still pending aborts immediately (restores Python's
    default SIGINT behaviour and re-raises), for when you really do just want to kill it.
    """
    checkpoint_steps = set(int(s) for s in checkpoint_steps)
    train_trajectories, train_params, train_instants = train_data
    val_trajectories, val_params, val_instants = val_data
    history: dict[str, list[float]] = {
        "step": [], "training_loss": [], "trajectory_loss": [], "validation_loss": [],
    }
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {0: copy.deepcopy(model.state_dict())}

    def l1_penalty() -> torch.Tensor:
        return sum(p.abs().sum() for p in model.parameters())

    def l2_penalty() -> torch.Tensor:
        return sum(p.pow(2).sum() for p in model.parameters())

    apply_manual_weight_decay = weight_decay > 0 and optimizer_name.lower() == "lbfgs"
    trajectory_loss_value = [0.0]

    progress = tqdm(range(1, training_steps + 1), desc="training", unit="step", dynamic_ncols=True)

    interrupt_count = [0]
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        interrupt_count[0] += 1
        if interrupt_count[0] == 1:
            progress.write(
                "\nKeyboardInterrupt: finishing the current step, then stopping training "
                "early to run checkpoint analysis/plotting on what's been trained so far. "
                "Press Ctrl-C again to abort immediately instead."
            )
        else:
            signal.signal(signal.SIGINT, previous_sigint_handler)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for step in progress:
            model.train()

            def closure():
                optimizer.zero_grad()
                trajectory_loss = residuals_fn(
                    train_trajectories, train_params, train_instants, trajectory_window, integrator
                )
                trajectory_loss_value[0] = float(trajectory_loss.detach().cpu())
                loss = trajectory_loss
                if l1_weight > 0:
                    loss = loss + l1_weight * l1_penalty()
                if apply_manual_weight_decay:
                    loss = loss + 0.5 * weight_decay * l2_penalty()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                return loss

            if optimizer_name.lower() == "lbfgs":
                training_loss = float(optimizer.step(closure).detach().cpu())
            else:
                loss = closure()
                optimizer.step()
                training_loss = float(loss.detach().cpu())

            model.eval()
            validation_loss = float(
                residuals_fn(
                    val_trajectories, val_params, val_instants, trajectory_window, integrator
                ).detach().cpu()
            )
            history["step"].append(step)
            history["training_loss"].append(training_loss)
            history["trajectory_loss"].append(trajectory_loss_value[0])
            history["validation_loss"].append(validation_loss)
            progress.set_postfix(train=f"{trajectory_loss_value[0]:.3e}", val=f"{validation_loss:.3e}")
            if step in checkpoint_steps:
                checkpoint_states[step] = copy.deepcopy(model.state_dict())
                progress.write(
                    f"step {step:5d} | trajectory {trajectory_loss_value[0]:.5e} | "
                    f"validation {validation_loss:.5e}"
                )

            if interrupt_count[0] > 0:
                if step not in checkpoint_states:
                    checkpoint_states[step] = copy.deepcopy(model.state_dict())
                progress.write(f"Stopped early at step {step} (of {training_steps} requested).")
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    return history, checkpoint_states


def write_result_json(result: dict[str, Any], output_dir: Path, step: int) -> None:
    (output_dir / f"sensitivity_step_{step:06d}.json").write_text(json.dumps(result, indent=2))


def write_csv_rows(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    """Plot optimisation and generalisation diagnostics.

    Uses ``trajectory_loss`` (the pure fit term, comparable to validation)
    where available, falling back to ``training_loss`` for older history
    dicts without it. When an L1 penalty was active, ``training_loss``
    (the actual optimised objective, fit + penalty) is also shown as a
    lighter dashed line so the two remain visually distinguishable.
    """
    steps = np.asarray(history["step"])
    training = np.asarray(history.get("trajectory_loss", history["training_loss"]))
    total_objective = np.asarray(history["training_loss"])
    validation = np.asarray(history["validation_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps, training, label="training (trajectory fit)", linewidth=1.6)
    if not np.allclose(training, total_objective):
        axes[0].plot(
            steps, total_objective, label="training (fit + L1 penalty)",
            linewidth=1.2, linestyle="--", color="tab:orange", alpha=0.7,
        )
    axes[0].plot(steps, validation, label="validation", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="training step", ylabel="trajectory MSE")
    axes[0].legend()

    gap = validation - training
    axes[1].plot(steps, gap, color="tab:purple", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="training step", ylabel="validation $-$ training loss")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output_dir / "training_diagnostics.png", dpi=200)
    fig.savefig(output_dir / "training_diagnostics.pdf")
    plt.close(fig)


def plot_ei_initial_vs_final(
    ei_initial: np.ndarray,
    ei_final: np.ndarray,
    parameter_slices: dict[str, slice],
    *,
    title: str,  # accepted for call-site compatibility but not rendered -- paper figures use a caption instead.
    output_stem: Path,
    quantity_label: str = "$E_i$",
    log_scale: bool = False,
) -> None:
    """Scatter of a per-parameter quantity (E_i by default) at random init vs. the final checkpoint.

    One point per parameter i, coloured by which named module (layer) it
    belongs to. Points on the y=x line are unchanged by training; points
    below it improved; points above it got worse. This is the per-parameter
    complement to the module-mean bar chart -- it shows the full spread
    within each layer, including outliers (e.g. zero-initialised biases that
    start exactly equivariant and drift away from it).

    ``log_scale=True`` is for quantities (like attribution coefficients c_i)
    that span many orders of magnitude, including exact zeros for pruned
    parameters -- those are floored to a small epsilon so they remain
    visible rather than vanishing at the origin.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    modules = list(parameter_slices.keys())
    colours = module_colours(modules)

    if log_scale:
        ei_initial = np.maximum(ei_initial, 1e-12)
        ei_final = np.maximum(ei_final, 1e-12)

    for name in modules:
        sl = parameter_slices[name]
        ax.scatter(
            ei_initial[sl], ei_final[sl], s=14, color=colours[name],
            label=prettify_parameter_name(name, modules), alpha=0.75, edgecolors="none",
        )

    if log_scale:
        lower = float(min(ei_initial.min(), ei_final.min())) / 1.5
        upper = float(max(ei_initial.max(), ei_final.max())) * 1.5
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="$y=x$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(lower, upper), ylim=(lower, upper))
    else:
        upper = float(max(ei_initial.max(), ei_final.max(), 1e-6)) * 1.05
        ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="$y=x$")
        ax.set(xlim=(0, upper), ylim=(0, upper))
    ax.set(xlabel=f"{quantity_label} at random init", ylabel=f"{quantity_label} at final checkpoint")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, which="both" if log_scale else "major")
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_vs_quantity(
    magnitude_initial: np.ndarray,
    magnitude_final: np.ndarray,
    quantity_initial: np.ndarray,
    quantity_final: np.ndarray,
    parameter_slices: dict[str, slice],
    *,
    quantity_label: str,
    title: str,  # accepted for call-site compatibility but not rendered -- paper figures use a caption instead.
    output_stem: Path,
    x_label: str = r"$|\theta_i|$ (parameter magnitude)",
) -> None:
    """Scatter of a per-parameter x quantity (|theta_i| by default) vs. a y quantity (S_i, E_i, or c_i).

    Two panels -- random init (left) and the final checkpoint (right) --
    coloured by module, using the same colour/label scheme as
    ``plot_ei_initial_vs_final``. Originally purpose-built to check whether
    an apparently small E_i (or S_i) for particular parameters is a genuine
    alignment effect, or simply an artefact of those parameters having been
    driven to near-zero magnitude (e.g. by an L1 penalty); the same
    scatter/floor treatment is equally useful for any other pair of
    per-parameter quantities (e.g. attribution c_i vs. equivariance E_i, via
    ``x_label``), since both can be exactly zero for pruned parameters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    modules = list(parameter_slices.keys())
    colours = module_colours(modules)

    for ax, x_values, quantity, subtitle in (
        (axes[0], magnitude_initial, quantity_initial, "random init"),
        (axes[1], magnitude_final, quantity_final, "trained"),
    ):
        for name in modules:
            sl = parameter_slices[name]
            ax.scatter(
                np.maximum(x_values[sl], 1e-12), np.maximum(quantity[sl], 1e-12),
                s=14, color=colours[name], label=prettify_parameter_name(name, modules),
                alpha=0.75, edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel=x_label, ylabel=quantity_label)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
