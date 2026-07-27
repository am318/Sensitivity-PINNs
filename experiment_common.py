"""Shared scaffolding for ASRNN functional-sensitivity experiment scripts.

Shared functions for the different ASRNN experiments
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


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
    l1_regularization: float,
) -> tuple[dict[str, list[float]], dict[int, dict[str, torch.Tensor]]]:
    """Train with Adam or LBFGS, snapshotting the model's state at ``checkpoint_steps``.

    Both the double-well and Hénon-Heiles ASRNN helper modules expose the same
    ``(trajectories, params, instants)`` data layout and the same
    ``residuals(trajectories, params, instants, window, integrator)`` /
    ``VerletIntegrator`` signatures, so this loop is identical across systems.
    """
    checkpoint_steps = set(int(s) for s in checkpoint_steps)
    train_trajectories, train_params, train_instants = train_data
    val_trajectories, val_params, val_instants = val_data
    history: dict[str, list[float]] = {"step": [], "training_loss": [], "validation_loss": []}
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {0: copy.deepcopy(model.state_dict())}

    progress = tqdm(range(1, training_steps + 1), desc="training", unit="step", dynamic_ncols=True)
    for step in progress:
        model.train()

        def closure():
            optimizer.zero_grad()

            if l1_regularization == 0.0:
                loss = residuals_fn(
                    train_trajectories, train_params, train_instants, trajectory_window, integrator
                )
                loss.backward()
                return loss
            else:
                l1_penalty = l1_regularization * sum(
                    p.abs().sum() for p in model.parameters()
                )
                loss = residuals_fn(
                                train_trajectories, train_params, train_instants, trajectory_window, integrator
                            )
                total_loss = loss + l1_penalty
                total_loss.backward()
                return total_loss

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
        history["validation_loss"].append(validation_loss)
        progress.set_postfix(train=f"{training_loss:.3e}", val=f"{validation_loss:.3e}")
        if step in checkpoint_steps:
            checkpoint_states[step] = copy.deepcopy(model.state_dict())
            progress.write(
                f"step {step:5d} | train {training_loss:.5e} | "
                f"validation {validation_loss:.5e}"
            )

    return history, checkpoint_states


def write_result_json(result: dict[str, Any], output_dir: Path, step: int) -> None:
    (output_dir / f"sensitivity_step_{step:06d}.json").write_text(json.dumps(result, indent=2))


def write_csv_rows(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(history: dict[str, list[float]], output_dir: Path) -> None:
    """Plot optimisation and generalisation diagnostics."""
    steps = np.asarray(history["step"])
    training = np.asarray(history["training_loss"])
    validation = np.asarray(history["validation_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(steps, training, label="training", linewidth=1.6)
    axes[0].plot(steps, validation, label="validation", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set(title="ASRNN training history", xlabel="training step", ylabel="trajectory MSE")
    axes[0].legend()

    gap = validation - training
    axes[1].plot(steps, gap, color="tab:purple", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Validation minus training loss",
        xlabel="training step",
        ylabel="generalisation gap",
    )
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
    title: str,
    output_stem: Path,
) -> None:
    """Scatter of per-parameter E_i (sec. 0.6) at random init vs. the final checkpoint.

    One point per parameter i, coloured by which named module (layer) it
    belongs to. Points on the y=x line are unchanged by training; points
    below it improved; points above it got *less* equivariant during
    training. This is the per-parameter complement to the module-mean bar
    chart -- it shows the full spread within each layer, including outliers
    (e.g. zero-initialised biases that start exactly equivariant and drift
    away from it).
    """
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    modules = list(parameter_slices.keys())
    cmap = plt.get_cmap("tab20" if len(modules) > 10 else "tab10")
    colours = cmap(np.linspace(0.0, 1.0, len(modules)))

    for colour, name in zip(colours, modules):
        sl = parameter_slices[name]
        ax.scatter(
            ei_initial[sl], ei_final[sl], s=14, color=colour, label=name,
            alpha=0.75, edgecolors="none",
        )

    upper = float(max(ei_initial.max(), ei_final.max(), 1e-6)) * 1.05
    ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="$y=x$ (unchanged)")
    ax.set(
        xlim=(0, upper), ylim=(0, upper),
        xlabel="$E_i$ at random init", ylabel="$E_i$ at final checkpoint",
        title=title,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
