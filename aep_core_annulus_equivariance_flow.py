"""Visual c/b "walk" experiment for the annulus classifier: show the trained MLP's
non-invariant part as we ITERATIVELY step its weights along the g = X_rot f attribution
(c) and along the defect-reducing -(delta = f - P f) attribution (-b), then read the
result off a single orbit as a function of angle.

Important: c and b are only exact to first order at the CURRENT weights, so how the step
is taken is the experiment, and ``--mode`` selects it:

``--mode resolved`` (default, writes outputs/aep_core_annulus_equivariance_flow)
    Take one small step along the current c (or -b), then RECOMPUTE it at the new weights
    before taking the next step. This follows the (curving) integral curve of the
    c-vector-field / b-vector-field through parameter space.

``--mode fixed`` (writes outputs/aep_core_annulus_equivariance_flow_fixed)
    Calculate the direction ONCE at t = 0 and step along that same vector repeatedly. The
    direction never changes, so k steps of dt z_0 are exactly one step of k dt z_0 and the
    trajectory is the straight line theta_0 + t z_0 -- the linear extrapolation the
    re-solved walk is a correction to. Both modes produce the identical figure set on the
    identical time grid, against the identical exact-flow predictions, so the two output
    directories can be read panel for panel.

Either way the walk is scored against the SAME exact solutions, and panel 6 of Figure 4
carries the diagnostic that separates the two: solid is how well the tangent space at the
current weights can represent the target, dotted is how much of that the frozen z_0 still
achieves. They start equal by construction and part company as the walk moves.

The step is NOT normalised to a fixed L2 length in parameter space, unlike
``run_potential_walk_experiment.py``. Here the raw solution z of J z = g is used with a
fixed time step dt, which makes the walk parameter a genuine FLOW TIME and buys two
exact solutions with no free parameters to compare against:

  - along -b:  df/dt = -(I - P) f  leaves P f fixed and contracts the rest, so
    f_t = P(f_0) + e^{-t} delta_0. The defect decays EXPONENTIALLY, reaching e^{-1}
    (63% removed) at t = 1 -- not 0, which is where the single linear step aims.
  - along c:   df/dt = X_rot f  integrates to f_t = f_0 . R_t exactly, so t is an angle
    in RADIANS and t = 1 is a 57.3-degree rotation of the decision boundary. ||(I-P)f||
    must stay FLAT: transport along the orbit moves the model without symmetrising it.

Because any z with J z = g gives the same df/dt, these two identities are insensitive to
which minimum-norm solution ``tangent_projection`` picks; only the parameter-space path
length changes. Normalising z to unit length would destroy both.

Six figures, all laid out as 2 rows (c-walk on top, -b-walk on bottom) x one column per
condition (wedge/full x initial/trained), except Figure 6 which transposes for space:

  - angular_deviation_flow.png (the primary, most legible figure): d(theta) = f(r,theta)
    - <f(r,.)> read off at a fixed radius as a function of angle, one curve per walk step
    (colour = flow time, dotted = the exact-flow prediction). Subtracting the orbit
    average IS (I - P) f restricted to that orbit, so this is a slice of the same delta
    the walk is defined by, and an exactly invariant model gives the flat line 0. A
    rotation shows up as a clean phase shift of the whole curve; a change in how
    invariant the network is shows up as a clean amplitude change with no phase shift.
  - angular_deviation_heatmap.png: the same profiles as a surface over (angle, flow
    time), using every step rather than the six drawn above -- -b fades in place, c
    shears into diagonal bands of slope 1.
  - angular_deviation_amplitude.png: RMS amplitude of that profile against flow time
    (normalised by its value at t = 0, so all conditions share one axis and the
    prediction is a single curve), and the residual from each mechanism's prediction.
  - equivariance_flow_<kind>.png: the scalar diagnostics along the walk -- defect, task
    metrics, distance to the exact trajectory and to the endpoint target, and whether
    the direction is still in the tangent space after the walk has moved.
  - flow_vs_fixed_step_<kind>.png: this run's walk against the other mode's, on the same
    time grid -- what re-solving buys. In resolved mode the partner straight line is
    computed here (it needs no Jacobians); in fixed mode the partner is read back from the
    resolved run's summary via --compare-summary.
  - decision_boundary_flow_<kind>.png: the decision boundary itself along the walk, in
    the same panel style as decision_boundary_before_after.png.

Usage:
    python3 aep_core_annulus_equivariance_flow.py outputs/aep_core_annulus_mlp outputs/aep_core_annulus_mlp_full
    python3 aep_core_annulus_equivariance_flow.py --mode fixed outputs/aep_core_annulus_mlp outputs/aep_core_annulus_mlp_full
    python3 aep_core_annulus_equivariance_flow.py outputs/aep_core_annulus_mlp --kinds b --n-steps 400
    python3 aep_core_annulus_equivariance_flow.py --replot outputs/aep_core_annulus_mlp
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import aep_core_annulus_mlp as m
from aep_core_annulus_tangent import (
    equivariance_defect,
    group_average,
    orbit_direction,
    sensitivity_jacobian,
)
from sensitivity_tools import tangent_projection

# Rows of every figure. "c" follows +z (a velocity to be followed); "b" follows -z (a
# residual to be removed). Fixing the sign here is what lets both walks run on a
# positive time axis, so the two mechanisms are directly comparable panel to panel.
KINDS = [
    ("c", +1.0, r"along $c$ ($g = X_{\rm rot}f$; rotates the shape)"),
    ("b", -1.0, r"along $-b$ ($\Delta = f - P(f)$; shrinks the defect)"),
]
KIND_TARGET = {"c": r"$X_{\rm rot}f$", "b": r"$\Delta = f - P(f)$"}
KIND_ENDPOINT = {"c": r"$f_0 \circ R_{t_{\max}}$", "b": r"$P(f_0)$"}
KIND_EXACT = {"c": r"$f_0 \circ R_t$", "b": r"$P(f_0) + e^{-t}\Delta_0$"}
KIND_TIME = {"c": r"flow time $t$ = rotation angle (rad)", "b": r"flow time $t$ along $-b$"}

# --mode resolved re-solves the direction at every step (the integral curve); --mode fixed
# calculates it ONCE at t = 0 and steps along that same vector repeatedly (the straight
# line). Everything else -- figures, targets, exact-flow predictions -- is identical, which
# is the point: the two output directories are directly comparable panel for panel.
MODE_LABEL = {"resolved": "re-solved at every step", "fixed": "one direction, calculated once at $t=0$"}
MODE_SUFFIX = {"resolved": "", "fixed": "_fixed"}
COMPARE_LABEL = {"resolved": "fixed $z_0$", "fixed": "re-solved"}


def flat_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_parameters(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Write a flat vector back in ``model.parameters()`` order -- the same order
    ``sensitivity_tools.parameter_gradient_row`` builds the Jacobian columns in, so index
    i of z really is the parameter column i of J differentiates with respect to."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n
    assert offset == flat.numel(), "flat vector length does not match the model"


def probe_grid(extent: float, n_per_axis: int, max_radius: float, device, dtype) -> torch.Tensor:
    """The square probe grid with the far corners dropped, so the solve is not dominated
    by the region beyond every data point where the model purely extrapolates."""
    axis = torch.linspace(-extent, extent, n_per_axis, device=device, dtype=dtype)
    mesh_x, mesh_y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)
    return points[points.norm(dim=1) <= max_radius]


def circle_points(radius: float, n_angles: int, device, dtype) -> tuple[np.ndarray, torch.Tensor]:
    """Angles and points of the single orbit the deviation profile is read off. The
    endpoint is excluded so the angles are a uniform quadrature of the circle --
    including both 0 and 2*pi would double-count one point in the orbit average."""
    angles = np.linspace(0.0, 2 * math.pi, n_angles, endpoint=False)
    points = torch.tensor(
        np.c_[radius * np.cos(angles), radius * np.sin(angles)], device=device, dtype=dtype
    )
    return angles, points


@torch.no_grad()
def angular_deviation(model: torch.nn.Module, circle: torch.Tensor) -> torch.Tensor:
    values = model(circle).squeeze(-1)
    return values - values.mean()


def target_field(model, points: torch.Tensor, kind: str, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(f, g)`` on the probe grid: the orbit velocity for c, the defect for b."""
    if kind == "c":
        return orbit_direction(model, points)
    return equivariance_defect(model, points, order)


def solve_direction(model, points: torch.Tensor, kind: str, order: int, cutoff: float):
    """Recompute c (kind='c') or b (kind='b') at the model's CURRENT weights. The sign that
    turns b into the defect-reducing -b is applied by the caller, via KINDS.

    J and g come back too, so a caller can ask how well some OTHER direction -- notably the
    one frozen at t = 0 -- still realises the target here, which is the whole question the
    ``--mode fixed`` walk is built to answer."""
    _, g = target_field(model, points, kind, order)
    jacobian = sensitivity_jacobian(model, points)
    residual, angle, vec, _, _ = tangent_projection(jacobian, g, cutoff)
    return vec.to(device=points.device, dtype=points.dtype), residual, angle, jacobian, g


def realisation_error(jacobian: torch.Tensor, g: torch.Tensor, z: torch.Tensor) -> float:
    """``||J z - g|| / ||g||``: how well direction z realises the target g AT THIS POINT.

    Scored against a re-solved z this is just the tangent-space projection error; scored
    against the frozen z_0 it is the quantity that decides whether stepping repeatedly along
    one initial direction can work at all."""
    predicted = jacobian.to(dtype=g.dtype) @ z.to(dtype=g.dtype)
    return float(torch.linalg.vector_norm(predicted - g) / torch.linalg.vector_norm(g).clamp_min(1e-15))


@torch.no_grad()
def rotated_values(model, points: torch.Tensor, angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=points.dtype, device=points.device)
    return model(points @ rotation.T).squeeze(-1)


def exact_values(kind: str, t: float, base_model, points, base_projection, base_defect) -> torch.Tensor:
    """Where the exact flow is at time t: ``f_0 . R_t`` for c, ``P(f_0) + e^{-t}delta_0``
    for -b. Both solve the corresponding function-space ODE exactly, so any gap is
    attributable to the tangent space or the Euler discretisation, not to the target."""
    if kind == "c":
        return rotated_values(base_model, points, t)
    return base_projection + math.exp(-t) * base_defect


def endpoint_target(kind: str, base_model, points, base_projection, time_max: float) -> torch.Tensor:
    """The fixed function the whole walk aims at -- NOT the same as ``exact_values`` at
    ``t = time_max`` for -b: the walk aims at the exactly-symmetrised ``P(f_0)``, while the
    exact flow only gets ``1 - e^{-1} = 63%`` of the way there. Keeping the two separate is
    what makes that shortfall visible instead of hiding it in a self-consistent zero."""
    if kind == "c":
        return rotated_values(base_model, points, time_max)
    return base_projection


def exact_profile(kind: str, profile_0: np.ndarray, angles: np.ndarray, t: float) -> np.ndarray:
    """The same two predictions read on the orbit. c rigidly rotates the profile,
    ``d_t(theta) = d_0(theta + t)`` -- the sign the flow actually produces, since
    ``f_t = f_0 . R_t`` at ``x(theta)`` is ``f_0(x(theta + t))``, so a feature at
    ``theta_0`` in ``d_0`` reappears at ``theta_0 - t``. -b contracts it in place."""
    if kind == "c":
        return np.interp(angles + t, angles, profile_0, period=2 * math.pi)
    return math.exp(-t) * profile_0


@torch.no_grad()
def task_metrics(model, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    logits = model(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    return float(loss), float(((logits > 0).to(y.dtype) == y).to(torch.float64).mean())


def walk(model, base_model, points, circle, x, y, kind: str, sign: float, cfg, args,
         device, dtype) -> tuple[dict, dict[float, torch.Tensor], np.ndarray]:
    """Iteratively step-then-recompute, following the vector field's actual integral
    curve. Returns the per-stop scalar record, the parameter vectors at the snapshot
    times, and the angular profile at EVERY stop.

    Unlike ``run_potential_walk_experiment.py`` this does not keep every intermediate
    state_dict: 200 stops x 19k parameters x 4 conditions x 2 kinds is a lot to hold, and
    only the handful of snapshot times are ever redrawn."""
    dt = args.time_max / args.n_steps
    base = flat_parameters(model)
    base_norm = float(torch.linalg.vector_norm(base))
    with torch.no_grad():
        base_values = model(points).squeeze(-1).clone()
    base_projection = group_average(model, points, args.group_order)
    base_defect = base_values - base_projection
    value_norm = float(torch.linalg.vector_norm(base_values).clamp_min(1e-15))
    endpoint = endpoint_target(kind, base_model, points, base_projection, args.time_max)

    wanted = {min(max(int(round(t / dt)), 0), args.n_steps): round(t, 9) for t in args.snapshot_times}
    snapshots: dict[float, torch.Tensor] = {}
    profiles: list[np.ndarray] = []
    record: dict[str, list[float]] = {k: [] for k in (
        "t", "defect_norm", "defect_prediction", "relative_defect", "exact_distance",
        "exact_baseline", "endpoint_distance", "endpoint_prediction", "solve_residual",
        "frozen_residual", "solve_angle", "loss", "accuracy", "invariance_error",
        "parameter_distance", "path_length",
    )}

    z_start: torch.Tensor | None = None
    path_length = 0.0
    x_numpy = x.cpu().numpy()
    for step in range(args.n_steps + 1):
        t = step * dt
        params = flat_parameters(model)
        values, defect = equivariance_defect(model, points, args.group_order)
        exact = exact_values(kind, t, base_model, points, base_projection, base_defect)
        loss, accuracy = task_metrics(model, x, y)
        # Same rotations at every stop: E(T) draws its angles at random, and resampling
        # them per step would add noise on top of the effect being measured.
        torch.manual_seed(args.seed)
        _, invariance_max = m.invariance_error(model, x_numpy, cfg, device, dtype)

        record["t"].append(t)
        record["defect_norm"].append(float(torch.linalg.vector_norm(defect)))
        record["defect_prediction"].append(
            float(torch.linalg.vector_norm(base_defect)) * (1.0 if kind == "c" else math.exp(-t))
        )
        record["relative_defect"].append(float(
            torch.linalg.vector_norm(defect) / torch.linalg.vector_norm(values).clamp_min(1e-15)
        ))
        record["exact_distance"].append(float(torch.linalg.vector_norm(values - exact)) / value_norm)
        record["exact_baseline"].append(float(torch.linalg.vector_norm(base_values - exact)) / value_norm)
        record["endpoint_distance"].append(float(torch.linalg.vector_norm(values - endpoint)) / value_norm)
        record["endpoint_prediction"].append(float(torch.linalg.vector_norm(exact - endpoint)) / value_norm)
        record["loss"].append(loss)
        record["accuracy"].append(accuracy)
        record["invariance_error"].append(invariance_max)
        record["parameter_distance"].append(float(torch.linalg.vector_norm(params - base)) / base_norm)
        record["path_length"].append(path_length / base_norm)
        # One forward pass, so recorded at every stop: a dense time axis is what makes the
        # drift/decay legible as a surface rather than as six curves.
        profiles.append(angular_deviation(model, circle).cpu().numpy())

        z, residual, angle, jacobian, g = solve_direction(
            model, points, kind, args.group_order, args.cutoff
        )
        if z_start is None:
            z_start = z.clone()
        # Both modes score the FROZEN z_0 here, not just the fixed one: on the re-solved
        # path it answers "how stale would one initial direction have gone by now?", which
        # is the same question from the other side. At t = 0 the two residuals coincide.
        record["solve_residual"].append(residual)
        record["frozen_residual"].append(realisation_error(jacobian, g, z_start))
        record["solve_angle"].append(angle)
        if step in wanted:
            snapshots[wanted[step]] = params.clone()
        if step == args.n_steps:
            break
        # --mode fixed: step repeatedly along the direction calculated ONCE at t = 0, so the
        # trajectory is the straight line theta_0 + t z_0 and k steps of dt z_0 are exactly
        # one step of k dt z_0. --mode resolved: follow the re-solved integral curve.
        step_z = z_start if args.mode == "fixed" else z
        path_length += dt * float(torch.linalg.vector_norm(step_z))
        set_flat_parameters(model, params + sign * dt * step_z)

    set_flat_parameters(model, base)
    return record, snapshots, np.stack(profiles)


def fixed_walk(model, base_model, points, x, y, kind: str, sign: float, args) -> dict:
    """The straight line ``theta_0 + s t z_0`` on the same time grid: z_0 reused at every
    stop, which is exactly the approximation the walk above is testing. No Jacobians."""
    dt = args.time_max / args.n_steps
    base = flat_parameters(model)
    with torch.no_grad():
        base_values = model(points).squeeze(-1).clone()
    base_projection = group_average(model, points, args.group_order)
    base_defect = base_values - base_projection
    value_norm = float(torch.linalg.vector_norm(base_values).clamp_min(1e-15))
    endpoint = endpoint_target(kind, base_model, points, base_projection, args.time_max)
    z0, _, _, _, _ = solve_direction(model, points, kind, args.group_order, args.cutoff)

    record: dict[str, list[float]] = {k: [] for k in (
        "t", "defect_norm", "relative_defect", "exact_distance", "endpoint_distance", "loss", "accuracy",
    )}
    for step in range(args.n_steps + 1):
        t = step * dt
        set_flat_parameters(model, base + sign * t * z0)
        values, defect = equivariance_defect(model, points, args.group_order)
        exact = exact_values(kind, t, base_model, points, base_projection, base_defect)
        loss, accuracy = task_metrics(model, x, y)
        record["t"].append(t)
        record["defect_norm"].append(float(torch.linalg.vector_norm(defect)))
        record["relative_defect"].append(float(
            torch.linalg.vector_norm(defect) / torch.linalg.vector_norm(values).clamp_min(1e-15)
        ))
        record["exact_distance"].append(float(torch.linalg.vector_norm(values - exact)) / value_norm)
        record["endpoint_distance"].append(float(torch.linalg.vector_norm(values - endpoint)) / value_norm)
        record["loss"].append(loss)
        record["accuracy"].append(accuracy)
    set_flat_parameters(model, base)
    return record


def condition_width(index: int, base: float = 2.4) -> float:
    """Earlier-drawn conditions get fatter lines. The two runs share an initialisation by
    construction, so their "initial" curves coincide EXACTLY -- without this the one drawn
    first hides underneath the other and the figure looks like it is missing a condition."""
    return base - 0.3 * index


def log_floor(axis, series_list) -> None:
    """Clip a log axis to where the curves live. Several of these quantities are
    identically 0 at t = 0, which otherwise stretches the scale over 16 empty decades."""
    positive = [v for series in series_list for v in series if v > 0.0]
    if positive:
        axis.set_ylim(bottom=0.5 * min(positive))


def figure_amplitude(profiles, angles, conditions, args, cmap, output_dir: Path) -> None:
    """RMS amplitude of the orbit profile and its residual from each exact prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True, squeeze=False)
    for col_idx, (kind, _, label) in enumerate(KINDS):
        for label_idx, name in enumerate(conditions):
            profile = profiles[kind][name]
            times = np.linspace(0.0, args.time_max, profile.shape[0])
            amplitude = np.sqrt((profile ** 2).mean(axis=1))
            reference = max(amplitude[0], 1e-300)
            colour = cmap(label_idx / max(len(conditions) - 1, 1))
            width = condition_width(label_idx)
            axes[0, col_idx].plot(times, amplitude / reference, color=colour, linewidth=width, label=name)
            residual = [
                np.sqrt(((profile[i] - exact_profile(kind, profile[0], angles, t)) ** 2).mean()) / reference
                for i, t in enumerate(times)
            ]
            axes[1, col_idx].plot(times, residual, color=colour, linewidth=width, label=name)
        grid = np.linspace(0.0, args.time_max, 200)
        predicted = np.ones_like(grid) if kind == "c" else np.exp(-grid)
        axes[0, col_idx].plot(grid, predicted, color="0.2", linewidth=1.2, linestyle=":",
                              label="exact flow: 1" if kind == "c" else r"exact flow: $e^{-t}$")
        axes[0, col_idx].set(xlabel=KIND_TIME[kind], yscale="log",
                             ylabel=r"RMS$_\theta(d_t)$ / RMS$_\theta(d_0)$",
                             title=f"amplitude of the defect on the orbit, {label}")
        axes[1, col_idx].set(xlabel=KIND_TIME[kind], yscale="log",
                             ylabel=r"RMS$_\theta(d_t - d_t^{\rm pred})$ / RMS$_\theta(d_0)$",
                             title=(r"is it a rigid rotation?  $d_t(\theta) = d_0(\theta + t)$" if kind == "c"
                                    else r"does it contract in place?  $d_t = e^{-t}d_0$"))
        log_floor(axes[1, col_idx], [[
            np.sqrt(((profiles[kind][n][i] - exact_profile(kind, profiles[kind][n][0], angles, t)) ** 2).mean())
            / max(np.sqrt((profiles[kind][n][0] ** 2).mean()), 1e-300)
            for i, t in enumerate(np.linspace(0.0, args.time_max, profiles[kind][n].shape[0]))
        ] for n in conditions])
        for row in (0, 1):
            axes[row, col_idx].grid(alpha=0.25)
            axes[row, col_idx].legend(fontsize=8)
    fig.suptitle("The two mechanisms on one orbit: rotation of the profile (left) vs "
                 f"contraction of it (right) -- {MODE_LABEL[args.mode]}")
    path = output_dir / "angular_deviation_amplitude.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def figure_diagnostics(results, kind: str, label: str, mode: str, cmap, output_dir: Path) -> None:
    """The scalar diagnostics along one walk, one panel per quantity."""
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.0), constrained_layout=True)
    axes = axes.ravel()
    names = list(results)
    for idx, name in enumerate(names):
        record = results[name]["walk"]
        colour = cmap(idx / max(len(names) - 1, 1))
        times = np.asarray(record["t"])
        for ax, key in zip(axes, ("defect_norm", "relative_defect", "invariance_error",
                                  "exact_distance", "endpoint_distance", "solve_residual",
                                  "loss", "accuracy", "parameter_distance")):
            ax.plot(times, record[key], color=colour, linewidth=condition_width(idx), label=name)
        axes[0].plot(times, record["defect_prediction"], color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
        axes[3].plot(times, record["exact_baseline"], color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
        axes[4].plot(times, record["endpoint_prediction"], color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
        axes[8].plot(times, record["path_length"], color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
        axes[5].plot(times, record["frozen_residual"], color=colour, linewidth=1.0,
                     linestyle=":", alpha=0.9)
        if kind == "b":
            # Star at t = 1: what the EXACT projection P(f_0) scores. The gap to it is the
            # part of the symmetrisation the walk has not done (the flow only removes 63%).
            axes[6].scatter([1.0], [results[name]["projected_loss"]], marker="*", s=140,
                            color=colour, edgecolors="black", linewidths=0.5, zorder=10)
            axes[7].scatter([1.0], [results[name]["projected_accuracy"]], marker="*", s=140,
                            color=colour, edgecolors="black", linewidths=0.5, zorder=10)

    defect_note = (r"dotted: $\|\Delta_0\|$ (transport leaves it fixed)" if kind == "c"
                   else r"dotted: $e^{-t}\|\Delta_0\|$ (exact flow)")
    star = "" if kind == "c" else r" (star: exact $P(f_0)$)"
    panels = [
        (f"non-invariant component\n({defect_note})", r"$\|(I-P)f\|$", "log"),
        ("relative non-invariance", r"$\|(I-P)f\| / \|f\|$", "log"),
        ("rotation-invariance error on the data", r"$\mathcal{E}(T)$", "log"),
        (f"distance to the exact flow {KIND_EXACT[kind]}\n(dotted: not moving)",
         r"$\|f_t - f_t^{\rm exact}\| / \|f_0\|$", "log"),
        (f"distance to the endpoint target {KIND_ENDPOINT[kind]}\n(dotted: where the exact flow would be)",
         r"$\|f_t - f_{\rm end}\| / \|f_0\|$", "log"),
        (f"is {KIND_TARGET[kind]} still in the tangent space?\n"
         r"(dotted: what the frozen $z_0$ still achieves here)",
         r"$\|Jz - g\| / \|g\|$", "log"),
        (f"task loss{star}", "BCE on training data", "log"),
        (f"task accuracy{star}", "accuracy", "linear"),
        ("distance travelled in parameter space\n(dotted: path length)",
         r"$\|\theta_t - \theta_0\| / \|\theta_0\|$", "linear"),
    ]
    for ax, (title, ylabel, yscale) in zip(axes, panels):
        ax.axvline(1.0, color="0.6", linewidth=0.8, zorder=0)
        ax.set(xlabel=KIND_TIME[kind], ylabel=ylabel, title=title, yscale=yscale)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    log_floor(axes[3], [results[n]["walk"]["exact_distance"][1:] for n in names])

    fig.suptitle(f"Diagnostics walking {label}, $t = 0 \\to 1$ -- {MODE_LABEL[mode]}")
    path = output_dir / f"equivariance_flow_{kind}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def figure_walk_comparison(results, kind: str, label: str, mode: str, cmap, output_dir: Path) -> None:
    """This run's walk (solid) against the other way of moving the same distance (dashed).

    In resolved mode the partner is the straight line through z_0, computed here for free.
    In fixed mode it is the re-solved walk, read back from that run's saved summary -- the
    same two curves either way, so the figure means the same thing in both directories."""
    if not any("compare" in r for r in results.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    names = list(results)
    for idx, name in enumerate(names):
        record, other = results[name]["walk"], results[name].get("compare")
        if other is None:
            continue
        colour = cmap(idx / max(len(names) - 1, 1))
        times = np.asarray(record["t"])
        for ax, key in zip(axes, ("defect_norm", "endpoint_distance", "accuracy")):
            ax.plot(times, record[key], color=colour, linewidth=condition_width(idx, 2.6),
                    label=f"{name} (this walk)")
            ax.plot(np.asarray(other["t"]), other[key], color=colour,
                    linewidth=condition_width(idx, 1.9), linestyle="--",
                    label=f"{name} ({COMPARE_LABEL[mode]})")
        axes[0].plot(times, record["defect_prediction"], color=colour, linewidth=1.0, linestyle=":", alpha=0.7)
    panels = [
        ("non-invariant component\n(dotted: exact-flow prediction)", r"$\|(I-P)f\|$", "log"),
        (f"distance to {KIND_ENDPOINT[kind]}", r"$\|f_t - f_{\rm end}\| / \|f_0\|$", "log"),
        ("task accuracy", "accuracy", "linear"),
    ]
    for ax, (title, ylabel, yscale) in zip(axes, panels):
        ax.axvline(1.0, color="0.6", linewidth=0.8, zorder=0)
        ax.set(xlabel=KIND_TIME[kind], ylabel=ylabel, title=title, yscale=yscale)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    title=f"solid = {MODE_LABEL[mode]}\ndashed = {COMPARE_LABEL[mode]}")
    fig.suptitle(f"Walking {label}: {MODE_LABEL[mode]} vs {COMPARE_LABEL[mode]}")
    path = output_dir / f"flow_vs_fixed_step_{kind}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def figure_boundaries(states, points, kind: str, label: str, args, cfg, device, dtype,
                      output_dir: Path) -> None:
    """Decision boundary along the walk: rows = condition, columns = flow time."""
    mesh = m.decision_grid(args.boundary_limit, args.boundary_grid, device, dtype)
    columns = [round(t, 9) for t in args.snapshot_times]
    fig, axes = plt.subplots(len(states), len(columns), squeeze=False,
                             figsize=(3.4 * len(columns), 4.0 * len(states)), constrained_layout=True)
    for row_idx, (name, state) in enumerate(states.items()):
        model = state["model"]
        for col_idx, t in enumerate(columns):
            ax = axes[row_idx, col_idx]
            set_flat_parameters(model, state["snapshots"][t])
            m.draw_decision_panel(ax, model, mesh, state["x_numpy"], state["y_numpy"],
                                  n_levels=args.boundary_levels)
            values, defect = equivariance_defect(model, points, args.group_order)
            relative = float(torch.linalg.vector_norm(defect)
                             / torch.linalg.vector_norm(values).clamp_min(1e-15))
            _, accuracy = task_metrics(model, state["x"], state["y"])
            ax.text(0.5, -0.02, f"$\\|(I-P)f\\|/\\|f\\|$ = {relative:.3f},  acc = {accuracy:.2f}",
                    ha="center", va="top", fontsize=12, transform=ax.transAxes)
            if row_idx == len(states) - 1:
                extra = f"\n(${math.degrees(t):.0f}^\\circ$)" if kind == "c" else ""
                ax.set_xlabel(f"$t = {t:g}${extra}", fontsize=17, labelpad=28)
        set_flat_parameters(model, state["snapshots"][columns[0]])
        axes[row_idx, 0].set_ylabel(name, fontsize=17, labelpad=12)
    axes[0, 0].set_title("start", fontsize=16, pad=8)
    axes[0, -1].set_title(KIND_ENDPOINT[kind] + "?", fontsize=16, pad=8)
    fig.suptitle(f"Decision boundary walking {label} -- {MODE_LABEL[args.mode]}", fontsize=15)
    path = output_dir / f"decision_boundary_flow_{kind}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dirs", type=Path, nargs="+",
                        help="Run directories holding checkpoints.pt; the condition label is read off each run's outer_arc_degrees.")
    parser.add_argument("--mode", choices=list(MODE_LABEL), default="resolved",
                        help="resolved: re-solve the direction at every step (the integral curve). "
                             "fixed: calculate it once at t=0 and step along that same vector repeatedly.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to outputs/aep_core_annulus_equivariance_flow, with _fixed appended in fixed mode.")
    parser.add_argument("--compare-summary", type=Path, default=None,
                        help="Summary JSON of the other mode's run, drawn as the dashed partner in Figure 5. "
                             "In fixed mode this defaults to the resolved run's summary.")
    parser.add_argument("--kinds", nargs="+", default=[k for k, _, _ in KINDS], choices=[k for k, _, _ in KINDS],
                        help="Which walks to run: c (rotate) and/or b (symmetrise).")
    parser.add_argument("--states", nargs="+", default=["initial", "trained"], choices=["initial", "trained"],
                        help="Which saved model states to walk from.")
    parser.add_argument("--time-max", type=float, default=1.0, help="Flow time to integrate to; for c this is an angle in radians.")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of forward-Euler steps from 0 to --time-max.")
    parser.add_argument("--snapshot-times", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help="Flow times drawn as curves/columns in the figures (snapped onto the Euler grid).")
    parser.add_argument("--probe-extent", type=float, default=3.5)
    parser.add_argument("--probe-n", type=int, default=24, help="Probe grid points per axis, before dropping the far corners.")
    parser.add_argument("--probe-max-radius", type=float, default=3.5)
    parser.add_argument("--group-order", type=int, default=64, help="Order m of the cyclic group C_m used as the quadrature for P.")
    parser.add_argument("--cutoff", type=float, default=1e-3, help="Relative SVD cutoff defining the numerically resolved tangent space.")
    parser.add_argument("--profile-radius", type=float, default=1.0,
                        help="Radius of the orbit the angular profile is read off; 1.0 is r_inner, the rim of the positive class.")
    parser.add_argument("--profile-n-angles", type=int, default=360)
    parser.add_argument("--boundary-limit", type=float, default=4.0)
    parser.add_argument("--boundary-grid", type=int, default=220)
    parser.add_argument("--boundary-levels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Fixes the rotations E(T) averages over, so it is comparable across stops.")
    parser.add_argument("--no-fixed-comparison", action="store_true", help="Skip the straight-line z_0 walk (Figure 5).")
    parser.add_argument("--replot", action="store_true", help="Redraw the figures from a previous run's saved records, without walking again.")
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.float64
    default_dir = Path("outputs/aep_core_annulus_equivariance_flow" + MODE_SUFFIX[args.mode])
    output_dir = args.output_dir if args.output_dir is not None else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_summary = args.compare_summary
    if compare_summary is None and args.mode == "fixed":
        compare_summary = Path("outputs/aep_core_annulus_equivariance_flow/equivariance_flow_summary.json")

    points = probe_grid(args.probe_extent, args.probe_n, args.probe_max_radius, device, dtype)
    angles, circle = circle_points(args.profile_radius, args.profile_n_angles, device, dtype)
    # Conditions become the columns of every figure. Two runs sharing an initialisation
    # give two identical "initial" columns by construction -- that is the control, not a bug.
    known = {f.name for f in fields(m.Config)}
    conditions: dict[str, dict] = {}
    for source_dir in args.source_dirs:
        checkpoint = torch.load(source_dir / "checkpoints.pt", weights_only=False)
        cfg = m.Config(**{k: v for k, v in checkpoint["config"].items() if k in known})
        arc = cfg.outer_arc_degrees
        run = "wedge" if arc < 359.0 else "full"
        x = torch.tensor(checkpoint["x"], device=device, dtype=dtype)
        y = torch.tensor((checkpoint["y"] > 0).astype(np.float64), device=device, dtype=dtype)
        for state in args.states:
            key = "initial_state_dict" if state == "initial" else "final_state_dict"
            models = []
            for _ in range(2):  # one to walk, one frozen copy for the reference
                model = m.SimpleMLP(hidden_dim=cfg.hidden_dim, depth=cfg.depth)
                model.load_state_dict(checkpoint[key])
                models.append(model.to(device=device, dtype=dtype).eval())
            conditions[f"{run}/{state}"] = {
                "model": models[0], "base_model": models[1], "cfg": cfg, "arc": arc,
                "x": x, "y": y, "x_numpy": checkpoint["x"], "y_numpy": checkpoint["y"],
            }

    cmap = plt.get_cmap("coolwarm")
    # Symmetric about 0 so t = 0 lands on coolwarm's pale midpoint and later steps deepen
    # into the warm half. The reference walks both ways and gets the pale baseline for
    # free; this walk is one-sided, and centring the norm on 0 anyway is what keeps the
    # baseline pale-and-thick instead of burying t = t_max/2 in the washed-out middle.
    norm = plt.Normalize(vmin=-args.time_max, vmax=args.time_max)
    condition_cmap = plt.get_cmap("viridis")
    summary_path = output_dir / "equivariance_flow_summary.json"
    profiles_path = output_dir / "angular_deviation_profiles.npz"

    if args.replot:
        stored = np.load(profiles_path)
        angles = stored["angles"]
        payload = json.loads(summary_path.read_text())
        profiles = {k: {n: stored[f"{k}/{n}"] for n in payload["kinds"][k]} for k in payload["kinds"]}
        summary = payload["kinds"]
        snapshots = np.load(output_dir / "equivariance_flow_parameters.npz")
    else:
        print(f"mode={args.mode} ({MODE_LABEL[args.mode]})")
        print(f"conditions={list(conditions)}")
        print(f"probe grid: {points.shape[0]} points; profile on |x| = {args.profile_radius:g} "
              f"at {args.profile_n_angles} angles")
        print(f"time_max={args.time_max}, n_steps={args.n_steps}, dt={args.time_max / args.n_steps:g}, "
              f"group_order={args.group_order}, cutoff={args.cutoff:g}\n")

        summary: dict[str, dict] = {}
        profiles: dict[str, dict[str, np.ndarray]] = {}
        snapshot_arrays: dict[str, np.ndarray] = {}
        walk_states: dict[str, dict[str, dict]] = {}
        for kind, sign, label in KINDS:
            if kind not in args.kinds:
                continue
            print(f"walking {kind} ({label})...")
            summary[kind], profiles[kind], walk_states[kind] = {}, {}, {}
            for name, condition in conditions.items():
                model, base_model = condition["model"], condition["base_model"]
                cfg, x, y = condition["cfg"], condition["x"], condition["y"]
                base = flat_parameters(model)
                record, snaps, profile = walk(model, base_model, points, circle, x, y, kind, sign,
                                              cfg, args, device, dtype)
                # In resolved mode the partner is the straight line through z_0, which costs
                # no Jacobians. In fixed mode the straight line IS this walk, so the partner
                # has to be read back from the resolved run rather than recomputed.
                compare = None
                if not args.no_fixed_comparison:
                    if args.mode == "resolved":
                        compare = fixed_walk(model, base_model, points, x, y, kind, sign, args)
                    elif compare_summary is not None and compare_summary.exists():
                        other = json.loads(compare_summary.read_text())["kinds"]
                        compare = other.get(kind, {}).get(name, {}).get("walk")
                projected_logits = group_average(model, x, args.group_order)
                projected_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(projected_logits, y))
                projected_accuracy = float(((projected_logits > 0).to(y.dtype) == y).to(torch.float64).mean())

                profiles[kind][name] = profile
                walk_states[kind][name] = {
                    "model": model, "snapshots": snaps, "x": x, "y": y,
                    "x_numpy": condition["x_numpy"], "y_numpy": condition["y_numpy"],
                }
                summary[kind][name] = {
                    "outer_arc_degrees": condition["arc"],
                    "defect_start": record["defect_norm"][0],
                    "defect_end": record["defect_norm"][-1],
                    "defect_predicted_end": record["defect_prediction"][-1],
                    "endpoint_distance_start": record["endpoint_distance"][0],
                    "endpoint_distance_end": record["endpoint_distance"][-1],
                    "endpoint_predicted_end": record["endpoint_prediction"][-1],
                    "exact_distance_end": record["exact_distance"][-1],
                    "solve_residual_start": record["solve_residual"][0],
                    "solve_residual_max": max(record["solve_residual"]),
                    "frozen_residual_end": record["frozen_residual"][-1],
                    "frozen_residual_max": max(record["frozen_residual"]),
                    "accuracy_start": record["accuracy"][0],
                    "accuracy_end": record["accuracy"][-1],
                    "parameter_distance_end": record["parameter_distance"][-1],
                    "path_length": record["path_length"][-1],
                    "projected_loss": projected_loss,
                    "projected_accuracy": projected_accuracy,
                    "walk": record,
                }
                if compare is not None:
                    summary[kind][name]["compare"] = compare
                for t, params in snaps.items():
                    snapshot_arrays[f"{kind}/{name}/{t:g}"] = params.cpu().numpy()
                print(f"  {name:>16s}: ||(I-P)f|| {record['defect_norm'][0]:.3e} -> "
                      f"{record['defect_norm'][-1]:.3e} (exact {record['defect_prediction'][-1]:.3e})"
                      f" | dist to end {record['endpoint_distance'][0]:.4f} -> "
                      f"{record['endpoint_distance'][-1]:.4f} (exact {record['endpoint_prediction'][-1]:.4f})"
                      f" | off exact path {record['exact_distance'][-1]:.4f}"
                      f" | ||Jz-g||/||g|| re-solved max {max(record['solve_residual']):.2e},"
                      f" frozen {record['frozen_residual'][-1]:.2e}"
                      f" | acc {record['accuracy'][0]:.3f} -> {record['accuracy'][-1]:.3f}")

        np.savez_compressed(profiles_path, angles=angles,
                            **{f"{k}/{n}": p for k, per in profiles.items() for n, p in per.items()})
        np.savez_compressed(output_dir / "equivariance_flow_parameters.npz", **snapshot_arrays)
        summary_path.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                            "kinds": summary}, indent=2))
        print(f"Done. Wrote {summary_path}\n")

    kinds = [(k, s, l) for k, s, l in KINDS if k in summary]
    names = list(summary[kinds[0][0]])

    # --- Figure 1 (primary): angular profile of the defect at fixed radius. Rows = {c, -b}, columns = condition. ---
    dt = args.time_max / args.n_steps
    drawn = [(t, min(max(int(round(t / dt)), 0), args.n_steps)) for t in args.snapshot_times]
    fig, axes = plt.subplots(len(kinds), len(names), figsize=(4.6 * len(names), 4.6 * len(kinds)),
                             constrained_layout=True, squeeze=False)
    for row_idx, (kind, _, label) in enumerate(kinds):
        for col_idx, name in enumerate(names):
            ax = axes[row_idx, col_idx]
            profile = profiles[kind][name]
            for t, step in drawn:
                colour = cmap(norm(t))
                lw = 2.2 if step == 0 else 1.3
                ax.plot(np.degrees(angles), profile[step], color=colour, linewidth=lw, label=f"$t = {t:g}$")
                ax.plot(np.degrees(angles), exact_profile(kind, profile[0], angles, t),
                        color=colour, linewidth=1.0, linestyle=":", alpha=0.9)
            ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set(xlabel=r"angle $\theta$ (deg) at $r=%.2f$" % args.profile_radius, title=name)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n" + r"$d(\theta) = f(r,\theta) - \overline{f}(r)$")
    axes[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       title="solid = walk, dotted = exact")
    fig.suptitle("Angular profile of the non-invariant part along the walk, evaluated across "
                 f"conditions -- {MODE_LABEL[args.mode]}")
    path = output_dir / "angular_deviation_flow.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")

    # --- Figure 2: the same profiles as a surface over (angle, flow time), using every step. ---
    fig, axes = plt.subplots(len(kinds), len(names), figsize=(4.6 * len(names), 4.0 * len(kinds)),
                             constrained_layout=True, squeeze=False)
    for row_idx, (kind, _, label) in enumerate(kinds):
        for col_idx, name in enumerate(names):
            ax = axes[row_idx, col_idx]
            profile = profiles[kind][name]
            times = np.linspace(0.0, args.time_max, profile.shape[0])
            scale = float(np.abs(profile).max())
            image = ax.pcolormesh(np.degrees(angles), times, profile, cmap="RdBu_r",
                                  vmin=-scale, vmax=scale, shading="nearest")
            if kind == "c":
                # The rigid-rotation prediction is a straight line of slope 1 in these axes:
                # follow each zero crossing of d_0 and see whether it tracks.
                for crossing in np.where(np.diff(np.signbit(profile[0])))[0]:
                    ax.plot((np.degrees(angles)[crossing] - np.degrees(times)) % 360.0, times,
                            ".", markersize=0.8, color="0.15", alpha=0.55)
            fig.colorbar(image, ax=ax, pad=0.015)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set(xlabel=r"angle $\theta$ (deg)", title=name)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n" + r"flow time $t$")
    fig.suptitle(r"Non-invariant part $d(\theta)$ on the orbit over the whole walk "
                 f"(dots: rigid-rotation prediction) -- {MODE_LABEL[args.mode]}")
    path = output_dir / "angular_deviation_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Wrote {path}")

    # --- Figure 3: amplitude of that profile, and the residual from each exact prediction. ---
    figure_amplitude(profiles, angles, names, args, condition_cmap, output_dir)

    # --- Figures 4-6: per-kind scalar diagnostics, the fixed-z_0 comparison, and the boundary. ---
    for kind, _, label in kinds:
        figure_diagnostics(summary[kind], kind, label, args.mode, condition_cmap, output_dir)
        figure_walk_comparison(summary[kind], kind, label, args.mode, condition_cmap, output_dir)
        if args.replot:
            states = {n: {"model": conditions[n]["model"],
                          "snapshots": {round(t, 9): torch.tensor(snapshots[f"{kind}/{n}/{t:g}"],
                                                                  device=device, dtype=dtype)
                                        for t in args.snapshot_times},
                          "x": conditions[n]["x"], "y": conditions[n]["y"],
                          "x_numpy": conditions[n]["x_numpy"], "y_numpy": conditions[n]["y_numpy"]}
                      for n in names}
        else:
            states = walk_states[kind]
        figure_boundaries(states, points, kind, label, args, conditions[names[0]]["cfg"],
                          device, dtype, output_dir)


if __name__ == "__main__":
    main()
