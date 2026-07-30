"""System-agnostic functional-tangent-space and symmetry-representation tools.

This module implements the geometric machinery from
``Functional_Sensitivity_of_Neural_Networks-2.pdf`` (secs. 0.2-0.6) in a form
shared across physical systems: the functional tangent space
``T_theta = Im(J_theta)``, projection of a symmetry-generator direction onto
``T_theta`` (``tangent_projection``), principal angles between ``T_theta`` and
a multi-generator subspace together with the resulting representation
dimension (``principal_angles_and_dimension``), and finite-group
invariance/equivariance residuals for both function values
(``finite_transform_residual``) and functional sensitivities
(``sensitivity_transform_residual`` / ``involution_energy_fraction``), which
generalise a plain sign-flip (Z2 parity) check to an arbitrary representation
matrix (e.g. a rotation).
"""

from __future__ import annotations

import numpy as np
import torch


def parameter_gradient_row(output: torch.Tensor, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten d(output)/d(params) into one 1D row, zero-filling unused parameters.

    ``output`` must be a scalar (0-dim) tensor with a graph depending on ``params``.
    """
    grads = torch.autograd.grad(output, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, grads)
        ]
    )


def tangent_projection(
    jacobian: torch.Tensor, target: torch.Tensor, relative_cutoff: float
) -> tuple[float, float, torch.Tensor, int, list[float]]:
    """Project ``target`` onto Im(J), returning error, angle, and min-norm c.

    A truncated SVD defines the numerically resolved tangent space. This avoids
    declaring every direction represented merely because an overparameterised
    Jacobian has tiny nonzero singular values on a finite probe grid.

    ``jacobian`` has shape ``[N, P]`` and ``target`` shape ``[N]``, where ``N``
    indexes flattened (probe point, output component) pairs.
    """
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    g = target.detach().cpu().to(dtype=work_dtype)
    u, singular_values, vh = torch.linalg.svd(j, full_matrices=False)
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        keep = torch.zeros_like(singular_values, dtype=torch.bool)
    else:
        keep = singular_values >= relative_cutoff * singular_values[0]
    resolved_rank = int(keep.sum())
    if resolved_rank:
        u_kept = u[:, keep]
        s_kept = singular_values[keep]
        vh_kept = vh[keep, :]
        coordinates = u_kept.T @ g
        coefficients = vh_kept.T @ (coordinates / s_kept)
        projection = u_kept @ coordinates
    else:
        coefficients = torch.zeros(j.shape[1], dtype=work_dtype)
        projection = torch.zeros_like(g)
    target_norm = torch.linalg.vector_norm(g).clamp_min(1e-15)
    relative_error = torch.linalg.vector_norm(g - projection) / target_norm
    cosine = torch.dot(g, projection) / (
        target_norm * torch.linalg.vector_norm(projection).clamp_min(1e-15)
    )
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    return (
        float(relative_error),
        float(angle),
        coefficients,
        resolved_rank,
        singular_values.cpu().tolist(),
    )


def principal_angles_and_dimension(
    jacobian: torch.Tensor,
    generator_matrix: torch.Tensor,
    relative_cutoff: float,
    angle_threshold_degrees: float = 10.0,
) -> dict:
    """Principal angles between T_theta and span(generators), and their count below threshold.

    ``jacobian`` has shape ``[N, P]``; ``generator_matrix`` has shape ``[N, K]``
    with one target direction per column, both indexed over the same flattened
    (probe point, output component) axis. Implements the PDF's "principal
    angles" and "symmetry representation dimension" (sec. 0.5): T_theta is
    resolved via a truncated SVD of ``jacobian`` (as in ``tangent_projection``),
    span(generators) is resolved via a truncated SVD of ``generator_matrix``,
    and the principal angles between the two resolved subspaces are the
    arccos of the singular values of the product of their orthonormal bases.
    """
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    gmat = generator_matrix.detach().cpu().to(dtype=work_dtype)

    u_t, s_t, _ = torch.linalg.svd(j, full_matrices=False)
    keep_t = s_t >= relative_cutoff * s_t[0] if s_t.numel() and s_t[0] > 0 else torch.zeros_like(s_t, dtype=torch.bool)
    basis_t = u_t[:, keep_t]

    u_g, s_g, _ = torch.linalg.svd(gmat, full_matrices=False)
    keep_g = s_g >= relative_cutoff * s_g[0] if s_g.numel() and s_g[0] > 0 else torch.zeros_like(s_g, dtype=torch.bool)
    basis_g = u_g[:, keep_g]

    resolved_tangent_rank = int(keep_t.sum())
    resolved_generator_rank = int(keep_g.sum())

    if resolved_tangent_rank == 0 or resolved_generator_rank == 0:
        angles = [90.0] * max(resolved_generator_rank, 0)
        return {
            "principal_angles_degrees": angles,
            "representation_dimension": 0,
            "resolved_tangent_rank": resolved_tangent_rank,
            "resolved_generator_rank": resolved_generator_rank,
        }

    cross = basis_t.T @ basis_g
    cosines = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosines))
    representation_dimension = int((angles <= angle_threshold_degrees).sum())
    return {
        "principal_angles_degrees": angles.tolist(),
        "representation_dimension": representation_dimension,
        "resolved_tangent_rank": resolved_tangent_rank,
        "resolved_generator_rank": resolved_generator_rank,
    }


def finite_transform_residual(
    values_x: torch.Tensor,
    values_gx: torch.Tensor,
    rep_matrix: torch.Tensor | None = None,
) -> float:
    """Relative residual of ``values_gx approx rho(g) @ values_x`` for a finite group element g.

    ``values_x``/``values_gx`` have shape ``[N, D]`` (D output components,
    evaluated at probe points x and transformed points g.x respectively).
    ``rep_matrix`` is the ``[D, D]`` representation of g acting on the output;
    ``None`` means the identity (an invariance check, e.g. for a scalar
    potential). This generalises a plain sign-flip parity check to any finite
    transform, including rotations, since it only requires values already
    evaluated at both x and g.x rather than assuming the probe grid is closed
    under the transform.
    """
    x = values_x.detach().cpu().to(dtype=torch.float64)
    gx = values_gx.detach().cpu().to(dtype=torch.float64)
    predicted = x if rep_matrix is None else x @ rep_matrix.detach().cpu().to(dtype=torch.float64).T
    residual = gx - predicted
    return float(
        torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(x).clamp_min(1e-15)
    )


def transform_defect(
    jacobian_x: torch.Tensor,
    jacobian_gx: torch.Tensor,
    rep_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw defect ``J(gx) - rho(g) @ J(x)`` (shape ``[N, D, P]``), before normalisation.

    Exposed separately from ``sensitivity_transform_residual`` so callers can
    also derive per-parameter attribution (e.g. which parameters most violate
    the expected equivariance).
    """
    jx = jacobian_x.detach().cpu().to(dtype=torch.float64)
    jgx = jacobian_gx.detach().cpu().to(dtype=torch.float64)
    predicted = jx if rep_matrix is None else torch.einsum(
        "dc,ncp->ndp", rep_matrix.detach().cpu().to(dtype=torch.float64), jx
    )
    return jgx - predicted


def sensitivity_transform_residual(
    jacobian_x: torch.Tensor,
    jacobian_gx: torch.Tensor,
    rep_matrix: torch.Tensor | None = None,
) -> float:
    """Relative Frobenius residual of ``J(gx) approx rho(g) @ J(x)`` (functional sensitivity equivariance).

    ``jacobian_x``/``jacobian_gx`` have shape ``[N, D, P]`` (D output
    components, P parameters), evaluated at probe points x and g.x. This is
    the sec. 0.6 sensitivity-equivariance check generalised from a plain sign
    flip to any finite-group representation ``rho(g)`` (``[D, D]``, or
    ``None`` for the identity).
    """
    defect = transform_defect(jacobian_x, jacobian_gx, rep_matrix)
    jx = jacobian_x.detach().cpu().to(dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(defect) / torch.linalg.vector_norm(jx).clamp_min(1e-15)
    )


def domain_parity_energy_fraction(
    jacobian_x: torch.Tensor, jacobian_flipped_x: torch.Tensor
) -> float:
    """Fraction of the Jacobian's Frobenius energy lying in its domain-odd part.

    ``jacobian_flipped_x`` must be ``jacobian_x`` re-indexed by a grid-closed
    domain involution (e.g. ``torch.flip`` along the probe axis for a
    reflection-symmetric grid) -- no output representation is applied here.
    This measures whether the sensitivities happen to be an odd function of
    the probe coordinate itself, which is what an exactly parity-equivariant
    network's Jacobian must be (since the true force is odd); it is a
    diagnostic of functional form, independent of, and complementary to,
    ``sensitivity_transform_residual``'s equivariance-defect check.
    """
    jx = jacobian_x.detach().cpu().to(dtype=torch.float64)
    jflip = jacobian_flipped_x.detach().cpu().to(dtype=torch.float64)
    odd_part = 0.5 * (jx - jflip)
    return float(
        torch.linalg.vector_norm(odd_part).square()
        / torch.linalg.vector_norm(jx).square().clamp_min(1e-30)
    )


def per_parameter_equivariance_error(
    jacobian_x: torch.Tensor,
    jacobian_gx: torch.Tensor,
    rep_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-parameter sensitivity-equivariance error E_i (PDF sec. 0.6), shape ``[P]``.

    ``E_i = ||S_i(gx) - rho(g) S_i(x)|| / ||S_i(x)||``, the norm taken over the
    probe points and output components for one fixed parameter i. Unlike
    ``sensitivity_transform_residual`` (one aggregate scalar over the whole
    Jacobian), this is computed independently for every parameter, so it can
    be ranked, or aggregated by module, to localise where a network's
    functional sensitivities fail to transform correctly under a symmetry
    generator -- even when the function itself has learned the symmetry well.
    """
    defect = transform_defect(jacobian_x, jacobian_gx, rep_matrix)
    jx = jacobian_x.detach().cpu().to(dtype=torch.float64)
    numerator = torch.linalg.vector_norm(defect, dim=(0, 1))
    denominator = torch.linalg.vector_norm(jx, dim=(0, 1)).clamp_min(1e-15)
    return numerator / denominator


def participation_ratio(values) -> float:
    """Effective number of entries carrying a per-parameter quantity (e.g. attribution c_i).

    PR = (sum v_i^2)^2 / sum v_i^4, a standard localisation-theory summary:
    PR = k if the "mass" v_i^2 is spread evenly over exactly k entries and
    zero elsewhere; PR = 1 if a single entry carries everything. Robust to
    counting "nonzero" entries directly, which is useless once L1/pruning
    leaves almost nothing exactly zero at floating-point precision -- PR
    instead answers "effectively how many parameters does this attribution
    actually live on," independent of how many are technically nonzero.
    """
    v = np.asarray(values, dtype=np.float64)
    sq = v**2
    denom = float(np.sum(sq**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(sq) ** 2 / denom)


def aggregate_by_module(values: torch.Tensor, slices: dict[str, slice]) -> dict[str, float]:
    return {
        name: float(torch.linalg.vector_norm(values[sl]).cpu())
        for name, sl in slices.items()
    }


def aggregate_by_module_mean(values: torch.Tensor, slices: dict[str, slice]) -> dict[str, float]:
    """Mean (rather than norm) within each module -- appropriate for already-normalised
    per-parameter ratios such as E_i, where a norm would just grow with module size."""
    return {
        name: float(values[sl].mean().cpu())
        for name, sl in slices.items()
    }


def noether_energy_drift(
    grad_e_p: torch.Tensor,
    grad_e_q: torch.Tensor,
    dpdt: torch.Tensor,
    dqdt: torch.Tensor,
) -> torch.Tensor:
    """Pointwise dE_true/dt under a *learned* dynamics (dpdt, dqdt).

    For an autonomous Hamiltonian system the true energy E_true(p,q) is
    exactly conserved by the true dynamics: dE/dt = grad_p E . dp/dt + grad_q
    E . dq/dt = 0 identically. This function evaluates that same expression
    with the model's own learned (dpdt, dqdt) in place of the true dynamics,
    using the *known, analytic* grad_e_p/grad_e_q (never the network's own
    energy) -- so it is exactly zero only if the learned flow happens to
    conserve the true physical energy, a genuine, non-tautological question
    (unlike a claimed symmetry that doesn't actually hold, this is an exact
    fact of the real physics for every parameter value).

    Shapes: ``grad_e_p``/``dpdt`` are ``[N, p_dim]``, ``grad_e_q``/``dqdt``
    are ``[N, q_dim]``. Returns ``[N]``.
    """
    return (grad_e_p * dpdt).sum(dim=-1) + (grad_e_q * dqdt).sum(dim=-1)


def relative_energy_drift(
    grad_e_p: torch.Tensor,
    grad_e_q: torch.Tensor,
    dpdt: torch.Tensor,
    dqdt: torch.Tensor,
) -> float:
    """RMS energy-conservation violation, relative to the typical magnitude of the two terms.

    0 for exact conservation; up to ~2 if the drift is comparable in size to
    the terms themselves (triangle-inequality bound).
    """
    drift = noether_energy_drift(grad_e_p, grad_e_q, dpdt, dqdt)
    p_term_scale = torch.linalg.vector_norm(grad_e_p, dim=-1) * torch.linalg.vector_norm(dpdt, dim=-1)
    q_term_scale = torch.linalg.vector_norm(grad_e_q, dim=-1) * torch.linalg.vector_norm(dqdt, dim=-1)
    scale = torch.sqrt(torch.mean((p_term_scale + q_term_scale) ** 2)).clamp_min(1e-15)
    return float(torch.sqrt(torch.mean(drift**2)) / scale)

def parameter_magnitude_ci_correlation(
    parameter_magnitude,
    ci,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """
    Pearson correlation between parameter magnitude |theta_i| and a
    per-parameter quantity (e.g. attribution coefficients, confidence
    interval, or other CI measure).

    Parameters
    ----------
    parameter_magnitude : array-like
        |theta_i| for every parameter.
    ci : array-like
        Per-parameter quantity of the same length.
    mask : array-like of bool, optional
        Boolean mask selecting parameters to include
        (e.g. only V_net parameters).

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1]. Returns NaN if the
        correlation is undefined.
    """
    magnitude = np.asarray(parameter_magnitude, dtype=float)
    ci = np.asarray(ci, dtype=float)

    if magnitude.shape != ci.shape:
        raise ValueError("parameter_magnitude and ci must have the same shape.")

    valid = np.isfinite(magnitude) & np.isfinite(ci)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    magnitude = magnitude[valid]
    ci = ci[valid]

    if magnitude.size < 2:
        return np.nan

    if np.std(magnitude) == 0 or np.std(ci) == 0:
        return np.nan

    return float(np.corrcoef(magnitude, ci)[0, 1])