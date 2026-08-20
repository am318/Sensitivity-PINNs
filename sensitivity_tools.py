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

import math

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats import spearmanr


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


def _resolve_tangent_svd(j: torch.Tensor, g: torch.Tensor, relative_cutoff: float):
    """Shared truncated-SVD tangent-space resolution used by both norm choices.

    Returns ``(u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank,
    singular_values, col_scale)``. ``col_scale`` is the per-parameter native
    sensitivity magnitude ``||J[:, i]||`` used to remove scale/basis bias from
    whichever norm is minimised over the underdetermined solution set (see
    ``tangent_projection`` and ``tangent_projection_l1``).
    """
    u, singular_values, vh = torch.linalg.svd(j, full_matrices=False)
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        keep = torch.zeros_like(singular_values, dtype=torch.bool)
    else:
        keep = singular_values >= relative_cutoff * singular_values[0]
    resolved_rank = int(keep.sum())
    # A *relative* floor (like relative_cutoff above), not an absolute one: an
    # absolute clamp_min(1e-15) is disastrous once P is in the thousands and
    # some parameters are genuinely dead (e.g. a saturated/pruned unit) --
    # floating-point noise in that parameter's row of V (which should be
    # exactly 0 but isn't, at machine precision) gets divided by 1e-15 and
    # blows up into a spurious coefficient of order 1e14, corrupting the
    # whole equality-constrained solve. Columns at or below the relative
    # floor are treated as exactly dead (see _normalized_constraint_matrix).
    raw_col_scale = j.norm(dim=0)
    max_col_scale = raw_col_scale.max().clamp_min(1e-300)
    dead_columns = raw_col_scale <= max_col_scale * 1e-8
    col_scale = raw_col_scale.clamp_min(max_col_scale * 1e-8)
    if resolved_rank:
        u_kept = u[:, keep]
        s_kept = singular_values[keep]
        vh_kept = vh[keep, :]
        coordinates = u_kept.T @ g
        projection = u_kept @ coordinates
    else:
        u_kept = s_kept = vh_kept = coordinates = None
        projection = torch.zeros_like(g)
    return u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, singular_values, col_scale, dead_columns


def _normalized_constraint_matrix(
    vh_kept: torch.Tensor, col_scale: torch.Tensor, dead_columns: torch.Tensor
) -> torch.Tensor:
    """``vh_kept / col_scale``, with dead (near-zero-sensitivity) columns forced to exactly 0.

    A dead parameter cannot affect the target regardless of its coefficient
    (its true column of ``J`` is ~0), so it must get exactly 0 attribution --
    not an arbitrary or numerically-blown-up value from dividing SVD noise
    by a small floor. Forcing the column to 0 here means the downstream
    solve has no way to route weight through it, and un-normalizing
    (``coefficients / col_scale``) then also yields exactly 0 for it.
    """
    a_eq = vh_kept / col_scale
    if dead_columns.any():
        a_eq = a_eq.clone()
        a_eq[:, dead_columns] = 0.0
    return a_eq


def _projection_error_and_angle(g: torch.Tensor, projection: torch.Tensor) -> tuple[float, float]:
    target_norm = torch.linalg.vector_norm(g).clamp_min(1e-15)
    relative_error = torch.linalg.vector_norm(g - projection) / target_norm
    cosine = torch.dot(g, projection) / (
        target_norm * torch.linalg.vector_norm(projection).clamp_min(1e-15)
    )
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    return float(relative_error), float(angle)


def tangent_projection(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
    normalize_columns: bool = True,
) -> tuple[float, float, torch.Tensor, int, list[float]]:
    """Project ``target`` onto Im(J), returning error, angle, and min-norm c.

    A truncated SVD defines the numerically resolved tangent space. This avoids
    declaring every direction represented merely because an overparameterised
    Jacobian has tiny nonzero singular values on a finite probe grid.

    ``jacobian`` has shape ``[N, P]`` and ``target`` shape ``[N]``, where ``N``
    indexes flattened (probe point, output component) pairs.

    Within the resolved tangent space, ``target``'s projection is unique, but
    the coefficient vector ``c`` reproducing it is not (P typically exceeds the
    resolved rank). ``tangent_projection`` returns the minimum-2-norm ``c``. If
    ``normalize_columns`` is true (default), the 2-norm is minimised after
    rescaling each parameter's column of ``J`` to unit norm and the scaling is
    undone before returning ``c`` -- otherwise a unit change in a parameter
    with a naturally small gradient column is treated as "cheap" purely due to
    units, biasing which parameters end up carrying attribution.
    """
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    g = target.detach().cpu().to(dtype=work_dtype)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, singular_values, col_scale, dead_columns = (
        _resolve_tangent_svd(j, g, relative_cutoff)
    )
    if resolved_rank:
        alpha = coordinates / s_kept
        if normalize_columns:
            a_eq = _normalized_constraint_matrix(vh_kept, col_scale, dead_columns)
            coefficients = a_eq.T @ torch.linalg.solve(a_eq @ a_eq.T, alpha)
            coefficients = coefficients / col_scale
        else:
            coefficients = vh_kept.T @ alpha
    else:
        coefficients = torch.zeros(j.shape[1], dtype=work_dtype)
    relative_error, angle = _projection_error_and_angle(g, projection)
    return (
        relative_error,
        angle,
        coefficients,
        resolved_rank,
        singular_values.cpu().tolist(),
    )


def tangent_projection_l1(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
    normalize_columns: bool = True,
) -> tuple[float, float, torch.Tensor, int, list[float]]:
    """Same as ``tangent_projection``, but returns the minimum-1-norm ``c``.

    ``relative_error``, ``angle``, ``resolved_rank`` and the singular-value
    spectrum are identical to ``tangent_projection`` -- both functions
    reconstruct the exact same projection of ``target`` onto the resolved
    tangent space; they differ only in which of the infinitely many
    coefficient vectors ``c`` reproducing that projection is reported.
    Minimising ``sum(|c_i|)`` instead of ``sum(c_i**2)`` tends to concentrate
    attribution onto a sparse subset of (resolved-rank-many) parameters
    instead of spreading it smoothly, which is the tradeoff this function
    exists to expose. The min-1-norm solve is done via linear programming
    (``scipy.optimize.linprog``, HiGHS), so it is considerably more expensive
    than the closed-form min-2-norm solve for large ``P``.
    """
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    g = target.detach().cpu().to(dtype=work_dtype)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, singular_values, col_scale, dead_columns = (
        _resolve_tangent_svd(j, g, relative_cutoff)
    )
    if resolved_rank:
        alpha = coordinates / s_kept
        a_eq = _normalized_constraint_matrix(vh_kept, col_scale, dead_columns) if normalize_columns else vh_kept
        c_tilde = _minimum_l1_norm_solution(a_eq.numpy(), alpha.numpy())
        coefficients = torch.from_numpy(c_tilde).to(dtype=work_dtype)
        if normalize_columns:
            coefficients = coefficients / col_scale
    else:
        coefficients = torch.zeros(j.shape[1], dtype=work_dtype)
    relative_error, angle = _projection_error_and_angle(g, projection)
    return (
        relative_error,
        angle,
        coefficients,
        resolved_rank,
        singular_values.cpu().tolist(),
    )


def _minimum_l1_norm_solution(a_eq: np.ndarray, b_eq: np.ndarray) -> np.ndarray:
    """Solve ``argmin ||c||_1 s.t. a_eq @ c == b_eq`` via linear programming.

    Split into non-negative parts ``c = p - q`` (standard L1-as-LP trick),
    minimising ``sum(p + q)``. ``a_eq`` has shape ``[rank, P]`` with ``rank``
    typically far smaller than ``P``, so the LP has few equality constraints
    despite ``2P`` variables.
    """
    n_params = a_eq.shape[1]
    cost = np.ones(2 * n_params)
    lp_matrix = np.hstack([a_eq, -a_eq])
    result = linprog(cost, A_eq=lp_matrix, b_eq=b_eq, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"minimum-L1-norm LP failed to converge: {result.message}")
    p, q = result.x[:n_params], result.x[n_params:]
    return p - q


def tangent_projection_elastic_net(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
    l1_ratio: float,
    normalize_columns: bool = True,
    rho: float = 1.0,
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> tuple[float, float, torch.Tensor, int, list[float]]:
    """Same as ``tangent_projection``, but returns the minimum-elastic-net-norm ``c``.

    Solves ``argmin l1_ratio*||c||_1 + (1-l1_ratio)/2*||c||_2**2`` subject to
    the exact tangent-space equality constraint, via ADMM. ``l1_ratio=1``
    recovers (up to solver tolerance) ``tangent_projection_l1``'s answer;
    ``l1_ratio=0`` recovers ``tangent_projection``'s closed-form min-2-norm
    answer.

    Because the equality constraint forces an *exact* reconstruction of
    ``target``'s projection (there is no fit-quality/sparsity tradeoff to
    absorb, unlike ordinary elastic-net regression), ``l1_ratio`` is the only
    free hyperparameter here -- it purely controls how ties are broken among
    (near-)collinear parameter columns, not how much of ``target`` gets
    explained. Pick it via a stability sweep (see ``docs`` / the paper
    discussion): the smallest ``l1_ratio`` at which the selected support
    stops changing across resampled probes or random seeds, rather than by
    cross-validating a fit metric.
    """
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    g = target.detach().cpu().to(dtype=work_dtype)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, singular_values, col_scale, dead_columns = (
        _resolve_tangent_svd(j, g, relative_cutoff)
    )
    if resolved_rank:
        alpha = coordinates / s_kept
        a_eq = _normalized_constraint_matrix(vh_kept, col_scale, dead_columns) if normalize_columns else vh_kept
        c_tilde = _elastic_net_admm(a_eq.numpy(), alpha.numpy(), l1_ratio, rho, max_iter, tol)
        coefficients = torch.from_numpy(c_tilde).to(dtype=work_dtype)
        if normalize_columns:
            coefficients = coefficients / col_scale
    else:
        coefficients = torch.zeros(j.shape[1], dtype=work_dtype)
    relative_error, angle = _projection_error_and_angle(g, projection)
    return (
        relative_error,
        angle,
        coefficients,
        resolved_rank,
        singular_values.cpu().tolist(),
    )


def _elastic_net_admm(
    a_eq: np.ndarray, b_eq: np.ndarray, l1_ratio: float, rho: float, max_iter: int, tol: float
) -> np.ndarray:
    """ADMM solve of ``argmin l1_ratio*||c||_1 + (1-l1_ratio)/2*||c||_2**2 s.t. a_eq @ c == b_eq``.

    Standard consensus splitting (Boyd et al., 2011, secs. 6.x / "Basis
    Pursuit"): introduce ``z = c``, put the smooth ridge term plus the affine
    constraint on the ``c``-update (closed form -- an affine projection using
    the ``rank x rank`` Gram matrix ``a_eq @ a_eq.T``, cached once) and the
    L1 term on the ``z``-update (elementwise soft-thresholding). At
    ``l1_ratio=1`` this is exactly ADMM for basis pursuit; at ``l1_ratio=0``
    it converges to the min-2-norm solution.
    """
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
    n_params = a_eq.shape[1]
    gram = a_eq @ a_eq.T
    gram_inv = np.linalg.inv(gram)
    smooth_weight = (1.0 - l1_ratio) + rho
    threshold = l1_ratio / rho

    def project_onto_constraint(y: np.ndarray) -> np.ndarray:
        return y - a_eq.T @ (gram_inv @ (a_eq @ y - b_eq))

    c = np.zeros(n_params)
    z = np.zeros(n_params)
    u = np.zeros(n_params)
    for _ in range(max_iter):
        c = project_onto_constraint((rho / smooth_weight) * (z - u))
        z_new = np.sign(c + u) * np.maximum(np.abs(c + u) - threshold, 0.0)
        primal_residual = np.linalg.norm(c - z_new)
        dual_residual = rho * np.linalg.norm(z_new - z)
        u = u + c - z_new
        z = z_new
        if primal_residual < tol and dual_residual < tol:
            break
    return c


def _n_eff(c: np.ndarray) -> float:
    """Inverse participation ratio: "effective number of parameters carrying attribution"."""
    sum_sq = float((c**2).sum())
    if sum_sq <= 0:
        return 0.0
    return float(np.abs(c).sum() ** 2) / sum_sq


def _debiased_refit(a_eq: np.ndarray, b_eq: np.ndarray, coefficients: np.ndarray, support_tol: float = 1e-6) -> np.ndarray:
    """Refit unbiased magnitudes on the support of ``coefficients``, zero elsewhere.

    L1 (and, to a lesser extent, elastic-net) solutions are known to shrink
    the magnitude of retained coefficients toward zero relative to what is
    actually needed to explain the target (the standard Lasso shrinkage
    bias). This support-restricted least-squares refit -- "Lasso + OLS
    refit" / Gauss-Dantzig-selector style debiasing -- keeps the sparsity
    pattern chosen by the original solve but replaces the magnitudes with
    the unbiased least-squares fit on just that support, which is what
    should be reported/compared against ablation effect sizes. A no-op (up
    to numerical cleanup) for the dense min-2-norm solution, since there is
    no shrinkage bias to remove there.
    """
    max_abs = np.abs(coefficients).max()
    if max_abs <= 0:
        return coefficients
    support = np.where(np.abs(coefficients) > support_tol * max_abs)[0]
    if support.size == 0:
        return coefficients
    refit, *_ = np.linalg.lstsq(a_eq[:, support], b_eq, rcond=None)
    debiased = np.zeros_like(coefficients)
    debiased[support] = refit
    return debiased


def _support_size(c: np.ndarray, support_tol: float = 1e-6) -> int:
    max_abs = np.abs(c).max()
    if max_abs <= 0:
        return 0
    return int((np.abs(c) > support_tol * max_abs).sum())


def choose_l1_ratio_for_sparsity(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
    normalize_columns: bool = True,
    sparsity_tolerance: float = 1.15,
    l1_ratio_grid: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0),
) -> float:
    """Smallest ``l1_ratio`` giving an elastic-net support within ``sparsity_tolerance`` of pure L1's.

    Because ``tangent_projection``'s constraint is an *exact* equality (no
    fit-quality/sparsity tradeoff), ``l1_ratio`` mostly behaves as a switch
    rather than a dial (see the elastic-net docstring): even a small nonzero
    ridge admixture usually collapses to near-L1 sparsity, so this grid
    search finds the least amount of ridge regularisation -- i.e. the most
    tie-breaking headroom for (near-)collinear parameter columns -- that
    still matches L1's support *size* (number of nonzero coefficients, at
    the same threshold ``_debiased_refit`` uses) to within
    ``sparsity_tolerance`` (default: within 15%). Matching on support size
    rather than ``n_eff`` keeps this criterion invariant to the downstream
    debiasing refit, which reshapes magnitudes within a support without
    changing which parameters are in it. Falls back to ``l1_ratio=1.0``
    (pure L1) if no smaller ratio in the grid qualifies.
    """
    _, _, c_l1, resolved_rank, _ = tangent_projection_l1(jacobian, target, relative_cutoff, normalize_columns)
    if resolved_rank == 0:
        return 1.0
    target_support = _support_size(c_l1.numpy())
    for ratio in l1_ratio_grid:
        if ratio >= 1.0:
            return 1.0
        _, _, c_en, _, _ = tangent_projection_elastic_net(
            jacobian, target, relative_cutoff, l1_ratio=ratio, normalize_columns=normalize_columns
        )
        if _support_size(c_en.numpy()) <= target_support * sparsity_tolerance:
            return ratio
    return 1.0


def tangent_projection_auto(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    relative_cutoff: float,
    method: str = "elastic_net",
    normalize_columns: bool = True,
    debias: bool = True,
    l1_ratio: float | None = None,
    sparsity_tolerance: float = 1.15,
) -> tuple[float, float, torch.Tensor, int, list[float]]:
    """Unified entry point for the experiment scripts: choose the norm, then debias.

    ``method`` is one of ``"l2"`` (``tangent_projection``), ``"l1"``
    (``tangent_projection_l1``), or ``"elastic_net"`` (default --
    ``tangent_projection_elastic_net`` with ``l1_ratio`` chosen via
    ``choose_l1_ratio_for_sparsity`` unless explicitly passed, i.e. the
    smallest ridge admixture that still matches L1's sparsity, so
    (near-)collinear parameter columns get tie-broken/split rather than one
    arbitrarily zeroed, without giving up L1's sparsity/reproducibility
    advantage -- see the module-level discussion in the project's
    attribution-methodology notes). Every path is followed by a
    support-restricted least-squares debiasing refit (``debias=True``,
    default) that removes L1/elastic-net's shrinkage bias on the retained
    coefficients; this is a no-op for ``"l2"``.

    Returns the same 5-tuple as ``tangent_projection`` (drop-in replacement
    at existing call sites): ``(relative_error, angle_degrees, coefficients,
    resolved_rank, singular_values)``.
    """
    if method not in ("l2", "l1", "elastic_net"):
        raise ValueError(f"method must be 'l2', 'l1', or 'elastic_net', got {method!r}")
    work_dtype = torch.float64
    j = jacobian.detach().cpu().to(dtype=work_dtype)
    g = target.detach().cpu().to(dtype=work_dtype)
    u_kept, s_kept, vh_kept, coordinates, projection, resolved_rank, singular_values, col_scale, dead_columns = (
        _resolve_tangent_svd(j, g, relative_cutoff)
    )
    if resolved_rank == 0:
        coefficients = torch.zeros(j.shape[1], dtype=work_dtype)
        relative_error, angle = _projection_error_and_angle(g, projection)
        return relative_error, angle, coefficients, resolved_rank, singular_values.cpu().tolist()

    alpha = coordinates / s_kept
    a_eq = _normalized_constraint_matrix(vh_kept, col_scale, dead_columns) if normalize_columns else vh_kept
    a_eq_np, alpha_np = a_eq.numpy(), alpha.numpy()

    if method == "l2":
        c_tilde = a_eq_np.T @ np.linalg.solve(a_eq_np @ a_eq_np.T, alpha_np)
    elif method == "l1":
        c_tilde = _minimum_l1_norm_solution(a_eq_np, alpha_np)
    else:
        chosen_ratio = l1_ratio if l1_ratio is not None else choose_l1_ratio_for_sparsity(
            jacobian, target, relative_cutoff, normalize_columns, sparsity_tolerance
        )
        c_tilde = _elastic_net_admm(a_eq_np, alpha_np, chosen_ratio, rho=1.0, max_iter=20000, tol=1e-12)

    if debias:
        c_tilde = _debiased_refit(a_eq_np, alpha_np, c_tilde)

    coefficients = torch.from_numpy(c_tilde).to(dtype=work_dtype)
    if normalize_columns:
        coefficients = coefficients / col_scale
    relative_error, angle = _projection_error_and_angle(g, projection)
    return (
        relative_error,
        angle,
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


def participation_ratio_l1l2(values) -> float:
    """PR(c) = (sum |v_i|)^2 / sum v_i^2 -- an L1/L2-ratio "effective count", distinct from
    ``participation_ratio``'s L2/L4-ratio definition above.

    Both are standard localisation-theory summaries and agree qualitatively
    (max when mass is spread evenly, 1 when concentrated on one entry), but
    are numerically different quantities -- this is the exact form used in
    the project's attribution-methodology review notes (as "PR(c)"), so it
    is kept as a separate named function rather than silently changing
    ``participation_ratio``'s existing formula (used elsewhere in the
    project). Use this one when comparing against that review's diagnostics
    (e.g. the null-control / Lorenz-curve checks); it is also exactly the
    ``n_eff`` computed ad hoc throughout the L1/L2/elastic-net comparison
    scripts in this project.
    """
    v = np.asarray(values, dtype=np.float64)
    sum_sq = float((v**2).sum())
    if sum_sq <= 0:
        return 0.0
    return float(np.abs(v).sum() ** 2) / sum_sq


def random_matched_norm_target(
    target: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A uniformly-random direction in R^N, rescaled to the same norm as ``target``.

    Null control for attribution concentration: solve ``tangent_projection*``
    against this instead of the real symmetry-generator target and compare
    the resulting participation ratio / Lorenz curve. If a random,
    physically meaningless target gives the same "a few large c_i" structure
    as the real generator, that structure is the conditioning of the
    Jacobian (small singular values amplified by the pseudo-inverse), not
    evidence of anything about symmetry specifically.
    """
    raw = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(target.device)
    raw_norm = torch.linalg.vector_norm(raw).clamp_min(1e-15)
    target_norm = torch.linalg.vector_norm(target)
    return raw * (target_norm / raw_norm)


def random_symmetry_test_matrix(
    dim: int = 2,
    scale_spread: float = 0.3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A random linear map with the same "gentleness" (operator scale ~1) as a rotation,
    but generically *not* orthogonal -- the null-control counterpart of
    ``random_matched_norm_target``, for E_i instead of c_i.

    Constructed as ``R1 @ diag(s1, s2) @ R2`` for random rotations ``R1, R2``
    and singular values ``s1 != s2`` drawn near 1 (``1 +/- scale_spread``).
    This keeps points roughly in-domain (unlike an unconstrained random
    matrix, which can blow the probe grid arbitrarily far outside where the
    model was ever evaluated during training, confounding "not equivariant"
    with "out of distribution"), while breaking the equal-singular-value
    property that makes a matrix orthogonal -- and for a system with full
    O(2) symmetry (as the isotropic Mexican-hat potential has: it is
    invariant under reflections too, not just proper rotations), even a
    reflection is *not* a valid non-symmetry null, so this must be
    non-orthogonal to genuinely fall outside the system's symmetry group.

    E_i(M) computed against this (see ``per_parameter_equivariance_error``,
    which accepts any ``rep_matrix``) should be high for essentially any
    single draw -- the model was never trained to respect an arbitrary M.
    That alone is not informative (as uninformative as a random *target*
    concentrating for c_i); what is informative is comparing E_i(true
    generator) against the *distribution* of E_i(M) over many draws.
    """
    if dim != 2:
        raise NotImplementedError("random_symmetry_test_matrix currently only supports dim=2")
    angles = torch.rand(2, generator=generator, dtype=dtype) * (2 * math.pi)
    r1 = _rotation_matrix_2d(angles[0])
    r2 = _rotation_matrix_2d(angles[1])
    spread = (torch.rand(2, generator=generator, dtype=dtype) - 0.5) * (2 * scale_spread)
    singular_values = 1.0 + spread
    m = r1 @ torch.diag(singular_values) @ r2
    return m.to(device=device) if device is not None else m


def linear_generator_target(
    positions: torch.Tensor, force: torch.Tensor, spatial_jacobian: torch.Tensor, generator_matrix: torch.Tensor
) -> torch.Tensor:
    """X_M F = (dF/dq)(M q) - M F(q), the Lie-derivative-style construction generalised
    to an arbitrary linear map M (not necessarily a true symmetry generator).

    With ``generator_matrix`` set to the true rotation generator, this
    reproduces the physical symmetry-generator target exactly (e.g.
    ``rotation_generator_target`` in the Mexican-hat script is this function
    specialised to Omega). With ``generator_matrix = random_symmetry_test_matrix(...)``,
    it produces a *matched-construction* null target for c_i's attribution:
    unlike ``random_matched_norm_target`` (isotropic noise, no spatial
    structure), this null shares the real target's smoothness and
    construction -- it is built from the same spatial-Jacobian machinery,
    just with a generic linear map standing in for the true generator --
    isolating "is this specifically about the true generator" from "is this
    just different from unstructured noise", which the isotropic null
    cannot do.

    ``positions`` has shape ``[N, d]`` (e.g. stacked q1, q2); ``force`` has
    shape ``[N, d]``; ``spatial_jacobian`` has shape ``[N, d, d]``;
    ``generator_matrix`` has shape ``[d, d]``.
    """
    m = generator_matrix.to(device=positions.device, dtype=positions.dtype)
    m_positions = positions @ m.T
    directional = torch.einsum("nkj,nj->nk", spatial_jacobian, m_positions)
    return directional - force @ m.T


def _rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def attribution_scale_diagnostics(
    coefficients: torch.Tensor, sensitivity: torch.Tensor, parameter_magnitude: torch.Tensor
) -> dict[str, float]:
    """Spearman correlations checking whether ``|c_i|`` is just recovering trivial per-parameter scale.

    ``rho_sensitivity``: correlation between ``|c_i|`` and the parameter's
    own sensitivity magnitude ``||S_i||`` (e.g. RMS Jacobian column norm).
    ``rho_magnitude``: correlation between ``|c_i|`` and the raw trained
    parameter magnitude ``|theta_i|``. High values for either indicate the
    attribution is largely reporting "this parameter is already big/already
    sensitive" rather than anything specific to the symmetry-generator
    target -- in particular, if training used an L1 penalty on the raw
    parameters (a *different* L1 from the attribution-method L1 -- see
    ``tangent_projection_l1``), a high ``rho_magnitude`` would mean the
    attribution's sparsity pattern is largely just reporting the training
    L1's own support, not a symmetry-specific finding.
    """
    c = np.abs(coefficients.detach().cpu().numpy())
    s = np.asarray(sensitivity.detach().cpu().numpy() if torch.is_tensor(sensitivity) else sensitivity).reshape(-1)
    m = np.abs(
        parameter_magnitude.detach().cpu().numpy() if torch.is_tensor(parameter_magnitude) else np.asarray(parameter_magnitude)
    ).reshape(-1)
    rho_sensitivity = float(spearmanr(c, s).statistic) if c.size > 1 else float("nan")
    rho_magnitude = float(spearmanr(c, m).statistic) if c.size > 1 else float("nan")
    return {"rho_sensitivity": rho_sensitivity, "rho_magnitude": rho_magnitude}


def scale_invariant_attribution_score(
    jacobian: torch.Tensor, target: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Signed share of the represented target that parameter i accounts for: sums to 1.

    ``a_i = c_i * <S_i, g> / ||P_T g||^2``, where ``<S_i, g>`` is the i-th
    entry of ``jacobian.T @ target``. Invariant under any per-parameter
    rescaling ``theta_i -> lambda_i theta_i`` (unlike raw ``|c_i|``, which is
    not), because rescaling divides ``c_i`` and multiplies ``<S_i, g>`` by
    the same factor. Signed, so a negative ``a_i`` means parameter i's
    natural direction opposes the represented target; the resulting vector
    can be sorted by |a_i| and Lorenz-curved ("top 10% of parameters carry
    X% of the represented symmetry"), independent of the raw units any
    parameter happens to live in.
    """
    j = jacobian.detach().cpu().to(dtype=torch.float64)
    g = target.detach().cpu().to(dtype=torch.float64)
    c = coefficients.detach().cpu().to(dtype=torch.float64)
    projection = j @ c
    denom = torch.dot(projection, projection).clamp_min(1e-30)
    return c * (j.T @ g) / denom


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