"""Planar two-body problem, kept in the full 8D lab frame (not reduced to relative coordinates).

    q = (q1x, q1y, q2x, q2y), p = (p1x, p1y, p2x, p2y), unit masses m1 = m2 = 1.
    V(q; alpha) = -alpha / r,   r = |q1 - q2|   (alpha > 0: attractive, "gravitational").
    dq/dt = p,
    dp1/dt = F1 = -alpha * (q1 - q2) / r**3,
    dp2/dt = -F1  (Newton's third law).

Deliberately *not* reduced to the relative coordinate / centre-of-mass frame:
V depends only on q1 - q2, so this system has two independent, exact
continuous symmetries the network must discover for itself directly in the
8D lab-frame representation:

1. SO(2) rotation, acting on (q1, q2) *and* (p1, p2) simultaneously by the
   same rotation matrix (both particles' position and momentum vectors
   rotate together).
2. 2D translation, acting on (q1, q2) by the same shift vector (momenta
   untouched -- translation only moves positions, it isn't a Galilean boost).

For an exact 1/r attractive potential, Bertrand's theorem guarantees every
bound orbit (E < 0) is a closed ellipse (or circle) -- no special parameter
tuning is needed to get closed orbits, just sample bound initial conditions.

Reuses the dimension-generic pieces of ``mexican_hat_dynamics.py``
(``generate_trajectories``, ``split_trajectory``, ``train_test_split``,
``GenericHamiltonianMLP``) directly -- only the physics (F, true_energy,
initial-condition sampling, data generation) is two-body-specific and
defined here. ``VerletIntegrator``/``DirectLeapfrogIntegrator`` are *not*
reused for training: see ``SubsteppedVerletIntegrator`` below for why this
system specifically needs a substepped training-time integrator that those
don't provide.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.autograd import grad

from direct_dynamics import DirectDynamicsMLP
from mexican_hat_dynamics import IntegratorBase, generate_trajectories, split_trajectory, train_test_split  # noqa: F401

# Reduced mass for m1 = m2 = 1: mu = m1*m2/(m1+m2) = 0.5. Kept as a module-level
# constant rather than a Config field since the mass-split arithmetic below
# (q_cm +/- 0.5*r_vec, p1 = 0.5*v_rel, etc.) is hard-derived for equal unit
# masses; generalising to arbitrary m1/m2 would need those formulas reworked.
REDUCED_MASS = 0.5


def F(p: torch.Tensor, q: torch.Tensor, alpha: float) -> torch.Tensor:
    q1x, q1y, q2x, q2y = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    rx, ry = q1x - q2x, q1y - q2y
    r = torch.sqrt(rx**2 + ry**2)
    f1x = -alpha * rx / r**3
    f1y = -alpha * ry / r**3
    dp1dt = torch.cat((f1x, f1y), dim=1)
    dp2dt = -dp1dt
    dq1dt, dq2dt = p[:, 0:2], p[:, 2:4]
    return torch.cat((dp1dt, dp2dt, dq1dt, dq2dt), dim=1)


def true_energy(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y, alpha):
    r = torch.sqrt((q1x - q2x) ** 2 + (q1y - q2y) ** 2)
    return 0.5 * (p1x**2 + p1y**2 + p2x**2 + p2y**2) - alpha / r


def generate_initial_conditions(
    alpha: float, in_conds: int, *,
    r_box: float = 1.5, v_box: float = 1.2, r_min: float = 0.3,
    energy_margin: float = 0.05, l_min: float = 0.15, cm_box: float = 1.0,
    max_attempts: int = 2000,
) -> torch.Tensor:
    """Sample bound (E < 0), non-radial two-body initial conditions and place them in the lab frame.

    Sampling is done in relative coordinates (r_vec = q1 - q2, v_rel = dq1/dt - dq2/dt)
    via rejection sampling (uniform box, reject anything unbound, too close to
    collision, or too close to purely radial infall/near-zero angular momentum
    -- the latter would need much finer time resolution than ``coarsening_factor``
    below provides to integrate stably through periapsis). ``energy_margin``
    keeps samples comfortably bound rather than marginal (E just under 0).

    The centre of mass is placed at a random offset in a box of half-width
    ``cm_box`` and left at rest (v_cm = 0) -- this is what makes translation
    invariance a genuine, non-trivial thing for the network to discover: the
    same relative orbit shape appears at many different absolute (q1, q2).
    Verified (see the session that built this file): with these defaults,
    ``dt=0.1, coarsening_factor=100`` conserves the true trajectory's energy
    to ~1e-4 relative over a full training window.

    ``max_attempts`` caps the rejection-sampling loop -- an incompatible combination
    of bounds (e.g. r_box too small for the required r_min/l_min/energy_margin) has
    near-zero acceptance rate and would otherwise hang forever rather than erroring,
    which is exactly the kind of silent failure you do not want in an unattended,
    multi-hour sweep (found the hard way while preparing one).
    """
    out = torch.empty((in_conds, 8), dtype=torch.float32)
    filled = 0
    attempts = 0
    while filled < in_conds:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"generate_initial_conditions: only filled {filled}/{in_conds} after "
                f"{max_attempts} attempts -- r_box={r_box}, v_box={v_box}, r_min={r_min}, "
                f"energy_margin={energy_margin}, l_min={l_min} are likely incompatible "
                "(near-zero acceptance rate). Widen r_box/v_box or relax r_min/l_min/energy_margin."
            )
        batch = max(in_conds - filled, 16) * 6
        r_vec = r_box * (2 * torch.rand(batch, 2) - 1)
        v_rel = v_box * (2 * torch.rand(batch, 2) - 1)
        r = r_vec.norm(dim=1)
        e_rel = 0.5 * REDUCED_MASS * (v_rel**2).sum(dim=1) - alpha / r.clamp_min(1e-6)
        l = REDUCED_MASS * (r_vec[:, 0] * v_rel[:, 1] - r_vec[:, 1] * v_rel[:, 0])
        keep = (r >= r_min) & (e_rel <= -energy_margin) & (l.abs() >= l_min)
        idx = keep.nonzero(as_tuple=True)[0]
        take = min(int(idx.numel()), in_conds - filled)
        if take == 0:
            continue
        sel = idx[:take]
        q_cm = cm_box * (2 * torch.rand(take, 2) - 1)
        q1 = q_cm + 0.5 * r_vec[sel]
        q2 = q_cm - 0.5 * r_vec[sel]
        p1 = 0.5 * v_rel[sel]
        p2 = -0.5 * v_rel[sel]
        out[filled : filled + take, 0:2] = p1
        out[filled : filled + take, 2:4] = p2
        out[filled : filled + take, 4:6] = q1
        out[filled : filled + take, 6:8] = q2
        filled += take
    return out


# ============================================================
# Analytic (exact) Kepler-orbit data generation.
#
# The old generate_initial_conditions/generate_data pair above samples a random box of
# (r_vec, v_rel) and integrates numerically -- periapsis distance is then whatever it happens
# to be, discovered only after simulating, and a training window that happens not to cross
# periapsis simply never shows the network that regime at all. Since the two-body problem is
# exactly solvable, we don't have to leave this to chance: periapsis distance is a closed-form
# function of the orbital elements (r_peri = a*(1-e)), so it can be *stratified* directly, and
# the exact analytic solution (Kepler's equation) gives ground truth with zero integration
# error at any eccentricity -- no coarsening_factor, no integrator-resolution tradeoffs, and no
# risk of the ground truth itself being wrong near a fast periapsis passage.
# ============================================================


def solve_kepler_equation(
    mean_anomaly: torch.Tensor, eccentricity: torch.Tensor, *, tol: float = 1e-10, max_iter: int = 50,
) -> torch.Tensor:
    """Newton-Raphson solve of Kepler's equation M = E - e*sin(E) for the eccentric anomaly E
    (elliptical orbits, e < 1).

    Verified (session that added this function) against direct fine-step RK4 integration of
    the true two-body equation of motion: in float64, the analytic orbit this feeds into
    matches the RK4 reference to ~1e-13 relative error (floating-point exact) at e=0.6; the
    float32 default here is fine for training data, matching every other system in this
    project.
    """
    M = torch.remainder(mean_anomaly + math.pi, 2 * math.pi) - math.pi  # wrap to [-pi, pi]: stable Newton start
    E = M.clone()
    for _ in range(max_iter):
        f = E - eccentricity * torch.sin(E) - M
        fp = 1 - eccentricity * torch.cos(E)
        step = f / fp
        E = E - step
        if step.abs().max() < tol:
            break
    return E


def orbit_state_at_time(
    a: torch.Tensor, e: torch.Tensor, omega: torch.Tensor, gm_eff: float, t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact planar two-body relative position/velocity (r_vec, v_rel) at time ``t`` since
    periapsis passage (t=0), for a bound orbit with semi-major axis ``a``, eccentricity ``e``,
    and periapsis-direction angle ``omega`` (all broadcastable with ``t``).

    ``gm_eff = alpha / REDUCED_MASS`` is the effective "GM" for the reduced one-body form of
    this problem (mu * r'' = -alpha * rhat / r**2 is exactly the standard Kepler problem with
    GM replaced by gm_eff). Returns ``(x, y, vx, vy)``.
    """
    n = torch.sqrt(torch.as_tensor(gm_eff, dtype=a.dtype, device=a.device) / a**3)  # mean motion
    E = solve_kepler_equation(n * t, e)
    cosE, sinE = torch.cos(E), torch.sin(E)
    x_p = a * (cosE - e)
    y_p = a * torch.sqrt(1 - e**2) * sinE
    denom = 1 - e * cosE
    xdot_p = -a * n * sinE / denom
    ydot_p = a * n * torch.sqrt(1 - e**2) * cosE / denom
    cosw, sinw = torch.cos(omega), torch.sin(omega)
    x = cosw * x_p - sinw * y_p
    y = sinw * x_p + cosw * y_p
    vx = cosw * xdot_p - sinw * ydot_p
    vy = sinw * xdot_p + cosw * ydot_p
    return x, y, vx, vy


def _log_uniform(n: int, low: float, high: float) -> torch.Tensor:
    u = torch.rand(n) * (math.log(high) - math.log(low)) + math.log(low)
    return torch.exp(u)


def sample_stratified_orbits(
    alpha: float, n: int, *, r_peri_min: float, r_peri_max: float, r_apo_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample bound orbital elements with *both* periapsis and apoapsis stratified
    log-uniformly, rather than periapsis alone with an independently-sampled eccentricity.

    Sampling periapsis and eccentricity independently (an earlier version of this function)
    lets apoapsis blow up uncontrollably: r_apo = r_peri*(1+e)/(1-e) reaches ~15 for
    r_peri=1.2, e_max=0.85, even though r_peri itself stays small elsewhere -- found the hard
    way when a network trained on this couldn't get anywhere, because the resulting position/
    velocity magnitudes spanned such a huge dynamic range (up to v~8, |q|~4, versus O(1)
    elsewhere) that the trajectory-fitting MSE loss was dominated by a handful of huge-orbit
    outliers, drowning out the close-approach learning signal the whole redesign was for.

    Sampling r_peri and r_apo directly and independently (both capped) avoids this
    entirely -- orbit *size* (r_apo) is controlled the same deliberate way as how *close* it
    gets (r_peri), and eccentricity/semi-major axis are simply derived from the two:
    a = (r_peri + r_apo)/2, e = (r_apo - r_peri)/(r_apo + r_peri).

    Returns ``(a, e, omega, gm_eff)``, each of shape ``[n]`` except ``gm_eff`` (a scalar).
    """
    gm_eff = alpha / REDUCED_MASS
    r_peri = _log_uniform(n, r_peri_min, r_peri_max)
    r_apo = torch.maximum(_log_uniform(n, r_peri_max, r_apo_max), r_peri * 1.05)
    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri)
    omega = 2 * math.pi * torch.rand(n)
    return a, e, omega, gm_eff


def generate_data_analytic(
    alphas: torch.Tensor, T_window: int, N: int, *,
    dt: float = 0.1, in_conds: int = 8, splits: int = 5,
    r_peri_min: float = 0.08, r_peri_max: float = 1.2, r_apo_max: float = 2.0,
    periapsis_centered_fraction: float = 0.5, cm_box: float = 0.8,
    augment_dataset: bool = False, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate trajectory windows via the exact analytic Kepler solution -- the preferred
    two-body data generator (see the module-level note above); no numerical integration is
    used for ground truth at all, so it is exact through periapsis regardless of eccentricity.

    For each of ``in_conds`` orbits sampled per alpha (periapsis *and* apoapsis stratified via
    ``sample_stratified_orbits`` -- apoapsis is capped too, so a wide periapsis range can't
    produce runaway-sized orbits), generates ``splits`` windows each: a
    ``periapsis_centered_fraction`` of them are deliberately time-shifted (with a little random
    jitter) so periapsis passage falls near the middle of the window -- guaranteeing the
    network actually trains on that orbit's own closest approach, rather than only seeing it
    when a randomly-phased window happens to cross it; the rest use a uniformly random phase
    across the whole orbital period (general diversity, matching the spirit of the old
    box-sampling approach). The centre of mass is placed at a random, per-window offset and
    left at rest, same as before -- what makes translation invariance non-trivial to learn.

    Returns ``(trajectories, params, indices)`` in exactly the format
    ``generic_residuals``/``residuals`` expect: ``trajectories`` has shape
    ``[N, batch, 8]`` = ``[p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y]`` at the ``N`` sampled instants,
    ``indices`` has shape ``[batch, N]`` giving which of the ``T_window`` outer integrator
    steps (always including step 0) each sampled instant corresponds to.
    """
    if device is None:
        device = alphas.device
    K = alphas.shape[0]
    n_centered = int(round(periapsis_centered_fraction * splits))
    traj_batches_all, sampled_idx_all, params_all = [], [], []

    for k in range(K):
        alpha_val = float(alphas[k].item())
        a, e, omega, gm_eff = sample_stratified_orbits(
            alpha_val, in_conds, r_peri_min=r_peri_min, r_peri_max=r_peri_max, r_apo_max=r_apo_max,
        )
        period = 2 * math.pi * torch.sqrt(a**3 / gm_eff)

        batch = in_conds * splits
        a_b = a.repeat_interleave(splits)
        e_b = e.repeat_interleave(splits)
        omega_b = omega.repeat_interleave(splits)
        period_b = period.repeat_interleave(splits)
        split_of = torch.arange(splits).repeat(in_conds)
        is_centered = split_of < n_centered

        centered_start = -0.5 * (T_window - 1) * dt + (torch.rand(batch) - 0.5) * 0.4 * (T_window - 1) * dt
        random_start = torch.rand(batch) * period_b
        t_start = torch.where(is_centered, centered_start, random_start)

        idx = torch.zeros(batch, N, dtype=torch.long)
        if N > 1:
            for b in range(batch):
                idx[b, 1:] = 1 + torch.randperm(T_window - 1)[: N - 1]
            idx, _ = torch.sort(idx, dim=1)

        t = t_start.unsqueeze(1) + idx.to(dtype=dtype) * dt  # [batch, N]
        x, y, vx, vy = orbit_state_at_time(
            a_b.unsqueeze(1), e_b.unsqueeze(1), omega_b.unsqueeze(1), gm_eff, t,
        )

        q_cm = cm_box * (2 * torch.rand(batch, 2) - 1)
        q1x, q1y = q_cm[:, 0:1] + 0.5 * x, q_cm[:, 1:2] + 0.5 * y
        q2x, q2y = q_cm[:, 0:1] - 0.5 * x, q_cm[:, 1:2] - 0.5 * y
        p1x, p1y = 0.5 * vx, 0.5 * vy
        p2x, p2y = -0.5 * vx, -0.5 * vy

        traj_batch = torch.stack([p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y], dim=-1)  # [batch, N, 8]
        traj_batch = traj_batch.permute(1, 0, 2).contiguous().to(device=device, dtype=dtype)  # [N, batch, 8]
        idx_batch = idx.to(device=device)
        params = torch.full((batch, 1), alpha_val, device=device, dtype=dtype)

        if augment_dataset:
            traj_batch = augment_with_random_rotation_and_translation(traj_batch, device, dtype)
            idx_batch = torch.cat([idx_batch, idx_batch], dim=0)
            params = torch.cat([params, params], dim=0)

        traj_batches_all.append(traj_batch)
        sampled_idx_all.append(idx_batch)
        params_all.append(params)

    trajectories = torch.cat(traj_batches_all, dim=1)
    params = torch.cat(params_all, dim=0)
    indices = torch.cat(sampled_idx_all, dim=0)

    n_total = trajectories.size(1)
    perm = torch.randperm(n_total, device=trajectories.device)
    return trajectories[:, perm, :], params[perm, :], indices[perm, :]


def generate_data(
    alphas: torch.Tensor, Ffun, L: int, T_window: int, N: int, *,
    dt: float = 0.1, in_conds: int = 8, coarsening_factor: int = 1,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
    augment_dataset: bool = False, ic_kwargs: Optional[dict] = None,
):
    """Two-body analogue of ``mexican_hat_dynamics.generate_data`` (p0/q0 split at dim 4, not 2).

    ``ic_kwargs`` is forwarded to ``generate_initial_conditions`` (r_box, v_box, r_min,
    energy_margin, l_min, cm_box) -- without this, the Config.ic_* fields in the analysis
    script would silently have no effect, since this function previously always called
    ``generate_initial_conditions`` with its hardcoded defaults regardless of what was
    configured (a real bug, fixed in the same session that added the substepped integrators).
    """
    if device is None:
        device = alphas.device
    ic_kwargs = ic_kwargs or {}
    K = alphas.shape[0]
    traj_batches_all, sampled_idx_all, params_all = [], [], []
    for k in range(K):
        alpha_val = float(alphas[k].item())
        ic = generate_initial_conditions(alpha_val, in_conds, **ic_kwargs)
        p0 = ic[:, :4].to(device=device, dtype=dtype)
        q0 = ic[:, 4:].to(device=device, dtype=dtype)
        traj = generate_trajectories(p0, q0, Ffun, alpha_val, L, dt, coarsening_factor)
        split_traj, sampled_idx = split_trajectory(traj, T_window, N)
        splits, Bk = split_traj.shape[0], traj.shape[1]
        tb = split_traj.permute(1, 0, 2, 3).contiguous().view(N, splits * Bk, split_traj.shape[-1])
        indices = sampled_idx.repeat_interleave(Bk, dim=0)
        params = torch.full((splits * Bk, 1), alpha_val, device=device, dtype=dtype)
        if augment_dataset:
            tb = augment_with_random_rotation_and_translation(tb, device, dtype)
            indices = torch.cat((indices, indices), dim=0)
            params = torch.cat((params, params), dim=0)
        traj_batches_all.append(tb)
        sampled_idx_all.append(indices)
        params_all.append(params)

    trajectories = torch.cat(traj_batches_all, dim=1)
    params = torch.cat(params_all, dim=0)
    indices = torch.cat(sampled_idx_all, dim=0)

    n = trajectories.size(1)
    perm = torch.randperm(n, device=trajectories.device)
    return trajectories[:, perm, :], params[perm, :], indices[perm, :]


def augment_with_random_rotation_and_translation(
    tb: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Double a trajectory batch with a copy transformed by both exact continuous symmetries.

    ``tb`` has layout ``[N, batch, 8]`` = ``[p1x,p1y,p2x,p2y,q1x,q1y,q2x,q2y]``.
    Applies one random SO(2) rotation to *both* (p1,p2) and (q1,q2) together
    (same matrix for both particles), plus an independent random shift applied
    to *both* q1 and q2 together (momenta untouched by translation) -- both
    exact symmetries of this system's true dynamics for every alpha > 0
    (verified in the session that built this file), so both together still
    give a genuinely valid trajectory. One random transform per batch
    element, shared across the whole time window.
    """
    batch = tb.shape[1]
    theta = 2.0 * torch.pi * torch.rand(batch, device=device, dtype=dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack((torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2)  # [batch, 2, 2]
    shift = 1.0 * (2.0 * torch.rand(batch, 2, device=device, dtype=dtype) - 1.0)  # random 2D shift

    tb_aug = tb.clone()

    def rotate_pair(x_slice):
        vec = tb[:, :, x_slice].permute(1, 0, 2)  # [batch, N, 2]
        return torch.bmm(vec, rot.transpose(1, 2)).permute(1, 0, 2)

    tb_aug[:, :, 0:2] = rotate_pair(slice(0, 2))
    tb_aug[:, :, 2:4] = rotate_pair(slice(2, 4))
    q1_rot = rotate_pair(slice(4, 6))
    q2_rot = rotate_pair(slice(6, 8))
    tb_aug[:, :, 4:6] = q1_rot + shift.unsqueeze(0)
    tb_aug[:, :, 6:8] = q2_rot + shift.unsqueeze(0)
    return torch.cat((tb, tb_aug), dim=1)


class SubsteppedVerletIntegrator(IntegratorBase):
    """Same V_net/K_net kick-drift-kick step as ``mexican_hat_dynamics.VerletIntegrator``,
    but each ``.step()`` call internally performs ``substeps`` sub-iterations of size
    ``dt / substeps``, fully differentiable throughout (autograd just sees a longer chain).

    Needed specifically for this system: ground-truth trajectories are generated with a
    fine internal ``coarsening_factor`` for accuracy, but the *training-time* differentiable
    rollout (``generic_residuals``/``residuals`` in every other script here) takes exactly
    one un-substepped step of size ``dt`` per outer trajectory-window index. For a smooth
    polynomial potential that's fine, but verified numerically (session that added this
    class): for this system's 1/r potential, a single dt=0.1 Verlet step already accumulates
    >50% relative trajectory error, for the *true* force law, for a majority of bound orbits
    sampled from the wide initial-condition box -- periapsis passage is simply too fast for
    that step size to resolve. Tightening the initial-condition sampler alone cannot fully
    fix this without also making the dataset unrepresentatively close to circular; combining
    modestly tighter initial conditions (see Config.ic_* recommended defaults) with
    ``substeps=10`` here reduces the same test to <3% relative error for every sampled orbit.
    """

    def __init__(self, model: torch.nn.Module, dt: float, substeps: int = 1):
        super().__init__(model=model, dt=dt)
        self.substeps = substeps

    def step(self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor):
        q, p, params = q.to(self.device), p.to(self.device), params.to(self.device)
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            q.requires_grad_(True)
            p.requires_grad_(True)
            V = self.model.V_net(torch.cat((q, params), dim=1))
            dpdt = -grad(V.sum(), q, create_graph=True)[0]
            p_half = p + dpdt * (sub_dt / 2)
            K = self.model.K_net(p_half)
            dqdt = grad(K.sum(), p_half, create_graph=True)[0]
            q_next = q + dqdt * sub_dt
            q_next.requires_grad_(True)
            V2 = self.model.V_net(torch.cat((q_next, params), dim=1))
            dpdt2 = -grad(V2.sum(), q_next, create_graph=True)[0]
            p_next = p_half + dpdt2 * (sub_dt / 2)
            p, q = p_next, q_next
        return p, q


class SubsteppedDirectLeapfrogIntegrator(torch.nn.Module):
    """Same kick-drift-kick *shape* as ``direct_dynamics.DirectLeapfrogIntegrator``, but
    substepped exactly like ``SubsteppedVerletIntegrator`` above, for the same reason: the
    periapsis-resolution problem is a property of the true dynamics' time scale, not of
    which architecture is being fit, so the direct-MLP architecture needs the same fix."""

    def __init__(self, model: DirectDynamicsMLP, dt: float, substeps: int = 1):
        super().__init__()
        self.model = model
        self.dt = dt
        self.substeps = substeps
        self.device = next(model.parameters()).device

    def step(self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor):
        q, p, params = q.to(self.device), p.to(self.device), params.to(self.device)
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            dpdt, _ = self.model(p, q, params)
            p_half = p + dpdt * (sub_dt / 2)
            _, dqdt = self.model(p_half, q, params)
            q_next = q + dqdt * sub_dt
            dpdt2, _ = self.model(p_half, q_next, params)
            p_next = p_half + dpdt2 * (sub_dt / 2)
            p, q = p_next, q_next
        return p, q
