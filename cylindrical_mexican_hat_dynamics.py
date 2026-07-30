"""3D "cylindrical Mexican hat": rotation (SO(2) in x,y) + translation (in z), together.

    V(q1, q2, q3; alpha) = 0.5*alpha*r^2 + 0.25*r^4,   r^2 = q1^2 + q2^2,
    dq/dt = p,
    dp1/dt = F1 = -q1*(alpha + r^2),
    dp2/dt = F2 = -q2*(alpha + r^2),
    dp3/dt = F3 = 0.

V has no q3-dependence at all, so this system has *two* independent, exact
continuous symmetries simultaneously: SO(2) rotation in the (q1, q2) plane
(inherited unchanged from the 2D Mexican hat), and continuous translation in
q3 (free/ignorable coordinate -- p3 is exactly conserved, matching Noether's
theorem for the manifestly q3-independent potential). Physically this is a
particle in a cylindrically-symmetric channel/trap.

Reuses the dimension-generic pieces of ``mexican_hat_dynamics.py``
(``generate_trajectories``, ``split_trajectory``, ``train_test_split``,
``GenericHamiltonianMLP``, ``VerletIntegrator``) and ``direct_dynamics.py``
(``DirectDynamicsMLP``, ``DirectLeapfrogIntegrator``, ``generic_residuals``)
directly -- only the physics (F, true_energy, initial-condition sampling,
data generation) is 3D-specific and defined here.
"""

from __future__ import annotations

from typing import Optional

import torch

from mexican_hat_dynamics import generate_trajectories, split_trajectory, train_test_split  # noqa: F401


def F(p: torch.Tensor, q: torch.Tensor, alpha: float) -> torch.Tensor:
    q1, q2, q3 = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    p1, p2, p3 = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    r2 = q1**2 + q2**2
    dp1dt = -q1 * (alpha + r2)
    dp2dt = -q2 * (alpha + r2)
    dp3dt = torch.zeros_like(q3)
    dq1dt, dq2dt, dq3dt = p1, p2, p3
    return torch.cat((dp1dt, dp2dt, dp3dt, dq1dt, dq2dt, dq3dt), dim=1)


def true_energy(p1, p2, p3, q1, q2, q3, alpha):
    r2 = q1**2 + q2**2
    return 0.5 * (p1**2 + p2**2 + p3**2) + 0.5 * alpha * r2 + 0.25 * r2**2


def generate_initial_conditions(alpha: float, in_conds: int, *, box: float = 2.0, energy_max: float = 3.0) -> torch.Tensor:
    """Uniform box + energy cap rejection sampling, 3D analogue of the 2D Mexican-hat sampler."""
    dat = torch.empty((in_conds, 6), dtype=torch.float32)
    filled = 0
    while filled < in_conds:
        batch = max(in_conds - filled, 16) * 4
        p = box * (2 * torch.rand(batch, 3) - 1)
        q = box * (2 * torch.rand(batch, 3) - 1)
        e = true_energy(p[:, 0], p[:, 1], p[:, 2], q[:, 0], q[:, 1], q[:, 2], alpha)
        keep = e <= energy_max
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        take = min(n_keep, in_conds - filled)
        dat[filled : filled + take, :3] = p[keep][:take]
        dat[filled : filled + take, 3:] = q[keep][:take]
        filled += take
    return dat


def generate_data(
    alphas: torch.Tensor, Ffun, L: int, T_window: int, N: int, *,
    dt: float = 0.1, in_conds: int = 8, coarsening_factor: int = 1,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
    augment_dataset: bool = False,
):
    """3D analogue of ``mexican_hat_dynamics.generate_data`` (p0/q0 split at dim 3, not 2)."""
    if device is None:
        device = alphas.device
    K = alphas.shape[0]
    traj_batches_all, sampled_idx_all, params_all = [], [], []
    for k in range(K):
        alpha_val = float(alphas[k].item())
        ic = generate_initial_conditions(alpha_val, in_conds)
        p0 = ic[:, :3].to(device=device, dtype=dtype)
        q0 = ic[:, 3:].to(device=device, dtype=dtype)
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

    ``tb`` has layout ``[N, batch, 6]`` = ``[p1, p2, p3, q1, q2, q3]``. Applies
    a random SO(2) rotation to (p1,p2)/(q1,q2) *and* an independent random
    shift to q3 (p3 untouched -- translation in q3 doesn't affect momenta),
    both exact symmetries of this system's true dynamics for every alpha
    (verified in the session that built this file), so both together still
    give a genuinely valid trajectory. One random transform per batch
    element, shared across the whole time window.
    """
    batch = tb.shape[1]
    theta = 2.0 * torch.pi * torch.rand(batch, device=device, dtype=dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack((torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2)  # [batch, 2, 2]
    shift = 2.0 * torch.rand(batch, device=device, dtype=dtype) - 1.0  # random z-shift in [-1, 1]

    tb_aug = tb.clone()
    p_xy = tb[:, :, :2].permute(1, 0, 2)  # [batch, N, 2]
    tb_aug[:, :, :2] = torch.bmm(p_xy, rot.transpose(1, 2)).permute(1, 0, 2)
    q_xy = tb[:, :, 3:5].permute(1, 0, 2)
    tb_aug[:, :, 3:5] = torch.bmm(q_xy, rot.transpose(1, 2)).permute(1, 0, 2)
    tb_aug[:, :, 5] = tb[:, :, 5] + shift.unsqueeze(0)
    return torch.cat((tb, tb_aug), dim=1)
