"""2D isotropic double well ("Mexican hat" potential):

    V(q1, q2; alpha) = 0.5*alpha*r^2 + 0.25*r^4,   r^2 = q1^2 + q2^2,
    dq/dt = p,
    dp1/dt = F1 = -q1*(alpha + r^2),
    dp2/dt = F2 = -q2*(alpha + r^2).

Unlike Henon-Heiles, V depends on q only through r^2 = q1^2+q2^2, so it is
invariant under continuous rotation for *every* alpha -- the direct
2D generalisation of the existing double well (alpha < 0 gives a ring of
minima instead of two points), used to test a continuous X_rot
generator.

Mirrors the structure of the ASRNN Double_Well_Code/Henon_Heiles_Code
``helper.py`` files (kick-drift-kick Verlet integrator, sparse/noisy
trajectory generator) but with a dimension-generic Hamiltonian MLP (the
ASRNN classes hardcode V_net/K_net input dims for their specific systems) and
a simpler uniform-box-plus-energy-cap initial-condition sampler (adequate for
a system outside that repo; no need to match its stratified rejection-sampling
convention).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
# escnn (E(n)-equivariant steerable networks, cloned locally) hard-imports
# lie_learn at package load time purely for SO(3) Wigner-D matrices, a
# dependency whose bundled C extension doesn't build against modern numpy.
# We only ever use escnn's SO(2) machinery (NormPool, Linear,
# PointwiseNonLinearity, no SO(3)/spherical-harmonics codepath is exercised),
# so a stub package satisfying the one unconditional import is sufficient --
# see vendor_stubs/lie_learn/representations/SO3/wigner_d.py.
sys.path.insert(0, str(_ROOT / "vendor_stubs"))
sys.path.insert(0, str(_ROOT / "escnn"))
from escnn import gspaces  # noqa: E402
from escnn import group  # noqa: E402
from escnn import nn as enn  # noqa: E402
from torch.autograd import grad


def F(p: torch.Tensor, q: torch.Tensor, alpha: float) -> torch.Tensor:
    q1, q2 = q[:, 0:1], q[:, 1:2]
    p1, p2 = p[:, 0:1], p[:, 1:2]
    r2 = q1**2 + q2**2
    dp1dt = -q1 * (alpha + r2)
    dp2dt = -q2 * (alpha + r2)
    dq1dt = p1
    dq2dt = p2
    return torch.cat((dp1dt, dp2dt, dq1dt, dq2dt), dim=1)


def true_energy(p1, p2, q1, q2, alpha):
    r2 = q1**2 + q2**2
    return 0.5 * (p1**2 + p2**2) + 0.5 * alpha * r2 + 0.25 * r2**2


def generate_initial_conditions(alpha: float, in_conds: int, *, box: float = 2.0, energy_max: float = 3.0) -> torch.Tensor:
    """Uniform box + energy cap rejection sampling (simpler than the ASRNN repo's stratified sampler)."""
    dat = torch.empty((in_conds, 4), dtype=torch.float32)
    filled = 0
    while filled < in_conds:
        batch = max(in_conds - filled, 16) * 4
        p = box * (2 * torch.rand(batch, 2) - 1)
        q = box * (2 * torch.rand(batch, 2) - 1)
        e = true_energy(p[:, 0], p[:, 1], q[:, 0], q[:, 1], alpha)
        keep = e <= energy_max
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        take = min(n_keep, in_conds - filled)
        dat[filled : filled + take, :2] = p[keep][:take]
        dat[filled : filled + take, 2:] = q[keep][:take]
        filled += take
    return dat


def _like(t: torch.Tensor):
    return dict(device=t.device, dtype=t.dtype)


def generate_trajectories(
    p0: torch.Tensor, q0: torch.Tensor, Ffun, alpha: float, T: int, dt: float, coarsening_factor: int = 1
) -> torch.Tensor:
    fine = torch.empty((T * coarsening_factor, p0.shape[0], 2 * p0.shape[1]), **_like(p0))
    dtau = dt / coarsening_factor
    p, q = p0, q0
    dim = p0.shape[1]
    time_drvt = Ffun(p, q, alpha)
    dpdt = time_drvt[:, :dim]
    for i in range(T * coarsening_factor):
        p_half = p + dpdt * (dtau / 2)
        fine[i, :, :dim] = p
        fine[i, :, dim:] = q
        time_drvt = Ffun(p_half, q, alpha)
        dqdt = time_drvt[:, dim:]
        q_next = q + dqdt * dtau
        time_drvt = Ffun(p_half, q_next, alpha)
        dpdt = time_drvt[:, :dim]
        p_next = p_half + dpdt * (dtau / 2)
        p, q = p_next, q_next
    return fine[torch.arange(T, device=p0.device) * coarsening_factor, :, :]


def split_trajectory(trajectory: torch.Tensor, T_window: int, N: int):
    L = len(trajectory)
    splits = L - T_window + 1
    B, D = trajectory.shape[1], trajectory.shape[2]
    out = torch.empty((splits, N, B, D), **_like(trajectory))
    idx = torch.empty((splits, N), dtype=torch.long, device=trajectory.device)
    for i in range(splits):
        idx[i, 0] = 0
        if N > 1:
            rand = torch.randperm(T_window - 1, device=trajectory.device)[: N - 1] + 1
            idx[i, 1:] = rand
        idx[i] = idx[i].sort()[0]
        window = trajectory[i : i + T_window]
        out[i] = window[idx[i]]
    return out, idx


def generate_data(
    alphas: torch.Tensor, Ffun, L: int, T_window: int, N: int, *,
    dt: float = 0.1, in_conds: int = 8, coarsening_factor: int = 1,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
):
    if device is None:
        device = alphas.device
    K = alphas.shape[0]
    traj_batches_all, sampled_idx_all, params_all = [], [], []
    for k in range(K):
        alpha_val = float(alphas[k].item())
        ic = generate_initial_conditions(alpha_val, in_conds)
        p0 = ic[:, :2].to(device=device, dtype=dtype)
        q0 = ic[:, 2:].to(device=device, dtype=dtype)
        traj = generate_trajectories(p0, q0, Ffun, alpha_val, L, dt, coarsening_factor)
        split_traj, sampled_idx = split_trajectory(traj, T_window, N)
        splits, Bk = split_traj.shape[0], traj.shape[1]
        tb = split_traj.permute(1, 0, 2, 3).contiguous().view(N, splits * Bk, split_traj.shape[-1])
        traj_batches_all.append(tb)
        sampled_idx_all.append(sampled_idx.repeat_interleave(Bk, dim=0))
        params_all.append(torch.full((splits * Bk, 1), alpha_val, device=device, dtype=dtype))

    trajectories = torch.cat(traj_batches_all, dim=1)
    params = torch.cat(params_all, dim=0)
    indices = torch.cat(sampled_idx_all, dim=0)

    n = trajectories.size(1)
    perm = torch.randperm(n, device=trajectories.device)
    return trajectories[:, perm, :], params[perm, :], indices[perm, :]


def train_test_split(trajectories, params, indices, val_size: float = 0.25):
    from sklearn.model_selection import train_test_split as sk_train_test_split

    N = trajectories.shape[1]
    all_idx = torch.arange(N)
    tr_idx, va_idx = sk_train_test_split(all_idx, test_size=val_size, random_state=42)
    return (
        (trajectories[:, tr_idx, :], params[tr_idx, :], indices[tr_idx, :]),
        (trajectories[:, va_idx, :], params[va_idx, :], indices[va_idx, :]),
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 50, n_hidden: int = 1):
        super().__init__()
        act = nn.Tanh()
        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(1, n_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class GenericHamiltonianMLP(nn.Module):
    """Same V_net/K_net split as the ASRNN Hamiltonian_MLP_Network classes, but dimension-generic
    (those classes hardcode their system's specific q_dim/p_dim/param_dim)."""

    def __init__(
        self, q_dim: int, p_dim: int, param_dim: int,
        kin_hidden_dim: int, kin_n_hidden: int, pot_hidden_dim: int, pot_n_hidden: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.V_net = MLP(in_dim=q_dim + param_dim, out_dim=1, hidden_dim=pot_hidden_dim, n_hidden=pot_n_hidden).to(self.device)
        self.K_net = MLP(in_dim=p_dim, out_dim=1, hidden_dim=kin_hidden_dim, n_hidden=kin_n_hidden).to(self.device)

    def forward(self, p, q, params):
        p, q, params = p.to(self.device), q.to(self.device), params.to(self.device)
        return self.K_net(p), self.V_net(torch.cat((q, params), dim=1))


class IntegratorBase(nn.Module):
    def __init__(self, model: nn.Module, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt
        self.device = next(model.parameters()).device


class VerletIntegrator(IntegratorBase):
    def step(self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor):
        q, p, params = q.to(self.device), p.to(self.device), params.to(self.device)
        q.requires_grad_(True)
        p.requires_grad_(True)
        V = self.model.V_net(torch.cat((q, params), dim=1))
        dpdt = -grad(V.sum(), q, create_graph=True)[0]
        p_half = p + dpdt * (self.dt / 2)
        K = self.model.K_net(p_half)
        dqdt = grad(K.sum(), p_half, create_graph=True)[0]
        q_next = q + dqdt * self.dt
        q_next.requires_grad_(True)
        V2 = self.model.V_net(torch.cat((q_next, params), dim=1))
        dpdt2 = -grad(V2.sum(), q_next, create_graph=True)[0]
        p_next = p_half + dpdt2 * (self.dt / 2)
        return p_next, q_next


def residuals(train_trajectories, train_params, train_instants, T, integrator):
    batch_size = train_trajectories.shape[1]
    N_idx = train_instants.shape[1]
    p_pred = train_trajectories[0, :, 0:2].clone().to(integrator.device).requires_grad_(True)
    q_pred = train_trajectories[0, :, 2:4].clone().to(integrator.device).requires_grad_(True)
    params = train_params.to(integrator.device)

    p_preds, q_preds = [p_pred], [q_pred]
    for _ in range(1, T):
        p_pred, q_pred = integrator.step(p_pred, q_pred, params)
        p_preds.append(p_pred)
        q_preds.append(q_pred)
    p_preds, q_preds = torch.stack(p_preds), torch.stack(q_preds)

    idx_b = torch.arange(batch_size, device=integrator.device)
    idx_t = train_instants.to(device=integrator.device, dtype=torch.long)
    sampled_p = torch.stack([p_preds[idx_t[:, i], idx_b] for i in range(N_idx)], dim=0)
    sampled_q = torch.stack([q_preds[idx_t[:, i], idx_b] for i in range(N_idx)], dim=0)

    p_true = train_trajectories[:, :, 0:2].to(integrator.device)
    q_true = train_trajectories[:, :, 2:4].to(integrator.device)
    return torch.mean((sampled_p - p_true) ** 2) + torch.mean((sampled_q - q_true) ** 2)


# ============================================================
# Genuinely equivariant architecture (escnn), Experiment 2's third arm.
#
# For a network built with escnn's typed equivariant layers, invariance of
# the scalar output holds identically for *every* parameter value, not just
# a specially-trained one -- so, unlike the Hamiltonian-split or direct-MLP
# architectures, sensitivity equivariance (E_i, PDF sec. 0.6) is predicted to
# be ~0 even at random initialisation, with nothing left for training to do.
# Verified numerically before use (see the module docstring below and the
# session's manual checks): force equivariance residual and E_i are both
# ~1e-7 (float32 noise) at random init, against ~1-2 for the other two
# architectures.
# ============================================================


class InvariantScalarMLP(nn.Module):
    """An SO(2)-invariant scalar function of a 2D vector plus optional extra invariant scalars.

    Built from escnn's ``NormPool`` (exact: the norm of a field is always
    exactly invariant, by definition, not an approximation) followed by
    ordinary ``Linear``/``PointwiseNonLinearity`` layers acting purely on
    trivial-representation (already-invariant) channels, where any linear
    map or pointwise nonlinearity trivially preserves invariance. No
    approximate/discretised equivariance (e.g. Fourier-sampled nonlinearities)
    is used anywhere, so invariance is exact to floating-point precision for
    every parameter value, not just a trained one.
    """

    def __init__(
        self, gspace, G, n_extra_scalars: int, hidden_dim: int, n_hidden: int, device=None
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.vector_type = gspace.type(G.standard_representation())
        self.norm_pool = enn.NormPool(self.vector_type)

        in_dim = 1 + n_extra_scalars
        scalar_in_type = gspace.type(*([G.trivial_representation] * in_dim))
        layers = []
        current_type = scalar_in_type
        for _ in range(n_hidden):
            hidden_type = gspace.type(*([G.trivial_representation] * hidden_dim))
            layers.append(enn.Linear(current_type, hidden_type))
            layers.append(enn.PointwiseNonLinearity(hidden_type, function="p_relu"))
            current_type = hidden_type
        out_type = gspace.type(G.trivial_representation)
        layers.append(enn.Linear(current_type, out_type))
        self.mlp = enn.SequentialModule(*layers).to(self.device)
        self.scalar_in_type = scalar_in_type

    def forward(self, vector: torch.Tensor, extra_scalars: torch.Tensor | None = None) -> torch.Tensor:
        v_gt = enn.GeometricTensor(vector, self.vector_type)
        vnorm = self.norm_pool(v_gt).tensor
        combined = torch.cat([vnorm, extra_scalars], dim=1) if extra_scalars is not None else vnorm
        combined_gt = enn.GeometricTensor(combined, self.scalar_in_type)
        return self.mlp(combined_gt).tensor


class _VNetAdapter(nn.Module):
    """Adapts InvariantScalarMLP to the model.V_net(cat(q, params)) calling convention."""

    def __init__(self, inner: InvariantScalarMLP):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x[:, :2], x[:, 2:])


class _KNetAdapter(nn.Module):
    """Adapts InvariantScalarMLP to the model.K_net(p) calling convention."""

    def __init__(self, inner: InvariantScalarMLP):
        super().__init__()
        self.inner = inner

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.inner(p, None)


class EquivariantHamiltonianMLP(nn.Module):
    """Drop-in replacement for GenericHamiltonianMLP with an escnn-built, exactly SO(2)-invariant V_net/K_net.

    Exposes the same ``.V_net``/``.K_net`` interface (each taking one
    concatenated tensor and returning a ``[batch, 1]`` scalar) as
    GenericHamiltonianMLP, so it plugs into the existing VerletIntegrator,
    ``residuals``, and analysis code unchanged.
    """

    def __init__(
        self,
        kin_hidden_dim: int,
        kin_n_hidden: int,
        pot_hidden_dim: int,
        pot_n_hidden: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        v_inner = InvariantScalarMLP(
            self.gspace, self.G, n_extra_scalars=1,
            hidden_dim=pot_hidden_dim, n_hidden=pot_n_hidden, device=self.device,
        )
        k_inner = InvariantScalarMLP(
            self.gspace, self.G, n_extra_scalars=0,
            hidden_dim=kin_hidden_dim, n_hidden=kin_n_hidden, device=self.device,
        )
        self.V_net = _VNetAdapter(v_inner)
        self.K_net = _KNetAdapter(k_inner)

    def forward(self, p, q, params):
        p, q, params = p.to(self.device), q.to(self.device), params.to(self.device)
        return self.K_net(p), self.V_net(torch.cat((q, params), dim=1))
