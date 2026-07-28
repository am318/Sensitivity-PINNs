"""Architecture-agnostic direct-dynamics MLP and matching leapfrog-shaped integrator.

Unlike the ASRNN Hamiltonian-structured networks (a V_net/K_net split whose
symplectic Verlet step architecturally guarantees conservation of the
network's *own* learned energy), this model outputs ``(dp/dt, dq/dt)``
directly, with no conservation structure at all

The integrator below keeps the *same* kick-drift-kick step shape as the
ASRNN ``VerletIntegrator`` (half-step p update, full-step q update using the
half-updated p, half-step p update using the new q) -- only the source of
``(dp/dt, dq/dt)`` changes, from V/K gradients to a direct model call.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectDynamicsMLP(nn.Module):
    """One plain MLP mapping (p, q, alpha) -> (dp/dt, dq/dt) directly."""

    def __init__(
        self,
        p_dim: int,
        q_dim: int,
        param_dim: int,
        hidden_dim: int = 50,
        n_hidden: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.device = device or torch.device("cpu")

        act = nn.SiLU()
        layers = [nn.Linear(p_dim + q_dim + param_dim, hidden_dim), act]
        for _ in range(1, n_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, p_dim + q_dim)]
        self.net = nn.Sequential(*layers).to(self.device)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, q, params = p.to(self.device), q.to(self.device), params.to(self.device)
        out = self.net(torch.cat((p, q, params), dim=1))
        return out[:, : self.p_dim], out[:, self.p_dim :]


class DirectLeapfrogIntegrator(nn.Module):
    """Same kick-drift-kick shape as the ASRNN VerletIntegrator, model outputs derivatives directly."""

    def __init__(self, model: DirectDynamicsMLP, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt
        self.device = next(model.parameters()).device

    def step(
        self, p: torch.Tensor, q: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, p, params = q.to(self.device), p.to(self.device), params.to(self.device)
        dpdt, _ = self.model(p, q, params)
        p_half = p + dpdt * (self.dt / 2)
        _, dqdt = self.model(p_half, q, params)
        q_next = q + dqdt * self.dt
        dpdt2, _ = self.model(p_half, q_next, params)
        p_next = p_half + dpdt2 * (self.dt / 2)
        return p_next, q_next


def generic_residuals(
    trajectories: torch.Tensor,
    params: torch.Tensor,
    instants: torch.Tensor,
    T: int,
    integrator: DirectLeapfrogIntegrator,
    p_dim: int,
) -> torch.Tensor:
    """Trajectory-fitting MSE loss, generalising the ASRNN per-system ``residuals()`` by p_dim.

    ``trajectories`` has layout ``[..., 0:p_dim]`` = p, ``[..., p_dim:2*p_dim]``
    = q, matching both the double-well (p_dim=1) and Henon-Heiles (p_dim=2)
    ASRNN data generators.
    """
    batch_size = trajectories.shape[1]
    n_idx = instants.shape[1]

    p_pred = trajectories[0, :, 0:p_dim].clone().to(integrator.device).requires_grad_(True)
    q_pred = trajectories[0, :, p_dim : 2 * p_dim].clone().to(integrator.device).requires_grad_(True)
    params_dev = params.to(integrator.device)

    p_preds = [p_pred]
    q_preds = [q_pred]
    for _ in range(1, T):
        p_pred, q_pred = integrator.step(p_pred, q_pred, params_dev)
        p_preds.append(p_pred)
        q_preds.append(q_pred)

    p_preds = torch.stack(p_preds)
    q_preds = torch.stack(q_preds)

    idx_b = torch.arange(batch_size, device=integrator.device)
    idx_t = instants.to(device=integrator.device, dtype=torch.long)

    sampled_p = torch.stack([p_preds[idx_t[:, i], idx_b] for i in range(n_idx)], dim=0)
    sampled_q = torch.stack([q_preds[idx_t[:, i], idx_b] for i in range(n_idx)], dim=0)

    p_true = trajectories[:, :, 0:p_dim].to(integrator.device)
    q_true = trajectories[:, :, p_dim : 2 * p_dim].to(integrator.device)

    loss_p = torch.mean((sampled_p - p_true) ** 2)
    loss_q = torch.mean((sampled_q - q_true) ** 2)
    return loss_p + loss_q
