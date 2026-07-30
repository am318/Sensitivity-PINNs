"""c_i-guided "surgical" fine-tuning: does perturbing the parameters tangent_projection
identifies as responsible for the rotation-equivariance violation actually reduce it faster
than perturbing an equal-size random or low-|c_i| subset?

Starting from an already-trained checkpoint (hamiltonian architecture, Mexican-hat), computes
xrot_score (alpha-averaged |c_i|) exactly as in asrnn_mexican_hat_symmetry_sensitivity.py, picks
K = round(participation_ratio) among V_net parameters, then runs four parallel fine-tunes from
the *same* starting point, each with all but one K-sized subset of V_net frozen:

  top-K      -- the hypothesis: these should be the efficient ones.
  bottom-K   -- inverse control: should barely help.
  random-K   -- baseline control.
  all        -- unrestricted continued training (upper-bound reference; K_net included).

Freezing is implemented via gradient masking (zero the .grad of frozen positions after
backward(), before optimizer.step()) rather than requires_grad, since the selected sets are
individual elements scattered within weight tensors, not whole parameter tensors.

Two objectives, run separately (--objective a|b):
  a: continue the ordinary trajectory-fitting loss, restricted to the unfrozen subset.
  b: trajectory loss + an explicit rotation-violation penalty (mean-squared X_rot F over the
     probe grid, the same quantity xrot_score's target is built from), restricted to the
     unfrozen subset -- the more direct causal test of whether these parameters can actually
     close the gap when explicitly asked to.

The violation penalty needs to be evaluated every fine-tuning step, so it's computed with a
fast, batched (vectorized) reimplementation of rotation_generator_target/evaluate_at_points
(the existing per-point Python loop is far too slow to call thousands of times) -- verified
to match the existing, already-validated point-by-point implementation exactly (max abs diff
0.0 on a random test case) before being trusted here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import asrnn_mexican_hat_symmetry_sensitivity as base
from experiment_common import parameter_layout, select_device, select_dtype
from mexican_hat_dynamics import residuals
from sensitivity_tools import participation_ratio

STRATEGIES = ("top_k", "bottom_k", "random_k", "all")


def batched_rotation_violation(model: torch.nn.Module, q1: torch.Tensor, q2: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fast, batched X_rot F -- verified against evaluate_at_points/rotation_generator_target
    (the existing, slow point-by-point implementation) to match exactly before use here."""
    q = torch.stack([q1, q2], dim=1).clone().requires_grad_(True)
    alpha_t = torch.full((q.shape[0], 1), float(alpha), device=q.device, dtype=q.dtype)
    potential = model.V_net(torch.cat((q, alpha_t), dim=1))
    force = -torch.autograd.grad(potential.sum(), q, create_graph=True)[0]
    d_f1_dq = torch.autograd.grad(force[:, 0].sum(), q, create_graph=True, retain_graph=True)[0]
    d_f2_dq = torch.autograd.grad(force[:, 1].sum(), q, create_graph=True)[0]
    spatial_jac = torch.stack([d_f1_dq, d_f2_dq], dim=1)
    omega_q = torch.stack([-q2, q1], dim=1)
    directional = torch.einsum("nkj,nj->nk", spatial_jac, omega_q)
    omega = base.ROTATION_LIE_GENERATOR.to(device=force.device, dtype=force.dtype)
    return directional - force @ omega.T


def load_checkpoint(checkpoint_dir: Path, device: torch.device, dtype: torch.dtype):
    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg = base.Config(**{k: v for k, v in cfg_dict.items() if k in base.Config.__dataclass_fields__})
    model, integrator = base.build_model(cfg, device, dtype)
    state_dict = torch.load(checkpoint_dir / "final_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    return cfg, model, integrator, state_dict


def compute_xrot_score(model, cfg, device, dtype, flat_names, parameter_slices) -> np.ndarray:
    result = base.analyse_checkpoint(model, 0, cfg, device, dtype, flat_names, parameter_slices)
    return np.asarray(result["xrot_score"])


def build_masks(
    xrot_score: np.ndarray, vnet_mask: np.ndarray, k: int, seed: int = 0
) -> dict[str, np.ndarray]:
    p = xrot_score.shape[0]
    vnet_indices = np.where(vnet_mask)[0]
    order = vnet_indices[np.argsort(xrot_score[vnet_indices])]  # ascending by |c_i|
    bottom_k_idx = order[:k]
    top_k_idx = order[-k:]
    rng = np.random.default_rng(seed)
    random_k_idx = rng.choice(vnet_indices, size=k, replace=False)

    masks = {}
    for name, idx in (("top_k", top_k_idx), ("bottom_k", bottom_k_idx), ("random_k", random_k_idx)):
        m = np.zeros(p, dtype=bool)
        m[idx] = True
        masks[name] = m
    masks["all"] = np.ones(p, dtype=bool)
    return masks


def mask_to_param_tensors(mask_flat: np.ndarray, parameter_slices: dict[str, slice], model: torch.nn.Module):
    mask_t = torch.from_numpy(mask_flat)
    out = {}
    for name, param in model.named_parameters():
        sl = parameter_slices[name]
        out[name] = mask_t[sl].reshape(param.shape).to(dtype=torch.float32)
    return out


def evaluate_metrics(model, integrator, cfg, device, dtype, flat_names, parameter_slices, train_data) -> dict[str, float]:
    result = base.analyse_checkpoint(model, 0, cfg, device, dtype, flat_names, parameter_slices)
    rows = result["alpha_results"]
    vnet_mask = np.array([n.startswith("V_net.") for n in flat_names])
    mean_ei = float(np.mean([np.asarray(r["rotation_equivariance_error_by_parameter"])[vnet_mask] for r in rows]))
    mean_xrot_proj_err = float(np.mean([r["xrot_projection_error"] for r in rows]))
    mean_force_residual = float(np.mean([r["rotation_force_residual"] for r in rows]))

    trajectories, params, instants = train_data
    # VerletIntegrator.step differentiates V w.r.t. q internally even at "eval" time
    # (it computes force via autograd), so this must NOT be wrapped in torch.no_grad().
    traj_loss = float(residuals(trajectories, params, instants, cfg.trajectory_window, integrator=integrator).detach())
    return {
        "mean_ei_vnet": mean_ei,
        "xrot_projection_error": mean_xrot_proj_err,
        "rotation_force_residual": mean_force_residual,
        "trajectory_loss": traj_loss,
    }


def run_finetune(
    state_dict, cfg, device, dtype, mask_tensors: dict[str, torch.Tensor],
    train_data, objective: str, finetune_steps: int, lambda_sym: float, lr: float,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    model, integrator = base.build_model(cfg, device, dtype)
    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    probe_q1, probe_q2 = base.build_probe_grid(cfg, device, dtype)
    trajectories, params, instants = train_data

    for _ in tqdm(range(finetune_steps), desc=f"finetune ({objective})", leave=False):
        optimizer.zero_grad()
        traj_loss = residuals(trajectories, params, instants, cfg.trajectory_window, integrator=integrator)
        if objective == "a":
            loss = traj_loss
        else:
            sym_loss = sum(
                (batched_rotation_violation(model, probe_q1, probe_q2, alpha) ** 2).mean()
                for alpha in cfg.analysis_alphas
            ) / len(cfg.analysis_alphas)
            loss = traj_loss + lambda_sym * sym_loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.mul_(mask_tensors[name].to(param.grad.device))
        optimizer.step()
    return model, integrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=["a", "b"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finetune-steps", type=int, default=3000)
    parser.add_argument("--lambda-sym", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = select_dtype("float32")

    cfg, model, integrator, state_dict = load_checkpoint(args.checkpoint_dir, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    vnet_mask = np.array([n.startswith("V_net.") for n in flat_names])

    print("Computing xrot_score at the starting checkpoint...")
    xrot_score = compute_xrot_score(model, cfg, device, dtype, flat_names, parameter_slices)
    k = max(1, round(participation_ratio(xrot_score[vnet_mask])))
    print(f"K (participation ratio, V_net only) = {k}")

    masks = build_masks(xrot_score, vnet_mask, k)

    print("Generating fine-tuning trajectory data (same alphas/sampling as the checkpoint's training config)...")
    train_data, _ = base.make_dataset(cfg, device, dtype)

    print("Evaluating BEFORE metrics...")
    before_metrics = evaluate_metrics(model, integrator, cfg, device, dtype, flat_names, parameter_slices, train_data)
    print("Before:", before_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "objective": args.objective,
        "k": k,
        "finetune_steps": args.finetune_steps,
        "lambda_sym": args.lambda_sym,
        "before": before_metrics,
        "strategies": {},
    }

    for strategy in STRATEGIES:
        print(f"=== strategy: {strategy} ===")
        mask_tensors = mask_to_param_tensors(masks[strategy], parameter_slices, model)
        finetuned_model, finetuned_integrator = run_finetune(
            state_dict, cfg, device, dtype, mask_tensors, train_data,
            args.objective, args.finetune_steps, args.lambda_sym, args.lr,
        )
        after_metrics = evaluate_metrics(
            finetuned_model, finetuned_integrator, cfg, device, dtype, flat_names, parameter_slices, train_data
        )
        print(f"After ({strategy}):", after_metrics)
        results["strategies"][strategy] = after_metrics

    (args.output_dir / "finetune_results.json").write_text(json.dumps(results, indent=2))
    plot_comparison(results, args.output_dir)
    print(f"Finished. Results written to {args.output_dir.resolve()}")


def plot_comparison(results: dict[str, Any], output_dir: Path) -> None:
    metrics = ["mean_ei_vnet", "xrot_projection_error", "rotation_force_residual", "trajectory_loss"]
    labels = [r"mean $E_i$ ($V_{\rm net}$)", r"$X_{\rm rot}F$ projection error", "rotation force residual", "trajectory loss"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    x = np.arange(len(STRATEGIES))
    for ax, metric, label in zip(axes, metrics, labels):
        before = results["before"][metric]
        after_vals = [results["strategies"][s][metric] for s in STRATEGIES]
        ax.axhline(before, color="0.4", linestyle="--", linewidth=1, label="before (all strategies start here)")
        ax.bar(x, after_vals, color=["tab:green", "tab:red", "tab:gray", "tab:blue"])
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
        ax.set(title=label, yscale="log")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"c_i-guided fine-tuning (objective {results['objective']}, K={results['k']})")
    fig.savefig(output_dir / "finetune_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "finetune_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
