"""One-off experiment: redo the c_i-vs-b_i analysis on an already-trained mexican-hat
checkpoint using a rotationally-symmetric POLAR probe grid instead of the default square
Cartesian one -- no retraining needed, since g/delta/c_i/b_i are all post-hoc diagnostics
computed from a fixed set of model weights.

Why: on the default square grid, cos(g, delta) is empirically ~-0.1 (not ~0 as the exact
proof <delta, X delta> = 0 predicts) and this does NOT shrink under grid refinement -- it's a
domain-boundary artefact (the proof needs a rotation-invariant integration domain; a square
isn't one). A polar (disk) grid is rotation-invariant and was verified to recover
cos(g, delta) ~ 1e-14. This script reruns the analysis on such a grid, for both the random
init and the final trained checkpoint (matching config.json's seed, so "random init" is
reproduced exactly, not re-derived), to see the c_i-vs-b_i relationship uncontaminated by
that artefact.

Usage:
    python3 run_polar_probe_experiment.py outputs/asrnn_mexican_hat_symmetry
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch

import asrnn_mexican_hat_symmetry_sensitivity as m
from experiment_common import parameter_layout, select_device, select_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Existing square-grid run to reanalyse with a polar probe grid.")
    parser.add_argument("--polar-n-radii", type=int, default=15)
    parser.add_argument("--polar-n-angles", type=int, default=32)
    args = parser.parse_args()

    saved_fields = json.loads((args.source_dir / "config.json").read_text())
    known = {f.name for f in fields(m.Config)}
    cfg = m.Config(**{k: v for k, v in saved_fields.items() if k in known})
    cfg = replace(
        cfg,
        probe_grid_shape="polar",
        polar_n_radii=args.polar_n_radii,
        polar_n_angles=args.polar_n_angles,
        output_dir=str(args.source_dir).rstrip("/") + "_polar_probe",
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = select_device(cfg.device)
    dtype = select_dtype(cfg.dtype)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, _ = m.build_model(cfg, device, dtype)
    flat_names, parameter_slices = parameter_layout(model)
    print(f"Model parameters: {len(flat_names):,}; probe grid: polar, "
          f"{cfg.polar_n_radii} radii x {cfg.polar_n_angles} angles = {cfg.polar_n_radii * cfg.polar_n_angles} points")

    print("Analysing random init...")
    init_result = m.analyse_checkpoint(model, 0, cfg, device, dtype, flat_names, parameter_slices)

    trained_step = max(json.loads((args.source_dir / "config.json").read_text())["checkpoint_steps"])
    model.load_state_dict(torch.load(args.source_dir / "final_model.pt", map_location=device))
    print(f"Analysing trained checkpoint (step {trained_step})...")
    trained_result = m.analyse_checkpoint(model, trained_step, cfg, device, dtype, flat_names, parameter_slices)

    all_results = [init_result, trained_result]
    for result in all_results:
        m.write_checkpoint_outputs(result, output_dir)
    (output_dir / "all_checkpoint_results.json").write_text(json.dumps(all_results, indent=2))
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    print("Plotting...")
    m.plot_equivariance_by_module(all_results, output_dir)
    m.plot_equivariance_scatter(all_results, parameter_slices, output_dir)
    m.plot_magnitude_diagnostics(all_results, parameter_slices, output_dir)
    m.plot_module_attribution(all_results, parameter_slices, output_dir)
    m.plot_attribution_scatter(all_results, parameter_slices, output_dir)
    m.plot_module_b_attribution(all_results, parameter_slices, output_dir)
    m.plot_b_attribution_scatter(all_results, parameter_slices, output_dir)
    m.plot_module_c_vs_b_comparison(all_results, parameter_slices, output_dir)
    m.plot_c_vs_b_scatter(all_results, parameter_slices, output_dir)
    m.plot_c_vs_b_scatter_signed(all_results, parameter_slices, output_dir)
    m.plot_module_symmetry_comparison(all_results, parameter_slices, output_dir)
    m.plot_symmetry_attribution_scatter(all_results, parameter_slices, output_dir)

    # Direct <g,delta> orthogonality check on this polar grid (same diagnostic used to
    # discover/confirm the square-grid artefact), for both checkpoints, at every analysis alpha.
    print("\ncos(g, delta) on the polar grid, per alpha (expect ~1e-14, not the square grid's ~-0.1):")
    q1_grid, q2_grid = m.build_probe_grid(cfg, device, dtype)
    for step, state in ((0, None), (trained_step, args.source_dir / "final_model.pt")):
        if state is not None:
            model.load_state_dict(torch.load(state, map_location=device))
        else:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            model, _ = m.build_model(cfg, device, dtype)
        print(f"  step {step}:")
        for alpha in cfg.analysis_alphas:
            v_x, f_x, j_x, spatial_jac_x = m.evaluate_at_points(
                model, q1_grid, q2_grid, alpha, cfg.architecture, device=device, dtype=dtype,
                need_spatial_jacobian=True,
            )
            g = m.rotation_generator_target(q1_grid, q2_grid, f_x, spatial_jac_x)
            pi_f = m.compute_orbit_average(model, cfg, device, dtype, alpha, q1_grid, q2_grid)
            delta = f_x - pi_f
            g64 = g.detach().cpu().double().reshape(-1)
            d64 = delta.detach().cpu().double().reshape(-1)
            ng, nd = torch.linalg.norm(g64).item(), torch.linalg.norm(d64).item()
            cos = torch.dot(g64, d64).item() / (ng * nd) if ng > 0 and nd > 0 else float("nan")
            print(f"    alpha={alpha:+.2f}  cos(g,delta)={cos: .3e}")

    print(f"\nDone. Results written to {output_dir}")


if __name__ == "__main__":
    main()
