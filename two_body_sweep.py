"""Two-phase sweep over data amount and architecture capacity for the two-body script.

Motivation (session that built this file): with trajectory_window/splits/sampled_instants=5,
training works but the network overfits -- training loss keeps falling while validation
loss plateaus early. The question this sweep answers: is there a data/architecture regime
where validation loss keeps improving alongside training loss, rather than plateauing?

Design (confirmed with the user before launching):
  - Data axis: initial_conditions_per_alpha, trajectory_splits (window stays fixed at 5,
    which was independently found necessary -- longer windows expose periapsis passages
    within-window at a rate that jumps from <1% at window=5 to >10% at window=6+, which a
    smooth MLP cannot yet fit accurately, creating an unrelated loss floor).
  - Architecture axis: V_net/K_net hidden_dim and hidden_layers (capacity) only --
    weight_decay and l1_weight are held fixed (not swept) per the user's explicit choice.
  - One extra arm: window=6 and window=8 with *tightened* initial-condition eccentricity
    bounds (verified separately, see the ic_r_min/ic_l_min/ic_energy_margin/ic_v_box/ic_r_box
    values below) that bring the periapsis-in-window rate back down near/below the window=5
    baseline -- testing whether that unlocks longer windows too.

Two phases:
  1. Screening: every config in SCREEN_CONFIGS gets a short, fixed wall-clock time budget.
     Each is launched as a subprocess; when its budget expires, this script sends it exactly
     one SIGINT (the same graceful-early-stop mechanism built into run_training_loop), so it
     finishes its current step, checkpoints, and runs its own analysis/plotting on whatever
     was trained -- never a hard kill unless something hangs well past its budget.
  2. Refinement: the TOP_N_TO_REFINE screening configs (ranked by final validation loss) are
     re-run with a much larger time budget, splitting whatever wall-clock time remains in
     REFINEMENT_TOTAL_BUDGET_SECONDS.

Every config's result (final/min train & validation loss, whether it hit its time budget or
finished on its own, steps actually reached) is written to sweep_results.json immediately
after that config finishes -- so the sweep is always safe to inspect or stop early and still
have complete, usable results for everything that has finished so far.

Ctrl-C handling, mirroring run_training_loop's own semantics:
  - One Ctrl-C: forwards a SIGINT to the currently running child (finishes its current step +
    analysis, same as if its own time budget had expired), then stops the sweep after that
    (does not launch the next config). Whatever's already in sweep_results.json is complete
    and usable.
  - A second Ctrl-C (while waiting for the child to wrap up): hard-kills the child immediately
    and exits the sweep right away.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "asrnn_two_body_symmetry_sensitivity.py"
SWEEP_OUTPUT_ROOT = ROOT / "outputs" / "two_body_sweep"
RESULTS_PATH = SWEEP_OUTPUT_ROOT / "sweep_results.json"

# Wall-clock budgets. Total sweep target: ~7h (under the 8h ceiling, leaving margin).
SCREEN_TIME_BUDGET_SECONDS = 15 * 60
REFINEMENT_TOTAL_BUDGET_SECONDS = int(4.3 * 3600)
TOP_N_TO_REFINE = 2
SWEEP_DEADLINE_SECONDS = int(7.5 * 3600)

# Capacity presets (kinetic/potential hidden_dim, hidden_layers), applied to both nets equally.
CAPACITY = {
    "small": dict(kinetic_hidden_dim=16, kinetic_hidden_layers=2, potential_hidden_dim=16, potential_hidden_layers=2),
    "medium": dict(kinetic_hidden_dim=32, kinetic_hidden_layers=2, potential_hidden_dim=32, potential_hidden_layers=2),
    "large": dict(kinetic_hidden_dim=64, kinetic_hidden_layers=3, potential_hidden_dim=64, potential_hidden_layers=3),
}

# Base fields shared by every screening config (kept off the data/architecture axes being swept).
BASE = dict(
    device="auto", optimizer="adam", learning_rate=1e-3, weight_decay=0.0, l1_weight=0.0,
    max_grad_norm=1.0, training_steps=10_000_000,  # effectively unbounded -- SIGINT ends it
    checkpoint_steps=[0],  # run_training_loop always also force-saves the actual final step
    analysis_alphas=[0.6, 1.0, 1.6, 2.5], plotting_alphas=[0.6, 1.6],
)


def with_base(overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(BASE)
    merged.update(overrides)
    return merged


SCREEN_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("baseline_ic8_splits5_medium", with_base({
        "initial_conditions_per_alpha": 8, "trajectory_splits": 5,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("data_ic20", with_base({
        "initial_conditions_per_alpha": 20, "trajectory_splits": 5,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("data_ic40", with_base({
        "initial_conditions_per_alpha": 40, "trajectory_splits": 5,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("data_splits15", with_base({
        "initial_conditions_per_alpha": 8, "trajectory_splits": 15,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("data_splits30", with_base({
        "initial_conditions_per_alpha": 8, "trajectory_splits": 30,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("arch_small", with_base({
        "initial_conditions_per_alpha": 8, "trajectory_splits": 5,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["small"],
    })),
    ("arch_large", with_base({
        "initial_conditions_per_alpha": 8, "trajectory_splits": 5,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["large"],
    })),
    ("combo_highdata_small", with_base({
        "initial_conditions_per_alpha": 40, "trajectory_splits": 15,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["small"],
    })),
    ("combo_highdata_medium", with_base({
        "initial_conditions_per_alpha": 40, "trajectory_splits": 15,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["medium"],
    })),
    ("combo_highdata_large", with_base({
        "initial_conditions_per_alpha": 40, "trajectory_splits": 15,
        "trajectory_window": 5, "sampled_instants": 5, **CAPACITY["large"],
    })),
    # Tighter-eccentricity longer-window arm (bounds verified separately: bring the
    # periapsis-in-window rate to ~0.5% at window=6 and ~0% at window=8, both at or below
    # the window=5 baseline's own ~2.2% rate).
    ("window6_tight_ecc", with_base({
        "initial_conditions_per_alpha": 20, "trajectory_splits": 6,
        "trajectory_window": 6, "sampled_instants": 6, **CAPACITY["medium"],
        "ic_r_box": 1.5, "ic_v_box": 0.5, "ic_r_min": 0.6, "ic_l_min": 0.45, "ic_energy_margin": 0.15,
    })),
    ("window8_tight_ecc", with_base({
        "initial_conditions_per_alpha": 20, "trajectory_splits": 8,
        "trajectory_window": 8, "sampled_instants": 8, **CAPACITY["medium"],
        "ic_r_box": 1.7, "ic_v_box": 0.55, "ic_r_min": 0.9, "ic_l_min": 0.7, "ic_energy_margin": 0.3,
    })),
]


@dataclass
class RunOutcome:
    name: str
    phase: str
    overrides: dict[str, Any]
    time_budget_seconds: float
    wall_seconds: float
    hit_time_budget: bool
    returncode: int | None
    final_step: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    min_val_loss: float | None = None
    min_val_loss_step: int | None = None
    val_loss_at_60pct: float | None = None
    still_improving: bool | None = None
    error: str | None = None


def load_history_summary(output_dir: Path) -> dict[str, Any]:
    npz_path = output_dir / "training_history.npz"
    if not npz_path.exists():
        return {"error": "no training_history.npz found"}
    h = np.load(npz_path)
    steps = h["step"]
    train_loss = h["trajectory_loss"]
    val_loss = h["validation_loss"]
    if len(steps) == 0:
        return {"error": "empty training history"}
    idx_60pct = int(0.6 * (len(steps) - 1))
    min_idx = int(np.argmin(val_loss))
    # "Still improving": is validation loss at the end meaningfully lower than at the 60% mark?
    # (rather than flat/rising, which is the plateau/overfitting signature we're screening for)
    still_improving = bool(val_loss[-1] < 0.9 * val_loss[idx_60pct])
    return {
        "final_step": int(steps[-1]),
        "final_train_loss": float(train_loss[-1]),
        "final_val_loss": float(val_loss[-1]),
        "min_val_loss": float(val_loss[min_idx]),
        "min_val_loss_step": int(steps[min_idx]),
        "val_loss_at_60pct": float(val_loss[idx_60pct]),
        "still_improving": still_improving,
    }


class SweepAborted(Exception):
    pass


def run_one_config(
    name: str, phase: str, overrides: dict[str, Any], time_budget_seconds: float, sweep_deadline: float,
    # sweep_deadline is accepted but not enforced mid-run -- the overall deadline is only
    # checked between configs in main(), so a single config can never be cut short by it.
) -> RunOutcome:
    output_dir = SWEEP_OUTPUT_ROOT / phase / name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "overrides.json"
    config_path.write_text(json.dumps(overrides, indent=2))

    device = overrides.get("device", "auto")
    cmd = [
        sys.executable, str(SCRIPT),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--device", device,
    ]
    log_path = output_dir / "run.log"
    print(f"\n=== [{phase}] {name} -- budget {time_budget_seconds/60:.1f} min -- {output_dir} ===", flush=True)
    start = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        hit_budget = False
        interrupt_count = [0]

        def forward_sigint(signum, frame):
            interrupt_count[0] += 1
            if interrupt_count[0] == 1:
                print(f"\n[sweep] Ctrl-C: telling '{name}' to stop early and finish analysis. "
                      "Sweep will stop after this config. Press Ctrl-C again to kill it immediately.", flush=True)
                proc.send_signal(signal.SIGINT)
            else:
                print(f"\n[sweep] second Ctrl-C: killing '{name}' immediately and aborting sweep.", flush=True)
                proc.kill()
                proc.wait()
                raise SweepAborted()

        previous_handler = signal.signal(signal.SIGINT, forward_sigint)
        try:
            while True:
                try:
                    proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - start
                    if not hit_budget and elapsed > time_budget_seconds:
                        hit_budget = True
                        proc.send_signal(signal.SIGINT)
                    elif hit_budget and elapsed > time_budget_seconds + 15 * 60:
                        # Analysis/plotting after a graceful stop shouldn't take anywhere near
                        # 15 extra minutes; this is a hang-safety fallback, not the expected path.
                        print(f"[sweep] '{name}' did not exit within 15 min of SIGINT -- killing.", flush=True)
                        proc.kill()
                        proc.wait()
                        break
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    wall_seconds = time.time() - start
    summary = load_history_summary(output_dir)
    outcome = RunOutcome(
        name=name, phase=phase, overrides=overrides, time_budget_seconds=time_budget_seconds,
        wall_seconds=wall_seconds, hit_time_budget=hit_budget, returncode=proc.returncode,
        error=summary.get("error"),
        **{k: v for k, v in summary.items() if k != "error"},
    )
    print(
        f"[sweep] '{name}' done in {wall_seconds/60:.1f} min (budget hit: {hit_budget}, "
        f"returncode={proc.returncode}). final_val_loss={outcome.final_val_loss} "
        f"min_val_loss={outcome.min_val_loss} still_improving={outcome.still_improving}",
        flush=True,
    )
    return outcome


def save_results(results: list[RunOutcome]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps([r.__dict__ for r in results], indent=2))


def rank_key(r: RunOutcome) -> float:
    if r.final_val_loss is None or not np.isfinite(r.final_val_loss):
        return float("inf")
    return r.final_val_loss


def print_summary_table(results: list[RunOutcome]) -> None:
    print("\n" + "=" * 100)
    print(f"{'name':28s} {'phase':11s} {'steps':>8s} {'final_val':>12s} {'min_val':>12s} {'still_improv':>13s}")
    for r in sorted(results, key=rank_key):
        print(
            f"{r.name:28s} {r.phase:11s} {str(r.final_step):>8s} "
            f"{('%.5f' % r.final_val_loss) if r.final_val_loss is not None else 'n/a':>12s} "
            f"{('%.5f' % r.min_val_loss) if r.min_val_loss is not None else 'n/a':>12s} "
            f"{str(r.still_improving):>13s}"
        )
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Device for every sweep run (auto/cpu/mps).")
    args = parser.parse_args()

    for _, overrides in SCREEN_CONFIGS:
        overrides["device"] = args.device

    sweep_start = time.time()
    deadline = sweep_start + SWEEP_DEADLINE_SECONDS
    results: list[RunOutcome] = []

    print(f"Starting screening phase: {len(SCREEN_CONFIGS)} configs, "
          f"{SCREEN_TIME_BUDGET_SECONDS/60:.1f} min each "
          f"(~{len(SCREEN_CONFIGS)*SCREEN_TIME_BUDGET_SECONDS/3600:.2f}h total budget).", flush=True)

    try:
        for name, overrides in SCREEN_CONFIGS:
            if time.time() > deadline:
                print("[sweep] Overall deadline reached during screening -- stopping here.", flush=True)
                break
            outcome = run_one_config(name, "screen", overrides, SCREEN_TIME_BUDGET_SECONDS, deadline)
            results.append(outcome)
            save_results(results)

        screened = [r for r in results if r.phase == "screen" and r.final_val_loss is not None]
        top = sorted(screened, key=rank_key)[:TOP_N_TO_REFINE]
        if top:
            remaining = max(deadline - time.time(), 0.0)
            refine_budget_each = min(REFINEMENT_TOTAL_BUDGET_SECONDS, remaining) / max(len(top), 1)
            print(f"\nRefinement phase: top {len(top)} config(s) by final validation loss: "
                  f"{[r.name for r in top]}, {refine_budget_each/3600:.2f}h each.", flush=True)
            for r in top:
                if time.time() > deadline:
                    print("[sweep] Overall deadline reached before refinement of all candidates.", flush=True)
                    break
                outcome = run_one_config(f"{r.name}_refined", "refine", r.overrides, refine_budget_each, deadline)
                results.append(outcome)
                save_results(results)
        else:
            print("[sweep] No screening config produced a usable validation loss -- skipping refinement.", flush=True)

    except SweepAborted:
        print("[sweep] Aborted by double Ctrl-C.", flush=True)
    except KeyboardInterrupt:
        # Lands here only if Ctrl-C arrives in the brief window *between* configs (no child
        # running to forward it to, so no custom handler is installed at that moment) --
        # treat it the same as "stop after current config": don't start the next one, but
        # still print the summary of everything that already finished.
        print("[sweep] Ctrl-C between configs -- stopping sweep here.", flush=True)

    save_results(results)
    print_summary_table(results)
    print(f"\nTotal sweep wall time: {(time.time()-sweep_start)/3600:.2f}h. "
          f"Results: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
