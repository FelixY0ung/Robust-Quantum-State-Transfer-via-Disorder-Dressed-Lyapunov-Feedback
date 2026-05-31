"""Standalone open-system adjoint horizon diagnostic.

The main open-system adjoint horizon is reference-assisted by an open-system
GRAPE pulse.  This diagnostic removes that terminal optimizer reference.  It
first trains a compact finite-candidate Lindblad horizon pulse, then uses that
pulse only as a trust-region initializer while optimizing the direct target
state score through the Liouville-space Lindblad model.

The result tests whether exact dissipative short-horizon gradients improve the
open-system horizon controller itself.  It is not a standalone convergence or
global optimality claim.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from horizon_lyapunov import problem
from open_system_adjoint_horizon import (
    OpenAdjointConfig,
    design_adjoint_horizon,
    summarize,
)
from open_system_grape_baseline import EVAL_NOISE_CASES, TRAIN_NOISE
from open_system_horizon_training import design_open_system_pulse
from open_system_noise import evolve_open_system
from paths import result_path


def evaluate_pulse(
    task: str,
    controller: str,
    pulse: np.ndarray,
    config: OpenAdjointConfig,
    seed_seconds: float,
    horizon_objective: float,
    horizon_iterations: int,
    horizon_success: bool,
    horizon_seconds: float,
    horizon_steps: int,
) -> list[dict[str, float | int | str]]:
    energy = float(np.mean(np.sum(pulse * pulse, axis=1)))
    rows: list[dict[str, float | int | str]] = []
    for noise in EVAL_NOISE_CASES:
        for seed in config.eval_seeds:
            final_fidelity, purity = evolve_open_system(
                task,
                pulse,
                config.eval_strength,
                seed,
                noise,
            )
            rows.append(
                {
                    "task": task,
                    "controller": controller,
                    "train_noise_case": TRAIN_NOISE.label,
                    "eval_noise_case": noise.label,
                    "disorder_strength": config.eval_strength,
                    "gamma_phi": noise.gamma_phi,
                    "gamma_relax": noise.gamma_relax,
                    "seed": seed,
                    "final_fidelity": final_fidelity,
                    "final_infidelity": 1.0 - final_fidelity,
                    "final_purity": purity,
                    "pulse_energy": energy,
                    "segments": config.segments,
                    "horizon_steps": horizon_steps,
                    "reference_objective": 0.0,
                    "reference_restart_seed": 0,
                    "reference_iterations": 0,
                    "reference_success": "True",
                    "reference_seconds": seed_seconds,
                    "horizon_objective": horizon_objective,
                    "horizon_iterations": horizon_iterations,
                    "horizon_success": str(horizon_success),
                    "horizon_seconds": horizon_seconds,
                }
            )
    return rows


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("open_system_standalone_adjoint_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("open_system_standalone_adjoint_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Standalone Open-System Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def config_from_args(args: argparse.Namespace) -> OpenAdjointConfig:
    if args.quick:
        return OpenAdjointConfig(
            segments=24,
            horizon_steps=2,
            horizon_maxiter=3,
            training_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
            reference_weight=0.0,
            target_weight=1.0,
            trust_radius=0.35,
        )
    return OpenAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        training_seeds=(0, 1, 2),
        reference_weight=0.0,
        target_weight=1.0,
        trust_radius=args.trust_radius,
    )


def run(tasks: tuple[str, ...], config: OpenAdjointConfig, quick: bool) -> None:
    rows: list[dict[str, float | int | str]] = []
    for task in tasks:
        print(f"designing finite open-system horizon seed for {task}", flush=True)
        start = time.perf_counter()
        seed_pulse = design_open_system_pulse(
            task,
            train_seeds=config.training_seeds,
            segments=config.segments,
            horizon_steps=2 if not quick else config.horizon_steps,
            amplitudes=(0.2, 0.5, 1.0) if quick else (0.5, 1.0, 2.0, 3.0, 4.0),
        )
        seed_seconds = time.perf_counter() - start
        rows.extend(
            evaluate_pulse(
                task,
                "standalone_open_seed",
                seed_pulse,
                config,
                seed_seconds,
                horizon_objective=0.0,
                horizon_iterations=0,
                horizon_success=True,
                horizon_seconds=0.0,
                horizon_steps=2 if not quick else config.horizon_steps,
            )
        )

        print(f"polishing standalone open-system horizon for {task}", flush=True)
        pulse, objective, iterations, success, horizon_seconds = design_adjoint_horizon(
            task,
            seed_pulse,
            config,
        )
        rows.extend(
            evaluate_pulse(
                task,
                "standalone_open_adjoint",
                pulse,
                config,
                seed_seconds,
                horizon_objective=objective,
                horizon_iterations=iterations,
                horizon_success=success,
                horizon_seconds=horizon_seconds,
                horizon_steps=config.horizon_steps,
            )
        )

    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('open_system_standalone_adjoint_results.csv')}")
    print(f"wrote summary to {result_path('open_system_standalone_adjoint_summary.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--horizon-maxiter", type=int, default=8)
    parser.add_argument("--trust-radius", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run(tuple(args.tasks), config, quick=args.quick)


if __name__ == "__main__":
    main()
