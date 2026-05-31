"""Standalone leakage-adjoint horizon diagnostic.

The main leakage benchmark includes an adjoint horizon polished around a
leakage-penalized GRAPE reference.  This diagnostic removes that terminal
optimizer reference: it first designs the finite-candidate path horizon, then
uses that horizon pulse as the local initializer for exact short-horizon
adjoint polishing.

The result tests whether leakage-aware adjoint information improves the
receding-horizon controller itself.  It remains a local nonconvex diagnostic,
not a global optimality or convergence claim.
"""

from __future__ import annotations

import argparse
import csv
import time

from paths import result_path
from transmon_leakage_horizon import (
    EVAL_STRENGTHS,
    design_adjoint_horizon,
    design_pulse,
    evaluate_pulse,
    summarize,
)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("transmon_standalone_adjoint_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("transmon_standalone_adjoint_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Standalone Leakage-Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def run(quick: bool = False) -> None:
    if quick:
        eval_strengths = (0.03,)
        test_seeds = range(10, 15)
        start = time.perf_counter()
        path_pulse = design_pulse(
            train_strengths=(0.03,),
            train_seeds=(0, 1),
            segments=36,
            horizon_steps=3,
            beam_width=3,
            amplitudes=(0.015, 0.035, 0.06),
        )
        path_seconds = time.perf_counter() - start
        adjoint_kwargs = {
            "train_strengths": (0.03,),
            "train_seeds": (0, 1),
            "horizon_steps": 3,
            "maxiter": 2,
            "trust_radius": 0.025,
            "leakage_weight": 0.7,
        }
    else:
        eval_strengths = EVAL_STRENGTHS
        test_seeds = range(10, 60)
        start = time.perf_counter()
        path_pulse = design_pulse()
        path_seconds = time.perf_counter() - start
        adjoint_kwargs = {
            "train_strengths": (0.01, 0.02, 0.03),
            "train_seeds": (0, 1, 2, 3),
            "horizon_steps": 5,
            "maxiter": 6,
            "trust_radius": 0.025,
            "leakage_weight": 0.8,
        }

    rows: list[dict[str, float | int | str]] = []
    for strength in eval_strengths:
        rows.extend(
            evaluate_pulse(
                path_pulse,
                strength,
                controller="standalone_path_seed",
                training_seconds=path_seconds,
                test_seeds=test_seeds,
            )
        )

    start = time.perf_counter()
    adjoint_pulse, objective, iterations, success = design_adjoint_horizon(
        path_pulse,
        **adjoint_kwargs,
    )
    adjoint_seconds = path_seconds + time.perf_counter() - start
    for strength in eval_strengths:
        rows.extend(
            evaluate_pulse(
                adjoint_pulse,
                strength,
                controller="standalone_adjoint_horizon",
                training_seconds=adjoint_seconds,
                training_objective=objective,
                optimizer_iterations=iterations,
                optimizer_success=success,
                test_seeds=test_seeds,
            )
        )

    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('transmon_standalone_adjoint_results.csv')}")
    print(f"wrote summary to {result_path('transmon_standalone_adjoint_summary.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small diagnostic run for syntax and pipeline checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
