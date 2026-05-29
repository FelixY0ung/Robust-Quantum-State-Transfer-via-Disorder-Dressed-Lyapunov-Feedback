"""Higher-fidelity terminal open-loop polish for robust Z/H transfer.

This script is intentionally separate from the Lyapunov feedback scripts. It
uses the same two-level model and held-out seeds as the CAC paper, but optimizes
longer piecewise-constant terminal pulses at the stronger disorder level
``delta = 0.08``. The result is a reproducible performance ceiling showing how
close to unit fidelity the same Hamiltonian resources can get under compact
ensemble terminal optimization.
"""

from __future__ import annotations

import csv

import numpy as np

from paths import result_path
from robustness_scan import evaluate_pulse, optimize_pulse


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["task"]), float(row["eval_strength"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (task, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        infids = np.array([float(row["final_infidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.9g}",
                "final_fidelity_min": f"{np.min(fids):.9g}",
                "final_fidelity_std": f"{np.std(fids):.9g}",
                "final_infidelity_mean": f"{np.mean(infids):.9g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.9g}",
                "training_objective": f"{np.mean([float(row['training_objective']) for row in items]):.9g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("polished_openloop_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("polished_openloop_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Polished Open-Loop Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    segments = 14
    umax = 4.0
    training_strength = 0.08
    training_seeds = tuple(range(8))
    restart_seeds = (101, 102, 701, 777, 900, 1001, 1002)
    strength_grid = (0.05, 0.08)

    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        pulse, value = optimize_pulse(
            task,
            training_strength=training_strength,
            training_seeds=training_seeds,
            segments=segments,
            umax=umax,
            maxiter=220,
            restart_seeds=restart_seeds,
        )
        rows.extend(
            evaluate_pulse(
                task,
                pulse,
                "polished_open_loop",
                training_strength,
                value,
                strength_grid,
                range(10, 60),
                segments,
                umax,
            )
        )

    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/polished_openloop_results.csv")
    print("wrote aggregate table to results/polished_openloop_summary.md")


if __name__ == "__main__":
    main()
