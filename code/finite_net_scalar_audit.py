"""Scalar finite-net style audit for two-level beam-horizon pulses.

Proposition 7 in the journal manuscript states a deterministic finite-net
robustness certificate under explicit covering assumptions.  The Monte Carlo
held-out seeds used in the main simulations are not such a cover.  This script
therefore provides a narrower executable diagnostic: for a small fixed set of
held-out disorder directions, evaluate the trained beam-horizon pulse on a
dense scalar strength grid over ``delta in [0, 0.12]``.

The result is a one-dimensional finite-net audit along the disorder-strength
coordinate for fixed directions.  It reports empirical adjacent-grid slopes to
calibrate how conservative a Lipschitz penalty might be, but it does not claim
coverage of all disorder directions.
"""

from __future__ import annotations

import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import (
    default_beam_width,
    design_pulse,
    evaluate_pulse,
)
from paths import figure_path, result_path


DEFAULT_SEEDS = (10, 13, 20, 37, 50)


def strength_grid(points: int, max_strength: float) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linspace(0.0, max_strength, points))


def design_task_pulse(task: str, quick: bool) -> tuple[np.ndarray, float, str]:
    start = time.perf_counter()
    if quick:
        pulse = design_pulse(
            task,
            train_strengths=(0.08,),
            train_seeds=(0, 1),
            segments=20,
            horizon_steps=3,
            beam_width=3,
            amplitudes=(0.5, 1.0, 2.0),
        )
        profile = "quick smoke-test beam"
    else:
        pulse = design_pulse(task, beam_width=default_beam_width(task))
        profile = "100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam"
    return pulse, time.perf_counter() - start, profile


def evaluate_grid(
    task: str,
    pulse: np.ndarray,
    design_seconds: float,
    resource_profile: str,
    strengths: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for strength in strengths:
        evaluated = evaluate_pulse(
            task,
            pulse,
            disorder_strength=strength,
            test_seeds=range(min(seeds), max(seeds) + 1),
        )
        wanted = {int(seed) for seed in seeds}
        for row in evaluated:
            seed = int(row["seed"])
            if seed not in wanted:
                continue
            fidelity = float(row["final_fidelity"])
            rows.append(
                {
                    "task": task,
                    "method": "beam_horizon",
                    "eval_strength": strength,
                    "seed": seed,
                    "final_fidelity": fidelity,
                    "final_infidelity": 1.0 - fidelity,
                    "pulse_energy": float(row["pulse_energy"]),
                    "design_seconds": design_seconds,
                    "resource_profile": resource_profile,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    by_task_seed: dict[tuple[str, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_task_seed.setdefault((str(row["task"]), int(row["seed"])), []).append(row)

    summaries: list[dict[str, str]] = []
    for (task, seed), items in sorted(by_task_seed.items()):
        ordered = sorted(items, key=lambda row: float(row["eval_strength"]))
        strengths = np.array([float(row["eval_strength"]) for row in ordered])
        infids = np.array([float(row["final_infidelity"]) for row in ordered])
        fids = 1.0 - infids
        slopes = np.abs(np.diff(infids)) / np.diff(strengths)
        summaries.append(
            {
                "task": task,
                "seed": str(seed),
                "strength_min": f"{float(strengths[0]):.6g}",
                "strength_max": f"{float(strengths[-1]):.6g}",
                "grid_points": str(len(strengths)),
                "grid_radius": f"{0.5 * float(np.max(np.diff(strengths))):.6g}",
                "fidelity_min": f"{float(np.min(fids)):.6g}",
                "fidelity_at_max_strength": f"{float(fids[-1]):.6g}",
                "max_adjacent_infidelity_slope": f"{float(np.max(slopes)):.6g}",
                "mean_adjacent_infidelity_slope": f"{float(np.mean(slopes)):.6g}",
            }
        )
    return summaries


def summarize_by_task(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    by_task: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)

    task_rows: list[dict[str, str]] = []
    for task, items in sorted(by_task.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        max_strength_items = [
            row
            for row in items
            if abs(float(row["eval_strength"]) - max(float(x["eval_strength"]) for x in items)) < 1e-12
        ]
        max_strength_fids = np.array([float(row["final_fidelity"]) for row in max_strength_items])
        seed_summaries = summarize(items)
        slopes = np.array(
            [float(row["max_adjacent_infidelity_slope"]) for row in seed_summaries]
        )
        first = items[0]
        task_rows.append(
            {
                "task": task,
                "directions": str(len({int(row["seed"]) for row in items})),
                "grid_points": str(len({float(row["eval_strength"]) for row in items})),
                "fidelity_min_all": f"{float(np.min(fids)):.6g}",
                "fidelity_mean_at_max_strength": f"{float(np.mean(max_strength_fids)):.6g}",
                "fidelity_min_at_max_strength": f"{float(np.min(max_strength_fids)):.6g}",
                "max_empirical_infidelity_slope": f"{float(np.max(slopes)):.6g}",
                "pulse_energy": f"{float(first['pulse_energy']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
                "resource_profile": str(first["resource_profile"]),
            }
        )
    return task_rows


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("finite_net_scalar_audit_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    seed_summary = summarize(rows)
    task_summary = summarize_by_task(rows)
    summary_file = result_path("finite_net_scalar_audit_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Scalar Finite-Net Audit\n\n")
        f.write(
            "Dense scalar disorder-strength grid for fixed held-out disorder directions. "
            "This is a deterministic one-dimensional grid audit, not a cover of all disorder directions.\n\n"
        )
        f.write("## Task Summary\n\n")
        headers = list(task_summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in task_summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

        f.write("\n## Direction Summary\n\n")
        seed_headers = list(seed_summary[0].keys())
        f.write("| " + " | ".join(seed_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(seed_headers)) + " |\n")
        for row in seed_summary:
            f.write("| " + " | ".join(row[h] for h in seed_headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('finite_net_scalar_audit.pdf')}")


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.8, 2.9), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        task_rows = [row for row in rows if str(row["task"]) == task]
        for seed in sorted({int(row["seed"]) for row in task_rows}):
            seed_rows = sorted(
                [row for row in task_rows if int(row["seed"]) == seed],
                key=lambda row: float(row["eval_strength"]),
            )
            ax.plot(
                [float(row["eval_strength"]) for row in seed_rows],
                [float(row["final_fidelity"]) for row in seed_rows],
                marker="o",
                markersize=2.3,
                linewidth=1.0,
                label=f"seed {seed}",
            )
        ax.set_title(task)
        ax.set_xlabel("Disorder strength")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Final fidelity")
    axes[-1].legend(loc="lower left", fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path("finite_net_scalar_audit.pdf"))
    fig.savefig(figure_path("finite_net_scalar_audit.png"), dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    strengths = strength_grid(args.grid_points, args.max_strength)
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        print(f"designing {task} beam-horizon pulse", flush=True)
        pulse, design_seconds, resource_profile = design_task_pulse(task, args.quick)
        rows.extend(evaluate_grid(task, pulse, design_seconds, resource_profile, strengths, seeds))
    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tasks", default="Z,H")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--grid-points", type=int, default=25)
    parser.add_argument("--max-strength", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
