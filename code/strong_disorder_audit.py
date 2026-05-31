"""Out-of-training-range robustness audit for two-level robust transfer.

The main two-level comparisons train the beam-horizon controller on
``delta in {0.05, 0.08}`` and the train-8 dCRAB comparator at ``delta=0.08``.
This script reruns those design protocols and evaluates the resulting pulses at
``delta in {0.08, 0.10, 0.12}`` on the same held-out seeds.  The stronger
disorder rows are therefore extrapolation tests rather than retuned designs.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dcrab_baseline import DCrabConfig, evaluate_task as evaluate_dcrab_task, optimize_task
from horizon_lyapunov import (
    default_beam_width,
    design_pulse,
    evaluate_pulse as evaluate_horizon_pulse,
)
from paths import figure_path, result_path


@dataclass(frozen=True)
class AuditConfig:
    eval_strengths: tuple[float, ...] = (0.08, 0.10, 0.12)
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    tasks: tuple[str, ...] = ("Z", "H")


def evaluate_beam_horizon(
    task: str,
    config: AuditConfig,
    quick: bool,
) -> list[dict[str, float | int | str]]:
    """Design and evaluate the existing beam-horizon protocol."""
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
        resource_profile = "quick smoke-test beam"
    else:
        pulse = design_pulse(task, beam_width=default_beam_width(task))
        resource_profile = "100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam"
    design_seconds = time.perf_counter() - start

    rows: list[dict[str, float | int | str]] = []
    for strength in config.eval_strengths:
        horizon_rows = evaluate_horizon_pulse(
            task,
            pulse,
            disorder_strength=strength,
            test_seeds=range(min(config.eval_seeds), max(config.eval_seeds) + 1),
        )
        for row in horizon_rows:
            rows.append(
                {
                    "task": task,
                    "method": "beam_horizon",
                    "eval_strength": strength,
                    "seed": int(row["seed"]),
                    "final_fidelity": float(row["final_fidelity"]),
                    "final_infidelity": 1.0 - float(row["final_fidelity"]),
                    "pulse_energy": float(row["pulse_energy"]),
                    "design_seconds": design_seconds,
                    "resource_profile": resource_profile,
                    "training_strengths": "0.05,0.08" if not quick else "0.08",
                }
            )
    return rows


def evaluate_train8_dcrab(
    task: str,
    config: AuditConfig,
    quick: bool,
) -> list[dict[str, float | int | str]]:
    """Design and evaluate the train-8 sequential-basis dCRAB comparator."""
    if quick:
        dcrab_config = DCrabConfig(
            segments=16,
            basis_count=2,
            refreshes=1,
            maxiter=2,
            popsize=3,
            training_seeds=(0, 1),
            eval_strengths=config.eval_strengths,
            eval_seeds=config.eval_seeds,
        )
        resource_profile = "quick smoke-test dCRAB"
        training_strengths = "0.08"
    else:
        dcrab_config = DCrabConfig(
            training_seeds=tuple(range(8)),
            eval_strengths=config.eval_strengths,
            eval_seeds=config.eval_seeds,
        )
        resource_profile = "40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes"
        training_strengths = "0.08"

    controls, logs = optimize_task(task, dcrab_config)
    dcrab_rows = evaluate_dcrab_task(
        task,
        controls,
        logs,
        dcrab_config,
        baseline_label="dcrab_train8",
    )
    rows: list[dict[str, float | int | str]] = []
    for row in dcrab_rows:
        rows.append(
            {
                "task": task,
                "method": "dcrab_train8",
                "eval_strength": float(row["eval_strength"]),
                "seed": int(row["seed"]),
                "final_fidelity": float(row["final_fidelity"]),
                "final_infidelity": float(row["final_infidelity"]),
                "pulse_energy": float(row["pulse_energy"]),
                "design_seconds": float(row["optimization_seconds"]),
                "resource_profile": resource_profile,
                "training_strengths": training_strengths,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["method"]), float(row["eval_strength"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, str]] = []
    for (task, method, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        ci95 = 1.96 * float(np.std(fids)) / np.sqrt(len(fids))
        summary.append(
            {
                "task": task,
                "method": method,
                "eval_strength": f"{strength:.2f}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "design_seconds": f"{float(items[0]['design_seconds']):.3f}",
                "resource_profile": str(items[0]["resource_profile"]),
                "training_strengths": str(items[0]["training_strengths"]),
            }
        )
    return summary


def paired_deltas(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    by_key = {
        (
            str(row["task"]),
            str(row["method"]),
            float(row["eval_strength"]),
            int(row["seed"]),
        ): float(row["final_fidelity"])
        for row in rows
    }
    tasks = sorted({str(row["task"]) for row in rows})
    strengths = sorted({float(row["eval_strength"]) for row in rows})
    deltas: list[dict[str, str]] = []
    for task in tasks:
        for strength in strengths:
            values = []
            for seed in sorted({int(row["seed"]) for row in rows}):
                left = by_key.get((task, "dcrab_train8", strength, seed))
                right = by_key.get((task, "beam_horizon", strength, seed))
                if left is not None and right is not None:
                    values.append(left - right)
            if values:
                arr = np.array(values, dtype=float)
                ci95 = 1.96 * float(np.std(arr)) / np.sqrt(len(arr))
                deltas.append(
                    {
                        "task": task,
                        "comparison": "dcrab_train8_minus_beam_horizon",
                        "eval_strength": f"{strength:.2f}",
                        "n": str(len(arr)),
                        "delta_mean": f"{np.mean(arr):.8g}",
                        "delta_min": f"{np.min(arr):.8g}",
                        "delta_max": f"{np.max(arr):.8g}",
                        "delta_ci95": f"{ci95:.8g}",
                    }
                )
    return deltas


def write_markdown(
    summary: list[dict[str, str]],
    deltas: list[dict[str, str]],
) -> None:
    with result_path("strong_disorder_audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Strong-Disorder Extrapolation Audit\n\n")
        f.write(
            "The beam-horizon controller is trained on delta 0.05/0.08 and the "
            "dCRAB comparator is trained at delta 0.08. Rows at delta 0.10 and "
            "0.12 are held-out extrapolation tests, not retuned designs.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("## Held-Out Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

        delta_headers = list(deltas[0].keys())
        f.write("\n## Paired Deltas\n\n")
        f.write("| " + " | ".join(delta_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(delta_headers)) + " |\n")
        for row in deltas:
            f.write("| " + " | ".join(row[h] for h in delta_headers) + " |\n")


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("strong_disorder_audit_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict[str, str]]) -> None:
    tasks = ["Z", "H"]
    strengths = [0.08, 0.10, 0.12]
    methods = ["beam_horizon", "dcrab_train8"]
    means = {
        (row["task"], row["method"], float(row["eval_strength"])): float(
            row["final_fidelity_mean"]
        )
        for row in summary
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    x = np.arange(len(strengths))
    width = 0.36
    for ax, task in zip(axes, tasks):
        for method_index, method in enumerate(methods):
            values = [means[(task, method, strength)] for strength in strengths]
            ax.bar(
                x + (method_index - 0.5) * width,
                values,
                width,
                label=method.replace("_", "-"),
            )
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{strength:.2f}" for strength in strengths])
        ax.set_xlabel("Disorder strength")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out final fidelity")
    axes[0].set_ylim(0.98, 1.0005)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path("strong_disorder_audit.pdf"))
    fig.savefig(figure_path("strong_disorder_audit.png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test audit.")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("beam_horizon", "dcrab_train8"),
        default=("beam_horizon", "dcrab_train8"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        config = AuditConfig(eval_seeds=tuple(range(10, 16)))
    else:
        config = AuditConfig()

    rows: list[dict[str, float | int | str]] = []
    for task in config.tasks:
        if "beam_horizon" in args.methods:
            print(f"designing and evaluating beam horizon for {task}", flush=True)
            rows.extend(evaluate_beam_horizon(task, config, args.quick))
        if "dcrab_train8" in args.methods:
            print(f"optimizing and evaluating train-8 dCRAB for {task}", flush=True)
            rows.extend(evaluate_train8_dcrab(task, config, args.quick))

    summary = summarize(rows)
    deltas = paired_deltas(rows)
    write_csv(rows)
    write_markdown(summary, deltas)
    if not args.skip_plot:
        plot_summary(summary)
    print(f"wrote {len(rows)} rows to {result_path('strong_disorder_audit_results.csv')}")
    print(f"wrote summary to {result_path('strong_disorder_audit_summary.md')}")
    if not args.skip_plot:
        print(f"wrote figure to {figure_path('strong_disorder_audit.pdf')}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
