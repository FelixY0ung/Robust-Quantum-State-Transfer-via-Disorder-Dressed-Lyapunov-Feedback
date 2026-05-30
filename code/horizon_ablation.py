"""Ablation study for the beam-horizon Lyapunov controller.

The main horizon benchmark reports the best configured controller.  This
script checks whether the gain is tied to a single parameter choice by varying
the lookahead length and the retained beam width, then evaluating each pulse on
held-out disorder seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import design_pulse, evaluate_pulse
from paths import figure_path, result_path


@dataclass(frozen=True)
class AblationConfig:
    label: str
    horizon_steps: int
    beam_width: int


DEFAULT_CONFIGS = (
    AblationConfig("q1_b1", 1, 1),
    AblationConfig("q2_b2", 2, 2),
    AblationConfig("q4_b4", 4, 4),
    AblationConfig("q6_b1", 6, 1),
    AblationConfig("q6_b3", 6, 3),
    AblationConfig("q6_b6", 6, 6),
    AblationConfig("q6_b8", 6, 8),
)


def run_ablation(
    tasks: tuple[str, ...],
    configs: tuple[AblationConfig, ...],
    disorder_strength: float,
    test_seeds: range,
    segments: int,
    train_strengths: tuple[float, ...],
    train_seeds: tuple[int, ...],
    amplitudes: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for task in tasks:
        for config in configs:
            pulse = design_pulse(
                task,
                train_strengths=train_strengths,
                train_seeds=train_seeds,
                segments=segments,
                horizon_steps=config.horizon_steps,
                beam_width=config.beam_width,
                amplitudes=amplitudes,
            )
            eval_rows = evaluate_pulse(
                task,
                pulse,
                disorder_strength=disorder_strength,
                test_seeds=test_seeds,
            )
            for row in eval_rows:
                row.update(
                    {
                        "task": task,
                        "config": config.label,
                        "horizon_steps": config.horizon_steps,
                        "beam_width": config.beam_width,
                        "segments": segments,
                        "n_train_strengths": len(train_strengths),
                        "n_train_seeds": len(train_seeds),
                    }
                )
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["config"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (task, config), items in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            int(str(item[0][1]).split("_")[0][1:]),
            int(str(item[0][1]).split("_")[1][1:]),
        ),
    ):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        tails = np.array([float(row["tail_infidelity_mean"]) for row in items])
        energies = np.array([float(row["pulse_energy"]) for row in items])
        q = int(items[0]["horizon_steps"])
        b = int(items[0]["beam_width"])
        summary.append(
            {
                "task": task,
                "config": config,
                "q": str(q),
                "B": str(b),
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "tail_infidelity_mean": f"{np.mean(tails):.6g}",
                "pulse_energy_mean": f"{np.mean(energies):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    csv_path = result_path("horizon_ablation_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("horizon_ablation_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Horizon Ablation Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    tasks = tuple(dict.fromkeys(row["task"] for row in summary))
    configs = tuple(dict.fromkeys(row["config"] for row in summary))
    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for offset, task in zip((-0.5, 0.5), tasks):
        rows = [row for row in summary if row["task"] == task]
        means = np.array([float(row["final_fidelity_mean"]) for row in rows])
        mins = np.array([float(row["final_fidelity_min"]) for row in rows])
        ax.bar(
            x + offset * width,
            1.0 - means,
            width=width,
            label=f"{task} mean infidelity",
        )
        ax.scatter(
            x + offset * width,
            1.0 - mins,
            s=18,
            marker="x",
            color="black",
            linewidths=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Held-out final infidelity at $\\delta=0.08$")
    ax.set_xlabel("Horizon/beam configuration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("horizon_ablation.pdf"))
    fig.savefig(figure_path("horizon_ablation.png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use fewer seeds and segments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        rows = run_ablation(
            tasks=("Z", "H"),
            configs=DEFAULT_CONFIGS,
            disorder_strength=0.08,
            test_seeds=range(10, 30),
            segments=60,
            train_strengths=(0.08,),
            train_seeds=(0, 1, 2, 3),
            amplitudes=(1.0, 2.0, 3.0, 4.0),
        )
    else:
        rows = run_ablation(
            tasks=("Z", "H"),
            configs=DEFAULT_CONFIGS,
            disorder_strength=0.08,
            test_seeds=range(10, 60),
            segments=100,
            train_strengths=(0.05, 0.08),
            train_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
            amplitudes=(0.5, 1.0, 2.0, 3.0, 4.0),
        )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('horizon_ablation_results.csv')}")
    print(f"wrote aggregate table to {result_path('horizon_ablation_summary.md')}")
    print(f"wrote figure to {figure_path('horizon_ablation.pdf')}")


if __name__ == "__main__":
    main()
