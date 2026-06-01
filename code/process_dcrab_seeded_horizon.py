"""dCRAB-seeded process receding-horizon diagnostic.

The process-GRAPE-seeded horizon shows that a high-fidelity process reference
can be carried into the receding-horizon architecture, but it still relies on a
gradient process optimizer.  This script replaces that reference by the
derivative-free process dCRAB pulse.  The result is still reference-assisted,
not a standalone process Lyapunov theorem, but it checks whether the horizon
architecture depends specifically on process-GRAPE.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dcrab_baseline import DCrabConfig
from ensemble_grape_baseline import average_gate_fidelity, target_gate
from horizon_lyapunov import dagger, fidelity, interaction_frame_operator, problem, random_disorder
from paths import figure_path, result_path
from process_dcrab_baseline import optimize_task
from process_seeded_horizon import SeededProcessConfig, design_seeded_horizon, step_unitaries


@dataclass(frozen=True)
class DCrabSeededConfig:
    dcrab: DCrabConfig
    horizon: SeededProcessConfig


def reference_metadata(logs: list[dict[str, float | int | bool]]) -> dict[str, float | int | bool]:
    return {
        "reference_objective": float(logs[-1]["objective"]),
        "reference_refreshes": len(logs),
        "reference_iterations": int(sum(int(log["optimizer_iterations"]) for log in logs)),
        "reference_success": all(bool(log["optimizer_success"]) for log in logs),
        "reference_seconds": float(sum(float(log["optimization_seconds"]) for log in logs)),
    }


def evaluate_pulse(
    task: str,
    controller: str,
    reference_type: str,
    pulse: np.ndarray,
    metadata: dict[str, float | int | bool],
    horizon_seconds: float,
    config: SeededProcessConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    pulse_energy = float(np.mean(np.sum(np.square(pulse), axis=1)))
    rows: list[dict[str, float | int | str]] = []

    for seed in config.eval_seeds:
        disorder = random_disorder(seed)
        unitary = np.eye(2, dtype=complex)
        for index, control in enumerate(pulse):
            t = float(t_eval[index])
            control_ops = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
            disorder_i = interaction_frame_operator(p, disorder, t)
            unitary = step_unitaries(
                (unitary,),
                (config.eval_strength,),
                control_ops,
                (disorder_i,),
                dt,
                control,
            )[0]
        rho_final = unitary @ p.initial @ dagger(unitary)
        rows.append(
            {
                "task": task,
                "controller": controller,
                "reference_type": reference_type,
                "disorder_strength": config.eval_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "average_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": pulse_energy,
                "segments": config.segments,
                "horizon_steps": config.horizon_steps
                if controller == "dcrab_seeded_process_horizon"
                else 0,
                "beam_width": config.beam_width
                if controller == "dcrab_seeded_process_horizon"
                else 0,
                "reference_objective": float(metadata["reference_objective"]),
                "reference_refreshes": int(metadata["reference_refreshes"]),
                "reference_iterations": int(metadata["reference_iterations"]),
                "reference_success": str(bool(metadata["reference_success"])),
                "reference_seconds": float(metadata["reference_seconds"]),
                "horizon_seconds": horizon_seconds,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["controller"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (task, controller), items in sorted(groups.items()):
        state = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate = np.array([float(row["average_gate_fidelity"]) for row in items])
        gate_ci = 1.96 * float(np.std(gate, ddof=1)) / np.sqrt(len(gate))
        summary.append(
            {
                "task": task,
                "controller": controller,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state):.6g}",
                "state_fidelity_min": f"{np.min(state):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate):.6g}",
                "avg_gate_fidelity_ci95": f"{gate_ci:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "horizon_steps": str(items[0]["horizon_steps"]),
                "beam_width": str(items[0]["beam_width"]),
                "reference_refreshes": str(items[0]["reference_refreshes"]),
                "reference_success": str(items[0]["reference_success"]),
                "reference_seconds": f"{float(items[0]['reference_seconds']):.3f}",
                "horizon_seconds": f"{float(items[0]['horizon_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]], output_prefix: str) -> None:
    with result_path(f"{output_prefix}_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path(f"{output_prefix}_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Process dCRAB-Seeded Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary, output_prefix)


def plot_summary(summary: list[dict[str, str]], output_prefix: str) -> None:
    labels = [f"{row['task']}\\n{row['controller'].replace('_', ' ')}" for row in summary]
    x = np.arange(len(labels))
    gate_means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])
    gate_mins = np.array([float(row["avg_gate_fidelity_min"]) for row in summary])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x, gate_means, width=0.46, label="mean")
    ax.scatter(x, gate_mins, marker="x", color="black", s=24, linewidths=0.9, label="min")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.97, 1.001)
    ax.set_ylabel("Held-out average gate fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(f"{output_prefix}.pdf"))
    fig.savefig(figure_path(f"{output_prefix}.png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    parser.add_argument("--output-prefix", default="process_dcrab_seeded_horizon")
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--dcrab-maxiter", type=int, default=DCrabConfig.maxiter)
    parser.add_argument("--dcrab-refreshes", type=int, default=DCrabConfig.refreshes)
    parser.add_argument("--training-seed-count", type=int, default=8)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DCrabSeededConfig:
    if args.quick:
        dcrab = DCrabConfig(
            segments=16,
            basis_count=2,
            refreshes=1,
            maxiter=2,
            popsize=3,
            training_strengths=(0.08,),
            training_seeds=(0, 1),
            eval_strengths=(0.08,),
            eval_seeds=tuple(range(10, 16)),
        )
        horizon = SeededProcessConfig(
            segments=16,
            horizon_steps=2,
            beam_width=3,
            train_strengths=(0.08,),
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 16)),
            local_radii=(0.25,),
            target_weight=0.08,
        )
        return DCrabSeededConfig(dcrab=dcrab, horizon=horizon)

    dcrab = DCrabConfig(
        segments=args.segments,
        refreshes=args.dcrab_refreshes,
        maxiter=args.dcrab_maxiter,
        training_strengths=(0.08,),
        training_seeds=tuple(range(args.training_seed_count)),
        eval_strengths=(0.08,),
        eval_seeds=tuple(range(10, 60)),
    )
    horizon = SeededProcessConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        beam_width=args.beam_width,
        train_strengths=(0.08,),
        train_seeds=tuple(range(args.training_seed_count)),
        eval_strength=0.08,
        eval_seeds=tuple(range(10, 60)),
        local_radii=(0.1, 0.25, 0.5),
        target_weight=0.08,
    )
    return DCrabSeededConfig(dcrab=dcrab, horizon=horizon)


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []

    for task in args.tasks:
        print(f"optimizing process dCRAB reference for {task}", flush=True)
        reference_pulse, logs = optimize_task(task, config.dcrab)
        metadata = reference_metadata(logs)
        print(
            f"  dCRAB objective={float(metadata['reference_objective']):.6g}, "
            f"refreshes={metadata['reference_refreshes']}, "
            f"success={metadata['reference_success']}, "
            f"seconds={float(metadata['reference_seconds']):.1f}",
            flush=True,
        )
        start = time.perf_counter()
        horizon_pulse = design_seeded_horizon(task, reference_pulse, config.horizon)
        horizon_seconds = time.perf_counter() - start
        print(f"  dCRAB-seeded horizon seconds={horizon_seconds:.1f}", flush=True)

        rows.extend(
            evaluate_pulse(
                task,
                "process_dcrab_reference",
                "process_dcrab",
                reference_pulse,
                metadata,
                0.0,
                config.horizon,
            )
        )
        rows.extend(
            evaluate_pulse(
                task,
                "dcrab_seeded_process_horizon",
                "process_dcrab",
                horizon_pulse,
                metadata,
                horizon_seconds,
                config.horizon,
            )
        )

    write_outputs(rows, args.output_prefix)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
