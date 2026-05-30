"""Terminal process-fidelity baseline for robust single-qubit gates.

The beam-horizon controller in the paper is optimized for one-state transfer.
This script adds a compact terminal optimizer that directly maximizes average
gate fidelity for the corresponding Z and Hadamard gates.  It serves as a
process-fidelity baseline/ceiling, not as a Lyapunov-feedback method.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from horizon_lyapunov import (
    dagger,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
    unitary2,
)
from paths import figure_path, result_path


@dataclass(frozen=True)
class OptimizerConfig:
    segments: int = 20
    umax: float = 4.0
    maxiter: int = 250
    training_strengths: tuple[float, ...] = (0.05, 0.08)
    training_seeds: tuple[int, ...] = (0, 1, 2, 3)
    restart_seeds: tuple[int, ...] = (11, 22, 33)
    worst_weight: float = 0.25
    energy_weight: float = 1e-4


def target_gate(task: str) -> np.ndarray:
    if task == "Z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    if task == "H":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    raise ValueError(task)


def average_gate_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    dim = actual.shape[0]
    overlap = abs(np.trace(dagger(target) @ actual)) ** 2
    return float((overlap + dim) / (dim * (dim + 1)))


def interaction_frame_unitary(
    task: str,
    controls_flat: np.ndarray,
    segments: int,
    disorder_strength: float,
    disorder: np.ndarray,
    umax: float,
) -> np.ndarray:
    p = problem(task)
    controls = np.clip(np.reshape(controls_flat, (segments, len(p.controls))), -umax, umax)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    unitary = np.eye(2, dtype=complex)

    for index, control in enumerate(controls):
        t = float(t_eval[index])
        h = disorder_strength * interaction_frame_operator(p, disorder, t)
        for coeff, hc in zip(control, p.controls):
            h = h + coeff * interaction_frame_operator(p, hc, t)
        unitary = unitary2(h, dt) @ unitary
    return unitary


def objective(
    controls_flat: np.ndarray,
    task: str,
    config: OptimizerConfig,
    scenarios: tuple[tuple[float, np.ndarray], ...],
) -> float:
    target = target_gate(task)
    gate_infids = np.array(
        [
            1.0
            - average_gate_fidelity(
                interaction_frame_unitary(
                    task,
                    controls_flat,
                    config.segments,
                    strength,
                    disorder,
                    config.umax,
                ),
                target,
            )
            for strength, disorder in scenarios
        ]
    )
    energy = float(np.mean(np.square(np.clip(controls_flat, -config.umax, config.umax))))
    return float(
        np.mean(gate_infids)
        + config.worst_weight * np.max(gate_infids)
        + config.energy_weight * energy
    )


def optimize_gate(task: str, config: OptimizerConfig) -> tuple[np.ndarray, float, int]:
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.training_strengths
        for seed in config.training_seeds
    )
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    best_seed = -1

    for seed in config.restart_seeds:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(scale=0.2, size=config.segments * len(problem(task).controls))
        result = minimize(
            objective,
            x0,
            args=(task, config, scenarios),
            method="Powell",
            bounds=[(-config.umax, config.umax)] * len(x0),
            options={
                "maxiter": config.maxiter,
                "xtol": 1e-4,
                "ftol": 1e-6,
                "disp": False,
            },
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x)
            best_seed = seed

    if best_x is None:
        raise RuntimeError(f"no optimizer restart completed for {task}")
    return best_x, best_fun, best_seed


def evaluate_gate(
    task: str,
    controls_flat: np.ndarray,
    training_objective: float,
    best_restart_seed: int,
    config: OptimizerConfig,
    disorder_strength: float = 0.08,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    rows: list[dict[str, float | int | str]] = []
    energy = float(np.mean(np.square(np.clip(controls_flat, -config.umax, config.umax))))

    for seed in test_seeds:
        unitary = interaction_frame_unitary(
            task,
            controls_flat,
            config.segments,
            disorder_strength,
            random_disorder(seed),
            config.umax,
        )
        rho_final = unitary @ p.initial @ dagger(unitary)
        rows.append(
            {
                "task": task,
                "baseline": "terminal_process_open_loop",
                "disorder_strength": disorder_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "average_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": energy,
                "segments": config.segments,
                "training_objective": training_objective,
                "best_restart_seed": best_restart_seed,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(str(row["task"]), []).append(row)

    summary: list[dict[str, str]] = []
    for task, items in sorted(groups.items()):
        state_fids = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state_fids):.6g}",
                "state_fidelity_min": f"{np.min(state_fids):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_std": f"{np.std(gate_fids):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "best_restart_seed": str(items[0]["best_restart_seed"]),
                "training_objective": f"{float(items[0]['training_objective']):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("gate_process_baseline_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("gate_process_baseline_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Gate Process Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [row["task"] for row in summary]
    x = np.arange(len(labels))
    means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])
    mins = np.array([float(row["avg_gate_fidelity_min"]) for row in summary])

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(x, means, width=0.45, label="mean")
    ax.scatter(x, mins, marker="x", s=24, color="black", linewidths=0.9, label="min")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 1.001)
    ax.set_ylabel("Held-out average gate fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("gate_process_baseline.pdf"))
    fig.savefig(figure_path("gate_process_baseline.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    config = OptimizerConfig()
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        controls, value, seed = optimize_gate(task, config)
        rows.extend(evaluate_gate(task, controls, value, seed, config))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('gate_process_baseline_results.csv')}")
    print(f"wrote aggregate table to {result_path('gate_process_baseline_summary.md')}")
    print(f"wrote figure to {figure_path('gate_process_baseline.pdf')}")


if __name__ == "__main__":
    main()
