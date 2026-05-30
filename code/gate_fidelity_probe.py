"""Gate-fidelity diagnostic for state-transfer pulses.

The main paper studies one-state transfer, not full gate synthesis.  This script
quantifies that distinction by evaluating the beam-horizon pulses as if they
were intended to implement the corresponding Z or Hadamard gate.  The reported
average gate fidelity is a scope diagnostic, not an optimization target.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import (
    dagger,
    default_beam_width,
    design_pulse,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
    unitary2,
)
from paths import figure_path, result_path


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
    pulse: np.ndarray,
    disorder_strength: float,
    disorder_seed: int,
) -> np.ndarray:
    p = problem(task)
    disorder = random_disorder(disorder_seed)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    unitary = np.eye(2, dtype=complex)

    for index, control in enumerate(pulse):
        t = float(t_eval[index])
        h = disorder_strength * interaction_frame_operator(p, disorder, t)
        for coeff, hc in zip(control, p.controls):
            h = h + coeff * interaction_frame_operator(p, hc, t)
        unitary = unitary2(h, dt) @ unitary
    return unitary


def run_probe(
    tasks: tuple[str, ...] = ("Z", "H"),
    disorder_strength: float = 0.08,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for task in tasks:
        p = problem(task)
        pulse = design_pulse(task, beam_width=default_beam_width(task))
        target = target_gate(task)
        energy = float(np.mean(np.sum(pulse * pulse, axis=1)))
        for seed in test_seeds:
            unitary = interaction_frame_unitary(task, pulse, disorder_strength, seed)
            rho_final = unitary @ p.initial @ dagger(unitary)
            rows.append(
                {
                    "task": task,
                    "disorder_strength": disorder_strength,
                    "seed": seed,
                    "state_transfer_fidelity": fidelity(rho_final, p.target),
                    "average_gate_fidelity": average_gate_fidelity(unitary, target),
                    "pulse_energy": energy,
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
                "state_fidelity_std": f"{np.std(state_fids):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_std": f"{np.std(gate_fids):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("gate_fidelity_probe_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("gate_fidelity_probe_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Gate Fidelity Probe Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [row["task"] for row in summary]
    x = np.arange(len(labels))
    width = 0.32
    state_mean = np.array([float(row["state_fidelity_mean"]) for row in summary])
    gate_mean = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])
    state_min = np.array([float(row["state_fidelity_min"]) for row in summary])
    gate_min = np.array([float(row["avg_gate_fidelity_min"]) for row in summary])

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.bar(x - width / 2, state_mean, width=width, label="state transfer")
    ax.bar(x + width / 2, gate_mean, width=width, label="average gate")
    ax.scatter(x - width / 2, state_min, marker="x", s=22, color="black", linewidths=0.8)
    ax.scatter(x + width / 2, gate_min, marker="x", s=22, color="black", linewidths=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Held-out fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("gate_fidelity_probe.pdf"))
    fig.savefig(figure_path("gate_fidelity_probe.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    rows = run_probe()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('gate_fidelity_probe_results.csv')}")
    print(f"wrote aggregate table to {result_path('gate_fidelity_probe_summary.md')}")
    print(f"wrote figure to {figure_path('gate_fidelity_probe.pdf')}")


if __name__ == "__main__":
    main()
