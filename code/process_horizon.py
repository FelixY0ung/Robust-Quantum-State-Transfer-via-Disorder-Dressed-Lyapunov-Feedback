"""Process-level receding-horizon controller prototype.

The main beam-horizon controller optimizes a one-state Lyapunov objective.  This
script tests the next journal-level extension: replace the state-transfer
terminal cost inside the horizon search with a process-fidelity cost for the
full unitary.  The controller is still a finite-candidate receding-horizon
method, but its score is now average gate infidelity over the disorder
ensemble.

Two scoring modes are available:

* ``tracking`` compares the predicted unitary with a smooth path from identity
  to the target gate at the corresponding horizon time.
* ``terminal`` compares every horizon rollout directly with the final target
  gate.

The script is intentionally separate from the paper's main state-transfer
controller so weak or intermediate results do not get confused with the main
claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm

from horizon_lyapunov import (
    candidate_controls,
    dagger,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
    unitary2,
)
from paths import figure_path, result_path


@dataclass(frozen=True)
class ProcessHorizonConfig:
    segments: int = 100
    horizon_steps: int = 6
    beam_width: int = 8
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    test_strength: float = 0.08
    test_seeds: tuple[int, ...] = tuple(range(10, 60))
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    energy_weight: float = 0.0


def target_gate(task: str) -> np.ndarray:
    if task == "Z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    if task == "H":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    raise ValueError(task)


def phase_adjusted_target(task: str) -> np.ndarray:
    target = target_gate(task)
    phase = np.exp(-0.5j * np.angle(np.linalg.det(target)))
    return phase * target


def target_path(task: str, steps: int) -> tuple[np.ndarray, ...]:
    target = phase_adjusted_target(task)
    generator = logm(target)
    return tuple(expm((index / steps) * generator) for index in range(steps + 1))


def average_gate_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    dim = actual.shape[0]
    overlap = abs(np.trace(dagger(target) @ actual)) ** 2
    return float((overlap + dim) / (dim * (dim + 1)))


def process_cost(
    unitaries: tuple[np.ndarray, ...],
    target: np.ndarray,
    energy_mean: float,
    worst_weight: float,
    energy_weight: float,
) -> float:
    infids = np.array([1.0 - average_gate_fidelity(unitary, target) for unitary in unitaries])
    return float(np.mean(infids) + worst_weight * np.max(infids) + energy_weight * energy_mean)


def step_precomputed(
    unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    next_unitaries = []
    for unitary, strength, disorder_i in zip(unitaries, strengths, disorders_i):
        h = strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            h = h + coeff * hc_i
        next_unitaries.append(unitary2(h, dt) @ unitary)
    return tuple(next_unitaries)


def select_control_beam(
    start_index: int,
    unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dt: float,
    candidates: tuple[np.ndarray, ...],
    targets: tuple[np.ndarray, ...],
    final_target: np.ndarray,
    mode: str,
    config: ProcessHorizonConfig,
) -> np.ndarray:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), unitaries, 0.0)
    ]
    for depth in range(config.horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        controls_i = controls_cache[cache_index]
        disorders_i = disorder_cache[cache_index]
        target_index = min(start_index + depth + 1, len(targets) - 1)
        score_target = targets[target_index] if mode == "tracking" else final_target
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_precomputed(
                    states, strengths, controls_i, disorders_i, dt, candidate
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = process_cost(
                    next_states,
                    score_target,
                    next_energy / float(depth + 1),
                    config.worst_weight,
                    config.energy_weight,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[: config.beam_width]
        ]
    return beams[0][0][0]


def design_pulse(task: str, mode: str, config: ProcessHorizonConfig) -> np.ndarray:
    if mode not in {"tracking", "terminal"}:
        raise ValueError(mode)

    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    candidates = candidate_controls(config.amplitudes)
    cache_times = tuple(
        float(t_eval[min(index, config.segments)])
        for index in range(config.segments + config.horizon_steps)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )
    targets = target_path(task, config.segments)
    final_target = phase_adjusted_target(task)
    pulse = []

    for index, t in enumerate(t_eval[:-1]):
        control = select_control_beam(
            index,
            unitaries,
            strengths,
            controls_cache,
            disorder_cache,
            dt,
            candidates,
            targets,
            final_target,
            mode,
            config,
        )
        pulse.append(control)
        unitaries = step_precomputed(
            unitaries,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dt,
            control,
        )
    return np.vstack(pulse)


def evaluate_pulse(
    task: str,
    mode: str,
    pulse: np.ndarray,
    config: ProcessHorizonConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    rows: list[dict[str, float | int | str]] = []
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    energy = float(np.mean(np.sum(np.square(pulse), axis=1)))

    for seed in config.test_seeds:
        disorder = random_disorder(seed)
        unitary = np.eye(2, dtype=complex)
        for index, control in enumerate(pulse):
            t = float(t_eval[index])
            h = config.test_strength * interaction_frame_operator(p, disorder, t)
            for coeff, hc in zip(control, p.controls):
                h = h + coeff * interaction_frame_operator(p, hc, t)
            unitary = unitary2(h, dt) @ unitary
        rho_final = unitary @ p.initial @ dagger(unitary)
        rows.append(
            {
                "task": task,
                "controller": "process_horizon",
                "mode": mode,
                "disorder_strength": config.test_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "average_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": energy,
                "segments": config.segments,
                "horizon_steps": config.horizon_steps,
                "beam_width": config.beam_width,
                "n_train_seeds": len(config.train_seeds),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["mode"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (task, mode), items in sorted(groups.items()):
        state_fids = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "mode": mode,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state_fids):.6g}",
                "state_fidelity_min": f"{np.min(state_fids):.6g}",
                "state_fidelity_std": f"{np.std(state_fids):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_std": f"{np.std(gate_fids):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "horizon_steps": str(items[0]["horizon_steps"]),
                "beam_width": str(items[0]["beam_width"]),
                "n_train_seeds": str(items[0]["n_train_seeds"]),
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("process_horizon_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("process_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Process Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [f"{row['task']} {row['mode']}" for row in summary]
    x = np.arange(len(labels))
    state_means = np.array([float(row["state_fidelity_mean"]) for row in summary])
    gate_means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.36
    ax.bar(x - width / 2, state_means, width=width, label="state transfer")
    ax.bar(x + width / 2, gate_means, width=width, label="average gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.001)
    ax.set_ylabel("Held-out fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("process_horizon.pdf"))
    fig.savefig(figure_path("process_horizon.png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test configuration.")
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    parser.add_argument("--modes", nargs="+", choices=("tracking", "terminal"), default=("tracking",))
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=8)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ProcessHorizonConfig:
    if args.quick:
        return ProcessHorizonConfig(
            segments=30,
            horizon_steps=3,
            beam_width=4,
            train_strengths=(0.08,),
            train_seeds=(0, 1, 2),
            test_seeds=tuple(range(10, 20)),
            amplitudes=(0.5, 1.0, 2.0, 4.0),
        )
    return ProcessHorizonConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        beam_width=args.beam_width,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        for mode in args.modes:
            print(
                f"designing process horizon for {task}/{mode}: "
                f"segments={config.segments}, q={config.horizon_steps}, B={config.beam_width}"
            )
            pulse = design_pulse(task, mode, config)
            rows.extend(evaluate_pulse(task, mode, pulse, config))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('process_horizon_results.csv')}")
    print(f"wrote aggregate table to {result_path('process_horizon_summary.md')}")
    print(f"wrote figure to {figure_path('process_horizon.pdf')}")


if __name__ == "__main__":
    main()
