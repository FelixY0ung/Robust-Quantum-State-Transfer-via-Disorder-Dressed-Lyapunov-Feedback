"""Informationally complete state-ensemble horizon controller.

The main beam-horizon controller optimizes one initial state, while the
process-horizon prototype optimizes average gate fidelity directly.  This
script tests an intermediate, Lyapunov-shaped gate-level extension: the
short-horizon score averages state infidelities over an informationally
complete set of qubit input states.  It remains a finite-candidate
receding-horizon controller and does not use a GRAPE or dCRAB reference pulse.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import (
    candidate_controls,
    dagger,
    fidelity,
    interaction_frame_operator,
    ket,
    normalize,
    problem,
    random_disorder,
    unitary2,
)
from paths import figure_path, result_path
from process_horizon import (
    ProcessHorizonConfig,
    average_gate_fidelity,
    phase_adjusted_target,
    target_gate,
    target_path,
)


@dataclass(frozen=True)
class StateEnsembleConfig:
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


def tomography_inputs() -> tuple[np.ndarray, ...]:
    z0 = ket(0)
    z1 = ket(1)
    plus = normalize(z0 + z1)
    plus_i = normalize(z0 + 1.0j * z1)
    return z0, z1, plus, plus_i


def state_ensemble_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    fids = []
    for psi in tomography_inputs():
        actual_psi = actual @ psi
        target_psi = target @ psi
        fids.append(abs((dagger(target_psi) @ actual_psi)[0, 0]) ** 2)
    return float(np.real(np.mean(fids)))


def state_ensemble_cost(
    unitaries: tuple[np.ndarray, ...],
    target: np.ndarray,
    energy_mean: float,
    worst_weight: float,
    energy_weight: float,
) -> float:
    infids = np.array([1.0 - state_ensemble_fidelity(unitary, target) for unitary in unitaries])
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
    config: StateEnsembleConfig,
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
                    states,
                    strengths,
                    controls_i,
                    disorders_i,
                    dt,
                    candidate,
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = state_ensemble_cost(
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


def design_pulse(task: str, mode: str, config: StateEnsembleConfig) -> np.ndarray:
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

    for index in range(config.segments):
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
    config: StateEnsembleConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    gate = target_gate(task)
    adjusted_gate = phase_adjusted_target(task)
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
                "controller": "state_ensemble_horizon",
                "mode": mode,
                "disorder_strength": config.test_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "state_ensemble_fidelity": state_ensemble_fidelity(unitary, adjusted_gate),
                "average_gate_fidelity": average_gate_fidelity(unitary, gate),
                "pulse_energy": energy,
                "segments": config.segments,
                "horizon_steps": config.horizon_steps,
                "beam_width": config.beam_width,
                "n_train_seeds": len(config.train_seeds),
                "n_train_strengths": len(config.train_strengths),
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
        ensemble_fids = np.array([float(row["state_ensemble_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        ci95 = 1.96 * float(np.std(gate_fids)) / np.sqrt(len(gate_fids))
        summary.append(
            {
                "task": task,
                "mode": mode,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state_fids):.6g}",
                "state_fidelity_min": f"{np.min(state_fids):.6g}",
                "state_ensemble_fidelity_mean": f"{np.mean(ensemble_fids):.6g}",
                "state_ensemble_fidelity_min": f"{np.min(ensemble_fids):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_ci95": f"{ci95:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "horizon_steps": str(items[0]["horizon_steps"]),
                "beam_width": str(items[0]["beam_width"]),
                "n_train_seeds": str(items[0]["n_train_seeds"]),
                "n_train_strengths": str(items[0]["n_train_strengths"]),
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("process_state_ensemble_horizon_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("process_state_ensemble_horizon_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Process State-Ensemble Horizon Summary\n\n")
        f.write(
            "Standalone finite-candidate horizons scored by an informationally "
            "complete set of four qubit input states.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [f"{row['task']} {row['mode']}" for row in summary]
    x = np.arange(len(labels))
    state_means = np.array([float(row["state_fidelity_mean"]) for row in summary])
    ensemble_means = np.array([float(row["state_ensemble_fidelity_mean"]) for row in summary])
    gate_means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.26
    ax.bar(x - width, state_means, width=width, label="task state")
    ax.bar(x, ensemble_means, width=width, label="state ensemble")
    ax.bar(x + width, gate_means, width=width, label="average gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.001)
    ax.set_ylabel("Held-out fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("process_state_ensemble_horizon.pdf"))
    fig.savefig(figure_path("process_state_ensemble_horizon.png"), dpi=220)
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


def config_from_args(args: argparse.Namespace) -> StateEnsembleConfig:
    if args.quick:
        return StateEnsembleConfig(
            segments=30,
            horizon_steps=3,
            beam_width=4,
            train_strengths=(0.08,),
            train_seeds=(0, 1, 2),
            test_seeds=tuple(range(10, 20)),
            amplitudes=(0.5, 1.0, 2.0, 4.0),
        )
    return StateEnsembleConfig(
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
                f"designing state-ensemble horizon for {task}/{mode}: "
                f"segments={config.segments}, q={config.horizon_steps}, B={config.beam_width}"
            )
            pulse = design_pulse(task, mode, config)
            rows.extend(evaluate_pulse(task, mode, pulse, config))
    write_outputs(rows)
    print(
        f"wrote {len(rows)} rows to "
        f"{result_path('process_state_ensemble_horizon_results.csv')}"
    )
    print(
        "wrote aggregate table to "
        f"{result_path('process_state_ensemble_horizon_summary.md')}"
    )
    print(f"wrote figure to {figure_path('process_state_ensemble_horizon.pdf')}")


if __name__ == "__main__":
    main()
