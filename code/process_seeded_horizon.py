"""Process-GRAPE-seeded receding-horizon diagnostic.

The finite-candidate process horizon in ``process_horizon.py`` shows that simply
replacing a state-transfer score by a process score is not enough.  This script
tests the next journal-level idea: use a gradient-optimized process pulse as a
reference trajectory, then run a local receding-horizon search around that
reference.

The result is a diagnostic bridge, not a standalone Lyapunov theorem.  It asks
whether process-level horizon control can preserve the performance of a strong
terminal process optimizer once a gradient reference is available.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ensemble_grape_baseline import (
    GrapeConfig,
    average_gate_fidelity,
    optimize_pulse,
    target_gate,
)
from horizon_lyapunov import dagger, fidelity, interaction_frame_operator, problem, random_disorder, unitary2
from paths import figure_path, result_path


@dataclass(frozen=True)
class SeededProcessConfig:
    segments: int = 60
    horizon_steps: int = 4
    beam_width: int = 6
    umax: float = 4.0
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = tuple(range(8))
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    local_radii: tuple[float, ...] = (0.15, 0.35, 0.7)
    reference_weight: float = 1.0
    target_weight: float = 0.08
    trust_weight: float = 2e-3
    energy_weight: float = 1e-4
    grape_maxiter: int = 120


def local_candidates(center: np.ndarray, config: SeededProcessConfig) -> tuple[np.ndarray, ...]:
    offsets = [np.zeros_like(center)]
    for radius in config.local_radii:
        offsets.extend(
            [
                np.array([radius, 0.0]),
                np.array([-radius, 0.0]),
                np.array([0.0, radius]),
                np.array([0.0, -radius]),
                np.array([radius, radius]) / np.sqrt(2.0),
                np.array([radius, -radius]) / np.sqrt(2.0),
                np.array([-radius, radius]) / np.sqrt(2.0),
                np.array([-radius, -radius]) / np.sqrt(2.0),
            ]
        )
    candidates = []
    seen: set[tuple[float, ...]] = set()
    for offset in offsets:
        candidate = np.clip(center + offset, -config.umax, config.umax)
        key = tuple(np.round(candidate, 12))
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return tuple(candidates)


def precompute_caches(
    task: str,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    config: SeededProcessConfig,
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[tuple[np.ndarray, ...], ...],
]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
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
    return controls_cache, disorder_cache


def step_unitaries(
    unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    next_unitaries = []
    for unitary, strength, disorder_i in zip(unitaries, strengths, disorders_i):
        hamiltonian = strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            hamiltonian = hamiltonian + coeff * hc_i
        next_unitaries.append(unitary2(hamiltonian, dt) @ unitary)
    return tuple(next_unitaries)


def rollout_reference_paths(
    task: str,
    reference_pulse: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    config: SeededProcessConfig,
) -> tuple[tuple[np.ndarray, ...], ...]:
    p = problem(task)
    dt = p.t_final / config.segments
    strengths = tuple(strength for strength, _ in scenarios)
    paths = [[np.eye(2, dtype=complex)] for _ in scenarios]
    unitaries = tuple(path[0] for path in paths)
    for index, control in enumerate(reference_pulse):
        unitaries = step_unitaries(
            unitaries,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dt,
            control,
        )
        for scenario_index, unitary in enumerate(unitaries):
            paths[scenario_index].append(unitary)
    return tuple(tuple(path) for path in paths)


def horizon_cost(
    states: tuple[np.ndarray, ...],
    reference_paths: tuple[tuple[np.ndarray, ...], ...],
    target: np.ndarray,
    target_index: int,
    control_energy: float,
    trust_energy: float,
    config: SeededProcessConfig,
) -> float:
    ref_infids = []
    target_infids = []
    for scenario_index, unitary in enumerate(states):
        reference = reference_paths[scenario_index][target_index]
        ref_infids.append(1.0 - average_gate_fidelity(unitary, reference))
        target_infids.append(1.0 - average_gate_fidelity(unitary, target))
    return float(
        config.reference_weight * np.mean(ref_infids)
        + config.target_weight * np.mean(target_infids)
        + config.energy_weight * control_energy
        + config.trust_weight * trust_energy
    )


def select_control(
    start_index: int,
    unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    reference_pulse: np.ndarray,
    reference_paths: tuple[tuple[np.ndarray, ...], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    target: np.ndarray,
    dt: float,
    config: SeededProcessConfig,
) -> np.ndarray:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float, float]] = [
        ((), unitaries, 0.0, 0.0)
    ]
    for depth in range(config.horizon_steps):
        cache_index = min(start_index + depth, config.segments - 1)
        center = reference_pulse[cache_index]
        candidates = local_candidates(center, config)
        expanded = []
        for sequence, states, energy_sum, trust_sum in beams:
            for candidate in candidates:
                next_states = step_unitaries(
                    states,
                    strengths,
                    controls_cache[cache_index],
                    disorder_cache[cache_index],
                    dt,
                    candidate,
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                next_trust = trust_sum + float(np.dot(candidate - center, candidate - center))
                target_index = min(start_index + depth + 1, config.segments)
                cost = horizon_cost(
                    next_states,
                    reference_paths,
                    target,
                    target_index,
                    next_energy / float(depth + 1),
                    next_trust / float(depth + 1),
                    config,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy, next_trust))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum, trust_sum)
            for _, sequence, states, energy_sum, trust_sum in expanded[: config.beam_width]
        ]
    return beams[0][0][0]


def design_reference(task: str, config: SeededProcessConfig) -> tuple[np.ndarray, float, int, int, bool, float]:
    grape_config = GrapeConfig(
        segments=config.segments,
        maxiter=config.grape_maxiter,
        training_strengths=config.train_strengths,
        training_seeds=config.train_seeds,
        eval_strength=config.eval_strength,
        eval_seeds=config.eval_seeds,
    )
    controls, value, seed, iterations, success, seconds = optimize_pulse(
        task,
        "process",
        grape_config,
    )
    return (
        np.reshape(controls, (config.segments, len(problem(task).controls))),
        value,
        seed,
        iterations,
        success,
        seconds,
    )


def design_seeded_horizon(
    task: str,
    reference_pulse: np.ndarray,
    config: SeededProcessConfig,
) -> np.ndarray:
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    reference_paths = rollout_reference_paths(
        task,
        reference_pulse,
        scenarios,
        controls_cache,
        disorder_cache,
        config,
    )
    dt = p.t_final / config.segments
    unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    pulse = []
    target = target_gate(task)
    for index in range(config.segments):
        control = select_control(
            index,
            unitaries,
            strengths,
            reference_pulse,
            reference_paths,
            controls_cache,
            disorder_cache,
            target,
            dt,
            config,
        )
        pulse.append(control)
        unitaries = step_unitaries(
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
    controller: str,
    pulse: np.ndarray,
    reference_objective: float,
    reference_restart_seed: int,
    reference_iterations: int,
    reference_success: bool,
    reference_seconds: float,
    horizon_seconds: float,
    config: SeededProcessConfig,
) -> list[dict[str, float | int | str | bool]]:
    p = problem(task)
    target = target_gate(task)
    rows: list[dict[str, float | int | str | bool]] = []
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    energy = float(np.mean(np.sum(np.square(pulse), axis=1)))
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
                "disorder_strength": config.eval_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "average_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": energy,
                "segments": config.segments,
                "horizon_steps": config.horizon_steps if controller == "seeded_process_horizon" else 0,
                "beam_width": config.beam_width if controller == "seeded_process_horizon" else 0,
                "reference_objective": reference_objective,
                "reference_restart_seed": reference_restart_seed,
                "reference_iterations": reference_iterations,
                "reference_success": str(reference_success),
                "reference_seconds": reference_seconds,
                "horizon_seconds": horizon_seconds,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["controller"])), []).append(row)
    summary: list[dict[str, str]] = []
    for (task, controller), items in sorted(groups.items()):
        state_fids = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "controller": controller,
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
                "reference_iterations": str(items[0]["reference_iterations"]),
                "reference_success": str(items[0]["reference_success"]),
                "reference_seconds": f"{float(items[0]['reference_seconds']):.3f}",
                "horizon_seconds": f"{float(items[0]['horizon_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str | bool]]) -> None:
    with result_path("process_seeded_horizon_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("process_seeded_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Process-Seeded Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [f"{row['task']}\\n{row['controller'].replace('_', ' ')}" for row in summary]
    x = np.arange(len(labels))
    state_means = np.array([float(row["state_fidelity_mean"]) for row in summary])
    gate_means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.36
    ax.bar(x - width / 2, state_means, width=width, label="state transfer")
    ax.bar(x + width / 2, gate_means, width=width, label="average gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.97, 1.001)
    ax.set_ylabel("Held-out fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("process_seeded_horizon.pdf"))
    fig.savefig(figure_path("process_seeded_horizon.png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--grape-maxiter", type=int, default=120)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SeededProcessConfig:
    if args.quick:
        return SeededProcessConfig(
            segments=12,
            horizon_steps=2,
            beam_width=3,
            train_strengths=(0.08,),
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
            local_radii=(0.25,),
            grape_maxiter=5,
        )
    return SeededProcessConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        beam_width=args.beam_width,
        grape_maxiter=args.grape_maxiter,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    rows: list[dict[str, float | int | str | bool]] = []
    for task in args.tasks:
        print(f"optimizing process-GRAPE reference for {task}", flush=True)
        reference_pulse, objective, restart_seed, iterations, success, ref_seconds = design_reference(
            task,
            config,
        )
        print(
            f"  reference objective={objective:.6g}, iterations={iterations}, "
            f"success={success}, seconds={ref_seconds:.1f}",
            flush=True,
        )
        start = time.perf_counter()
        seeded_pulse = design_seeded_horizon(task, reference_pulse, config)
        horizon_seconds = time.perf_counter() - start
        print(f"  seeded horizon seconds={horizon_seconds:.1f}", flush=True)
        rows.extend(
            evaluate_pulse(
                task,
                "process_grape_reference",
                reference_pulse,
                objective,
                restart_seed,
                iterations,
                success,
                ref_seconds,
                0.0,
                config,
            )
        )
        rows.extend(
            evaluate_pulse(
                task,
                "seeded_process_horizon",
                seeded_pulse,
                objective,
                restart_seed,
                iterations,
                success,
                ref_seconds,
                horizon_seconds,
                config,
            )
        )
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
