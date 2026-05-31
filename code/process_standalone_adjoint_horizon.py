"""Standalone process-adjoint receding-horizon diagnostic.

This script tests a process-level horizon objective without a terminal
process-GRAPE reference.  A finite-candidate process horizon supplies only the
local initializer for each short horizon; the short process score is then
optimized continuously with exact Frechet derivatives.

The result is a diagnostic for the journal extension path.  It is not a global
optimal-control solver and it does not claim convergence of the nonconvex
process objective.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from ensemble_grape_baseline import average_gate_fidelity, target_gate
from horizon_lyapunov import (
    candidate_controls,
    dagger,
    default_beam_width,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
)
from paths import result_path


@dataclass(frozen=True)
class StandaloneProcessConfig:
    segments: int = 60
    horizon_steps: int = 4
    umax: float = 4.0
    horizon_maxiter: int = 8
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = tuple(range(8))
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    energy_weight: float = 1e-4
    trust_weight: float = 2e-3


@dataclass(frozen=True)
class DesignStats:
    objective_mean: float
    iterations: int
    successful_steps: int
    seconds: float


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
        next_unitaries.append(expm(-1.0j * hamiltonian * dt) @ unitary)
    return tuple(next_unitaries)


def precompute_caches(
    task: str,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    config: StandaloneProcessConfig,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], tuple[tuple[np.ndarray, ...], ...]]:
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


def process_cost(
    unitaries: tuple[np.ndarray, ...],
    target: np.ndarray,
    energy_mean: float,
    config: StandaloneProcessConfig,
) -> float:
    infids = np.array([1.0 - average_gate_fidelity(unitary, target) for unitary in unitaries])
    return float(
        np.mean(infids)
        + config.worst_weight * np.max(infids)
        + config.energy_weight * energy_mean
    )


def select_seed_sequence(
    unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    target: np.ndarray,
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    config: StandaloneProcessConfig,
) -> tuple[np.ndarray, ...]:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), unitaries, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        expanded = []
        for sequence, states, energy_sum in beams:
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
                cost = process_cost(
                    next_states,
                    target,
                    next_energy / float(depth + 1),
                    config,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0]


def process_fidelity_gradient(
    final_unitary: np.ndarray,
    target: np.ndarray,
    prefixes: list[np.ndarray],
    suffixes: list[np.ndarray],
    generators: list[np.ndarray],
    frechet_dirs: list[list[np.ndarray]],
) -> tuple[float, np.ndarray]:
    dim = target.shape[0]
    n_steps = len(generators)
    n_controls = len(frechet_dirs[0])
    overlap = np.trace(dagger(target) @ final_unitary)
    fidelity_value = average_gate_fidelity(final_unitary, target)
    gradient = np.zeros((n_steps, n_controls), dtype=float)
    scale = float(dim * (dim + 1))

    for depth in range(n_steps):
        for control_index in range(n_controls):
            d_unitary = expm_frechet(
                generators[depth],
                frechet_dirs[depth][control_index],
                compute_expm=False,
            )
            d_total = suffixes[depth] @ d_unitary @ prefixes[depth]
            d_overlap = np.trace(dagger(target) @ d_total)
            gradient[depth, control_index] = (
                2.0 * float(np.real(np.conjugate(overlap) * d_overlap)) / scale
            )
    return fidelity_value, gradient.reshape(-1)


def horizon_objective_and_gradient(
    controls_flat: np.ndarray,
    start_unitaries: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    seed_flat: np.ndarray,
    target: np.ndarray,
    start_index: int,
    dt: float,
    config: StandaloneProcessConfig,
) -> tuple[float, np.ndarray]:
    n_controls = len(controls_cache[0])
    controls = np.reshape(controls_flat, (-1, n_controls))
    horizon_steps = controls.shape[0]
    costs = []
    grads = []

    for scenario_index, (unitary0, strength) in enumerate(zip(start_unitaries, strengths)):
        prefixes = [unitary0]
        propagators = []
        generators = []
        frechet_dirs = []
        for depth, coeffs in enumerate(controls):
            cache_index = min(start_index + depth, len(controls_cache) - 1)
            hamiltonian = strength * disorder_cache[cache_index][scenario_index]
            dirs = []
            for coeff, hc_i in zip(coeffs, controls_cache[cache_index]):
                hamiltonian = hamiltonian + coeff * hc_i
                dirs.append(-1.0j * hc_i * dt)
            generator = -1.0j * hamiltonian * dt
            propagator = expm(generator)
            generators.append(generator)
            propagators.append(propagator)
            frechet_dirs.append(dirs)
            prefixes.append(propagator @ prefixes[-1])

        suffixes: list[np.ndarray] = [
            np.eye(target.shape[0], dtype=complex) for _ in range(horizon_steps)
        ]
        running = np.eye(target.shape[0], dtype=complex)
        for depth in reversed(range(horizon_steps)):
            suffixes[depth] = running
            running = running @ propagators[depth]

        final_unitary = prefixes[-1]
        fid, fid_grad = process_fidelity_gradient(
            final_unitary,
            target,
            prefixes,
            suffixes,
            generators,
            frechet_dirs,
        )
        costs.append(1.0 - fid)
        grads.append(-fid_grad)

    costs_array = np.array(costs, dtype=float)
    grads_array = np.vstack(grads)
    worst_index = int(np.argmax(costs_array))
    trust_delta = controls_flat - seed_flat
    objective = float(
        np.mean(costs_array)
        + config.worst_weight * costs_array[worst_index]
        + config.energy_weight * np.mean(np.square(controls_flat))
        + config.trust_weight * np.mean(np.square(trust_delta))
    )
    gradient = (
        np.mean(grads_array, axis=0)
        + config.worst_weight * grads_array[worst_index]
        + config.energy_weight * (2.0 / controls_flat.size) * controls_flat
        + config.trust_weight * (2.0 / controls_flat.size) * trust_delta
    )
    return objective, gradient


def design_seed_pulse(
    task: str,
    beam_width: int,
    config: StandaloneProcessConfig,
) -> np.ndarray:
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    candidates = candidate_controls(config.amplitudes)
    target = target_gate(task)
    dt = p.t_final / config.segments
    unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    pulse = []

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        sequence = select_seed_sequence(
            unitaries,
            strengths,
            controls_cache,
            disorder_cache,
            target,
            index,
            dt,
            candidates,
            horizon,
            beam_width,
            config,
        )
        control = sequence[0]
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


def design_adjoint_pulse(
    task: str,
    beam_width: int,
    config: StandaloneProcessConfig,
) -> tuple[np.ndarray, DesignStats]:
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    candidates = candidate_controls(config.amplitudes)
    target = target_gate(task)
    dt = p.t_final / config.segments
    unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    pulse = []
    objectives = []
    iterations = []
    successes = []
    start = time.perf_counter()

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        seed_sequence = select_seed_sequence(
            unitaries,
            strengths,
            controls_cache,
            disorder_cache,
            target,
            index,
            dt,
            candidates,
            horizon,
            beam_width,
            config,
        )
        seed_flat = np.reshape(np.vstack(seed_sequence), -1)
        result = minimize(
            lambda x: horizon_objective_and_gradient(
                x,
                unitaries,
                strengths,
                controls_cache,
                disorder_cache,
                seed_flat,
                target,
                index,
                dt,
                config,
            ),
            seed_flat,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-config.umax, config.umax)] * seed_flat.size,
            options={"maxiter": config.horizon_maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        control = np.reshape(result.x, (horizon, len(p.controls)))[0]
        pulse.append(control)
        objectives.append(float(result.fun))
        iterations.append(int(result.nit))
        successes.append(bool(result.success))
        unitaries = step_unitaries(
            unitaries,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dt,
            control,
        )

    return np.vstack(pulse), DesignStats(
        objective_mean=float(np.mean(objectives)),
        iterations=int(sum(iterations)),
        successful_steps=int(sum(successes)),
        seconds=time.perf_counter() - start,
    )


def evaluate_pulse(
    task: str,
    controller: str,
    pulse: np.ndarray,
    stats: DesignStats,
    config: StandaloneProcessConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    rows: list[dict[str, float | int | str]] = []
    energy = float(np.mean(np.sum(np.square(pulse), axis=1)))
    for seed in config.eval_seeds:
        disorder = random_disorder(seed)
        unitary = np.eye(2, dtype=complex)
        for index, control in enumerate(pulse):
            t = float(t_eval[index])
            controls_i = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
            disorder_i = interaction_frame_operator(p, disorder, t)
            unitary = step_unitaries(
                (unitary,),
                (config.eval_strength,),
                controls_i,
                (disorder_i,),
                dt,
                control,
            )[0]
        rho_final = unitary @ p.initial @ dagger(unitary)
        rows.append(
            {
                "task": task,
                "controller": controller,
                "eval_strength": config.eval_strength,
                "seed": seed,
                "state_fidelity": fidelity(rho_final, p.target),
                "avg_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": energy,
                "segments": len(pulse),
                "horizon_steps": config.horizon_steps,
                "objective_mean": stats.objective_mean,
                "horizon_iterations": stats.iterations,
                "successful_steps": stats.successful_steps,
                "design_seconds": stats.seconds,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["controller"])), []).append(row)

    summary = []
    for (task, controller), items in sorted(groups.items()):
        state = np.array([float(row["state_fidelity"]) for row in items])
        gate = np.array([float(row["avg_gate_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "controller": controller,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state):.6g}",
                "state_fidelity_min": f"{np.min(state):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "objective_mean": f"{float(items[0]['objective_mean']):.6g}",
                "horizon_iterations": str(int(items[0]["horizon_iterations"])),
                "successful_steps": str(int(items[0]["successful_steps"])),
                "design_seconds": f"{float(items[0]['design_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("process_standalone_adjoint_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("process_standalone_adjoint_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Standalone Process Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def gradient_check() -> None:
    config = StandaloneProcessConfig(
        segments=12,
        horizon_steps=3,
        train_strengths=(0.08,),
        train_seeds=(0, 1),
        eval_seeds=tuple(range(10, 14)),
    )
    task = "Z"
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    start_unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    rng = np.random.default_rng(321)
    x = rng.normal(scale=0.25, size=config.horizon_steps * len(problem(task).controls))
    seed = np.zeros_like(x)
    direction = rng.normal(size=x.size)
    direction = direction / np.linalg.norm(direction)
    dt = problem(task).t_final / config.segments
    target = target_gate(task)
    value, grad = horizon_objective_and_gradient(
        x,
        start_unitaries,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        target,
        0,
        dt,
        config,
    )
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        x + eps * direction,
        start_unitaries,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        target,
        0,
        dt,
        config,
    )[0]
    minus = horizon_objective_and_gradient(
        x - eps * direction,
        start_unitaries,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        target,
        0,
        dt,
        config,
    )[0]
    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(np.dot(grad, direction))
    print(
        f"value={value:.8g}, finite_diff={finite_difference:.8g}, "
        f"analytic={analytic:.8g}, error={abs(finite_difference - analytic):.3g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StandaloneProcessConfig:
    if args.quick:
        return StandaloneProcessConfig(
            segments=20,
            horizon_steps=3,
            horizon_maxiter=3,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
        )
    return StandaloneProcessConfig()


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        beam_width = default_beam_width(task)
        print(f"designing finite-candidate process seed for {task}", flush=True)
        seed_start = time.perf_counter()
        seed_pulse = design_seed_pulse(task, beam_width, config)
        seed_stats = DesignStats(
            objective_mean=0.0,
            iterations=0,
            successful_steps=config.segments,
            seconds=time.perf_counter() - seed_start,
        )
        rows.extend(evaluate_pulse(task, "finite_process_seed", seed_pulse, seed_stats, config))

        print(f"designing standalone process-adjoint horizon for {task}", flush=True)
        adjoint_pulse, adjoint_stats = design_adjoint_pulse(task, beam_width, config)
        rows.extend(
            evaluate_pulse(
                task,
                "standalone_process_adjoint",
                adjoint_pulse,
                adjoint_stats,
                config,
            )
        )
        print(
            f"  objective={adjoint_stats.objective_mean:.6g}, "
            f"iterations={adjoint_stats.iterations}, "
            f"successful_steps={adjoint_stats.successful_steps}, "
            f"seconds={adjoint_stats.seconds:.1f}",
            flush=True,
        )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('process_standalone_adjoint_results.csv')}")
    print(f"wrote aggregate table to {result_path('process_standalone_adjoint_summary.md')}")


if __name__ == "__main__":
    main()
