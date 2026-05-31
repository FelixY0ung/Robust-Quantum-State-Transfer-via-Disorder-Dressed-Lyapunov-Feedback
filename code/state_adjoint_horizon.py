"""Standalone adjoint-assisted state-transfer horizon diagnostic.

The process, leakage, and open-system adjoint horizons in the manuscript are
reference assisted.  This script tests a less reference-dependent step for the
main two-level state-transfer task: use the finite-candidate beam horizon only
as a local initializer, then continuously optimize the same short state-transfer
Lyapunov score with exact Frechet derivatives of the segment propagators.

This is still a local nonconvex receding-horizon solver, not a global
convergence proof.  Its role is to test whether adjoint information can improve
the proposed horizon architecture without a terminal GRAPE reference.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from horizon_lyapunov import (
    TAIL_FRACTION,
    candidate_controls,
    dagger,
    default_beam_width,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    problem,
    random_disorder,
    scenario_cost,
)
from paths import result_path


@dataclass(frozen=True)
class StateAdjointConfig:
    segments: int = 60
    horizon_steps: int = 4
    umax: float = 4.0
    horizon_maxiter: int = 8
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = tuple(range(8))
    eval_strengths: tuple[float, ...] = (0.05, 0.08)
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


def step_density_exact(
    rho: np.ndarray,
    strength: float,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    dt: float,
    control: np.ndarray,
) -> np.ndarray:
    hamiltonian = strength * disorder_i
    for coeff, hc_i in zip(control, controls_i):
        hamiltonian = hamiltonian + coeff * hc_i
    unitary = expm(-1.0j * hamiltonian * dt)
    return hermitize_trace_one(unitary @ rho @ dagger(unitary))


def step_precomputed_exact(
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        step_density_exact(rho, strength, controls_i, disorder_i, dt, control)
        for rho, strength, disorder_i in zip(rhos, strengths, disorders_i)
    )


def precompute_caches(
    task: str,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    config: StateAdjointConfig,
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


def select_seed_sequence(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    config: StateAdjointConfig,
) -> tuple[np.ndarray, ...]:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_precomputed_exact(
                    states,
                    strengths,
                    controls_cache[cache_index],
                    disorder_cache[cache_index],
                    dt,
                    candidate,
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = scenario_cost(
                    p,
                    next_states,
                    next_energy / float(depth + 1),
                    config.worst_weight,
                    config.energy_weight,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0]


def fidelity_and_gradient(
    rho0: np.ndarray,
    target: np.ndarray,
    strength: float,
    controls: np.ndarray,
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    scenario_index: int,
    start_index: int,
    dt: float,
) -> tuple[float, np.ndarray]:
    horizon_steps, n_controls = controls.shape
    dim = rho0.shape[0]
    prefixes = [np.eye(dim, dtype=complex)]
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
        np.eye(dim, dtype=complex) for _ in range(horizon_steps)
    ]
    running = np.eye(dim, dtype=complex)
    for depth in reversed(range(horizon_steps)):
        suffixes[depth] = running
        running = running @ propagators[depth]

    final_unitary = prefixes[-1]
    rho_final = final_unitary @ rho0 @ dagger(final_unitary)
    fid = float(np.real(np.trace(target @ rho_final)))
    grad = np.zeros((horizon_steps, n_controls), dtype=float)
    final_adjoint = dagger(final_unitary)

    for depth in range(horizon_steps):
        for control_index in range(n_controls):
            d_propagator = expm_frechet(
                generators[depth],
                frechet_dirs[depth][control_index],
                compute_expm=False,
            )
            d_unitary = suffixes[depth] @ d_propagator @ prefixes[depth]
            d_fid = 2.0 * np.real(np.trace(target @ d_unitary @ rho0 @ final_adjoint))
            grad[depth, control_index] = float(d_fid)
    return fid, grad.reshape(-1)


def horizon_objective_and_gradient(
    controls_flat: np.ndarray,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    seed_flat: np.ndarray,
    target: np.ndarray,
    start_index: int,
    dt: float,
    config: StateAdjointConfig,
) -> tuple[float, np.ndarray]:
    n_controls = len(controls_cache[0])
    controls = np.reshape(controls_flat, (-1, n_controls))
    costs = []
    grads = []
    for scenario_index, (rho0, strength) in enumerate(zip(rhos, strengths)):
        fid, fid_grad = fidelity_and_gradient(
            rho0,
            target,
            strength,
            controls,
            controls_cache,
            disorder_cache,
            scenario_index,
            start_index,
            dt,
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
    config: StateAdjointConfig,
) -> np.ndarray:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    candidates = candidate_controls(config.amplitudes)
    pulse = []

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        sequence = select_seed_sequence(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            index,
            dt,
            candidates,
            horizon,
            beam_width,
            config,
        )
        control = sequence[0]
        pulse.append(control)
        rhos = step_precomputed_exact(
            rhos,
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
    config: StateAdjointConfig,
) -> tuple[np.ndarray, DesignStats]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    candidates = candidate_controls(config.amplitudes)
    pulse = []
    objectives = []
    iterations = []
    successes = []
    start = time.perf_counter()

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        seed_sequence = select_seed_sequence(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            index,
            dt,
            candidates,
            horizon,
            beam_width,
            config,
        )
        seed_flat = np.reshape(np.vstack(seed_sequence), -1)
        bounds = [(-config.umax, config.umax)] * seed_flat.size
        result = minimize(
            lambda x: horizon_objective_and_gradient(
                x,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                seed_flat,
                p.target,
                index,
                dt,
                config,
            ),
            seed_flat,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": config.horizon_maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        control = np.reshape(result.x, (horizon, len(p.controls)))[0]
        pulse.append(control)
        objectives.append(float(result.fun))
        iterations.append(int(result.nit))
        successes.append(bool(result.success))
        rhos = step_precomputed_exact(
            rhos,
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
    config: StateAdjointConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    rows: list[dict[str, float | int | str]] = []
    energy = float(np.mean(np.sum(np.square(pulse), axis=1)))
    for strength in config.eval_strengths:
        for seed in config.eval_seeds:
            disorder = random_disorder(seed)
            rho = p.initial.copy()
            fids = []
            for index, t in enumerate(t_eval):
                rho = hermitize_trace_one(rho)
                fids.append(fidelity(rho, p.target))
                if index < len(pulse):
                    controls_i = tuple(
                        interaction_frame_operator(p, hc, float(t)) for hc in p.controls
                    )
                    disorder_i = interaction_frame_operator(p, disorder, float(t))
                    rho = step_density_exact(
                        rho,
                        strength,
                        controls_i,
                        disorder_i,
                        dt,
                        pulse[index],
                    )
            inf = np.maximum(0.0, 1.0 - np.array(fids))
            tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
            rows.append(
                {
                    "task": task,
                    "controller": controller,
                    "disorder_strength": strength,
                    "seed": seed,
                    "final_fidelity": float(fids[-1]),
                    "tail_infidelity_mean": float(np.mean(tail)),
                    "tail_stability_range": float(np.max(tail) - np.min(tail)),
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
    groups: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["task"]), str(row["controller"]), float(row["disorder_strength"])),
            [],
        ).append(row)

    summary = []
    for (task, controller, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        tails = np.array([float(row["tail_infidelity_mean"]) for row in items])
        summary.append(
            {
                "task": task,
                "controller": controller,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "tail_infidelity_mean": f"{np.mean(tails):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "objective_mean": f"{float(items[0]['objective_mean']):.6g}",
                "horizon_iterations": str(int(items[0]["horizon_iterations"])),
                "successful_steps": str(int(items[0]["successful_steps"])),
                "design_seconds": f"{float(items[0]['design_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("state_adjoint_horizon_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("state_adjoint_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# State Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def gradient_check() -> None:
    config = StateAdjointConfig(
        segments=12,
        horizon_steps=3,
        train_strengths=(0.08,),
        train_seeds=(0, 1),
        eval_seeds=tuple(range(10, 14)),
    )
    task = "Z"
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    rng = np.random.default_rng(123)
    x = rng.normal(scale=0.3, size=config.horizon_steps * len(p.controls))
    seed = np.zeros_like(x)
    direction = rng.normal(size=x.size)
    direction = direction / np.linalg.norm(direction)
    dt = p.t_final / config.segments
    value, grad = horizon_objective_and_gradient(
        x,
        rhos,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        p.target,
        0,
        dt,
        config,
    )
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        x + eps * direction,
        rhos,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        p.target,
        0,
        dt,
        config,
    )[0]
    minus = horizon_objective_and_gradient(
        x - eps * direction,
        rhos,
        strengths,
        controls_cache,
        disorder_cache,
        seed,
        p.target,
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


def config_from_args(args: argparse.Namespace) -> StateAdjointConfig:
    if args.quick:
        return StateAdjointConfig(
            segments=20,
            horizon_steps=3,
            horizon_maxiter=3,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
        )
    return StateAdjointConfig()


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        beam_width = default_beam_width(task)
        print(f"designing finite-candidate seed pulse for {task}", flush=True)
        seed_start = time.perf_counter()
        seed_pulse = design_seed_pulse(task, beam_width, config)
        seed_stats = DesignStats(
            objective_mean=0.0,
            iterations=0,
            successful_steps=config.segments,
            seconds=time.perf_counter() - seed_start,
        )
        rows.extend(evaluate_pulse(task, "finite_candidate_seed", seed_pulse, seed_stats, config))

        print(f"designing adjoint state horizon for {task}", flush=True)
        adjoint_pulse, adjoint_stats = design_adjoint_pulse(task, beam_width, config)
        rows.extend(
            evaluate_pulse(task, "adjoint_state_horizon", adjoint_pulse, adjoint_stats, config)
        )
        print(
            f"  objective={adjoint_stats.objective_mean:.6g}, "
            f"iterations={adjoint_stats.iterations}, "
            f"successful_steps={adjoint_stats.successful_steps}, "
            f"seconds={adjoint_stats.seconds:.1f}",
            flush=True,
        )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('state_adjoint_horizon_results.csv')}")
    print(f"wrote aggregate table to {result_path('state_adjoint_horizon_summary.md')}")


if __name__ == "__main__":
    main()
