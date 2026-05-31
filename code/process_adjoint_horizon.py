"""Adjoint-polished process receding-horizon diagnostic.

This script upgrades the process-GRAPE-seeded finite-candidate diagnostic by
putting exact process-fidelity gradients inside each short receding-horizon
optimization.  A terminal process-GRAPE pulse is still used as a reference
trajectory and trust-region center, but the horizon score itself is optimized
continuously by Frechet derivatives of the segment propagators.

The result should be interpreted conservatively: it is evidence that
process-level adjoint information can be integrated into the horizon
architecture, not a standalone global-convergence result for beam search or
for the nonconvex process objective.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from ensemble_grape_baseline import (
    GrapeConfig,
    average_gate_fidelity,
    optimize_pulse,
    target_gate,
)
from horizon_lyapunov import dagger, fidelity, interaction_frame_operator, problem, random_disorder
from paths import figure_path, result_path


@dataclass(frozen=True)
class ProcessAdjointConfig:
    segments: int = 60
    horizon_steps: int = 4
    umax: float = 4.0
    trust_radius: float = 0.45
    horizon_maxiter: int = 8
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = tuple(range(8))
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    reference_weight: float = 1.0
    target_weight: float = 0.08
    worst_weight: float = 0.25
    trust_weight: float = 2e-3
    energy_weight: float = 1e-4
    grape_maxiter: int = 120


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
    config: ProcessAdjointConfig,
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


def rollout_reference_paths(
    task: str,
    reference_pulse: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    config: ProcessAdjointConfig,
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
    reference_paths: tuple[tuple[np.ndarray, ...], ...],
    reference_flat: np.ndarray,
    final_target: np.ndarray,
    start_index: int,
    dt: float,
    config: ProcessAdjointConfig,
) -> tuple[float, np.ndarray]:
    n_controls = len(controls_cache[0])
    horizon_steps = controls_flat.size // n_controls
    controls = np.reshape(controls_flat, (horizon_steps, n_controls))
    scenario_costs = []
    scenario_grads = []

    for scenario_index, (unitary0, strength) in enumerate(zip(start_unitaries, strengths)):
        prefixes = [unitary0]
        unitaries = []
        generators = []
        frechet_dirs = []
        for depth, coeffs in enumerate(controls):
            cache_index = min(start_index + depth, len(controls_cache) - 1)
            hamiltonian = strength * disorder_cache[cache_index][scenario_index]
            for coeff, hc_i in zip(coeffs, controls_cache[cache_index]):
                hamiltonian = hamiltonian + coeff * hc_i
            generator = -1.0j * hamiltonian * dt
            propagator = expm(generator)
            generators.append(generator)
            unitaries.append(propagator)
            frechet_dirs.append([-1.0j * hc_i * dt for hc_i in controls_cache[cache_index]])
            prefixes.append(propagator @ prefixes[-1])

        suffixes: list[np.ndarray] = [np.eye(final_target.shape[0], dtype=complex) for _ in range(horizon_steps)]
        running = np.eye(final_target.shape[0], dtype=complex)
        for depth in reversed(range(horizon_steps)):
            suffixes[depth] = running
            running = running @ unitaries[depth]

        final_unitary = prefixes[-1]
        reference_index = min(start_index + horizon_steps, len(reference_paths[scenario_index]) - 1)
        reference_target = reference_paths[scenario_index][reference_index]
        ref_fid, ref_grad = process_fidelity_gradient(
            final_unitary,
            reference_target,
            prefixes,
            suffixes,
            generators,
            frechet_dirs,
        )
        target_fid, target_grad = process_fidelity_gradient(
            final_unitary,
            final_target,
            prefixes,
            suffixes,
            generators,
            frechet_dirs,
        )
        cost = (
            config.reference_weight * (1.0 - ref_fid)
            + config.target_weight * (1.0 - target_fid)
        )
        grad = -config.reference_weight * ref_grad - config.target_weight * target_grad
        scenario_costs.append(cost)
        scenario_grads.append(grad)

    costs = np.array(scenario_costs, dtype=float)
    grads = np.vstack(scenario_grads)
    worst_index = int(np.argmax(costs))
    trust_delta = controls_flat - reference_flat
    objective = float(
        np.mean(costs)
        + config.worst_weight * costs[worst_index]
        + config.energy_weight * np.mean(np.square(controls_flat))
        + config.trust_weight * np.mean(np.square(trust_delta))
    )
    gradient = (
        np.mean(grads, axis=0)
        + config.worst_weight * grads[worst_index]
        + config.energy_weight * (2.0 / controls_flat.size) * controls_flat
        + config.trust_weight * (2.0 / controls_flat.size) * trust_delta
    )
    return objective, gradient


def design_reference(task: str, config: ProcessAdjointConfig) -> tuple[np.ndarray, float, int, int, bool, float]:
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


def design_adjoint_horizon(
    task: str,
    reference_pulse: np.ndarray,
    config: ProcessAdjointConfig,
) -> tuple[np.ndarray, float, int, bool, float]:
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
    target = target_gate(task)
    unitaries = tuple(np.eye(2, dtype=complex) for _ in scenarios)
    pulse = []
    objectives = []
    iterations = []
    successes = []
    start = time.perf_counter()

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        reference_block = reference_pulse[index : index + horizon].reshape(-1).copy()
        lower = np.maximum(-config.umax, reference_block - config.trust_radius)
        upper = np.minimum(config.umax, reference_block + config.trust_radius)
        result = minimize(
            lambda x: horizon_objective_and_gradient(
                x,
                unitaries,
                strengths,
                controls_cache,
                disorder_cache,
                reference_paths,
                reference_block,
                target,
                index,
                dt,
                config,
            ),
            reference_block,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper)),
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

    return (
        np.vstack(pulse),
        float(np.mean(objectives)),
        int(sum(iterations)),
        all(successes),
        time.perf_counter() - start,
    )


def evaluate_pulse(
    task: str,
    controller: str,
    pulse: np.ndarray,
    reference_objective: float,
    reference_restart_seed: int,
    reference_iterations: int,
    reference_success: bool,
    reference_seconds: float,
    horizon_objective: float,
    horizon_iterations: int,
    horizon_success: bool,
    horizon_seconds: float,
    config: ProcessAdjointConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    rows: list[dict[str, float | int | str]] = []
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
                "horizon_steps": config.horizon_steps if controller == "adjoint_process_horizon" else 0,
                "reference_objective": reference_objective,
                "reference_restart_seed": reference_restart_seed,
                "reference_iterations": reference_iterations,
                "reference_success": str(reference_success),
                "reference_seconds": reference_seconds,
                "horizon_objective": horizon_objective,
                "horizon_iterations": horizon_iterations,
                "horizon_success": str(horizon_success),
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
        state_fids = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        state_ci = 1.96 * float(np.std(state_fids)) / np.sqrt(len(state_fids))
        gate_ci = 1.96 * float(np.std(gate_fids)) / np.sqrt(len(gate_fids))
        summary.append(
            {
                "task": task,
                "controller": controller,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state_fids):.6g}",
                "state_fidelity_min": f"{np.min(state_fids):.6g}",
                "state_fidelity_ci95": f"{state_ci:.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_ci95": f"{gate_ci:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "horizon_steps": str(items[0]["horizon_steps"]),
                "reference_iterations": str(items[0]["reference_iterations"]),
                "reference_success": str(items[0]["reference_success"]),
                "horizon_iterations": str(items[0]["horizon_iterations"]),
                "horizon_success": str(items[0]["horizon_success"]),
                "reference_seconds": f"{float(items[0]['reference_seconds']):.3f}",
                "horizon_seconds": f"{float(items[0]['horizon_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("process_adjoint_horizon_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("process_adjoint_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Process-Adjoint Horizon Summary\n\n")
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
    fig.savefig(figure_path("process_adjoint_horizon.pdf"))
    fig.savefig(figure_path("process_adjoint_horizon.png"), dpi=220)
    plt.close(fig)


def gradient_check() -> None:
    rng = np.random.default_rng(123)
    config = ProcessAdjointConfig(
        segments=6,
        horizon_steps=3,
        train_strengths=(0.08,),
        train_seeds=(0,),
        eval_seeds=(10,),
        grape_maxiter=2,
    )
    task = "Z"
    scenarios = ((0.08, random_disorder(0)),)
    controls_cache, disorder_cache = precompute_caches(task, scenarios, config)
    reference = rng.normal(scale=0.05, size=(config.segments, len(problem(task).controls)))
    reference_paths = rollout_reference_paths(
        task,
        reference,
        scenarios,
        controls_cache,
        disorder_cache,
        config,
    )
    x = rng.normal(scale=0.05, size=config.horizon_steps * len(problem(task).controls))
    unitary0 = (np.eye(2, dtype=complex),)
    dt = problem(task).t_final / config.segments
    value, grad = horizon_objective_and_gradient(
        x,
        unitary0,
        (0.08,),
        controls_cache,
        disorder_cache,
        reference_paths,
        reference[: config.horizon_steps].reshape(-1),
        target_gate(task),
        0,
        dt,
        config,
    )
    direction = rng.normal(size=x.size)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        x + eps * direction,
        unitary0,
        (0.08,),
        controls_cache,
        disorder_cache,
        reference_paths,
        reference[: config.horizon_steps].reshape(-1),
        target_gate(task),
        0,
        dt,
        config,
    )[0]
    minus = horizon_objective_and_gradient(
        x - eps * direction,
        unitary0,
        (0.08,),
        controls_cache,
        disorder_cache,
        reference_paths,
        reference[: config.horizon_steps].reshape(-1),
        target_gate(task),
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
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--horizon-maxiter", type=int, default=8)
    parser.add_argument("--grape-maxiter", type=int, default=120)
    parser.add_argument("--trust-radius", type=float, default=0.45)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ProcessAdjointConfig:
    if args.quick:
        return ProcessAdjointConfig(
            segments=12,
            horizon_steps=2,
            horizon_maxiter=3,
            train_strengths=(0.08,),
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
            trust_radius=0.35,
            grape_maxiter=5,
        )
    return ProcessAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        grape_maxiter=args.grape_maxiter,
        trust_radius=args.trust_radius,
    )


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
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
        pulse, horizon_objective, horizon_iterations, horizon_success, horizon_seconds = (
            design_adjoint_horizon(task, reference_pulse, config)
        )
        print(
            f"  adjoint horizon objective={horizon_objective:.6g}, "
            f"iterations={horizon_iterations}, success={horizon_success}, "
            f"seconds={horizon_seconds:.1f}",
            flush=True,
        )
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
                0,
                True,
                0.0,
                config,
            )
        )
        rows.extend(
            evaluate_pulse(
                task,
                "adjoint_process_horizon",
                pulse,
                objective,
                restart_seed,
                iterations,
                success,
                ref_seconds,
                horizon_objective,
                horizon_iterations,
                horizon_success,
                horizon_seconds,
                config,
            )
        )
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
