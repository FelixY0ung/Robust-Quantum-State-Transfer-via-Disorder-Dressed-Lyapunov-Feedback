"""Adjoint-polished open-system receding-horizon diagnostic.

The compact Lindblad finite-candidate horizon currently underperforms a
terminal open-system GRAPE baseline.  This script tests the next method step:
use the open-system GRAPE pulse as a reference trajectory, then optimize each
short receding horizon continuously through the Liouville-space Lindblad model
with exact Frechet derivatives.

The result is a reference-assisted diagnostic.  It shows whether dissipative
state-transfer performance can be carried into a horizon architecture when
adjoint information is used inside the horizon score; it is not a standalone
open-system Lyapunov convergence theorem.
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

from horizon_lyapunov import dagger, fidelity, interaction_frame_operator, ket, problem, random_disorder
from open_system_grape_baseline import (
    EVAL_NOISE_CASES,
    TRAIN_NOISE,
    OpenGrapeConfig,
    commutator_super,
    liouvillian,
    mat,
    optimize_open_grape,
    vec,
)
from open_system_noise import NoiseCase, evolve_open_system
from paths import figure_path, result_path


@dataclass(frozen=True)
class OpenAdjointConfig:
    segments: int = 40
    horizon_steps: int = 4
    umax: float = 4.0
    trust_radius: float = 0.35
    horizon_maxiter: int = 8
    training_strength: float = 0.08
    training_seeds: tuple[int, ...] = (0, 1, 2, 3)
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    reference_weight: float = 1.0
    target_weight: float = 0.25
    worst_weight: float = 0.25
    trust_weight: float = 2e-3
    energy_weight: float = 2e-4
    grape_maxiter: int = 100


def precompute_caches(
    task: str,
    disorders: tuple[np.ndarray, ...],
    config: OpenAdjointConfig,
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[tuple[np.ndarray, ...], ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
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
        tuple(interaction_frame_operator(p, disorder, t) for disorder in disorders)
        for t in cache_times
    )
    dephasing_cache = tuple(interaction_frame_operator(p, p.h0, t) for t in cache_times)
    relaxation_operator = ket(1) @ dagger(ket(0))
    relaxation_cache = tuple(
        interaction_frame_operator(p, relaxation_operator, t) for t in cache_times
    )
    return controls_cache, disorder_cache, dephasing_cache, relaxation_cache


def step_liouville(
    state_vectors: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dephasing_i: np.ndarray,
    relaxation_i: np.ndarray,
    dt: float,
    control: np.ndarray,
    noise: NoiseCase,
) -> tuple[np.ndarray, ...]:
    next_vectors = []
    for state, strength, disorder_i in zip(state_vectors, strengths, disorders_i):
        hamiltonian = strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            hamiltonian = hamiltonian + coeff * hc_i
        generator = liouvillian(
            hamiltonian,
            dephasing_i,
            relaxation_i,
            noise.gamma_phi,
            noise.gamma_relax,
        )
        next_vectors.append(expm(generator * dt) @ state)
    return tuple(next_vectors)


def rollout_reference_paths(
    task: str,
    reference_pulse: np.ndarray,
    disorders: tuple[np.ndarray, ...],
    config: OpenAdjointConfig,
) -> tuple[tuple[np.ndarray, ...], ...]:
    p = problem(task)
    dt = p.t_final / config.segments
    strengths = tuple(config.training_strength for _ in disorders)
    controls_cache, disorder_cache, dephasing_cache, relaxation_cache = precompute_caches(
        task,
        disorders,
        config,
    )
    paths = [[vec(p.initial)] for _ in disorders]
    states = tuple(path[0] for path in paths)
    for index, control in enumerate(reference_pulse):
        states = step_liouville(
            states,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dephasing_cache[index],
            relaxation_cache[index],
            dt,
            control,
            TRAIN_NOISE,
        )
        for scenario_index, state in enumerate(states):
            paths[scenario_index].append(state)
    return tuple(tuple(path) for path in paths)


def horizon_objective_and_gradient(
    controls_flat: np.ndarray,
    start_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dephasing_cache: tuple[np.ndarray, ...],
    relaxation_cache: tuple[np.ndarray, ...],
    reference_paths: tuple[tuple[np.ndarray, ...], ...],
    reference_flat: np.ndarray,
    task: str,
    start_index: int,
    dt: float,
    config: OpenAdjointConfig,
) -> tuple[float, np.ndarray]:
    p = problem(task)
    n_controls = len(p.controls)
    horizon_steps = controls_flat.size // n_controls
    controls = np.reshape(controls_flat, (horizon_steps, n_controls))
    target_row = vec(p.target.T)
    scenario_costs = []
    scenario_grads = []

    for scenario_index, (state0, strength) in enumerate(zip(start_states, strengths)):
        state_vectors = [state0]
        propagators = []
        generators = []
        frechet_dirs = []
        for depth, coeffs in enumerate(controls):
            cache_index = min(start_index + depth, len(controls_cache) - 1)
            hamiltonian = strength * disorder_cache[cache_index][scenario_index]
            for coeff, hc_i in zip(coeffs, controls_cache[cache_index]):
                hamiltonian = hamiltonian + coeff * hc_i
            generator_dt = (
                liouvillian(
                    hamiltonian,
                    dephasing_cache[cache_index],
                    relaxation_cache[cache_index],
                    TRAIN_NOISE.gamma_phi,
                    TRAIN_NOISE.gamma_relax,
                )
                * dt
            )
            propagator = expm(generator_dt)
            generators.append(generator_dt)
            propagators.append(propagator)
            frechet_dirs.append(
                [commutator_super(hc_i) * dt for hc_i in controls_cache[cache_index]]
            )
            state_vectors.append(propagator @ state_vectors[-1])

        final_state = state_vectors[-1]
        reference_index = min(start_index + horizon_steps, len(reference_paths[scenario_index]) - 1)
        reference_state = reference_paths[scenario_index][reference_index]
        reference_row = vec(mat(reference_state).T)
        reference_score = float(np.real(reference_row @ final_state))
        target_score = float(np.real(target_row @ final_state))
        cost = (
            config.reference_weight * (1.0 - reference_score)
            + config.target_weight * (1.0 - target_score)
        )
        scenario_costs.append(cost)

        costates: list[np.ndarray] = [
            np.zeros_like(target_row) for _ in range(horizon_steps + 1)
        ]
        costates[-1] = -config.reference_weight * reference_row - config.target_weight * target_row
        for depth in reversed(range(horizon_steps)):
            costates[depth] = propagators[depth].T @ costates[depth + 1]

        grad = np.zeros((horizon_steps, n_controls), dtype=float)
        for depth in range(horizon_steps):
            for control_index in range(n_controls):
                d_propagator = expm_frechet(
                    generators[depth],
                    frechet_dirs[depth][control_index],
                    compute_expm=False,
                )
                value = costates[depth + 1] @ (d_propagator @ state_vectors[depth])
                grad[depth, control_index] = float(np.real(value))
        scenario_grads.append(grad.reshape(-1))

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


def design_reference(task: str, config: OpenAdjointConfig) -> tuple[np.ndarray, float, int, int, bool, float]:
    grape_config = OpenGrapeConfig(
        segments=config.segments,
        maxiter=config.grape_maxiter,
        training_strength=config.training_strength,
        training_seeds=config.training_seeds,
        eval_strength=config.eval_strength,
        eval_seeds=config.eval_seeds,
    )
    controls, value, seed, iterations, success, seconds = optimize_open_grape(task, grape_config)
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
    config: OpenAdjointConfig,
) -> tuple[np.ndarray, float, int, bool, float]:
    p = problem(task)
    disorders = tuple(random_disorder(seed) for seed in config.training_seeds)
    strengths = tuple(config.training_strength for _ in disorders)
    controls_cache, disorder_cache, dephasing_cache, relaxation_cache = precompute_caches(
        task,
        disorders,
        config,
    )
    reference_paths = rollout_reference_paths(task, reference_pulse, disorders, config)
    states = tuple(vec(p.initial) for _ in disorders)
    dt = p.t_final / config.segments
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
                states,
                strengths,
                controls_cache,
                disorder_cache,
                dephasing_cache,
                relaxation_cache,
                reference_paths,
                reference_block,
                task,
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
        states = step_liouville(
            states,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dephasing_cache[index],
            relaxation_cache[index],
            dt,
            control,
            TRAIN_NOISE,
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
    config: OpenAdjointConfig,
) -> list[dict[str, float | int | str]]:
    energy = float(np.mean(np.sum(pulse * pulse, axis=1)))
    rows: list[dict[str, float | int | str]] = []
    for noise in EVAL_NOISE_CASES:
        for seed in config.eval_seeds:
            final_fidelity, purity = evolve_open_system(
                task,
                pulse,
                config.eval_strength,
                seed,
                noise,
            )
            rows.append(
                {
                    "task": task,
                    "controller": controller,
                    "train_noise_case": TRAIN_NOISE.label,
                    "eval_noise_case": noise.label,
                    "disorder_strength": config.eval_strength,
                    "gamma_phi": noise.gamma_phi,
                    "gamma_relax": noise.gamma_relax,
                    "seed": seed,
                    "final_fidelity": final_fidelity,
                    "final_infidelity": 1.0 - final_fidelity,
                    "final_purity": purity,
                    "pulse_energy": energy,
                    "segments": config.segments,
                    "horizon_steps": config.horizon_steps if controller == "adjoint_open_horizon" else 0,
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
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["task"]), str(row["controller"]), str(row["eval_noise_case"])),
            [],
        ).append(row)
    summary: list[dict[str, str]] = []
    for (task, controller, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        ci95 = 1.96 * float(np.std(fids)) / np.sqrt(len(fids))
        summary.append(
            {
                "task": task,
                "controller": controller,
                "eval_noise_case": noise_case,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
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
    with result_path("open_system_adjoint_horizon_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("open_system_adjoint_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Open-System Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    cases = ("static_only", "deph_0.005", "relax_0.002", "combined")
    controllers = ("open_grape_reference", "adjoint_open_horizon")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    width = 0.35
    x = np.arange(len(cases))
    for ax, task in zip(axes, ("Z", "H")):
        for offset, controller in zip((-width / 2, width / 2), controllers):
            rows = [
                row
                for row in summary
                if row["task"] == task and row["controller"] == controller
            ]
            by_case = {row["eval_noise_case"]: row for row in rows}
            means = np.array([float(by_case[case]["final_fidelity_mean"]) for case in cases])
            ax.bar(x + offset, means, width=width, label=controller.replace("_", " "))
        ax.set_title(f"{task} transfer")
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out final fidelity")
    axes[0].set_ylim(0.94, 1.001)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path("open_system_adjoint_horizon.pdf"))
    fig.savefig(figure_path("open_system_adjoint_horizon.png"), dpi=220)
    plt.close(fig)


def gradient_check() -> None:
    task = "Z"
    rng = np.random.default_rng(123)
    config = OpenAdjointConfig(
        segments=6,
        horizon_steps=3,
        training_seeds=(0,),
        eval_seeds=(10,),
        grape_maxiter=2,
    )
    p = problem(task)
    disorders = (random_disorder(0),)
    strengths = (config.training_strength,)
    controls_cache, disorder_cache, dephasing_cache, relaxation_cache = precompute_caches(
        task,
        disorders,
        config,
    )
    reference = rng.normal(scale=0.05, size=(config.segments, len(p.controls)))
    reference_paths = rollout_reference_paths(task, reference, disorders, config)
    x = rng.normal(scale=0.05, size=config.horizon_steps * len(p.controls))
    reference_block = reference[: config.horizon_steps].reshape(-1)
    value, grad = horizon_objective_and_gradient(
        x,
        (vec(p.initial),),
        strengths,
        controls_cache,
        disorder_cache,
        dephasing_cache,
        relaxation_cache,
        reference_paths,
        reference_block,
        task,
        0,
        p.t_final / config.segments,
        config,
    )
    direction = rng.normal(size=x.size)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        x + eps * direction,
        (vec(p.initial),),
        strengths,
        controls_cache,
        disorder_cache,
        dephasing_cache,
        relaxation_cache,
        reference_paths,
        reference_block,
        task,
        0,
        p.t_final / config.segments,
        config,
    )[0]
    minus = horizon_objective_and_gradient(
        x - eps * direction,
        (vec(p.initial),),
        strengths,
        controls_cache,
        disorder_cache,
        dephasing_cache,
        relaxation_cache,
        reference_paths,
        reference_block,
        task,
        0,
        p.t_final / config.segments,
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
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--horizon-maxiter", type=int, default=8)
    parser.add_argument("--grape-maxiter", type=int, default=100)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> OpenAdjointConfig:
    if args.quick:
        return OpenAdjointConfig(
            segments=12,
            horizon_steps=2,
            horizon_maxiter=3,
            training_seeds=(0, 1),
            eval_seeds=tuple(range(10, 14)),
            grape_maxiter=5,
        )
    return OpenAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        grape_maxiter=args.grape_maxiter,
    )


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        print(f"optimizing open-system GRAPE reference for {task}", flush=True)
        reference_pulse, objective, restart_seed, iterations, success, ref_seconds = (
            design_reference(task, config)
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
                "open_grape_reference",
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
                "adjoint_open_horizon",
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
