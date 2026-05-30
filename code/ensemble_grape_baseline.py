"""Ensemble GRAPE-style optimal-control baselines.

This script adds a stronger terminal optimal-control comparator for the journal
version.  It uses exact Frechet derivatives of the segment propagators and
optimizes either a one-state transfer objective or an average-gate-fidelity
objective over the same interaction-frame disorder model used by the
beam-horizon Lyapunov controller.

The resulting pulses are terminal open-loop baselines, not Lyapunov feedback
laws.  They are included to separate what the Hamiltonian resources can achieve
from what the proposed Lyapunov-shaped controller achieves.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from horizon_lyapunov import dagger, fidelity, interaction_frame_operator, problem, random_disorder
from paths import figure_path, result_path


@dataclass(frozen=True)
class GrapeConfig:
    segments: int = 60
    umax: float = 4.0
    maxiter: int = 120
    training_strengths: tuple[float, ...] = (0.05, 0.08)
    training_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    restart_seeds: tuple[int, ...] = (11, 22)
    worst_weight: float = 0.25
    energy_weight: float = 1e-4
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))


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


def segment_data(
    task: str,
    controls_flat: np.ndarray,
    segments: int,
    strength: float,
    disorder: np.ndarray,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[np.ndarray]]:
    p = problem(task)
    controls = np.reshape(controls_flat, (segments, len(p.controls)))
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    unitaries: list[np.ndarray] = []
    frechet_dirs: list[list[np.ndarray]] = []
    generators: list[np.ndarray] = []

    for index, coeffs in enumerate(controls):
        t = float(t_eval[index])
        control_ops = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        h = strength * interaction_frame_operator(p, disorder, t)
        for coeff, hc_i in zip(coeffs, control_ops):
            h = h + coeff * hc_i
        generator = -1.0j * h * dt
        unitaries.append(expm(generator))
        frechet_dirs.append([-1.0j * hc_i * dt for hc_i in control_ops])
        generators.append(generator)

    return unitaries, frechet_dirs, generators


def final_unitary(
    task: str,
    controls_flat: np.ndarray,
    segments: int,
    strength: float,
    disorder: np.ndarray,
) -> np.ndarray:
    unitaries, _, _ = segment_data(task, controls_flat, segments, strength, disorder)
    total = np.eye(2, dtype=complex)
    for unitary in unitaries:
        total = unitary @ total
    return total


def state_fidelity_and_gradient(
    task: str,
    controls_flat: np.ndarray,
    config: GrapeConfig,
    strength: float,
    disorder: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = problem(task)
    unitaries, frechet_dirs, generators = segment_data(
        task, controls_flat, config.segments, strength, disorder
    )
    n_controls = len(p.controls)

    rhos = [p.initial.copy()]
    for unitary in unitaries:
        rhos.append(unitary @ rhos[-1] @ dagger(unitary))

    lambdas: list[np.ndarray] = [np.zeros_like(p.target) for _ in range(config.segments + 1)]
    lambdas[-1] = p.target
    for index in reversed(range(config.segments)):
        lambdas[index] = dagger(unitaries[index]) @ lambdas[index + 1] @ unitaries[index]

    grad = np.zeros((config.segments, n_controls), dtype=float)
    for index in range(config.segments):
        for control_index in range(n_controls):
            d_unitary = expm_frechet(
                generators[index],
                frechet_dirs[index][control_index],
                compute_expm=False,
            )
            value = np.trace(
                lambdas[index + 1] @ d_unitary @ rhos[index] @ dagger(unitaries[index])
            )
            grad[index, control_index] = 2.0 * float(np.real(value))

    return fidelity(rhos[-1], p.target), grad.reshape(-1)


def process_fidelity_and_gradient(
    task: str,
    controls_flat: np.ndarray,
    config: GrapeConfig,
    strength: float,
    disorder: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = problem(task)
    target = target_gate(task)
    unitaries, frechet_dirs, generators = segment_data(
        task, controls_flat, config.segments, strength, disorder
    )
    n_controls = len(p.controls)
    dim = target.shape[0]

    prefixes = [np.eye(dim, dtype=complex)]
    for unitary in unitaries:
        prefixes.append(unitary @ prefixes[-1])
    total = prefixes[-1]

    suffixes: list[np.ndarray] = [np.eye(dim, dtype=complex) for _ in range(config.segments)]
    running = np.eye(dim, dtype=complex)
    for index in reversed(range(config.segments)):
        suffixes[index] = running
        running = running @ unitaries[index]

    overlap = np.trace(dagger(target) @ total)
    grad = np.zeros((config.segments, n_controls), dtype=float)
    scale = float(dim * (dim + 1))
    for index in range(config.segments):
        for control_index in range(n_controls):
            d_unitary = expm_frechet(
                generators[index],
                frechet_dirs[index][control_index],
                compute_expm=False,
            )
            d_total = suffixes[index] @ d_unitary @ prefixes[index]
            d_overlap = np.trace(dagger(target) @ d_total)
            grad[index, control_index] = 2.0 * float(np.real(np.conjugate(overlap) * d_overlap)) / scale

    return average_gate_fidelity(total, target), grad.reshape(-1)


def objective_and_gradient(
    controls_flat: np.ndarray,
    task: str,
    objective_kind: str,
    config: GrapeConfig,
    scenarios: tuple[tuple[float, np.ndarray], ...],
) -> tuple[float, np.ndarray]:
    metric_fn: Callable[
        [str, np.ndarray, GrapeConfig, float, np.ndarray],
        tuple[float, np.ndarray],
    ]
    if objective_kind == "state":
        metric_fn = state_fidelity_and_gradient
    elif objective_kind == "process":
        metric_fn = process_fidelity_and_gradient
    else:
        raise ValueError(objective_kind)

    fidelities = []
    fidelity_grads = []
    for strength, disorder in scenarios:
        fid, grad_fid = metric_fn(task, controls_flat, config, strength, disorder)
        fidelities.append(fid)
        fidelity_grads.append(grad_fid)

    fids = np.array(fidelities, dtype=float)
    grads = np.vstack(fidelity_grads)
    infids = 1.0 - fids
    worst_index = int(np.argmax(infids))

    objective = float(np.mean(infids) + config.worst_weight * infids[worst_index])
    grad = -np.mean(grads, axis=0) - config.worst_weight * grads[worst_index]

    energy = float(np.mean(np.square(controls_flat)))
    objective += config.energy_weight * energy
    grad += config.energy_weight * (2.0 / controls_flat.size) * controls_flat
    return objective, grad


def optimize_pulse(
    task: str,
    objective_kind: str,
    config: GrapeConfig,
) -> tuple[np.ndarray, float, int, int, bool, float]:
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.training_strengths
        for seed in config.training_seeds
    )
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    best_seed = -1
    best_iters = 0
    best_success = False
    start = time.perf_counter()

    for restart_seed in config.restart_seeds:
        rng = np.random.default_rng(restart_seed)
        x0 = rng.normal(scale=0.15, size=config.segments * len(problem(task).controls))
        result = minimize(
            lambda x: objective_and_gradient(x, task, objective_kind, config, scenarios),
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-config.umax, config.umax)] * x0.size,
            options={"maxiter": config.maxiter, "ftol": 1e-11, "gtol": 1e-7},
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x)
            best_seed = restart_seed
            best_iters = int(result.nit)
            best_success = bool(result.success)

    if best_x is None:
        raise RuntimeError(f"no GRAPE restart completed for {task}/{objective_kind}")
    return best_x, best_fun, best_seed, best_iters, best_success, time.perf_counter() - start


def evaluate_pulse(
    task: str,
    objective_kind: str,
    controls_flat: np.ndarray,
    training_objective: float,
    restart_seed: int,
    n_iterations: int,
    success: bool,
    optimization_seconds: float,
    config: GrapeConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    target = target_gate(task)
    rows: list[dict[str, float | int | str]] = []
    energy = float(np.mean(np.square(controls_flat)))

    for seed in config.eval_seeds:
        disorder = random_disorder(seed)
        unitary = final_unitary(
            task,
            controls_flat,
            config.segments,
            config.eval_strength,
            disorder,
        )
        rho_final = unitary @ p.initial @ dagger(unitary)
        rows.append(
            {
                "task": task,
                "baseline": "ensemble_grape",
                "objective_kind": objective_kind,
                "disorder_strength": config.eval_strength,
                "seed": seed,
                "state_transfer_fidelity": fidelity(rho_final, p.target),
                "average_gate_fidelity": average_gate_fidelity(unitary, target),
                "pulse_energy": energy,
                "segments": config.segments,
                "training_objective": training_objective,
                "best_restart_seed": restart_seed,
                "optimizer_iterations": n_iterations,
                "optimizer_success": str(success),
                "optimization_seconds": optimization_seconds,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["objective_kind"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (task, objective_kind), items in sorted(groups.items()):
        state_fids = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate_fids = np.array([float(row["average_gate_fidelity"]) for row in items])
        state_ci = 1.96 * float(np.std(state_fids)) / np.sqrt(len(state_fids))
        gate_ci = 1.96 * float(np.std(gate_fids)) / np.sqrt(len(gate_fids))
        summary.append(
            {
                "task": task,
                "objective_kind": objective_kind,
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state_fids):.6g}",
                "state_fidelity_min": f"{np.min(state_fids):.6g}",
                "state_fidelity_std": f"{np.std(state_fids):.6g}",
                "state_fidelity_ci95": f"{state_ci:.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate_fids):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate_fids):.6g}",
                "avg_gate_fidelity_std": f"{np.std(gate_fids):.6g}",
                "avg_gate_fidelity_ci95": f"{gate_ci:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "best_restart_seed": str(items[0]["best_restart_seed"]),
                "training_objective": f"{float(items[0]['training_objective']):.6g}",
                "optimizer_iterations": str(items[0]["optimizer_iterations"]),
                "optimizer_success": str(items[0]["optimizer_success"]),
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("ensemble_grape_baseline_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("ensemble_grape_baseline_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Ensemble GRAPE Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [f"{row['task']} {row['objective_kind']}" for row in summary]
    x = np.arange(len(labels))
    state_means = np.array([float(row["state_fidelity_mean"]) for row in summary])
    gate_means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.36
    ax.bar(x - width / 2, state_means, width=width, label="state transfer")
    ax.bar(x + width / 2, gate_means, width=width, label="average gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.90, 1.001)
    ax.set_ylabel("Held-out fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("ensemble_grape_baseline.pdf"))
    fig.savefig(figure_path("ensemble_grape_baseline.png"), dpi=220)
    plt.close(fig)


def gradient_check() -> None:
    config = GrapeConfig(
        segments=5,
        maxiter=2,
        training_strengths=(0.05,),
        training_seeds=(0,),
        restart_seeds=(11,),
        eval_seeds=(10,),
    )
    rng = np.random.default_rng(123)
    x = rng.normal(scale=0.05, size=config.segments * len(problem("Z").controls))
    scenarios = ((0.05, random_disorder(0)),)
    for objective_kind in ("state", "process"):
        value, grad = objective_and_gradient(x, "Z", objective_kind, config, scenarios)
        direction = rng.normal(size=x.size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        plus = objective_and_gradient(x + eps * direction, "Z", objective_kind, config, scenarios)[0]
        minus = objective_and_gradient(x - eps * direction, "Z", objective_kind, config, scenarios)[0]
        finite_diff = (plus - minus) / (2.0 * eps)
        analytic = float(np.dot(grad, direction))
        error = abs(finite_diff - analytic)
        print(
            f"{objective_kind}: value={value:.8g}, "
            f"finite_diff={finite_diff:.8g}, analytic={analytic:.8g}, error={error:.3g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a smaller smoke-test configuration.")
    parser.add_argument("--gradient-check", action="store_true", help="Run a directional gradient check and exit.")
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--objectives", nargs="+", choices=("state", "process"), default=("state", "process"))
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> GrapeConfig:
    if args.quick:
        return GrapeConfig(
            segments=16,
            maxiter=20,
            training_strengths=(0.08,),
            training_seeds=(0, 1),
            restart_seeds=(11,),
            eval_seeds=tuple(range(10, 16)),
        )
    return GrapeConfig(segments=args.segments, maxiter=args.maxiter)


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        for objective_kind in args.objectives:
            print(f"optimizing {task}/{objective_kind} with {config.segments} segments")
            controls, value, seed, iterations, success, seconds = optimize_pulse(
                task, objective_kind, config
            )
            print(
                f"  best objective={value:.6g}, restart={seed}, "
                f"iterations={iterations}, success={success}, seconds={seconds:.1f}"
            )
            rows.extend(
                evaluate_pulse(
                    task,
                    objective_kind,
                    controls,
                    value,
                    seed,
                    iterations,
                    success,
                    seconds,
                    config,
                )
            )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('ensemble_grape_baseline_results.csv')}")
    print(f"wrote aggregate table to {result_path('ensemble_grape_baseline_summary.md')}")
    print(f"wrote figure to {figure_path('ensemble_grape_baseline.pdf')}")


if __name__ == "__main__":
    main()
