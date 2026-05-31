"""Reduced-parameter CRAB baseline for robust two-level state transfer.

The beam-horizon controller is compared mostly with open-loop and GRAPE-style
terminal optimizers.  This script adds an independent reduced-basis optimal-
control comparator: each control channel is parameterized by a small randomized
Fourier basis and optimized over the same interaction-frame disorder ensemble.

The method is a terminal open-loop baseline, not Lyapunov feedback.  Its role is
to test whether the reported robustness is an artifact of comparing only against
piecewise-constant GRAPE/open-loop parameterizations.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

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
class CrabConfig:
    segments: int = 40
    basis_count: int = 3
    umax: float = 4.0
    maxiter: int = 35
    training_strengths: tuple[float, ...] = (0.08,)
    training_seeds: tuple[int, ...] = tuple(range(4))
    eval_strengths: tuple[float, ...] = (0.05, 0.08)
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    frequency_seeds: tuple[int, ...] = (31,)
    restart_seeds: tuple[int, ...] = (101,)
    worst_weight: float = 0.25
    energy_weight: float = 2e-4


def crab_frequencies(task: str, frequency_seed: int, config: CrabConfig) -> np.ndarray:
    """Return randomized Fourier frequencies for each control channel."""
    p = problem(task)
    rng = np.random.default_rng(frequency_seed)
    base = 2.0 * np.pi * np.arange(1, config.basis_count + 1) / p.t_final
    jitter = rng.uniform(0.75, 1.25, size=(len(p.controls), config.basis_count))
    return jitter * base[None, :]


def controls_from_coefficients(
    task: str,
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    config: CrabConfig,
) -> np.ndarray:
    """Expand CRAB coefficients into clipped piecewise-constant controls."""
    p = problem(task)
    n_controls = len(p.controls)
    coeffs = np.reshape(
        coefficients,
        (n_controls, 1 + 2 * config.basis_count),
    )
    t_grid = (np.arange(config.segments) + 0.5) * p.t_final / config.segments
    envelope = np.sin(np.pi * t_grid / p.t_final) ** 2
    controls = np.zeros((config.segments, n_controls), dtype=float)
    for control_index in range(n_controls):
        values = np.full(config.segments, coeffs[control_index, 0], dtype=float)
        for basis_index in range(config.basis_count):
            omega = frequencies[control_index, basis_index]
            sin_coeff = coeffs[control_index, 1 + basis_index]
            cos_coeff = coeffs[control_index, 1 + config.basis_count + basis_index]
            values += envelope * (
                sin_coeff * np.sin(omega * t_grid)
                + cos_coeff * np.cos(omega * t_grid)
            )
        controls[:, control_index] = np.clip(values, -config.umax, config.umax)
    return controls


def precompute_scenario_cache(
    task: str,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    config: CrabConfig,
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[tuple[np.ndarray, ...], ...],
]:
    p = problem(task)
    t_grid = np.linspace(0.0, p.t_final, config.segments + 1)[:-1]
    controls_cache = []
    disorders_cache = []
    for t in t_grid:
        controls_i = tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        disorders_i = tuple(
            interaction_frame_operator(p, disorder, float(t))
            for _, disorder in scenarios
        )
        controls_cache.append(controls_i)
        disorders_cache.append(disorders_i)
    return tuple(controls_cache), tuple(disorders_cache)


def evolve_cached(
    task: str,
    controls: np.ndarray,
    strength: float,
    scenario_index: int,
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorders_cache: tuple[tuple[np.ndarray, ...], ...],
    config: CrabConfig,
) -> np.ndarray:
    p = problem(task)
    dt = p.t_final / config.segments
    rho = p.initial.copy()
    for segment_index, coeffs in enumerate(controls):
        hamiltonian = strength * disorders_cache[segment_index][scenario_index]
        for coeff, hc_i in zip(coeffs, controls_cache[segment_index]):
            hamiltonian = hamiltonian + coeff * hc_i
        unitary = unitary2(hamiltonian, dt)
        rho = unitary @ rho @ dagger(unitary)
    return 0.5 * (rho + dagger(rho)) / np.trace(rho)


def objective(
    coefficients: np.ndarray,
    task: str,
    frequencies: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorders_cache: tuple[tuple[np.ndarray, ...], ...],
    config: CrabConfig,
) -> float:
    p = problem(task)
    controls = controls_from_coefficients(task, coefficients, frequencies, config)
    infidelities = []
    for scenario_index, (strength, _) in enumerate(scenarios):
        rho = evolve_cached(
            task,
            controls,
            strength,
            scenario_index,
            controls_cache,
            disorders_cache,
            config,
        )
        infidelities.append(1.0 - fidelity(rho, p.target))
    infids = np.array(infidelities, dtype=float)
    energy = float(np.mean(np.square(controls)))
    return float(
        np.mean(infids)
        + config.worst_weight * np.max(infids)
        + config.energy_weight * energy
    )


def optimize_task(task: str, config: CrabConfig) -> tuple[np.ndarray, np.ndarray, int, int, float, int, bool, float]:
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.training_strengths
        for seed in config.training_seeds
    )
    controls_cache, disorders_cache = precompute_scenario_cache(task, scenarios, config)
    dim = len(p.controls) * (1 + 2 * config.basis_count)
    best_coefficients: np.ndarray | None = None
    best_frequencies: np.ndarray | None = None
    best_frequency_seed = -1
    best_restart_seed = -1
    best_fun = float("inf")
    best_iters = 0
    best_success = False
    start = time.perf_counter()

    for frequency_seed in config.frequency_seeds:
        frequencies = crab_frequencies(task, frequency_seed, config)
        for restart_seed in config.restart_seeds:
            result = differential_evolution(
                objective,
                args=(
                    task,
                    frequencies,
                    scenarios,
                    controls_cache,
                    disorders_cache,
                    config,
                ),
                bounds=[(-config.umax, config.umax)] * dim,
                maxiter=config.maxiter,
                popsize=6,
                tol=1e-4,
                polish=True,
                seed=restart_seed + 1000 * frequency_seed + len(task),
                workers=1,
                updating="immediate",
            )
            if float(result.fun) < best_fun:
                best_fun = float(result.fun)
                best_coefficients = np.asarray(result.x, dtype=float)
                best_frequencies = frequencies
                best_frequency_seed = frequency_seed
                best_restart_seed = restart_seed
                best_iters = int(result.nit)
                best_success = bool(result.success)

    if best_coefficients is None or best_frequencies is None:
        raise RuntimeError(f"no CRAB restart completed for {task}")
    elapsed = time.perf_counter() - start
    return (
        best_coefficients,
        best_frequencies,
        best_frequency_seed,
        best_restart_seed,
        best_fun,
        best_iters,
        best_success,
        elapsed,
    )


def evaluate_task(
    task: str,
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    frequency_seed: int,
    restart_seed: int,
    training_objective: float,
    optimizer_iterations: int,
    optimizer_success: bool,
    optimization_seconds: float,
    config: CrabConfig,
) -> list[dict[str, float | int | str | bool]]:
    p = problem(task)
    controls = controls_from_coefficients(task, coefficients, frequencies, config)
    rows: list[dict[str, float | int | str | bool]] = []
    for strength in config.eval_strengths:
        scenarios = tuple((strength, random_disorder(seed)) for seed in config.eval_seeds)
        controls_cache, disorders_cache = precompute_scenario_cache(task, scenarios, config)
        for scenario_index, (_, disorder) in enumerate(scenarios):
            seed = config.eval_seeds[scenario_index]
            rho = evolve_cached(
                task,
                controls,
                strength,
                scenario_index,
                controls_cache,
                disorders_cache,
                config,
            )
            fid = fidelity(rho, p.target)
            rows.append(
                {
                    "task": task,
                    "baseline": "crab",
                    "eval_strength": strength,
                    "seed": seed,
                    "final_fidelity": fid,
                    "final_infidelity": 1.0 - fid,
                    "pulse_energy": float(np.mean(np.square(controls))),
                    "basis_count": config.basis_count,
                    "segments": config.segments,
                    "frequency_seed": frequency_seed,
                    "restart_seed": restart_seed,
                    "training_objective": training_objective,
                    "optimizer_iterations": optimizer_iterations,
                    "optimizer_success": optimizer_success,
                    "optimization_seconds": optimization_seconds,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), float(row["eval_strength"])), []).append(row)
    summary = []
    for (task, strength), items in sorted(groups.items()):
        fids = np.array([float(item["final_fidelity"]) for item in items])
        energies = np.array([float(item["pulse_energy"]) for item in items])
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.3g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "pulse_energy_mean": f"{np.mean(energies):.6g}",
                "basis_count": str(int(items[0]["basis_count"])),
                "segments": str(int(items[0]["segments"])),
                "frequency_seed": str(int(items[0]["frequency_seed"])),
                "restart_seed": str(int(items[0]["restart_seed"])),
                "optimizer_iterations": str(int(items[0]["optimizer_iterations"])),
                "optimizer_success": str(bool(items[0]["optimizer_success"])),
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str | bool]]) -> None:
    with result_path("crab_baseline_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("crab_baseline_summary.md").open("w", encoding="utf-8") as f:
        f.write("# CRAB Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def plot_results(rows: list[dict[str, float | int | str | bool]]) -> None:
    summary = summarize(rows)
    tasks = ["Z", "H"]
    strengths = [0.05, 0.08]
    means = {
        (row["task"], float(row["eval_strength"])): float(row["final_fidelity_mean"])
        for row in summary
    }
    mins = {
        (row["task"], float(row["eval_strength"])): float(row["final_fidelity_min"])
        for row in summary
    }

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(tasks))
    width = 0.34
    for index, strength in enumerate(strengths):
        values = [means[(task, strength)] for task in tasks]
        low = [means[(task, strength)] - mins[(task, strength)] for task in tasks]
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            yerr=[low, [0.0, 0.0]],
            capsize=3,
            label=f"$\\delta={strength:.2f}$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Held-out final fidelity")
    ax.set_ylim(0.94, 1.0005)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("crab_baseline.pdf"))
    fig.savefig(figure_path("crab_baseline.png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=CrabConfig.maxiter)
    parser.add_argument("--segments", type=int, default=CrabConfig.segments)
    parser.add_argument("--basis-count", type=int, default=CrabConfig.basis_count)
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CrabConfig(
        maxiter=args.maxiter,
        segments=args.segments,
        basis_count=args.basis_count,
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for task in ("Z", "H"):
        print(f"optimizing CRAB baseline for {task}", flush=True)
        (
            coefficients,
            frequencies,
            frequency_seed,
            restart_seed,
            training_objective,
            optimizer_iterations,
            optimizer_success,
            optimization_seconds,
        ) = optimize_task(task, config)
        rows.extend(
            evaluate_task(
                task,
                coefficients,
                frequencies,
                frequency_seed,
                restart_seed,
                training_objective,
                optimizer_iterations,
                optimizer_success,
                optimization_seconds,
                config,
            )
        )
    write_outputs(rows)
    if not args.skip_plot:
        plot_results(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
