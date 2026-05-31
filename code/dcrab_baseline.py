"""Sequential-basis dCRAB-style baseline for robust two-level transfer.

This script strengthens the reduced-basis CRAB comparator by refreshing the
randomized Fourier basis over several optimization rounds. Each round optimizes
an additive correction to the current pulse on the same training disorder
ensemble. The final pulse is then evaluated on the same held-out split used by
the other two-level baselines.

The method is a terminal open-loop optimizer, not Lyapunov feedback. Its role is
to provide a stronger derivative-free terminal-control ceiling for the
beam-horizon Lyapunov controller.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from crab_baseline import precompute_scenario_cache, evolve_cached
from horizon_lyapunov import fidelity, problem, random_disorder
from paths import figure_path, result_path


@dataclass(frozen=True)
class DCrabConfig:
    segments: int = 40
    basis_count: int = 3
    refreshes: int = 3
    umax: float = 4.0
    correction_bound: float = 2.0
    maxiter: int = 18
    popsize: int = 5
    training_strengths: tuple[float, ...] = (0.08,)
    training_seeds: tuple[int, ...] = tuple(range(4))
    eval_strengths: tuple[float, ...] = (0.05, 0.08)
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    frequency_seeds: tuple[int, ...] = (31, 43, 59, 71, 83)
    optimizer_seed: int = 701
    worst_weight: float = 0.25
    energy_weight: float = 2e-4


def dcrab_frequencies(task: str, refresh_index: int, config: DCrabConfig) -> np.ndarray:
    """Return a refreshed randomized Fourier basis for each control channel."""
    p = problem(task)
    seed = config.frequency_seeds[refresh_index % len(config.frequency_seeds)]
    rng = np.random.default_rng(seed + 1009 * len(task) + 37 * refresh_index)
    base = 2.0 * np.pi * np.arange(1, config.basis_count + 1) / p.t_final
    jitter = rng.uniform(0.65, 1.35, size=(len(p.controls), config.basis_count))
    return jitter * base[None, :]


def correction_controls(
    task: str,
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    config: DCrabConfig,
) -> np.ndarray:
    """Expand one randomized-basis correction into segment controls."""
    p = problem(task)
    n_controls = len(p.controls)
    coeffs = np.reshape(coefficients, (n_controls, 1 + 2 * config.basis_count))
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
        controls[:, control_index] = values
    return controls


def apply_correction(
    task: str,
    base_controls: np.ndarray,
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    config: DCrabConfig,
) -> np.ndarray:
    """Add a randomized-basis correction and clip to the amplitude bound."""
    correction = correction_controls(task, coefficients, frequencies, config)
    return np.clip(base_controls + correction, -config.umax, config.umax)


def objective(
    coefficients: np.ndarray,
    task: str,
    base_controls: np.ndarray,
    frequencies: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorders_cache: tuple[tuple[np.ndarray, ...], ...],
    config: DCrabConfig,
) -> float:
    """Robust terminal state-transfer objective for one basis-refresh round."""
    controls = apply_correction(task, base_controls, coefficients, frequencies, config)
    return controls_objective(
        task,
        controls,
        scenarios,
        controls_cache,
        disorders_cache,
        config,
    )


def controls_objective(
    task: str,
    controls: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorders_cache: tuple[tuple[np.ndarray, ...], ...],
    config: DCrabConfig,
) -> float:
    """Robust terminal state-transfer objective for a concrete pulse."""
    p = problem(task)
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


def optimize_task(task: str, config: DCrabConfig) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Run sequential randomized-basis refreshes for one task."""
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.training_strengths
        for seed in config.training_seeds
    )
    controls_cache, disorders_cache = precompute_scenario_cache(task, scenarios, config)
    dim = len(p.controls) * (1 + 2 * config.basis_count)
    controls = np.zeros((config.segments, len(p.controls)), dtype=float)
    logs: list[dict[str, float | int | bool]] = []

    for refresh_index in range(config.refreshes):
        frequencies = dcrab_frequencies(task, refresh_index, config)
        bound = config.umax if refresh_index == 0 else config.correction_bound
        current_objective = controls_objective(
            task,
            controls,
            scenarios,
            controls_cache,
            disorders_cache,
            config,
        )
        start = time.perf_counter()
        result = differential_evolution(
            objective,
            args=(
                task,
                controls,
                frequencies,
                scenarios,
                controls_cache,
                disorders_cache,
                config,
            ),
            bounds=[(-bound, bound)] * dim,
            maxiter=config.maxiter,
            popsize=config.popsize,
            tol=1e-4,
            polish=True,
            seed=config.optimizer_seed + 1000 * refresh_index + 17 * len(task),
            workers=1,
            updating="immediate",
        )
        elapsed = time.perf_counter() - start
        candidate_controls = apply_correction(
            task,
            controls,
            np.asarray(result.x, dtype=float),
            frequencies,
            config,
        )
        candidate_objective = controls_objective(
            task,
            candidate_controls,
            scenarios,
            controls_cache,
            disorders_cache,
            config,
        )
        accepted = candidate_objective <= current_objective
        if accepted:
            controls = candidate_controls
        logs.append(
            {
                "refresh_index": refresh_index,
                "frequency_seed": config.frequency_seeds[refresh_index % len(config.frequency_seeds)],
                "previous_objective": current_objective,
                "candidate_objective": candidate_objective,
                "objective": candidate_objective if accepted else current_objective,
                "accepted": accepted,
                "optimizer_iterations": int(result.nit),
                "optimizer_success": bool(result.success),
                "optimization_seconds": elapsed,
                "correction_bound": bound,
            }
        )

    return controls, logs


def evaluate_task(
    task: str,
    controls: np.ndarray,
    logs: list[dict[str, float | int | bool]],
    config: DCrabConfig,
    baseline_label: str,
) -> list[dict[str, float | int | str | bool]]:
    p = problem(task)
    rows: list[dict[str, float | int | str | bool]] = []
    total_seconds = float(sum(float(log["optimization_seconds"]) for log in logs))
    success_all = all(bool(log["optimizer_success"]) for log in logs)
    final_objective = float(logs[-1]["objective"])
    frequency_seed_path = ",".join(str(int(log["frequency_seed"])) for log in logs)
    for strength in config.eval_strengths:
        scenarios = tuple((strength, random_disorder(seed)) for seed in config.eval_seeds)
        controls_cache, disorders_cache = precompute_scenario_cache(task, scenarios, config)
        for scenario_index, _ in enumerate(scenarios):
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
                    "baseline": baseline_label,
                    "eval_strength": strength,
                    "seed": seed,
                    "final_fidelity": fid,
                    "final_infidelity": 1.0 - fid,
                    "pulse_energy": float(np.mean(np.square(controls))),
                    "basis_count": config.basis_count,
                    "segments": config.segments,
                    "refreshes": config.refreshes,
                    "frequency_seed_path": frequency_seed_path,
                    "training_objective": final_objective,
                    "optimizer_success_all": success_all,
                    "optimization_seconds": total_seconds,
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
                "refreshes": str(int(items[0]["refreshes"])),
                "frequency_seed_path": str(items[0]["frequency_seed_path"]),
                "optimizer_success_all": str(bool(items[0]["optimizer_success_all"])),
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(
    rows: list[dict[str, float | int | str | bool]],
    logs_by_task: dict[str, list[dict[str, float | int | bool]]],
    output_prefix: str,
) -> None:
    with result_path(f"{output_prefix}_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path(f"{output_prefix}_summary.md").open("w", encoding="utf-8") as f:
        f.write("# dCRAB-Style Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
        f.write("\n## Refresh Logs\n\n")
        log_headers = [
            "task",
            "refresh_index",
            "frequency_seed",
            "previous_objective",
            "candidate_objective",
            "objective",
            "accepted",
            "optimizer_iterations",
            "optimizer_success",
            "optimization_seconds",
            "correction_bound",
        ]
        f.write("| " + " | ".join(log_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(log_headers)) + " |\n")
        for task, logs in logs_by_task.items():
            for log in logs:
                row = {"task": task, **log}
                f.write(
                    "| "
                    + " | ".join(
                        f"{float(row[h]):.6g}" if isinstance(row[h], float) else str(row[h])
                        for h in log_headers
                    )
                    + " |\n"
                )


def plot_results(rows: list[dict[str, float | int | str | bool]], output_prefix: str) -> None:
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
    fig.savefig(figure_path(f"{output_prefix}.pdf"))
    fig.savefig(figure_path(f"{output_prefix}.png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=DCrabConfig.maxiter)
    parser.add_argument("--segments", type=int, default=DCrabConfig.segments)
    parser.add_argument("--basis-count", type=int, default=DCrabConfig.basis_count)
    parser.add_argument("--refreshes", type=int, default=DCrabConfig.refreshes)
    parser.add_argument("--popsize", type=int, default=DCrabConfig.popsize)
    parser.add_argument("--training-seed-count", type=int, default=len(DCrabConfig.training_seeds))
    parser.add_argument("--output-prefix", default="dcrab_baseline")
    parser.add_argument("--baseline-label", default="dcrab")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DCrabConfig(
        maxiter=args.maxiter,
        segments=args.segments,
        basis_count=args.basis_count,
        refreshes=args.refreshes,
        popsize=args.popsize,
        training_seeds=tuple(range(args.training_seed_count)),
    )
    rows: list[dict[str, float | int | str | bool]] = []
    logs_by_task: dict[str, list[dict[str, float | int | bool]]] = {}
    for task in ("Z", "H"):
        print(f"optimizing dCRAB-style baseline for {task}", flush=True)
        controls, logs = optimize_task(task, config)
        logs_by_task[task] = logs
        rows.extend(evaluate_task(task, controls, logs, config, args.baseline_label))
    write_outputs(rows, logs_by_task, args.output_prefix)
    if not args.skip_plot:
        plot_results(rows, args.output_prefix)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
