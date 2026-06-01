"""dCRAB-style terminal process-fidelity baseline.

This script adds an independent derivative-free process-level comparator for
the Z and Hadamard gate tasks.  It reuses the sequential randomized Fourier
basis refresh used by the state-transfer dCRAB baseline, but replaces the
terminal score with robust average gate infidelity in the interaction frame.

The result is a terminal open-loop baseline, not a Lyapunov or receding-horizon
controller.  Its role is to separate process-level reachability from the
limitations of the finite-candidate process horizon.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from dcrab_baseline import DCrabConfig, apply_correction, dcrab_frequencies
from ensemble_grape_baseline import average_gate_fidelity, target_gate
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
class ProcessCache:
    dt: float
    controls: tuple[tuple[np.ndarray, ...], ...]
    disorders: tuple[tuple[np.ndarray, ...], ...]


def precompute_process_cache(
    task: str,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    config: DCrabConfig,
) -> ProcessCache:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    controls = tuple(
        tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        for t in t_eval[:-1]
    )
    disorders = tuple(
        tuple(interaction_frame_operator(p, disorder, float(t)) for t in t_eval[:-1])
        for _, disorder in scenarios
    )
    return ProcessCache(dt=dt, controls=controls, disorders=disorders)


def final_unitary_cached(
    task: str,
    controls: np.ndarray,
    strength: float,
    scenario_index: int,
    cache: ProcessCache,
) -> np.ndarray:
    p = problem(task)
    unitary = np.eye(2, dtype=complex)
    clipped = np.clip(controls, -4.0, 4.0)
    for segment_index, coeffs in enumerate(clipped):
        hamiltonian = strength * cache.disorders[scenario_index][segment_index]
        for coeff, hc_i in zip(coeffs, cache.controls[segment_index]):
            hamiltonian = hamiltonian + coeff * hc_i
        unitary = unitary2(hamiltonian, cache.dt) @ unitary
    return unitary


def controls_objective(
    task: str,
    controls: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    cache: ProcessCache,
    config: DCrabConfig,
) -> float:
    target = target_gate(task)
    infidelities = []
    for scenario_index, (strength, _) in enumerate(scenarios):
        unitary = final_unitary_cached(task, controls, strength, scenario_index, cache)
        infidelities.append(1.0 - average_gate_fidelity(unitary, target))
    infids = np.array(infidelities, dtype=float)
    energy = float(np.mean(np.square(controls)))
    return float(
        np.mean(infids)
        + config.worst_weight * np.max(infids)
        + config.energy_weight * energy
    )


def objective(
    coefficients: np.ndarray,
    task: str,
    base_controls: np.ndarray,
    frequencies: np.ndarray,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    cache: ProcessCache,
    config: DCrabConfig,
) -> float:
    controls = apply_correction(task, base_controls, coefficients, frequencies, config)
    return controls_objective(task, controls, scenarios, cache, config)


def optimize_task(
    task: str,
    config: DCrabConfig,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    p = problem(task)
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.training_strengths
        for seed in config.training_seeds
    )
    cache = precompute_process_cache(task, scenarios, config)
    dim = len(p.controls) * (1 + 2 * config.basis_count)
    controls = np.zeros((config.segments, len(p.controls)), dtype=float)
    logs: list[dict[str, float | int | bool]] = []

    for refresh_index in range(config.refreshes):
        frequencies = dcrab_frequencies(task, refresh_index, config)
        bound = config.umax if refresh_index == 0 else config.correction_bound
        current_objective = controls_objective(task, controls, scenarios, cache, config)
        start = time.perf_counter()
        result = differential_evolution(
            objective,
            args=(task, controls, frequencies, scenarios, cache, config),
            bounds=[(-bound, bound)] * dim,
            maxiter=config.maxiter,
            popsize=config.popsize,
            tol=1e-4,
            polish=True,
            seed=config.optimizer_seed + 2000 * refresh_index + 19 * len(task),
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
            cache,
            config,
        )
        accepted = candidate_objective <= current_objective
        if accepted:
            controls = candidate_controls
        logs.append(
            {
                "refresh_index": refresh_index,
                "frequency_seed": config.frequency_seeds[
                    refresh_index % len(config.frequency_seeds)
                ],
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
    target = target_gate(task)
    rows: list[dict[str, float | int | str | bool]] = []
    total_seconds = float(sum(float(log["optimization_seconds"]) for log in logs))
    success_all = all(bool(log["optimizer_success"]) for log in logs)
    final_objective = float(logs[-1]["objective"])
    frequency_seed_path = ",".join(str(int(log["frequency_seed"])) for log in logs)

    for strength in config.eval_strengths:
        scenarios = tuple((strength, random_disorder(seed)) for seed in config.eval_seeds)
        cache = precompute_process_cache(task, scenarios, config)
        for scenario_index, _ in enumerate(scenarios):
            seed = config.eval_seeds[scenario_index]
            unitary = final_unitary_cached(task, controls, strength, scenario_index, cache)
            rho_final = unitary @ p.initial @ dagger(unitary)
            state_fid = fidelity(rho_final, p.target)
            gate_fid = average_gate_fidelity(unitary, target)
            rows.append(
                {
                    "task": task,
                    "baseline": baseline_label,
                    "eval_strength": strength,
                    "seed": seed,
                    "state_transfer_fidelity": state_fid,
                    "average_gate_fidelity": gate_fid,
                    "average_gate_infidelity": 1.0 - gate_fid,
                    "pulse_energy": float(np.mean(np.square(controls))),
                    "basis_count": config.basis_count,
                    "segments": config.segments,
                    "refreshes": config.refreshes,
                    "n_training_seeds": len(config.training_seeds),
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
        state = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate = np.array([float(row["average_gate_fidelity"]) for row in items])
        ci95 = 1.96 * float(np.std(gate, ddof=1)) / np.sqrt(len(gate))
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.3g}",
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state):.6g}",
                "state_fidelity_min": f"{np.min(state):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate):.6g}",
                "avg_gate_fidelity_std": f"{np.std(gate):.6g}",
                "avg_gate_fidelity_ci95": f"{ci95:.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "basis_count": str(int(items[0]["basis_count"])),
                "segments": str(int(items[0]["segments"])),
                "refreshes": str(int(items[0]["refreshes"])),
                "n_training_seeds": str(int(items[0]["n_training_seeds"])),
                "frequency_seed_path": str(items[0]["frequency_seed_path"]),
                "optimizer_success_all": str(bool(items[0]["optimizer_success_all"])),
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
                "training_objective": f"{float(items[0]['training_objective']):.6g}",
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
        f.write("# Process dCRAB Baseline Summary\n\n")
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
    plot_results(summary, output_prefix)


def plot_results(summary: list[dict[str, str]], output_prefix: str) -> None:
    labels = [row["task"] for row in summary]
    x = np.arange(len(labels))
    means = np.array([float(row["avg_gate_fidelity_mean"]) for row in summary])
    mins = np.array([float(row["avg_gate_fidelity_min"]) for row in summary])

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(x, means, width=0.45, label="mean")
    ax.scatter(x, mins, marker="x", color="black", s=24, linewidths=0.9, label="min")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 1.001)
    ax.set_ylabel("Held-out average gate fidelity at $\\delta=0.08$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(f"{output_prefix}.pdf"))
    fig.savefig(figure_path(f"{output_prefix}.png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--maxiter", type=int, default=DCrabConfig.maxiter)
    parser.add_argument("--segments", type=int, default=DCrabConfig.segments)
    parser.add_argument("--basis-count", type=int, default=DCrabConfig.basis_count)
    parser.add_argument("--refreshes", type=int, default=DCrabConfig.refreshes)
    parser.add_argument("--popsize", type=int, default=DCrabConfig.popsize)
    parser.add_argument("--training-seed-count", type=int, default=8)
    parser.add_argument("--output-prefix", default="process_dcrab_baseline")
    parser.add_argument("--baseline-label", default="process_dcrab")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DCrabConfig:
    if args.quick:
        return DCrabConfig(
            segments=16,
            basis_count=2,
            refreshes=1,
            maxiter=2,
            popsize=3,
            training_seeds=(0, 1),
            eval_strengths=(0.08,),
            eval_seeds=tuple(range(10, 16)),
        )
    return DCrabConfig(
        maxiter=args.maxiter,
        segments=args.segments,
        basis_count=args.basis_count,
        refreshes=args.refreshes,
        popsize=args.popsize,
        training_seeds=tuple(range(args.training_seed_count)),
        eval_strengths=(0.08,),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    rows: list[dict[str, float | int | str | bool]] = []
    logs_by_task: dict[str, list[dict[str, float | int | bool]]] = {}
    for task in ("Z", "H"):
        print(f"optimizing process dCRAB for {task}", flush=True)
        controls, logs = optimize_task(task, config)
        logs_by_task[task] = logs
        rows.extend(evaluate_task(task, controls, logs, config, args.baseline_label))
    write_outputs(rows, logs_by_task, args.output_prefix)
    print(f"wrote {len(rows)} rows to {result_path(f'{args.output_prefix}_results.csv')}")
    print(f"wrote summary to {result_path(f'{args.output_prefix}_summary.md')}")
    print(f"wrote figure to {figure_path(f'{args.output_prefix}.pdf')}")


if __name__ == "__main__":
    main()
