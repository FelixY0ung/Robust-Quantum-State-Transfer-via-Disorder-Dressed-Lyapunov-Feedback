"""Krotov-package ensemble baseline for robust two-level transfer.

This optional script uses the external ``krotov`` package to add a direct
Krotov-family comparator for the journal version. The package currently depends
on QuTiP 4.x, so it is kept optional and is not required by the core
NumPy/SciPy reproduction scripts.

The optimization is performed in the laboratory frame with a final target
rotated by the known drift. This is equivalent to the interaction-frame
state-transfer objective used by the beam-horizon and GRAPE evaluations. The
resulting pulse is evaluated by the existing interaction-frame evaluator on
held-out disorder seeds.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ensemble_grape_baseline import average_gate_fidelity, final_unitary, target_gate
from horizon_lyapunov import dagger, fidelity, problem, random_disorder
from paths import figure_path, result_path


@dataclass(frozen=True)
class KrotovBaselineConfig:
    segments: int = 40
    iterations: int = 60
    lambda_a: float = 5.0
    training_strengths: tuple[float, ...] = (0.08,)
    training_seeds: tuple[int, ...] = tuple(range(8))
    eval_strengths: tuple[float, ...] = (0.05, 0.08)
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))


def require_krotov() -> tuple[Any, Any, Any]:
    """Import optional Krotov dependencies with a useful failure message."""
    try:
        import krotov
        import qutip
        from krotov.conversions import control_onto_interval
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The optional Krotov baseline requires krotov==1.3.0 and "
            "qutip<5. In this workspace it was prototyped with Python 3.9. "
            "Install those optional dependencies in a compatible environment "
            "before running code/krotov_baseline.py."
        ) from exc
    return krotov, qutip, control_onto_interval


def pure_vectors(task: str) -> tuple[np.ndarray, np.ndarray]:
    z0 = np.array([[1.0], [0.0]], dtype=complex)
    z1 = np.array([[0.0], [1.0]], dtype=complex)
    plus = (z0 + z1) / np.sqrt(2.0)
    minus = (z0 - z1) / np.sqrt(2.0)
    if task == "Z":
        return plus, minus
    if task == "H":
        return z0, plus
    raise ValueError(task)


def qket(qutip: Any, vector: np.ndarray) -> Any:
    return qutip.Qobj(vector, dims=[[2], [1]])


def lab_frame_target(task: str) -> np.ndarray:
    """Return the Schrödinger-picture final target for the interaction objective."""
    p = problem(task)
    _, target = pure_vectors(task)
    return expm(-1.0j * p.h0 * p.t_final) @ target


def build_objectives(
    task: str,
    config: KrotovBaselineConfig,
    krotov: Any,
    qutip: Any,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    p = problem(task)
    initial, _ = pure_vectors(task)
    target = lab_frame_target(task)
    ux = np.zeros(config.segments + 1, dtype=float)
    uy = np.zeros(config.segments + 1, dtype=float)
    objectives = []
    for strength in config.training_strengths:
        for seed in config.training_seeds:
            h0 = qutip.Qobj(p.h0 + strength * random_disorder(seed))
            hamiltonian = [
                h0,
                [qutip.Qobj(p.controls[0]), ux],
                [qutip.Qobj(p.controls[1]), uy],
            ]
            objectives.append(
                krotov.Objective(
                    initial_state=qket(qutip, initial),
                    target=qket(qutip, target),
                    H=hamiltonian,
                )
            )
    return objectives, ux, uy


def optimize_task(
    task: str,
    config: KrotovBaselineConfig,
) -> tuple[np.ndarray, int, float, float]:
    krotov, qutip, control_onto_interval = require_krotov()
    p = problem(task)
    objectives, ux, uy = build_objectives(task, config, krotov, qutip)
    tlist = np.linspace(0.0, p.t_final, config.segments + 1)
    pulse_options = {
        id(ux): {"lambda_a": config.lambda_a, "update_shape": 1},
        id(uy): {"lambda_a": config.lambda_a, "update_shape": 1},
    }
    start = time.perf_counter()
    result = krotov.optimize_pulses(
        objectives,
        pulse_options,
        tlist,
        propagator=krotov.propagators.expm,
        chi_constructor=krotov.functionals.chis_ss,
        iter_stop=config.iterations,
        store_all_pulses=False,
        limit_thread_pool=1,
    )
    seconds = time.perf_counter() - start
    ux_pulse = control_onto_interval(np.asarray(result.optimized_controls[0], dtype=float))
    uy_pulse = control_onto_interval(np.asarray(result.optimized_controls[1], dtype=float))
    controls = np.column_stack([ux_pulse, uy_pulse])
    return controls, int(result.iters[-1]), float(seconds), float(result.info_vals[-1]) if result.info_vals else float("nan")


def evaluate_task(
    task: str,
    controls: np.ndarray,
    iterations: int,
    seconds: float,
    terminal_info: float,
    config: KrotovBaselineConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    rows: list[dict[str, float | int | str]] = []
    energy = float(np.mean(np.square(controls)))
    max_abs_control = float(np.max(np.abs(controls)))
    for strength in config.eval_strengths:
        for seed in config.eval_seeds:
            disorder = random_disorder(seed)
            unitary = final_unitary(
                task,
                controls.reshape(-1),
                config.segments,
                strength,
                disorder,
            )
            rho_final = unitary @ p.initial @ dagger(unitary)
            rows.append(
                {
                    "task": task,
                    "baseline": "ensemble_krotov",
                    "eval_strength": strength,
                    "seed": seed,
                    "state_transfer_fidelity": fidelity(rho_final, p.target),
                    "average_gate_fidelity": average_gate_fidelity(unitary, target_gate(task)),
                    "pulse_energy": energy,
                    "max_abs_control": max_abs_control,
                    "segments": config.segments,
                    "training_strengths": ";".join(f"{x:g}" for x in config.training_strengths),
                    "training_seed_count": len(config.training_seeds),
                    "krotov_iterations": iterations,
                    "lambda_a": config.lambda_a,
                    "terminal_info": terminal_info,
                    "optimization_seconds": seconds,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), float(row["eval_strength"])), []).append(row)
    summary = []
    for (task, strength), items in sorted(groups.items()):
        state = np.array([float(row["state_transfer_fidelity"]) for row in items])
        gate = np.array([float(row["average_gate_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:g}",
                "n": str(len(items)),
                "state_fidelity_mean": f"{np.mean(state):.6g}",
                "state_fidelity_min": f"{np.min(state):.6g}",
                "state_fidelity_std": f"{np.std(state):.6g}",
                "state_fidelity_ci95": f"{1.96 * np.std(state) / np.sqrt(len(state)):.6g}",
                "avg_gate_fidelity_mean": f"{np.mean(gate):.6g}",
                "avg_gate_fidelity_min": f"{np.min(gate):.6g}",
                "pulse_energy_mean": f"{float(items[0]['pulse_energy']):.6g}",
                "max_abs_control": f"{float(items[0]['max_abs_control']):.6g}",
                "segments": str(items[0]["segments"]),
                "training_strengths": str(items[0]["training_strengths"]),
                "training_seed_count": str(items[0]["training_seed_count"]),
                "krotov_iterations": str(items[0]["krotov_iterations"]),
                "lambda_a": f"{float(items[0]['lambda_a']):.6g}",
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("krotov_baseline_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("krotov_baseline_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Ensemble Krotov Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    labels = [f"{row['task']} $\\delta={row['eval_strength']}$" for row in summary]
    means = np.array([float(row["state_fidelity_mean"]) for row in summary])
    mins = np.array([float(row["state_fidelity_min"]) for row in summary])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.bar(x, means, width=0.58, color="#5271a3", label="mean")
    ax.scatter(x, mins, color="#232323", s=18, zorder=3, label="worst")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Held-out state fidelity")
    ax.set_ylim(0.88, 1.001)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("krotov_baseline.pdf"))
    fig.savefig(figure_path("krotov_baseline.png"), dpi=220)
    plt.close(fig)


def config_from_args(args: argparse.Namespace) -> KrotovBaselineConfig:
    if args.quick:
        return KrotovBaselineConfig(
            segments=16,
            iterations=3,
            training_strengths=(0.08,),
            training_seeds=(0, 1),
            eval_strengths=(0.08,),
            eval_seeds=tuple(range(10, 16)),
            lambda_a=args.lambda_a,
        )
    return KrotovBaselineConfig(
        segments=args.segments,
        iterations=args.iterations,
        lambda_a=args.lambda_a,
        training_strengths=tuple(args.train_strengths),
        training_seeds=tuple(range(args.training_seed_count)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test configuration.")
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--lambda-a", type=float, default=5.0)
    parser.add_argument("--training-seed-count", type=int, default=8)
    parser.add_argument("--train-strengths", nargs="+", type=float, default=[0.08])
    parser.add_argument("--tasks", nargs="+", choices=("Z", "H"), default=("Z", "H"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks:
        print(
            f"optimizing {task} with Krotov: segments={config.segments}, "
            f"iterations={config.iterations}, train strengths={config.training_strengths}"
        )
        controls, iterations, seconds, terminal_info = optimize_task(task, config)
        print(
            f"  completed {iterations} iterations in {seconds:.1f}s; "
            f"max |u|={np.max(np.abs(controls)):.4g}"
        )
        rows.extend(evaluate_task(task, controls, iterations, seconds, terminal_info, config))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('krotov_baseline_results.csv')}")
    print(f"wrote summary to {result_path('krotov_baseline_summary.md')}")


if __name__ == "__main__":
    main()
