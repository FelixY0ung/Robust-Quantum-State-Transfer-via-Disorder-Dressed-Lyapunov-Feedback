"""Five-level leakage benchmark with Lindblad noise.

This diagnostic combines the weakly anharmonic leakage model with Markovian
dephasing and relaxation.  The controllers are still designed by the leakage
benchmark machinery, then evaluated under the open-system model.  The result is
a physical stress test, not an open-system optimal-control claim.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from paths import figure_path, result_path
from transmon_leakage_horizon import (
    TAIL_FRACTION,
    dagger,
    design_adjoint_horizon,
    design_pulse,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    leakage,
    optimize_grape_pulse,
    problem,
    random_disorder,
)


@dataclass(frozen=True)
class NoiseCase:
    label: str
    gamma_phi: float
    gamma_relax: float


NOISE_CASES = (
    NoiseCase("static_only", 0.0, 0.0),
    NoiseCase("deph_0.001", 0.001, 0.0),
    NoiseCase("relax_0.0005", 0.0, 0.0005),
    NoiseCase("combined", 0.001, 0.0005),
)


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def annihilation(dim: int) -> np.ndarray:
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(float(n))
    return a


def dissipator(operator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    odag_o = dagger(operator) @ operator
    return operator @ rho @ dagger(operator) - 0.5 * (odag_o @ rho + rho @ odag_o)


def derivative(
    rho: np.ndarray,
    hamiltonian: np.ndarray,
    dephasing_operator: np.ndarray,
    relaxation_operator: np.ndarray,
    noise: NoiseCase,
) -> np.ndarray:
    drho = -1.0j * comm(hamiltonian, rho)
    if noise.gamma_phi:
        drho = drho + noise.gamma_phi * dissipator(dephasing_operator, rho)
    if noise.gamma_relax:
        drho = drho + noise.gamma_relax * dissipator(relaxation_operator, rho)
    return drho


def rk4_open_step(
    rho: np.ndarray,
    dt: float,
    hamiltonian: np.ndarray,
    dephasing_operator: np.ndarray,
    relaxation_operator: np.ndarray,
    noise: NoiseCase,
) -> np.ndarray:
    k1 = derivative(rho, hamiltonian, dephasing_operator, relaxation_operator, noise)
    k2 = derivative(
        rho + 0.5 * dt * k1,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        noise,
    )
    k3 = derivative(
        rho + 0.5 * dt * k2,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        noise,
    )
    k4 = derivative(
        rho + dt * k3,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        noise,
    )
    return hermitize_trace_one(rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def evolve_open_leakage(
    pulse: np.ndarray,
    disorder_strength: float,
    disorder_seed: int,
    noise: NoiseCase,
) -> tuple[float, float, float, float, float]:
    p = problem()
    dim = p.h0.shape[0]
    disorder = random_disorder(disorder_seed, dim)
    number_operator = np.diag(np.arange(dim, dtype=float)).astype(complex)
    relaxation_operator = annihilation(dim)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    rho = p.initial.copy()
    fids: list[float] = []
    leaks: list[float] = []

    for index, t in enumerate(t_eval):
        rho = hermitize_trace_one(rho)
        fids.append(fidelity(rho, p.target))
        leaks.append(leakage(rho, p))
        if index == len(pulse):
            break

        controls_i = tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        disorder_i = interaction_frame_operator(p, disorder, float(t))
        dephasing_i = interaction_frame_operator(p, number_operator, float(t))
        relaxation_i = interaction_frame_operator(p, relaxation_operator, float(t))
        hamiltonian = disorder_strength * disorder_i
        for coeff, hc_i in zip(pulse[index], controls_i):
            hamiltonian = hamiltonian + coeff * hc_i
        rho = rk4_open_step(rho, dt, hamiltonian, dephasing_i, relaxation_i, noise)

    inf = np.maximum(0.0, 1.0 - np.array(fids))
    tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
    purity = float(np.real(np.trace(rho @ rho)))
    return float(fids[-1]), float(leaks[-1]), float(np.max(leaks)), float(np.mean(tail)), purity


def evaluate_pulse(
    pulse: np.ndarray,
    controller: str,
    training_seconds: float,
    training_objective: float | None,
    optimizer_iterations: int | None,
    optimizer_success: bool | None,
    disorder_strength: float = 0.03,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    p = problem()
    for noise in NOISE_CASES:
        for seed in test_seeds:
            final_fidelity, final_leakage, max_leakage, tail_infidelity, purity = (
                evolve_open_leakage(pulse, disorder_strength, seed, noise)
            )
            rows.append(
                {
                    "system": p.name,
                    "controller": controller,
                    "eval_strength": disorder_strength,
                    "noise_case": noise.label,
                    "gamma_phi": noise.gamma_phi,
                    "gamma_relax": noise.gamma_relax,
                    "seed": seed,
                    "final_fidelity": final_fidelity,
                    "final_leakage": final_leakage,
                    "max_leakage": max_leakage,
                    "tail_infidelity_mean": tail_infidelity,
                    "final_purity": purity,
                    "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
                    "segments": len(pulse),
                    "training_seconds": training_seconds,
                    "training_objective": "" if training_objective is None else training_objective,
                    "optimizer_iterations": "" if optimizer_iterations is None else optimizer_iterations,
                    "optimizer_success": "" if optimizer_success is None else str(optimizer_success),
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), str(row["noise_case"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (controller, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        final_leaks = np.array([float(row["final_leakage"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        first = items[0]
        ci95 = 1.96 * float(np.std(fids)) / np.sqrt(len(fids))
        summary.append(
            {
                "system": str(first["system"]),
                "controller": controller,
                "noise_case": noise_case,
                "eval_strength": f"{float(first['eval_strength']):.4g}",
                "gamma_phi": f"{float(first['gamma_phi']):.4g}",
                "gamma_relax": f"{float(first['gamma_relax']):.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(first["segments"]),
                "training_seconds": f"{float(first['training_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("transmon_open_system_leakage_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("transmon_open_system_leakage_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Transmon Open-System Leakage Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    controllers = tuple(dict.fromkeys(row["controller"] for row in summary))
    noise_cases = tuple(dict.fromkeys(row["noise_case"] for row in summary))
    x = np.arange(len(noise_cases))
    width = 0.8 / float(len(controllers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    for idx, controller in enumerate(controllers):
        rows = [row for row in summary if row["controller"] == controller]
        means = np.array([float(row["final_fidelity_mean"]) for row in rows])
        mins = np.array([float(row["final_fidelity_min"]) for row in rows])
        leaks = np.array([float(row["max_leakage_mean"]) for row in rows])
        offsets = x - 0.4 + width * (idx + 0.5)
        ax1.bar(offsets, means, width=width, label=controller)
        ax1.scatter(offsets, mins, s=16, marker="x", color="black", linewidths=0.7)
        ax2.plot(x, leaks, marker="o", linewidth=1.2, label=controller)

    ax1.set_ylabel("Held-out final fidelity")
    ax1.set_ylim(0.72, 1.002)
    ax1.grid(axis="y", alpha=0.25)
    ax2.set_ylabel("Mean max leakage")
    ax2.set_ylim(0.0, 0.12)
    ax2.grid(True, alpha=0.25)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(noise_cases, rotation=25, ha="right")
    ax1.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path("transmon_open_system_leakage.pdf"))
    fig.savefig(figure_path("transmon_open_system_leakage.png"), dpi=220)
    plt.close(fig)


def run(quick: bool = False) -> None:
    if quick:
        test_seeds = range(10, 15)
        start = time.perf_counter()
        path_pulse = design_pulse(
            train_strengths=(0.03,),
            train_seeds=(0, 1),
            segments=36,
            horizon_steps=3,
            beam_width=3,
            amplitudes=(0.015, 0.035, 0.06),
        )
        path_seconds = time.perf_counter() - start
        grape_kwargs = {
            "segments": 36,
            "maxiter": 5,
            "train_strengths": (0.02, 0.03),
            "train_seeds": (0, 1),
            "restart_seeds": (3,),
            "leakage_weight": 0.8,
        }
        reference_kwargs = dict(grape_kwargs)
        adjoint_kwargs = {
            "train_strengths": (0.03,),
            "train_seeds": (0, 1),
            "horizon_steps": 3,
            "maxiter": 2,
        }
    else:
        test_seeds = range(10, 60)
        start = time.perf_counter()
        path_pulse = design_pulse()
        path_seconds = time.perf_counter() - start
        grape_kwargs = {"leakage_weight": 0.8}
        reference_kwargs = {
            "maxiter": 45,
            "train_strengths": (0.01, 0.02, 0.03),
            "train_seeds": (0, 1, 2, 3),
            "restart_seeds": (3,),
            "leakage_weight": 0.8,
        }
        adjoint_kwargs = {}

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        evaluate_pulse(
            path_pulse,
            "path_horizon",
            path_seconds,
            None,
            None,
            None,
            test_seeds=test_seeds,
        )
    )

    (
        leakage_grape_pulse,
        leakage_grape_objective,
        leakage_grape_iters,
        leakage_grape_success,
        leakage_grape_seconds,
    ) = optimize_grape_pulse(**grape_kwargs)
    rows.extend(
        evaluate_pulse(
            leakage_grape_pulse,
            "leakage_penalized_grape",
            leakage_grape_seconds,
            leakage_grape_objective,
            leakage_grape_iters,
            leakage_grape_success,
            test_seeds=test_seeds,
        )
    )

    (
        seeded_reference_pulse,
        _seeded_reference_objective,
        _seeded_reference_iters,
        _seeded_reference_success,
        seeded_reference_seconds,
    ) = optimize_grape_pulse(**reference_kwargs)

    start = time.perf_counter()
    adjoint_pulse, adjoint_objective, adjoint_iters, adjoint_success = (
        design_adjoint_horizon(seeded_reference_pulse, **adjoint_kwargs)
    )
    adjoint_seconds = seeded_reference_seconds + time.perf_counter() - start
    rows.extend(
        evaluate_pulse(
            adjoint_pulse,
            "adjoint_horizon",
            adjoint_seconds,
            adjoint_objective,
            adjoint_iters,
            adjoint_success,
            test_seeds=test_seeds,
        )
    )

    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('transmon_open_system_leakage_results.csv')}")
    print(f"wrote summary to {result_path('transmon_open_system_leakage_summary.md')}")
    print(f"wrote figure to {figure_path('transmon_open_system_leakage.pdf')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced training and five held-out seeds for pipeline checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
