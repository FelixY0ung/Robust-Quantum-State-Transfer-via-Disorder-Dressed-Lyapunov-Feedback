"""Open-system GRAPE baseline for dissipative state transfer.

This script optimizes terminal state-transfer fidelity through the same
interaction-frame Lindblad dynamics used in the open-system stress tests.  It is
an open-loop terminal optimizer, not a Lyapunov feedback law.  Its role is to
separate the limits of the compact finite-candidate horizon search from the
reachability of the noisy state-transfer problem.
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

from horizon_lyapunov import (
    dagger,
    fidelity,
    interaction_frame_operator,
    ket,
    problem,
    random_disorder,
)
from open_system_noise import NoiseCase, dissipator, evolve_open_system
from paths import figure_path, result_path


TRAIN_NOISE = NoiseCase("combined", gamma_phi=0.005, gamma_relax=0.002)
EVAL_NOISE_CASES = (
    NoiseCase("static_only", 0.0, 0.0),
    NoiseCase("deph_0.005", 0.005, 0.0),
    NoiseCase("relax_0.002", 0.0, 0.002),
    TRAIN_NOISE,
)


@dataclass(frozen=True)
class OpenGrapeConfig:
    segments: int = 40
    umax: float = 4.0
    maxiter: int = 100
    training_strength: float = 0.08
    training_seeds: tuple[int, ...] = (0, 1, 2, 3)
    restart_seeds: tuple[int, ...] = (17, 29)
    worst_weight: float = 0.25
    energy_weight: float = 2e-4
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))


def vec(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=complex).reshape(-1, order="F")


def mat(vector: np.ndarray) -> np.ndarray:
    dim = int(round(np.sqrt(vector.size)))
    return np.asarray(vector, dtype=complex).reshape((dim, dim), order="F")


def commutator_super(hamiltonian: np.ndarray) -> np.ndarray:
    dim = hamiltonian.shape[0]
    ident = np.eye(dim, dtype=complex)
    return -1.0j * (np.kron(ident, hamiltonian) - np.kron(hamiltonian.T, ident))


def dissipator_super(operator: np.ndarray) -> np.ndarray:
    dim = operator.shape[0]
    ident = np.eye(dim, dtype=complex)
    odag_o = dagger(operator) @ operator
    return (
        np.kron(np.conjugate(operator), operator)
        - 0.5 * np.kron(ident, odag_o)
        - 0.5 * np.kron(odag_o.T, ident)
    )


def liouvillian(
    hamiltonian: np.ndarray,
    dephasing_operator: np.ndarray,
    relaxation_operator: np.ndarray,
    gamma_phi: float,
    gamma_relax: float,
) -> np.ndarray:
    generator = commutator_super(hamiltonian)
    if gamma_phi:
        generator = generator + gamma_phi * dissipator_super(dephasing_operator)
    if gamma_relax:
        generator = generator + gamma_relax * dissipator_super(relaxation_operator)
    return generator


def segment_operators(
    task: str,
    controls_flat: np.ndarray,
    config: OpenGrapeConfig,
    strength: float,
    disorder: np.ndarray,
    noise: NoiseCase,
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    p = problem(task)
    controls = np.reshape(controls_flat, (config.segments, len(p.controls)))
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    relaxation_operator = ket(1) @ dagger(ket(0))

    propagators: list[np.ndarray] = []
    frechet_dirs: list[list[np.ndarray]] = []
    for index, coeffs in enumerate(controls):
        t = float(t_eval[index])
        control_ops = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        disorder_i = interaction_frame_operator(p, disorder, t)
        dephasing_i = interaction_frame_operator(p, p.h0, t)
        relaxation_i = interaction_frame_operator(p, relaxation_operator, t)
        hamiltonian = strength * disorder_i
        for coeff, hc_i in zip(coeffs, control_ops):
            hamiltonian = hamiltonian + coeff * hc_i
        generator = liouvillian(
            hamiltonian,
            dephasing_i,
            relaxation_i,
            noise.gamma_phi,
            noise.gamma_relax,
        )
        propagators.append(expm(generator * dt))
        frechet_dirs.append([commutator_super(hc_i) * dt for hc_i in control_ops])
    return propagators, frechet_dirs


def state_fidelity_and_gradient(
    task: str,
    controls_flat: np.ndarray,
    config: OpenGrapeConfig,
    strength: float,
    disorder: np.ndarray,
    noise: NoiseCase,
) -> tuple[float, np.ndarray]:
    p = problem(task)
    propagators, frechet_dirs = segment_operators(
        task, controls_flat, config, strength, disorder, noise
    )
    state_vectors = [vec(p.initial)]
    for propagator in propagators:
        state_vectors.append(propagator @ state_vectors[-1])

    target_row = vec(p.target.T)
    costates: list[np.ndarray] = [
        np.zeros_like(target_row) for _ in range(config.segments + 1)
    ]
    costates[-1] = target_row
    for index in reversed(range(config.segments)):
        costates[index] = propagators[index].T @ costates[index + 1]

    gradient = np.zeros((config.segments, len(p.controls)), dtype=float)
    for index in range(config.segments):
        # Reconstruct the generator logarithm is numerically undesirable; use
        # expm_frechet on the segment generator by rebuilding the local data.
        # This path is compact enough for the two-level benchmark.
        local_controls = np.reshape(controls_flat, (config.segments, len(p.controls)))
        t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
        dt = float(t_eval[1] - t_eval[0])
        t = float(t_eval[index])
        control_ops = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        disorder_i = interaction_frame_operator(p, disorder, t)
        dephasing_i = interaction_frame_operator(p, p.h0, t)
        relaxation_i = interaction_frame_operator(p, ket(1) @ dagger(ket(0)), t)
        hamiltonian = strength * disorder_i
        for coeff, hc_i in zip(local_controls[index], control_ops):
            hamiltonian = hamiltonian + coeff * hc_i
        generator_dt = (
            liouvillian(
                hamiltonian,
                dephasing_i,
                relaxation_i,
                noise.gamma_phi,
                noise.gamma_relax,
            )
            * dt
        )
        for control_index, direction in enumerate(frechet_dirs[index]):
            d_propagator = expm_frechet(
                generator_dt,
                direction,
                compute_expm=False,
            )
            value = costates[index + 1] @ (d_propagator @ state_vectors[index])
            gradient[index, control_index] = float(np.real(value))

    final_rho = mat(state_vectors[-1])
    return fidelity(final_rho, p.target), gradient.reshape(-1)


def objective_and_gradient(
    controls_flat: np.ndarray,
    task: str,
    config: OpenGrapeConfig,
    scenarios: tuple[tuple[float, np.ndarray], ...],
) -> tuple[float, np.ndarray]:
    fidelities = []
    gradients = []
    for strength, disorder in scenarios:
        fid, grad = state_fidelity_and_gradient(
            task, controls_flat, config, strength, disorder, TRAIN_NOISE
        )
        fidelities.append(fid)
        gradients.append(grad)
    fids = np.array(fidelities)
    grads = np.vstack(gradients)
    infids = 1.0 - fids
    worst_index = int(np.argmax(infids))

    objective = float(np.mean(infids) + config.worst_weight * infids[worst_index])
    grad_obj = -np.mean(grads, axis=0) - config.worst_weight * grads[worst_index]

    energy = float(np.mean(np.square(controls_flat)))
    objective += config.energy_weight * energy
    grad_obj += config.energy_weight * (2.0 / controls_flat.size) * controls_flat
    return objective, grad_obj


def optimize_open_grape(
    task: str,
    config: OpenGrapeConfig,
) -> tuple[np.ndarray, float, int, int, bool, float]:
    scenarios = tuple(
        (config.training_strength, random_disorder(seed))
        for seed in config.training_seeds
    )
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    best_seed = -1
    best_iters = 0
    best_success = False
    start = time.perf_counter()

    for seed in config.restart_seeds:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(scale=0.2, size=config.segments * len(problem(task).controls))
        result = minimize(
            lambda x: objective_and_gradient(x, task, config, scenarios),
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-config.umax, config.umax)] * x0.size,
            options={"maxiter": config.maxiter, "ftol": 1e-11, "gtol": 1e-7},
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x)
            best_seed = seed
            best_iters = int(result.nit)
            best_success = bool(result.success)

    if best_x is None:
        raise RuntimeError(f"no open-system GRAPE restart completed for {task}")
    return best_x, best_fun, best_seed, best_iters, best_success, time.perf_counter() - start


def evaluate_open_grape(
    task: str,
    controls_flat: np.ndarray,
    training_objective: float,
    restart_seed: int,
    n_iterations: int,
    success: bool,
    optimization_seconds: float,
    config: OpenGrapeConfig,
) -> list[dict[str, float | int | str]]:
    pulse = np.reshape(controls_flat, (config.segments, len(problem(task).controls)))
    energy = float(np.mean(np.sum(pulse * pulse, axis=1)))
    rows: list[dict[str, float | int | str]] = []
    for noise in EVAL_NOISE_CASES:
        for seed in config.eval_seeds:
            final_fidelity, purity = evolve_open_system(
                task, pulse, config.eval_strength, seed, noise
            )
            rows.append(
                {
                    "task": task,
                    "baseline": "open_system_grape",
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
        groups.setdefault((str(row["task"]), str(row["eval_noise_case"])), []).append(row)
    summary: list[dict[str, str]] = []
    for (task, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        summary.append(
            {
                "task": task,
                "eval_noise_case": noise_case,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "training_objective": f"{float(items[0]['training_objective']):.6g}",
                "best_restart_seed": str(items[0]["best_restart_seed"]),
                "optimizer_iterations": str(items[0]["optimizer_iterations"]),
                "optimizer_success": str(items[0]["optimizer_success"]),
                "optimization_seconds": f"{float(items[0]['optimization_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("open_system_grape_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("open_system_grape_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Open-System GRAPE Baseline Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    cases = tuple(dict.fromkeys(row["eval_noise_case"] for row in summary))
    x = np.arange(len(cases))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, task in zip(axes, ("Z", "H")):
        rows = [row for row in summary if row["task"] == task]
        means = np.array([float(row["final_fidelity_mean"]) for row in rows])
        mins = np.array([float(row["final_fidelity_min"]) for row in rows])
        ax.bar(x, means, width=width, color="#2ca02c", label="mean")
        ax.scatter(x, mins, marker="x", s=18, color="black", linewidths=0.8, label="min")
        ax.set_title(f"{task} transfer")
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out final fidelity")
    axes[0].set_ylim(0.92, 1.001)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path("open_system_grape.pdf"))
    fig.savefig(figure_path("open_system_grape.png"), dpi=220)
    plt.close(fig)


def gradient_check() -> None:
    config = OpenGrapeConfig(segments=5, maxiter=2, training_seeds=(0,), restart_seeds=(1,))
    task = "Z"
    rng = np.random.default_rng(123)
    x = rng.normal(scale=0.1, size=config.segments * len(problem(task).controls))
    direction = rng.normal(size=x.size)
    direction /= np.linalg.norm(direction)
    scenarios = ((config.training_strength, random_disorder(0)),)
    value, grad = objective_and_gradient(x, task, config, scenarios)
    eps = 1e-6
    plus, _ = objective_and_gradient(x + eps * direction, task, config, scenarios)
    minus, _ = objective_and_gradient(x - eps * direction, task, config, scenarios)
    finite_diff = (plus - minus) / (2.0 * eps)
    analytic = float(np.dot(grad, direction))
    print(
        "open-system GRAPE gradient check: "
        f"value={value:.8g}, finite_diff={finite_diff:.8g}, "
        f"analytic={analytic:.8g}, error={abs(finite_diff - analytic):.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-check", action="store_true")
    args = parser.parse_args()
    if args.gradient_check:
        gradient_check()
        return

    config = OpenGrapeConfig()
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        print(f"optimizing open-system GRAPE baseline for {task}", flush=True)
        controls, value, seed, iterations, success, seconds = optimize_open_grape(task, config)
        rows.extend(
            evaluate_open_grape(
                task,
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
    print(f"wrote {len(rows)} rows to {result_path('open_system_grape_results.csv')}")
    print(f"wrote aggregate table to {result_path('open_system_grape_summary.md')}")
    print(f"wrote figure to {figure_path('open_system_grape.pdf')}")


if __name__ == "__main__":
    main()
