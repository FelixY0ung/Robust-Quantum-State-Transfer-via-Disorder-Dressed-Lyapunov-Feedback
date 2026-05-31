"""Standalone five-level Lindblad leakage-adjoint horizon diagnostic.

The existing leakage-plus-Lindblad stress test evaluates closed-system leakage
pulses under simultaneous leakage and Markovian noise.  This script takes the
next journal-level step: it starts from the finite-candidate path horizon and
then optimizes each short receding horizon directly through the five-level
Lindblad model with an explicit running leakage penalty.

The diagnostic remains local and nonconvex.  It does not use a terminal GRAPE
reference pulse, and it does not claim global optimality or convergence.
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

from open_system_grape_baseline import commutator_super, liouvillian, vec
from paths import figure_path, result_path
from transmon_leakage_horizon import (
    design_pulse,
    interaction_frame_operator,
    path_projector,
    problem,
    random_disorder,
)
from transmon_open_system_leakage import (
    NOISE_CASES,
    NoiseCase,
    annihilation,
    evaluate_pulse,
    summarize,
)


TRAIN_NOISE = NoiseCase("combined", gamma_phi=0.001, gamma_relax=0.0005)
PLOT_CONTROLLERS = (
    ("open_leakage_path_seed", "Path horizon"),
    ("standalone_open_leakage_adjoint", "Direct adjoint"),
    ("adjoint_horizon", "Ref.-adjoint"),
    ("leakage_penalized_grape", "Leakage-GRAPE"),
)


@dataclass(frozen=True)
class OpenLeakageAdjointConfig:
    segments: int = 120
    horizon_steps: int = 5
    horizon_maxiter: int = 4
    train_strength: float = 0.03
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    eval_strength: float = 0.03
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    umax: float = 0.12
    trust_radius: float = 0.025
    worst_weight: float = 0.25
    leakage_weight: float = 0.8
    energy_weight: float = 1e-5
    trust_weight: float = 1e-3


def precompute_caches(
    disorders: tuple[np.ndarray, ...],
    config: OpenLeakageAdjointConfig,
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[tuple[np.ndarray, ...], ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    p = problem()
    dim = p.h0.shape[0]
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
    number_operator = np.diag(np.arange(dim, dtype=float)).astype(complex)
    relaxation_operator = annihilation(dim)
    dephasing_cache = tuple(
        interaction_frame_operator(p, number_operator, t) for t in cache_times
    )
    relaxation_cache = tuple(
        interaction_frame_operator(p, relaxation_operator, t) for t in cache_times
    )
    return controls_cache, disorder_cache, dephasing_cache, relaxation_cache


def step_liouville(
    state_vectors: tuple[np.ndarray, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dephasing_i: np.ndarray,
    relaxation_i: np.ndarray,
    dt: float,
    control: np.ndarray,
    config: OpenLeakageAdjointConfig,
) -> tuple[np.ndarray, ...]:
    next_vectors: list[np.ndarray] = []
    for state, disorder_i in zip(state_vectors, disorders_i):
        hamiltonian = config.train_strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            hamiltonian = hamiltonian + coeff * hc_i
        generator = liouvillian(
            hamiltonian,
            dephasing_i,
            relaxation_i,
            TRAIN_NOISE.gamma_phi,
            TRAIN_NOISE.gamma_relax,
        )
        next_vectors.append(expm(generator * dt) @ state)
    return tuple(next_vectors)


def horizon_objective_and_gradient(
    controls_flat: np.ndarray,
    current_states: tuple[np.ndarray, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dephasing_cache: tuple[np.ndarray, ...],
    relaxation_cache: tuple[np.ndarray, ...],
    reference_flat: np.ndarray,
    start_index: int,
    dt: float,
    config: OpenLeakageAdjointConfig,
) -> tuple[float, np.ndarray]:
    p = problem()
    n_controls = len(p.controls)
    horizon_steps = controls_flat.size // n_controls
    controls = np.reshape(controls_flat, (horizon_steps, n_controls))
    score_target = path_projector(
        min(config.segments, start_index + horizon_steps) / float(config.segments),
        p.h0.shape[0],
    )
    target_row = vec(score_target.T)
    leakage_observable = np.eye(p.h0.shape[0], dtype=complex) - p.computational_projector
    leakage_row = vec(leakage_observable.T)
    scenario_costs: list[float] = []
    scenario_grads: list[np.ndarray] = []

    for scenario_index, state0 in enumerate(current_states):
        states = [state0]
        propagators: list[np.ndarray] = []
        generators: list[np.ndarray] = []
        frechet_dirs: list[list[np.ndarray]] = []

        for depth, coeffs in enumerate(controls):
            cache_index = min(start_index + depth, len(controls_cache) - 1)
            hamiltonian = config.train_strength * disorder_cache[cache_index][scenario_index]
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
            states.append(propagator @ states[-1])

        target_score = float(np.real(target_row @ states[-1]))
        leakage_scores = [float(np.real(leakage_row @ state)) for state in states[1:]]
        scenario_cost = 1.0 - target_score + config.leakage_weight * float(
            np.mean(leakage_scores)
        )
        scenario_costs.append(scenario_cost)

        state_weight = config.leakage_weight / float(horizon_steps)
        costates: list[np.ndarray] = [
            np.zeros_like(target_row) for _ in range(horizon_steps + 1)
        ]
        costates[-1] = -target_row + state_weight * leakage_row
        for depth in reversed(range(1, horizon_steps)):
            costates[depth] = (
                state_weight * leakage_row + propagators[depth].T @ costates[depth + 1]
            )

        grad = np.zeros((horizon_steps, n_controls), dtype=float)
        for depth in range(horizon_steps):
            for control_index in range(n_controls):
                d_propagator = expm_frechet(
                    generators[depth],
                    frechet_dirs[depth][control_index],
                    compute_expm=False,
                )
                value = costates[depth + 1] @ (d_propagator @ states[depth])
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


def design_open_leakage_adjoint_horizon(
    reference_pulse: np.ndarray,
    config: OpenLeakageAdjointConfig,
) -> tuple[np.ndarray, float, int, bool, float]:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    states = tuple(vec(p.initial) for _ in disorders)
    controls_cache, disorder_cache, dephasing_cache, relaxation_cache = precompute_caches(
        disorders,
        config,
    )
    pulse: list[np.ndarray] = []
    objectives: list[float] = []
    iterations: list[int] = []
    successes: list[bool] = []
    start = time.perf_counter()

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        initial = reference_pulse[index : index + horizon].reshape(-1).copy()
        lower = np.maximum(-config.umax, initial - config.trust_radius)
        upper = np.minimum(config.umax, initial + config.trust_radius)
        result = minimize(
            lambda x: horizon_objective_and_gradient(
                x,
                states,
                controls_cache,
                disorder_cache,
                dephasing_cache,
                relaxation_cache,
                initial,
                index,
                dt,
                config,
            ),
            initial,
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
            controls_cache[index],
            disorder_cache[index],
            dephasing_cache[index],
            relaxation_cache[index],
            dt,
            control,
            config,
        )

    return (
        np.vstack(pulse),
        float(np.mean(objectives)),
        int(sum(iterations)),
        all(successes),
        time.perf_counter() - start,
    )


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("transmon_open_leakage_adjoint_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("transmon_open_leakage_adjoint_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Standalone Open Leakage-Adjoint Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_combined_results(rows)


def load_result_rows(filename: str) -> list[dict[str, float | int | str]]:
    path = result_path(filename)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def plot_combined_results(rows: list[dict[str, float | int | str]]) -> None:
    stress_rows = load_result_rows("transmon_open_system_leakage_results.csv")
    selected_rows: list[dict[str, float | int | str]] = []
    selected_rows.extend(
        row
        for row in rows
        if str(row["controller"]) in {"open_leakage_path_seed", "standalone_open_leakage_adjoint"}
    )
    selected_rows.extend(
        row
        for row in stress_rows
        if str(row["controller"]) in {"adjoint_horizon", "leakage_penalized_grape"}
    )
    if not selected_rows:
        return

    summary = summarize(selected_rows)
    by_key = {
        (str(row["controller"]), str(row["noise_case"])): row
        for row in summary
    }
    noise_cases = tuple(case.label for case in NOISE_CASES)
    x = np.arange(len(noise_cases))
    width = 0.82 / float(len(PLOT_CONTROLLERS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.05), sharex=True)
    handles = []
    labels = []
    for idx, (controller, label) in enumerate(PLOT_CONTROLLERS):
        controller_rows = [by_key[(controller, noise)] for noise in noise_cases]
        means = np.array([float(row["final_fidelity_mean"]) for row in controller_rows])
        mins = np.array([float(row["final_fidelity_min"]) for row in controller_rows])
        leaks = np.array([float(row["max_leakage_mean"]) for row in controller_rows])
        offsets = x - 0.41 + width * (idx + 0.5)
        bar = ax1.bar(offsets, means, width=width, label=label)
        handles.append(bar[0])
        labels.append(label)
        ax1.scatter(offsets, mins, s=15, marker="x", color="black", linewidths=0.7)
        ax2.plot(x, leaks, marker="o", linewidth=1.2, label=label)

    ax1.set_ylabel("Held-out final fidelity")
    ax1.set_ylim(0.48, 1.002)
    ax1.grid(axis="y", alpha=0.25)
    ax2.set_ylabel("Mean max leakage")
    ax2.set_ylim(0.0, 0.065)
    ax2.grid(True, alpha=0.25)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(noise_cases, rotation=25, ha="right")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figure_path("transmon_open_leakage_combined.pdf"))
    fig.savefig(figure_path("transmon_open_leakage_combined.png"), dpi=220)
    plt.close(fig)


def plot_existing_results() -> None:
    rows = load_result_rows("transmon_open_leakage_adjoint_results.csv")
    if not rows:
        raise FileNotFoundError(result_path("transmon_open_leakage_adjoint_results.csv"))
    plot_combined_results(rows)
    print(f"wrote figure to {figure_path('transmon_open_leakage_combined.pdf')}")


def gradient_check() -> None:
    config = OpenLeakageAdjointConfig(
        segments=12,
        horizon_steps=3,
        horizon_maxiter=1,
        train_seeds=(0, 1),
        eval_seeds=tuple(range(10, 12)),
    )
    p = problem()
    rng = np.random.default_rng(7)
    controls = rng.normal(scale=0.015, size=(config.horizon_steps, len(p.controls)))
    reference = controls + rng.normal(scale=0.002, size=controls.shape)
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    states = tuple(vec(p.initial) for _ in disorders)
    caches = precompute_caches(disorders, config)
    dt = p.t_final / float(config.segments)
    value, grad = horizon_objective_and_gradient(
        controls.reshape(-1),
        states,
        *caches,
        reference.reshape(-1),
        2,
        dt,
        config,
    )
    direction = rng.normal(size=grad.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        controls.reshape(-1) + eps * direction,
        states,
        *caches,
        reference.reshape(-1),
        2,
        dt,
        config,
    )[0]
    minus = horizon_objective_and_gradient(
        controls.reshape(-1) - eps * direction,
        states,
        *caches,
        reference.reshape(-1),
        2,
        dt,
        config,
    )[0]
    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(np.dot(grad, direction))
    print(
        "value={:.8g} analytic={:.8g} finite_difference={:.8g} error={:.3g}".format(
            value,
            analytic,
            finite_difference,
            abs(analytic - finite_difference),
        )
    )


def config_from_args(args: argparse.Namespace) -> OpenLeakageAdjointConfig:
    if args.quick:
        return OpenLeakageAdjointConfig(
            segments=30,
            horizon_steps=3,
            horizon_maxiter=2,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 15)),
            trust_radius=0.025,
        )
    return OpenLeakageAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        trust_radius=args.trust_radius,
    )


def run(config: OpenLeakageAdjointConfig, quick: bool) -> None:
    if quick:
        path_kwargs = {
            "train_strengths": (config.train_strength,),
            "train_seeds": config.train_seeds,
            "segments": config.segments,
            "horizon_steps": config.horizon_steps,
            "beam_width": 3,
            "amplitudes": (0.015, 0.035, 0.06),
            "leakage_weight": 0.5,
        }
    else:
        path_kwargs = {"segments": config.segments}

    print("designing finite-candidate path horizon seed", flush=True)
    start = time.perf_counter()
    path_pulse = design_pulse(**path_kwargs)
    path_seconds = time.perf_counter() - start
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        evaluate_pulse(
            path_pulse,
            "open_leakage_path_seed",
            path_seconds,
            None,
            None,
            None,
            disorder_strength=config.eval_strength,
            test_seeds=range(min(config.eval_seeds), max(config.eval_seeds) + 1),
        )
    )

    print("polishing direct Lindblad leakage-aware horizon", flush=True)
    pulse, objective, iterations, success, horizon_seconds = design_open_leakage_adjoint_horizon(
        path_pulse,
        config,
    )
    rows.extend(
        evaluate_pulse(
            pulse,
            "standalone_open_leakage_adjoint",
            path_seconds + horizon_seconds,
            objective,
            iterations,
            success,
            disorder_strength=config.eval_strength,
            test_seeds=range(min(config.eval_seeds), max(config.eval_seeds) + 1),
        )
    )

    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('transmon_open_leakage_adjoint_results.csv')}")
    print(f"wrote summary to {result_path('transmon_open_leakage_adjoint_summary.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    parser.add_argument("--trust-radius", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return
    if args.plot_only:
        plot_existing_results()
        return
    run(config_from_args(args), quick=args.quick)


if __name__ == "__main__":
    main()
