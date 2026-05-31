"""Weakly anharmonic five-level leakage benchmark.

This script adds a more physical finite-dimensional stress test for the
beam-horizon idea.  The model is a rotating-frame weakly anharmonic oscillator
with five levels, nearest-neighbor quadrature controls weighted by oscillator
matrix elements, static Hermitian disorder, and an explicit leakage metric
outside the computational subspace ``{|0>, |1>}``.

The task is robust ``|0> -> |1>`` transfer.  Unlike the two-level examples, the
benchmark scores a slow reference path inside the computational subspace during
the horizon rollout.  This avoids rewarding unrealistically fast intermediate
motion that would populate the weakly detuned leakage levels.
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

from paths import figure_path, result_path


TAIL_FRACTION = 0.2
EVAL_STRENGTHS = (0.0, 0.01, 0.02, 0.03)


def dagger(a: np.ndarray) -> np.ndarray:
    return np.conjugate(a.T)


def ket(index: int, dim: int) -> np.ndarray:
    v = np.zeros((dim, 1), dtype=complex)
    v[index, 0] = 1.0
    return v


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + dagger(rho))
    return rho / np.trace(rho)


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def unitary(h: np.ndarray, t: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1.0j * vals * t)) @ dagger(vecs)


@dataclass(frozen=True)
class Problem:
    name: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    computational_projector: np.ndarray
    t_final: float


def oscillator_quadratures(dim: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(float(n))
    adag = dagger(a)
    x = a + adag
    y = -1.0j * (a - adag)
    return x, y


def problem(dim: int = 5, anharmonicity: float = 0.28, t_final: float = 60.0) -> Problem:
    # Rotating frame at the 0-1 transition of a weakly anharmonic oscillator.
    levels = np.arange(dim, dtype=float)
    h0 = np.diag(-0.5 * anharmonicity * levels * (levels - 1.0)).astype(complex)
    x, y = oscillator_quadratures(dim)
    comp = projector(ket(0, dim)) + projector(ket(1, dim))
    return Problem(
        name="five_level_weak_anharmonic",
        h0=h0,
        controls=(x, y),
        initial=projector(ket(0, dim)),
        target=projector(ket(1, dim)),
        computational_projector=comp,
        t_final=t_final,
    )


def path_projector(progress: float, dim: int) -> np.ndarray:
    """Reference projector for a slow resonant transfer in the 0-1 subspace."""
    clipped = min(1.0, max(0.0, progress))
    theta = 0.5 * np.pi * clipped
    v = np.zeros((dim, 1), dtype=complex)
    v[0, 0] = np.cos(theta)
    v[1, 0] = -1.0j * np.sin(theta)
    return projector(v)


def random_disorder(seed: int, dim: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diagonal = np.diag(rng.normal(size=dim)).astype(complex)
    raw = rng.normal(scale=0.35, size=(dim, dim)) + 1.0j * rng.normal(
        scale=0.35, size=(dim, dim)
    )
    hermitian = 0.5 * (raw + dagger(raw))
    h = diagonal + hermitian
    h = h - (np.trace(h) / dim) * np.eye(dim)
    norm = np.linalg.norm(h, ord="fro")
    return h / norm if norm > 0 else h


def interaction_frame_operator(p: Problem, operator: np.ndarray, t: float) -> np.ndarray:
    u0 = unitary(p.h0, t)
    return dagger(u0) @ operator @ u0


def leakage(rho: np.ndarray, p: Problem) -> float:
    return float(max(0.0, 1.0 - np.real(np.trace(p.computational_projector @ rho))))


def rk_step(
    rho: np.ndarray,
    dt: float,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    strength: float,
    control: np.ndarray,
) -> np.ndarray:
    h = strength * disorder_i
    for coeff, hc_i in zip(control, controls_i):
        h = h + coeff * hc_i
    u = unitary(h, dt)
    return hermitize_trace_one(u @ rho @ dagger(u))


def step_precomputed(
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        rk_step(rho, dt, controls_i, disorder_i, strength, control)
        for rho, strength, disorder_i in zip(rhos, strengths, disorders_i)
    )


def candidate_controls(amplitudes: tuple[float, ...]) -> tuple[np.ndarray, ...]:
    controls = [np.array([0.0, 0.0])]
    for amp in amplitudes:
        controls.extend(
            [
                np.array([amp, 0.0]),
                np.array([-amp, 0.0]),
                np.array([0.0, amp]),
                np.array([0.0, -amp]),
                np.array([amp, amp]) / np.sqrt(2.0),
                np.array([amp, -amp]) / np.sqrt(2.0),
                np.array([-amp, amp]) / np.sqrt(2.0),
                np.array([-amp, -amp]) / np.sqrt(2.0),
            ]
        )
    return tuple(controls)


def local_seeded_controls(
    center: np.ndarray,
    radii: tuple[float, ...] = (0.0, 0.01, 0.02, 0.04),
    umax: float = 0.12,
) -> tuple[np.ndarray, ...]:
    directions = (
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([1.0, -1.0]) / np.sqrt(2.0),
        np.array([-1.0, 1.0]) / np.sqrt(2.0),
        np.array([-1.0, -1.0]) / np.sqrt(2.0),
    )
    controls: list[np.ndarray] = []
    for radius in radii:
        for direction in directions:
            candidate = np.clip(center + radius * direction, -umax, umax)
            if not any(np.linalg.norm(candidate - existing) < 1e-12 for existing in controls):
                controls.append(candidate)
    return tuple(controls)


def scenario_cost(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    score_target: np.ndarray,
    energy_mean: float,
    worst_weight: float,
    leakage_weight: float,
    energy_weight: float,
) -> float:
    infids = np.array([1.0 - fidelity(rho, score_target) for rho in rhos])
    leaks = np.array([leakage(rho, p) for rho in rhos])
    return float(
        np.mean(infids)
        + worst_weight * np.max(infids)
        + leakage_weight * np.mean(leaks)
        + energy_weight * energy_mean
    )


def select_control_beam(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    path_targets: tuple[np.ndarray, ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    worst_weight: float,
    leakage_weight: float,
    energy_weight: float,
) -> np.ndarray:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        controls_i = controls_cache[cache_index]
        disorders_i = disorder_cache[cache_index]
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_precomputed(
                    states, strengths, controls_i, disorders_i, dt, candidate
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = scenario_cost(
                    p,
                    next_states,
                    path_targets[cache_index + 1],
                    next_energy / float(depth + 1),
                    worst_weight,
                    leakage_weight,
                    energy_weight,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0][0]


def design_pulse(
    train_strengths: tuple[float, ...] = (0.03, 0.05),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    segments: int = 120,
    horizon_steps: int = 5,
    beam_width: int = 6,
    amplitudes: tuple[float, ...] = (0.015, 0.025, 0.04, 0.06, 0.08),
    worst_weight: float = 0.25,
    leakage_weight: float = 0.5,
    energy_weight: float = 1e-4,
) -> np.ndarray:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed, p.h0.shape[0]))
        for strength in train_strengths
        for seed in train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(amplitudes)
    cache_times = tuple(float(t_eval[min(j, segments)]) for j in range(segments + horizon_steps))
    path_targets = tuple(
        path_projector(min(j, segments) / float(segments), p.h0.shape[0])
        for j in range(segments + horizon_steps + 1)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )
    pulse = []
    for index in range(segments):
        control = select_control_beam(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            path_targets,
            index,
            dt,
            candidates,
            horizon_steps,
            beam_width,
            worst_weight,
            leakage_weight,
            energy_weight,
        )
        pulse.append(control)
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dt,
            control,
        )
    return np.vstack(pulse)


def select_gradient_seeded_control(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    path_targets: tuple[np.ndarray, ...],
    reference_pulse: np.ndarray,
    start_index: int,
    dt: float,
    horizon_steps: int,
    beam_width: int,
    worst_weight: float,
    leakage_weight: float,
    energy_weight: float,
) -> np.ndarray:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        ref_index = min(cache_index, len(reference_pulse) - 1)
        controls_i = controls_cache[cache_index]
        disorders_i = disorder_cache[cache_index]
        candidates = local_seeded_controls(reference_pulse[ref_index])
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_precomputed(
                    states, strengths, controls_i, disorders_i, dt, candidate
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = scenario_cost(
                    p,
                    next_states,
                    path_targets[cache_index + 1],
                    next_energy / float(depth + 1),
                    worst_weight,
                    leakage_weight,
                    energy_weight,
                )
                expanded.append((cost, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0][0]


def design_gradient_seeded_horizon(
    reference_pulse: np.ndarray,
    train_strengths: tuple[float, ...] = (0.01, 0.02, 0.03),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3),
    horizon_steps: int = 5,
    beam_width: int = 6,
    worst_weight: float = 0.25,
    leakage_weight: float = 0.8,
    energy_weight: float = 1e-5,
) -> np.ndarray:
    p = problem()
    segments = len(reference_pulse)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed, p.h0.shape[0]))
        for strength in train_strengths
        for seed in train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    cache_times = tuple(
        float(t_eval[min(j, segments)]) for j in range(segments + horizon_steps)
    )
    path_targets = tuple(
        path_projector(min(j, segments) / float(segments), p.h0.shape[0])
        for j in range(segments + horizon_steps + 1)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )
    pulse = []
    for index in range(segments):
        control = select_gradient_seeded_control(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            path_targets,
            reference_pulse,
            index,
            dt,
            horizon_steps,
            beam_width,
            worst_weight,
            leakage_weight,
            energy_weight,
        )
        pulse.append(control)
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[index],
            disorder_cache[index],
            dt,
            control,
        )
    return np.vstack(pulse)


def horizon_objective_and_gradient(
    controls_flat: np.ndarray,
    current_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    score_target: np.ndarray,
    p: Problem,
    leakage_weight: float,
    worst_weight: float,
    energy_weight: float,
) -> tuple[float, np.ndarray]:
    n_controls = len(p.controls)
    horizon_steps = controls_flat.size // n_controls
    controls = np.reshape(controls_flat, (horizon_steps, n_controls))
    leakage_observable = np.eye(p.h0.shape[0], dtype=complex) - p.computational_projector
    scenario_costs = []
    scenario_grads = []

    for rho0, strength, disorders_i in zip(current_states, strengths, disorder_cache):
        rhos = [rho0]
        unitaries = []
        generators = []
        frechet_dirs = []
        for depth, coeffs in enumerate(controls):
            index = start_index + depth
            h = strength * disorders_i[index]
            for coeff, hc_i in zip(coeffs, controls_cache[index]):
                h = h + coeff * hc_i
            generator = -1.0j * h * dt
            unitary_i = expm(generator)
            generators.append(generator)
            unitaries.append(unitary_i)
            frechet_dirs.append([-1.0j * hc_i * dt for hc_i in controls_cache[index]])
            rhos.append(unitary_i @ rhos[-1] @ dagger(unitary_i))

        leakage_values = [
            float(np.real(np.trace(leakage_observable @ rho))) for rho in rhos[1:]
        ]
        scenario_cost = (
            1.0
            - fidelity(rhos[-1], score_target)
            + leakage_weight * float(np.mean(leakage_values))
        )
        scenario_costs.append(scenario_cost)

        state_weight = leakage_weight / float(horizon_steps)
        lambdas: list[np.ndarray] = [
            np.zeros_like(p.target) for _ in range(horizon_steps + 1)
        ]
        lambdas[-1] = -score_target + state_weight * leakage_observable
        for depth in reversed(range(1, horizon_steps)):
            lambdas[depth] = (
                state_weight * leakage_observable
                + dagger(unitaries[depth]) @ lambdas[depth + 1] @ unitaries[depth]
            )

        grad = np.zeros((horizon_steps, n_controls), dtype=float)
        for depth in range(horizon_steps):
            for control_index in range(n_controls):
                d_unitary = expm_frechet(
                    generators[depth],
                    frechet_dirs[depth][control_index],
                    compute_expm=False,
                )
                value = np.trace(
                    lambdas[depth + 1]
                    @ d_unitary
                    @ rhos[depth]
                    @ dagger(unitaries[depth])
                )
                grad[depth, control_index] = 2.0 * float(np.real(value))
        scenario_grads.append(grad.reshape(-1))

    costs = np.array(scenario_costs, dtype=float)
    grads = np.vstack(scenario_grads)
    worst_index = int(np.argmax(costs))
    objective = float(
        np.mean(costs)
        + worst_weight * costs[worst_index]
        + energy_weight * np.mean(np.square(controls_flat))
    )
    grad = (
        np.mean(grads, axis=0)
        + worst_weight * grads[worst_index]
        + energy_weight * (2.0 / controls_flat.size) * controls_flat
    )
    return objective, grad


def design_adjoint_horizon(
    reference_pulse: np.ndarray,
    train_strengths: tuple[float, ...] = (0.01, 0.02, 0.03),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3),
    horizon_steps: int = 5,
    maxiter: int = 6,
    trust_radius: float = 0.02,
    umax: float = 0.12,
    worst_weight: float = 0.25,
    leakage_weight: float = 0.8,
    energy_weight: float = 1e-5,
) -> tuple[np.ndarray, float, int, bool]:
    p = problem()
    segments = len(reference_pulse)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed, p.h0.shape[0]))
        for strength in train_strengths
        for seed in train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    cache_times = tuple(
        float(t_eval[min(j, segments - 1)]) for j in range(segments + horizon_steps)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for t in cache_times)
        for _, disorder in scenarios
    )
    pulse = []
    objectives = []
    iterations = []
    successes = []

    for index in range(segments):
        horizon = min(horizon_steps, segments - index)
        initial = reference_pulse[index : index + horizon].reshape(-1).copy()
        target = path_projector((index + horizon) / float(segments), p.h0.shape[0])
        lower = np.maximum(-umax, initial - trust_radius)
        upper = np.minimum(umax, initial + trust_radius)
        result = minimize(
            lambda x: horizon_objective_and_gradient(
                x,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                index,
                dt,
                target,
                p,
                leakage_weight,
                worst_weight,
                energy_weight,
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper)),
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        control = np.reshape(result.x, (horizon, len(p.controls)))[0]
        pulse.append(control)
        objectives.append(float(result.fun))
        iterations.append(int(result.nit))
        successes.append(bool(result.success))
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[index],
            tuple(disorders[index] for disorders in disorder_cache),
            dt,
            control,
        )

    return np.vstack(pulse), float(np.mean(objectives)), int(sum(iterations)), all(successes)


def evaluate_pulse(
    pulse: np.ndarray,
    disorder_strength: float,
    controller: str,
    training_seconds: float,
    training_objective: float | None = None,
    optimizer_iterations: int | None = None,
    optimizer_success: bool | None = None,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        for t in t_eval[:-1]
    )
    rows = []
    for seed in test_seeds:
        disorder = random_disorder(seed, p.h0.shape[0])
        rho = p.initial.copy()
        fids = []
        leaks = []
        for j, t in enumerate(t_eval):
            rho = hermitize_trace_one(rho)
            fids.append(fidelity(rho, p.target))
            leaks.append(leakage(rho, p))
            if j < len(pulse):
                disorder_i = interaction_frame_operator(p, disorder, float(t))
                rho = rk_step(
                    rho,
                    dt,
                    controls_cache[j],
                    disorder_i,
                    disorder_strength,
                    pulse[j],
                )
        inf = np.maximum(0.0, 1.0 - np.array(fids))
        tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
        rows.append(
            {
                "system": p.name,
                "controller": controller,
                "eval_strength": disorder_strength,
                "seed": seed,
                "final_fidelity": float(fids[-1]),
                "final_leakage": float(leaks[-1]),
                "max_leakage": float(np.max(leaks)),
                "tail_infidelity_mean": float(np.mean(tail)),
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
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), float(row["eval_strength"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (controller, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        final_leaks = np.array([float(row["final_leakage"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        ci95 = 1.96 * float(np.std(fids)) / np.sqrt(len(fids))
        summary.append(
            {
                "system": str(items[0]["system"]),
                "controller": controller,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "max_leakage_max": f"{np.max(max_leaks):.6g}",
                "tail_infidelity_mean": f"{np.mean([float(row['tail_infidelity_mean']) for row in items]):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "segments": str(items[0]["segments"]),
                "training_seconds": f"{float(items[0]['training_seconds']):.3f}",
                "training_objective": str(items[0]["training_objective"]),
                "optimizer_iterations": str(items[0]["optimizer_iterations"]),
                "optimizer_success": str(items[0]["optimizer_success"]),
            }
        )
    return summary


def segment_data(
    controls_flat: np.ndarray,
    segments: int,
    strength: float,
    disorder: np.ndarray,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[np.ndarray]]:
    p = problem()
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


def state_fidelity_and_gradient(
    controls_flat: np.ndarray,
    segments: int,
    strength: float,
    disorder: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = problem()
    unitaries, frechet_dirs, generators = segment_data(
        controls_flat, segments, strength, disorder
    )
    n_controls = len(p.controls)

    rhos = [p.initial.copy()]
    for propagator in unitaries:
        rhos.append(propagator @ rhos[-1] @ dagger(propagator))

    lambdas: list[np.ndarray] = [np.zeros_like(p.target) for _ in range(segments + 1)]
    lambdas[-1] = p.target
    for index in reversed(range(segments)):
        lambdas[index] = dagger(unitaries[index]) @ lambdas[index + 1] @ unitaries[index]

    grad = np.zeros((segments, n_controls), dtype=float)
    for index in range(segments):
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


def state_cost_and_gradient(
    controls_flat: np.ndarray,
    segments: int,
    strength: float,
    disorder: np.ndarray,
    leakage_weight: float,
) -> tuple[float, np.ndarray]:
    p = problem()
    unitaries, frechet_dirs, generators = segment_data(
        controls_flat, segments, strength, disorder
    )
    n_controls = len(p.controls)
    leakage_observable = np.eye(p.h0.shape[0], dtype=complex) - p.computational_projector

    rhos = [p.initial.copy()]
    for propagator in unitaries:
        rhos.append(propagator @ rhos[-1] @ dagger(propagator))

    final_infidelity = 1.0 - fidelity(rhos[-1], p.target)
    leakage_values = [float(np.real(np.trace(leakage_observable @ rho))) for rho in rhos[1:]]
    state_weight = leakage_weight / float(segments)
    objective = final_infidelity + leakage_weight * float(np.mean(leakage_values))

    lambdas: list[np.ndarray] = [np.zeros_like(p.target) for _ in range(segments + 1)]
    lambdas[-1] = -p.target + state_weight * leakage_observable
    for index in reversed(range(1, segments)):
        lambdas[index] = (
            state_weight * leakage_observable
            + dagger(unitaries[index]) @ lambdas[index + 1] @ unitaries[index]
        )

    grad = np.zeros((segments, n_controls), dtype=float)
    for index in range(segments):
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

    return objective, grad.reshape(-1)


def grape_objective_and_gradient(
    controls_flat: np.ndarray,
    segments: int,
    scenarios: tuple[tuple[float, np.ndarray], ...],
    worst_weight: float,
    energy_weight: float,
    leakage_weight: float = 0.0,
) -> tuple[float, np.ndarray]:
    metrics = []
    metric_grads = []
    for strength, disorder in scenarios:
        if leakage_weight == 0.0:
            fid, grad_fid = state_fidelity_and_gradient(
                controls_flat, segments, strength, disorder
            )
            metrics.append(1.0 - fid)
            metric_grads.append(-grad_fid)
        else:
            metric, grad_metric = state_cost_and_gradient(
                controls_flat, segments, strength, disorder, leakage_weight
            )
            metrics.append(metric)
            metric_grads.append(grad_metric)

    metric_values = np.array(metrics, dtype=float)
    grads = np.vstack(metric_grads)
    worst_index = int(np.argmax(metric_values))

    objective = float(np.mean(metric_values) + worst_weight * metric_values[worst_index])
    grad = np.mean(grads, axis=0) + worst_weight * grads[worst_index]
    energy = float(np.mean(np.square(controls_flat)))
    objective += energy_weight * energy
    grad += energy_weight * (2.0 / controls_flat.size) * controls_flat
    return objective, grad


def optimize_grape_pulse(
    segments: int = 80,
    maxiter: int = 40,
    train_strengths: tuple[float, ...] = (0.01, 0.02),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3),
    restart_seeds: tuple[int, ...] = (3, 17),
    umax: float = 0.12,
    worst_weight: float = 0.25,
    energy_weight: float = 1e-5,
    leakage_weight: float = 0.0,
) -> tuple[np.ndarray, float, int, bool, float]:
    p = problem()
    scenarios = tuple(
        (strength, random_disorder(seed, p.h0.shape[0]))
        for strength in train_strengths
        for seed in train_seeds
    )
    base = np.tile(
        np.array([np.pi / (2.0 * p.t_final), 0.0]), (segments, 1)
    ).reshape(-1)
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    best_iterations = 0
    best_success = False
    start = time.perf_counter()

    for restart_seed in restart_seeds:
        rng = np.random.default_rng(restart_seed)
        x0 = base + rng.normal(scale=0.005, size=base.size)
        result = minimize(
            lambda x: grape_objective_and_gradient(
                x, segments, scenarios, worst_weight, energy_weight, leakage_weight
            ),
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-umax, umax)] * x0.size,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        if float(result.fun) < best_fun:
            best_x = np.asarray(result.x)
            best_fun = float(result.fun)
            best_iterations = int(result.nit)
            best_success = bool(result.success)

    if best_x is None:
        raise RuntimeError("no transmon GRAPE restart completed")
    return (
        np.reshape(best_x, (segments, len(p.controls))),
        best_fun,
        best_iterations,
        best_success,
        time.perf_counter() - start,
    )


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("transmon_leakage_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("transmon_leakage_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Transmon-Like Leakage Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    controllers = sorted({row["controller"] for row in summary})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8), sharex=True)
    for controller in controllers:
        rows = [row for row in summary if row["controller"] == controller]
        strengths = np.array([float(row["eval_strength"]) for row in rows])
        fids = np.array([float(row["final_fidelity_mean"]) for row in rows])
        fid_ci = np.array([float(row["final_fidelity_ci95"]) for row in rows])
        leaks = np.array([float(row["max_leakage_mean"]) for row in rows])
        ax1.errorbar(strengths, fids, yerr=fid_ci, marker="o", capsize=3, label=controller)
        ax2.plot(strengths, leaks, marker="s", label=controller)
    ax1.set_xlabel("Disorder strength")
    ax1.set_ylabel("Held-out final fidelity")
    ax1.set_ylim(0.80, 1.002)
    ax1.grid(True, alpha=0.25)
    ax2.set_xlabel("Disorder strength")
    ax2.set_ylabel("Mean max leakage")
    ax2.set_ylim(0.0, 0.16)
    ax2.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("transmon_leakage.pdf"))
    fig.savefig(figure_path("transmon_leakage.png"), dpi=220)
    plt.close(fig)


def gradient_check() -> None:
    rng = np.random.default_rng(123)
    segments = 5
    p = problem()
    controls = rng.normal(scale=0.01, size=segments * len(p.controls))
    scenarios = ((0.01, random_disorder(0, p.h0.shape[0])),)
    for leakage_weight in (0.0, 0.8):
        value, grad = grape_objective_and_gradient(
            controls,
            segments,
            scenarios,
            worst_weight=0.25,
            energy_weight=1e-5,
            leakage_weight=leakage_weight,
        )
        direction = rng.normal(size=controls.size)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        plus = grape_objective_and_gradient(
            controls + eps * direction,
            segments,
            scenarios,
            worst_weight=0.25,
            energy_weight=1e-5,
            leakage_weight=leakage_weight,
        )[0]
        minus = grape_objective_and_gradient(
            controls - eps * direction,
            segments,
            scenarios,
            worst_weight=0.25,
            energy_weight=1e-5,
            leakage_weight=leakage_weight,
        )[0]
        finite_difference = (plus - minus) / (2.0 * eps)
        analytic = float(np.dot(grad, direction))
        print(
            f"leakage_weight={leakage_weight:.1f}: value={value:.8g}, "
            f"finite_diff={finite_difference:.8g}, analytic={analytic:.8g}, "
            f"error={abs(finite_difference - analytic):.3g}"
        )

    horizon_steps = 3
    t_eval = np.linspace(0.0, p.t_final, 81)
    dt = float(t_eval[1] - t_eval[0])
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        for t in t_eval[:horizon_steps]
    )
    disorder = random_disorder(1, p.h0.shape[0])
    disorder_cache = (
        tuple(interaction_frame_operator(p, disorder, float(t)) for t in t_eval[:horizon_steps]),
    )
    horizon_controls = rng.normal(scale=0.01, size=horizon_steps * len(p.controls))
    target = path_projector(horizon_steps / 80.0, p.h0.shape[0])
    value, grad = horizon_objective_and_gradient(
        horizon_controls,
        (p.initial.copy(),),
        (0.01,),
        controls_cache,
        disorder_cache,
        0,
        dt,
        target,
        p,
        leakage_weight=0.8,
        worst_weight=0.25,
        energy_weight=1e-5,
    )
    direction = rng.normal(size=horizon_controls.size)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = horizon_objective_and_gradient(
        horizon_controls + eps * direction,
        (p.initial.copy(),),
        (0.01,),
        controls_cache,
        disorder_cache,
        0,
        dt,
        target,
        p,
        leakage_weight=0.8,
        worst_weight=0.25,
        energy_weight=1e-5,
    )[0]
    minus = horizon_objective_and_gradient(
        horizon_controls - eps * direction,
        (p.initial.copy(),),
        (0.01,),
        controls_cache,
        disorder_cache,
        0,
        dt,
        target,
        p,
        leakage_weight=0.8,
        worst_weight=0.25,
        energy_weight=1e-5,
    )[0]
    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(np.dot(grad, direction))
    print(
        f"horizon: value={value:.8g}, finite_diff={finite_difference:.8g}, "
        f"analytic={analytic:.8g}, error={abs(finite_difference - analytic):.3g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gradient-check",
        action="store_true",
        help="Run directional derivative checks for the GRAPE objectives and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return

    rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()
    horizon_pulse = design_pulse()
    horizon_seconds = time.perf_counter() - start
    for strength in EVAL_STRENGTHS:
        rows.extend(
            evaluate_pulse(
                horizon_pulse,
                strength,
                controller="path_horizon",
                training_seconds=horizon_seconds,
            )
        )

    grape_pulse, grape_objective, grape_iters, grape_success, grape_seconds = (
        optimize_grape_pulse()
    )
    for strength in EVAL_STRENGTHS:
        rows.extend(
            evaluate_pulse(
                grape_pulse,
                strength,
                controller="terminal_grape",
                training_seconds=grape_seconds,
                training_objective=grape_objective,
                optimizer_iterations=grape_iters,
                optimizer_success=grape_success,
            )
        )

    (
        leakage_grape_pulse,
        leakage_grape_objective,
        leakage_grape_iters,
        leakage_grape_success,
        leakage_grape_seconds,
    ) = optimize_grape_pulse(leakage_weight=0.8)
    for strength in EVAL_STRENGTHS:
        rows.extend(
            evaluate_pulse(
                leakage_grape_pulse,
                strength,
                controller="leakage_penalized_grape",
                training_seconds=leakage_grape_seconds,
                training_objective=leakage_grape_objective,
                optimizer_iterations=leakage_grape_iters,
                optimizer_success=leakage_grape_success,
            )
        )

    (
        seeded_reference_pulse,
        seeded_reference_objective,
        seeded_reference_iters,
        seeded_reference_success,
        seeded_reference_seconds,
    ) = optimize_grape_pulse(
        maxiter=45,
        train_strengths=(0.01, 0.02, 0.03),
        train_seeds=(0, 1, 2, 3),
        restart_seeds=(3,),
        leakage_weight=0.8,
    )
    start = time.perf_counter()
    seeded_horizon_pulse = design_gradient_seeded_horizon(seeded_reference_pulse)
    seeded_horizon_seconds = seeded_reference_seconds + time.perf_counter() - start
    for strength in EVAL_STRENGTHS:
        rows.extend(
            evaluate_pulse(
                seeded_horizon_pulse,
                strength,
                controller="gradient_seeded_horizon",
                training_seconds=seeded_horizon_seconds,
                training_objective=seeded_reference_objective,
                optimizer_iterations=seeded_reference_iters,
                optimizer_success=seeded_reference_success,
            )
        )

    start = time.perf_counter()
    adjoint_horizon_pulse, adjoint_objective, adjoint_iters, adjoint_success = (
        design_adjoint_horizon(seeded_reference_pulse)
    )
    adjoint_seconds = seeded_reference_seconds + time.perf_counter() - start
    for strength in EVAL_STRENGTHS:
        rows.extend(
            evaluate_pulse(
                adjoint_horizon_pulse,
                strength,
                controller="adjoint_horizon",
                training_seconds=adjoint_seconds,
                training_objective=adjoint_objective,
                optimizer_iterations=adjoint_iters,
                optimizer_success=adjoint_success,
            )
        )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('transmon_leakage_results.csv')}")
    print(f"wrote summary to {result_path('transmon_leakage_summary.md')}")
    print(f"wrote figure to {figure_path('transmon_leakage.pdf')}")


if __name__ == "__main__":
    main()
