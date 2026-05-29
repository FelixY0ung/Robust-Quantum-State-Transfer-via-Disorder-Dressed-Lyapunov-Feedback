"""Beam-horizon ensemble Lyapunov controller for robust two-level transfer.

This script strengthens the one-step ensemble Lyapunov diagnostic.  Instead of
choosing the shared control only from the instantaneous averaged derivative, it
evaluates a finite set of candidate control sequences over a short lookahead
horizon in the interaction frame.  The selected first control minimizes a
terminal ensemble Lyapunov value with a small held-out-style worst-case term.

The design remains Lyapunov-shaped, but it allows temporary non-monotone moves
that a purely one-step descent law rejects.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from paths import result_path

import numpy as np


TAIL_FRACTION = 0.2


def dagger(a: np.ndarray) -> np.ndarray:
    return np.conjugate(a.T)


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def ket(index: int) -> np.ndarray:
    v = np.zeros((2, 1), dtype=complex)
    v[index, 0] = 1.0
    return v


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + dagger(rho))
    return rho / np.trace(rho)


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def unitary2(h: np.ndarray, t: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1.0j * vals * t)) @ dagger(vecs)


@dataclass(frozen=True)
class Problem:
    task: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    t_final: float


def problem(task: str, t_final: float = 8.0) -> Problem:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    z0 = ket(0)
    z1 = ket(1)
    plus = normalize(z0 + z1)
    minus = normalize(z0 - z1)
    if task == "Z":
        initial, target = plus, minus
    elif task == "H":
        initial, target = z0, plus
    else:
        raise ValueError(task)
    return Problem(
        task=f"horizon_lyapunov_{task}",
        h0=sz,
        controls=(sx, sy),
        initial=projector(initial),
        target=projector(target),
        t_final=t_final,
    )


def random_disorder(seed: int) -> np.ndarray:
    p = problem("Z")
    sx, sy = p.controls
    sz = p.h0
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def interaction_frame_operator(p: Problem, operator: np.ndarray, t: float) -> np.ndarray:
    u0 = unitary2(p.h0, t)
    return dagger(u0) @ operator @ u0


def derivative(
    rho: np.ndarray,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    disorder_strength: float,
    u: np.ndarray,
) -> np.ndarray:
    h = disorder_strength * disorder_i
    for coeff, hc_i in zip(u, controls_i):
        h = h + coeff * hc_i
    return -1.0j * comm(h, rho)


def rk4_step(
    rho: np.ndarray,
    dt: float,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    disorder_strength: float,
    u: np.ndarray,
) -> np.ndarray:
    k1 = derivative(rho, controls_i, disorder_i, disorder_strength, u)
    k2 = derivative(rho + 0.5 * dt * k1, controls_i, disorder_i, disorder_strength, u)
    k3 = derivative(rho + 0.5 * dt * k2, controls_i, disorder_i, disorder_strength, u)
    k4 = derivative(rho + dt * k3, controls_i, disorder_i, disorder_strength, u)
    return hermitize_trace_one(rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


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


def scenario_cost(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    energy_mean: float,
    worst_weight: float,
    energy_weight: float,
) -> float:
    infids = np.array([1.0 - fidelity(rho, p.target) for rho in rhos])
    return float(
        np.mean(infids)
        + worst_weight * np.max(infids)
        + energy_weight * energy_mean
    )


def step_scenarios(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    scenarios: tuple[tuple[float, np.ndarray], ...],
    t: float,
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    controls_i = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
    next_rhos = []
    for rho, (strength, disorder) in zip(rhos, scenarios):
        disorder_i = interaction_frame_operator(p, disorder, t)
        next_rhos.append(rk4_step(rho, dt, controls_i, disorder_i, strength, control))
    return tuple(next_rhos)


def step_precomputed(
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        rk4_step(rho, dt, controls_i, disorder_i, strength, control)
        for rho, strength, disorder_i in zip(rhos, strengths, disorders_i)
    )


def select_control_beam(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    worst_weight: float,
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
                    next_energy / float(depth + 1),
                    worst_weight,
                    energy_weight,
                )
                expanded.append(
                    (cost, sequence + (candidate,), next_states, next_energy)
                )
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0][0]


def design_pulse(
    task: str,
    train_strengths: tuple[float, ...] = (0.05, 0.08),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    segments: int = 100,
    horizon_steps: int = 6,
    beam_width: int = 6,
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0),
    worst_weight: float = 0.25,
    energy_weight: float = 0.0,
) -> np.ndarray:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in train_strengths
        for seed in train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(amplitudes)
    cache_times = tuple(float(t_eval[min(j, segments)]) for j in range(segments + horizon_steps))
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )
    pulse = []

    for j, t in enumerate(t_eval[:-1]):
        u = select_control_beam(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            j,
            dt,
            candidates,
            horizon_steps,
            beam_width,
            worst_weight,
            energy_weight,
        )
        pulse.append(u)
        rhos = step_precomputed(rhos, strengths, controls_cache[j], disorder_cache[j], dt, u)

    return np.asarray(pulse)


def default_beam_width(task: str) -> int:
    if task == "H":
        return 8
    return 6


def evaluate_pulse(
    task: str,
    pulse: np.ndarray,
    disorder_strength: float = 0.05,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    rows = []
    for seed in test_seeds:
        disorder = random_disorder(seed)
        rho = p.initial.copy()
        fids = []
        for j, t in enumerate(t_eval):
            rho = hermitize_trace_one(rho)
            fids.append(fidelity(rho, p.target))
            if j < len(pulse):
                controls_i = tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
                disorder_i = interaction_frame_operator(p, disorder, float(t))
                rho = rk4_step(
                    rho,
                    float(t_eval[j + 1] - t_eval[j]),
                    controls_i,
                    disorder_i,
                    disorder_strength,
                    pulse[j],
                )
        inf = np.maximum(0.0, 1.0 - np.array(fids))
        tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
        rows.append(
            {
                "system": p.task,
                "disorder_strength": disorder_strength,
                "seed": seed,
                "tail_infidelity_mean": float(np.mean(tail)),
                "tail_stability_range": float(np.max(tail) - np.min(tail)),
                "final_fidelity": float(fids[-1]),
                "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["system"]), float(row["disorder_strength"])), []
        ).append(row)
    summary = []
    for (system, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        tails = np.array([float(row["tail_infidelity_mean"]) for row in items])
        summary.append(
            {
                "system": system,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "tail_infidelity_mean": f"{np.mean(tails):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("horizon_lyapunov_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("horizon_lyapunov_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Horizon Lyapunov Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = []
    for task in ("Z", "H"):
        pulse = design_pulse(task, beam_width=default_beam_width(task))
        for strength in (0.0, 0.02, 0.05, 0.08):
            rows.extend(evaluate_pulse(task, pulse, disorder_strength=strength))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/horizon_lyapunov_results.csv")
    print("wrote aggregate table to results/horizon_lyapunov_summary.md")


if __name__ == "__main__":
    main()
