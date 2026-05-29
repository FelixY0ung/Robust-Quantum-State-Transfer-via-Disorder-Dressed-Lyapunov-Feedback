"""Three-level beam-horizon Lyapunov transfer benchmark.

This script checks whether the horizon Lyapunov idea survives beyond the
single-qubit examples.  The model is a three-level chain with no direct
0-to-2 control coupling, so the controller must route population through the
intermediate level.  Static disorder is a normalized Hermitian perturbation and
the pulse is trained on an ensemble of disorder samples.
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


def link_operator(dim: int, i: int, j: int, kind: str) -> np.ndarray:
    op = np.zeros((dim, dim), dtype=complex)
    if kind == "x":
        op[i, j] = 1.0
        op[j, i] = 1.0
    elif kind == "y":
        op[i, j] = -1.0j
        op[j, i] = 1.0j
    else:
        raise ValueError(kind)
    return op


@dataclass(frozen=True)
class Problem:
    name: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    t_final: float


def problem() -> Problem:
    dim = 3
    return Problem(
        name="three_level_chain",
        h0=np.diag([0.0, 1.0, 1.7]).astype(complex),
        controls=(
            link_operator(dim, 0, 1, "x"),
            link_operator(dim, 0, 1, "y"),
            link_operator(dim, 1, 2, "x"),
            link_operator(dim, 1, 2, "y"),
        ),
        initial=projector(ket(0, dim)),
        target=projector(ket(2, dim)),
        t_final=12.0,
    )


def random_disorder(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(3, 3)) + 1.0j * rng.normal(size=(3, 3))
    h = 0.5 * (raw + dagger(raw))
    h = h - (np.trace(h) / 3.0) * np.eye(3)
    norm = np.linalg.norm(h, ord="fro")
    return h / norm if norm > 0 else h


def interaction_frame_operator(p: Problem, operator: np.ndarray, t: float) -> np.ndarray:
    u0 = unitary(p.h0, t)
    return dagger(u0) @ operator @ u0


def step_scenarios(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    scenarios: tuple[tuple[float, np.ndarray], ...],
    t: float,
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    controls_i = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
    out = []
    for rho, (strength, disorder) in zip(rhos, scenarios):
        h = strength * interaction_frame_operator(p, disorder, t)
        for coeff, hc_i in zip(control, controls_i):
            h = h + coeff * hc_i
        u = unitary(h, dt)
        out.append(hermitize_trace_one(u @ rho @ dagger(u)))
    return tuple(out)


def candidate_controls(amplitudes: tuple[float, ...]) -> tuple[np.ndarray, ...]:
    controls = [np.zeros(4)]
    for amp in amplitudes:
        for index in range(4):
            u = np.zeros(4)
            u[index] = amp
            controls.append(u.copy())
            u[index] = -amp
            controls.append(u.copy())
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                ux = np.zeros(4)
                ux[0] = s1 * amp / np.sqrt(2.0)
                ux[2] = s2 * amp / np.sqrt(2.0)
                controls.append(ux)
                uy = np.zeros(4)
                uy[1] = s1 * amp / np.sqrt(2.0)
                uy[3] = s2 * amp / np.sqrt(2.0)
                controls.append(uy)
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


def select_control_beam(
    p: Problem,
    rhos: tuple[np.ndarray, ...],
    scenarios: tuple[tuple[float, np.ndarray], ...],
    t: float,
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
        tau = t + depth * dt
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_scenarios(p, states, scenarios, tau, dt, candidate)
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
    train_strengths: tuple[float, ...] = (0.03, 0.05),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3),
    segments: int = 80,
    horizon_steps: int = 8,
    beam_width: int = 5,
    amplitudes: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    worst_weight: float = 0.25,
    energy_weight: float = 0.0,
) -> np.ndarray:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in train_strengths
        for seed in train_seeds
    )
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(amplitudes)
    pulse = []
    for t in t_eval[:-1]:
        u = select_control_beam(
            p,
            rhos,
            scenarios,
            float(t),
            dt,
            candidates,
            horizon_steps,
            beam_width,
            worst_weight,
            energy_weight,
        )
        pulse.append(u)
        rhos = step_scenarios(p, rhos, scenarios, float(t), dt, u)
    return np.asarray(pulse)


def evaluate_pulse(
    pulse: np.ndarray,
    disorder_strength: float,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    rows = []
    for seed in test_seeds:
        disorder = random_disorder(seed)
        rho = p.initial.copy()
        fids = []
        for j, t in enumerate(t_eval):
            rho = hermitize_trace_one(rho)
            fids.append(fidelity(rho, p.target))
            if j < len(pulse):
                rho = step_scenarios(
                    p,
                    (rho,),
                    ((disorder_strength, disorder),),
                    float(t),
                    dt,
                    pulse[j],
                )[0]
        inf = np.maximum(0.0, 1.0 - np.array(fids))
        tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
        rows.append(
            {
                "system": p.name,
                "eval_strength": disorder_strength,
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
        groups.setdefault((str(row["system"]), float(row["eval_strength"])), []).append(row)
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
    with result_path("multilevel_horizon_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("multilevel_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Multilevel Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    pulse = design_pulse()
    rows = []
    for strength in (0.0, 0.03, 0.05, 0.08):
        rows.extend(evaluate_pulse(pulse, strength))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/multilevel_horizon_results.csv")
    print("wrote aggregate table to results/multilevel_horizon_summary.md")


if __name__ == "__main__":
    main()
