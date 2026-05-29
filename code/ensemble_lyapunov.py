"""Ensemble Lyapunov tracking controller for two-level robust transfer.

This script bridges the feedback Lyapunov experiments and the robust open-loop
baseline. A finite training ensemble of disorder realizations is propagated in
the interaction frame. At each digital control step, a single shared control
vector is computed from the ensemble-averaged Lyapunov derivative,

    dV_bar = a_bar + u dot b_bar,

with

    u = -a_bar b_bar / (||b_bar||^2 + eps) - K b_bar.

The resulting pulse is then evaluated on held-out disorder realizations. This
is still an open-loop pulse after design, but its construction is explicitly
Lyapunov-based and disorder-aware.
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


def clip_vector(u: np.ndarray, umax: float) -> np.ndarray:
    norm = float(np.linalg.norm(u))
    if norm == 0.0 or norm <= umax:
        return u
    return (umax / norm) * u


@dataclass(frozen=True)
class Problem:
    task: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    t_final: float


def problem(task: str, t_final: float = 25.0) -> Problem:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    z0 = ket(0)
    z1 = ket(1)
    plus = normalize(z0 + z1)
    minus = normalize(z0 - z1)
    if task == "X":
        initial, target = z0, z1
    elif task == "Z":
        initial, target = plus, minus
    elif task == "H":
        initial, target = z0, plus
    else:
        raise ValueError(task)
    return Problem(
        task=f"ensemble_lyapunov_{task}",
        h0=sz,
        controls=(sx, sy),
        initial=projector(initial),
        target=projector(target),
        t_final=t_final,
    )


def random_disorder(seed: int) -> np.ndarray:
    p = problem("X")
    sx, sy = p.controls
    sz = p.h0
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def interaction_frame_operator(p: Problem, operator: np.ndarray, t: float) -> np.ndarray:
    u0 = unitary2(p.h0, t)
    return dagger(u0) @ operator @ u0


def lyapunov_terms(
    rho: np.ndarray,
    target: np.ndarray,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    disorder_strength: float,
) -> tuple[float, np.ndarray]:
    disorder_drift = -1.0j * comm(disorder_strength * disorder_i, rho)
    a = -float(np.real(np.trace(target @ disorder_drift)))
    b = np.array(
        [
            -float(np.real(np.trace(target @ (-1.0j * comm(hc_i, rho)))))
            for hc_i in controls_i
        ]
    )
    return a, b


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


def design_pulse(
    task: str,
    disorder_strength: float = 0.05,
    train_seeds: tuple[int, ...] = (0, 1, 2, 3),
    points: int = 301,
    gain: float = 3.0,
    eps: float = 1e-2,
    umax: float = 3.0,
) -> np.ndarray:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, points)
    disorders = tuple(random_disorder(seed) for seed in train_seeds)
    rhos = [p.initial.copy() for _ in disorders]
    pulse = []

    for j, t in enumerate(t_eval[:-1]):
        controls_i = tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        terms = []
        for rho, disorder in zip(rhos, disorders):
            rho = hermitize_trace_one(rho)
            disorder_i = interaction_frame_operator(p, disorder, float(t))
            terms.append(lyapunov_terms(rho, p.target, controls_i, disorder_i, disorder_strength))
        a_bar = float(np.mean([item[0] for item in terms]))
        b_bar = np.mean([item[1] for item in terms], axis=0)
        if np.linalg.norm(b_bar) < 1e-8:
            u = np.zeros_like(b_bar)
            u[-1] = 0.2
        else:
            u = -(a_bar * b_bar) / (float(np.dot(b_bar, b_bar)) + eps) - gain * b_bar
            u = clip_vector(u, umax)
        pulse.append(u)

        dt = float(t_eval[j + 1] - t_eval[j])
        next_rhos = []
        for rho, disorder in zip(rhos, disorders):
            disorder_i = interaction_frame_operator(p, disorder, float(t))
            next_rhos.append(rk4_step(rho, dt, controls_i, disorder_i, disorder_strength, u))
        rhos = next_rhos

    return np.asarray(pulse)


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
    groups: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(str(row["system"]), []).append(row)
    summary = []
    for system, items in sorted(groups.items()):
        fids = np.array([float(x["final_fidelity"]) for x in items])
        tails = np.array([float(x["tail_infidelity_mean"]) for x in items])
        summary.append(
            {
                "system": system,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "tail_infidelity_mean": f"{np.mean(tails):.6g}",
                "pulse_energy_mean": f"{np.mean([float(x['pulse_energy']) for x in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("ensemble_lyapunov_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("ensemble_lyapunov_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Ensemble Lyapunov Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = []
    for task in ("Z", "H"):
        pulse = design_pulse(task)
        rows.extend(evaluate_pulse(task, pulse))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/ensemble_lyapunov_results.csv")
    print("wrote aggregate table to results/ensemble_lyapunov_summary.md")


if __name__ == "__main__":
    main()
