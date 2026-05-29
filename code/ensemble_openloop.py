"""Ensemble-trained open-loop baseline for two-level robust transfer.

The Lyapunov feedback scripts diagnose why fixed-target and interaction-frame
tracking are insufficient for Z/H tasks under disorder. This script adds a
small, reproducible robust-control baseline: piecewise-constant controls are
optimized over a finite ensemble of static disorder samples.

It is intentionally compact and uses SciPy's derivative-free optimizer so it
does not require an autodiff stack.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from paths import result_path

import numpy as np
from scipy.optimize import minimize


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


def unitary2(h: np.ndarray, dt: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1.0j * vals * dt)) @ dagger(vecs)


@dataclass(frozen=True)
class Problem:
    task: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    t_final: float


def problem(task: str) -> Problem:
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
        task=f"ensemble_{task}",
        h0=sz,
        controls=(sx, sy),
        initial=projector(initial),
        target=projector(target),
        t_final=8.0,
    )


def random_disorder(seed: int) -> np.ndarray:
    p = problem("X")
    sx, sy = p.controls
    sz = p.h0
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def evolve(
    p: Problem,
    controls_flat: np.ndarray,
    segments: int,
    disorder: np.ndarray,
    disorder_strength: float,
    umax: float,
) -> np.ndarray:
    controls = np.reshape(controls_flat, (segments, len(p.controls)))
    controls = np.clip(controls, -umax, umax)
    dt = p.t_final / segments
    rho = p.initial.copy()
    for coeffs in controls:
        h = p.h0 + disorder_strength * disorder
        for coeff, hc in zip(coeffs, p.controls):
            h = h + coeff * hc
        u = unitary2(h, dt)
        rho = u @ rho @ dagger(u)
    return hermitize_trace_one(rho)


def objective(
    x: np.ndarray,
    p: Problem,
    segments: int,
    training_disorders: tuple[np.ndarray, ...],
    disorder_strength: float,
    umax: float,
    energy_weight: float,
) -> float:
    infids = []
    for disorder in training_disorders:
        rho = evolve(p, x, segments, disorder, disorder_strength, umax)
        infids.append(1.0 - fidelity(rho, p.target))
    energy = float(np.mean(np.square(np.clip(x, -umax, umax))))
    return float(np.mean(infids) + energy_weight * energy)


def optimize_task(
    task: str,
    disorder_strength: float = 0.05,
    segments: int = 10,
    umax: float = 4.0,
    maxiter: int = 250,
) -> tuple[np.ndarray, float]:
    p = problem(task)
    training = tuple(random_disorder(seed) for seed in range(4))
    rng = np.random.default_rng(100 + len(task))
    x0 = rng.normal(scale=0.2, size=segments * len(p.controls))
    result = minimize(
        objective,
        x0,
        args=(p, segments, training, disorder_strength, umax, 1e-4),
        method="Powell",
        bounds=[(-umax, umax)] * len(x0),
        options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
    )
    return np.asarray(result.x), float(result.fun)


def evaluate(
    task: str,
    pulse: np.ndarray,
    disorder_strength: float = 0.05,
    segments: int = 10,
    umax: float = 4.0,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    rows = []
    for seed in test_seeds:
        rho = evolve(p, pulse, segments, random_disorder(seed), disorder_strength, umax)
        rows.append(
            {
                "system": p.task,
                "disorder_strength": disorder_strength,
                "seed": seed,
                "final_fidelity": fidelity(rho, p.target),
                "final_infidelity": 1.0 - fidelity(rho, p.target),
                "pulse_energy": float(np.mean(np.square(np.clip(pulse, -umax, umax)))),
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
        infids = np.array([float(x["final_infidelity"]) for x in items])
        summary.append(
            {
                "system": system,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_infidelity_mean": f"{np.mean(infids):.6g}",
                "pulse_energy_mean": f"{np.mean([float(x['pulse_energy']) for x in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("ensemble_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("ensemble_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Ensemble Open-Loop Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = []
    for task in ("Z", "H"):
        pulse, value = optimize_task(task)
        task_rows = evaluate(task, pulse)
        for row in task_rows:
            row["training_objective"] = value
        rows.extend(task_rows)
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/ensemble_results.csv")
    print("wrote aggregate table to results/ensemble_summary.md")


if __name__ == "__main__":
    main()
