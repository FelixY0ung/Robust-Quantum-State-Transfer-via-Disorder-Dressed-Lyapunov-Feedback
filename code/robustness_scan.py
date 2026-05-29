"""Held-out robustness scan for nominal and ensemble-trained pulses.

The CAC paper uses this script to separate reachability from feedback-law
limitations. For each Z/H transfer, a nominal open-loop pulse is trained at
zero disorder and an ensemble pulse is trained at finite disorder. Both are
evaluated on 50 held-out disorder realizations over a strength grid.
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
    if task == "Z":
        initial, target = plus, minus
    elif task == "H":
        initial, target = z0, plus
    else:
        raise ValueError(task)
    return Problem(
        task=task,
        h0=sz,
        controls=(sx, sy),
        initial=projector(initial),
        target=projector(target),
        t_final=8.0,
    )


def random_disorder(seed: int) -> np.ndarray:
    p = problem("Z")
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
    training_strength: float,
    umax: float,
    energy_weight: float,
) -> float:
    infids = []
    for disorder in training_disorders:
        rho = evolve(p, x, segments, disorder, training_strength, umax)
        infids.append(1.0 - fidelity(rho, p.target))
    energy = float(np.mean(np.square(np.clip(x, -umax, umax))))
    return float(np.mean(infids) + energy_weight * energy)


def optimize_pulse(
    task: str,
    training_strength: float,
    training_seeds: tuple[int, ...],
    segments: int,
    umax: float,
    maxiter: int,
    restart_seeds: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    p = problem(task)
    training_disorders = tuple(random_disorder(seed) for seed in training_seeds)
    best_x: np.ndarray | None = None
    best_fun = float("inf")
    for seed in restart_seeds:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(scale=0.2, size=segments * len(p.controls))
        result = minimize(
            objective,
            x0,
            args=(p, segments, training_disorders, training_strength, umax, 1e-4),
            method="Powell",
            bounds=[(-umax, umax)] * len(x0),
            options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5, "disp": False},
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_x = np.asarray(result.x)
    if best_x is None:
        raise RuntimeError("no optimizer restart completed")
    return best_x, best_fun


def evaluate_pulse(
    task: str,
    pulse: np.ndarray,
    pulse_type: str,
    training_strength: float,
    training_objective: float,
    strength_grid: tuple[float, ...],
    test_seeds: range,
    segments: int,
    umax: float,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    energy = float(np.mean(np.square(np.clip(pulse, -umax, umax))))
    rows = []
    for strength in strength_grid:
        for seed in test_seeds:
            rho = evolve(p, pulse, segments, random_disorder(seed), strength, umax)
            fid = fidelity(rho, p.target)
            rows.append(
                {
                    "task": task,
                    "pulse_type": pulse_type,
                    "training_strength": training_strength,
                    "eval_strength": strength,
                    "seed": seed,
                    "final_fidelity": fid,
                    "final_infidelity": 1.0 - fid,
                    "pulse_energy": energy,
                    "training_objective": training_objective,
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["pulse_type"]), float(row["eval_strength"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (task, pulse_type, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        infids = np.array([float(row["final_infidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "pulse_type": pulse_type,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_infidelity_mean": f"{np.mean(infids):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("robustness_scan_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("robustness_scan_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Robustness Scan Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    segments = 10
    umax = 4.0
    strength_grid = (0.0, 0.02, 0.05, 0.08)
    restart_seeds = (101, 102, 701, 777, 900)
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        nominal, nominal_value = optimize_pulse(
            task,
            training_strength=0.0,
            training_seeds=(0,),
            segments=segments,
            umax=umax,
            maxiter=180,
            restart_seeds=restart_seeds,
        )
        robust, robust_value = optimize_pulse(
            task,
            training_strength=0.05,
            training_seeds=(0, 1, 2, 3),
            segments=segments,
            umax=umax,
            maxiter=250,
            restart_seeds=restart_seeds,
        )
        rows.extend(
            evaluate_pulse(
                task,
                nominal,
                "nominal_open_loop",
                0.0,
                nominal_value,
                strength_grid,
                range(10, 60),
                segments,
                umax,
            )
        )
        rows.extend(
            evaluate_pulse(
                task,
                robust,
                "ensemble_open_loop",
                0.05,
                robust_value,
                strength_grid,
                range(10, 60),
                segments,
                umax,
            )
        )
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/robustness_scan_results.csv")
    print("wrote aggregate table to results/robustness_scan_summary.md")


if __name__ == "__main__":
    main()
