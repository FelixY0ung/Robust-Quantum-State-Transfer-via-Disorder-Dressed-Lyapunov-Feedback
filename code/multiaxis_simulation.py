"""Multi-axis Lyapunov smoke test for the two-level tasks.

This script extends the single-axis experiment by allowing independent
controls along sigma_x and sigma_y. It is a compact feasibility check for the
paper's next theoretical step: vector-valued Lyapunov feedback.
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


def ket(index: int, dim: int) -> np.ndarray:
    v = np.zeros((dim, 1), dtype=complex)
    v[index, 0] = 1.0
    return v


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + dagger(rho))
    return rho / np.trace(rho)


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def clip_vector(u: np.ndarray, umax: float) -> np.ndarray:
    norm = float(np.linalg.norm(u))
    if norm <= umax or norm == 0.0:
        return u
    return (umax / norm) * u


@dataclass(frozen=True)
class System:
    name: str
    h0: np.ndarray
    controls: tuple[np.ndarray, ...]
    initial: np.ndarray
    target: np.ndarray
    t_final: float


def qubit_system(task: str) -> System:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    z0 = ket(0, 2)
    z1 = ket(1, 2)
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
    return System(
        name=f"qubit_{task}",
        h0=sz,
        controls=(sx, sy),
        initial=projector(initial),
        target=projector(target),
        t_final=25.0,
    )


def random_disorder(rng: np.random.Generator) -> np.ndarray:
    sx, sy = qubit_system("X").controls
    sz = qubit_system("X").h0
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def control_vector(
    rho: np.ndarray,
    system: System,
    gain: float,
    umax: float,
    eps: float,
    kick: float,
) -> np.ndarray:
    drift = -1.0j * comm(system.h0, rho)
    a = -float(np.real(np.trace(system.target @ drift)))
    b = np.array(
        [
            -float(np.real(np.trace(system.target @ (-1.0j * comm(hc, rho)))))
            for hc in system.controls
        ]
    )
    if np.linalg.norm(b) < 1e-8 and 1.0 - fidelity(rho, system.target) > 1e-4:
        pulse = np.zeros_like(b)
        pulse[-1] = kick
        return pulse
    raw = -(a * b) / (float(np.dot(b, b)) + eps) - gain * b
    return clip_vector(raw, umax)


def derivative(
    rho: np.ndarray,
    system: System,
    disorder_strength: float,
    disorder: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    h = system.h0 + disorder_strength * disorder
    for coeff, hc in zip(u, system.controls):
        h = h + coeff * hc
    return -1.0j * comm(h, rho)


def rk4(
    rho: np.ndarray,
    dt: float,
    system: System,
    disorder_strength: float,
    disorder: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    k1 = derivative(rho, system, disorder_strength, disorder, u)
    k2 = derivative(rho + 0.5 * dt * k1, system, disorder_strength, disorder, u)
    k3 = derivative(rho + 0.5 * dt * k2, system, disorder_strength, disorder, u)
    k4 = derivative(rho + dt * k3, system, disorder_strength, disorder, u)
    return hermitize_trace_one(rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def run(task: str, disorder_strength: float, seed: int, points: int = 301) -> dict[str, float | int | str]:
    system = qubit_system(task)
    rng = np.random.default_rng(seed)
    disorder = random_disorder(rng)
    t_eval = np.linspace(0.0, system.t_final, points)
    rho = system.initial.copy()
    fids = []
    energies = []
    for j, _t in enumerate(t_eval):
        rho = hermitize_trace_one(rho)
        fids.append(fidelity(rho, system.target))
        u = control_vector(rho, system, gain=3.0, umax=3.0, eps=1e-2, kick=0.2)
        energies.append(float(np.dot(u, u)))
        if j + 1 < len(t_eval):
            rho = rk4(rho, float(t_eval[j + 1] - t_eval[j]), system, disorder_strength, disorder, u)
    inf = np.maximum(0.0, 1.0 - np.array(fids))
    tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
    threshold = 0.1 * max(float(inf[0]), 1e-12)
    reached = np.where(inf <= threshold)[0]
    return {
        "system": system.name,
        "disorder_strength": disorder_strength,
        "seed": seed,
        "tail_infidelity_mean": float(np.mean(tail)),
        "tail_stability_range": float(np.max(tail) - np.min(tail)),
        "final_fidelity": float(fids[-1]),
        "response_time": float(t_eval[reached[0]]) if len(reached) else float("nan"),
        "control_energy": float(np.trapezoid(np.array(energies), t_eval)),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["system"]), float(row["disorder_strength"])), []).append(row)
    out = []
    for (system, disorder), items in sorted(groups.items()):
        out.append(
            {
                "system": system,
                "disorder": f"{disorder:.4g}",
                "n": str(len(items)),
                "tail_infidelity_mean": f"{np.mean([float(x['tail_infidelity_mean']) for x in items]):.6g}",
                "tail_stability_mean": f"{np.mean([float(x['tail_stability_range']) for x in items]):.6g}",
                "final_fidelity_mean": f"{np.mean([float(x['final_fidelity']) for x in items]):.6g}",
                "response_time_mean": f"{np.nanmean([float(x['response_time']) for x in items]):.6g}",
                "control_energy_mean": f"{np.mean([float(x['control_energy']) for x in items]):.6g}",
            }
        )
    return out


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("multiaxis_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("multiaxis_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Multi-Axis Simulation Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = []
    for task in ("X", "Z", "H"):
        for disorder in (0.0, 0.05):
            for seed in range(3):
                rows.append(run(task, disorder, seed))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/multiaxis_results.csv")
    print("wrote aggregate table to results/multiaxis_summary.md")


if __name__ == "__main__":
    main()
