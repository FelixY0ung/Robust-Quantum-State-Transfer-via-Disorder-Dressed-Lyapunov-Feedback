"""Interaction-frame target-tracking simulation for two-level transfers.

The fixed-target Lyapunov tests show that Z/H transfers fail when the target
projector is not stationary under H0. This script tests the standard remedy:
design the Lyapunov controller in the interaction frame, where the control
Hamiltonians rotate and the target projector is fixed.
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


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + dagger(rho))
    return rho / np.trace(rho)


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def unitary(h: np.ndarray, t: float) -> np.ndarray:
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
    rho_initial: np.ndarray
    rho_target_interaction: np.ndarray
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
        task=f"tracking_{task}",
        h0=sz,
        controls=(sx, sy),
        rho_initial=projector(initial),
        rho_target_interaction=projector(target),
        t_final=25.0,
    )


def random_disorder(rng: np.random.Generator) -> np.ndarray:
    p = problem("X")
    sx, sy = p.controls
    sz = p.h0
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def interaction_controls(p: Problem, t: float) -> tuple[np.ndarray, ...]:
    u0 = unitary(p.h0, t)
    return tuple(dagger(u0) @ hc @ u0 for hc in p.controls)


def control_vector(
    rho_i: np.ndarray,
    target_i: np.ndarray,
    controls_i: tuple[np.ndarray, ...],
    gain: float,
    umax: float,
    kick: float,
) -> np.ndarray:
    b = np.array(
        [
            -float(np.real(np.trace(target_i @ (-1.0j * comm(hc, rho_i)))))
            for hc in controls_i
        ]
    )
    if np.linalg.norm(b) < 1e-8 and 1.0 - fidelity(rho_i, target_i) > 1e-4:
        pulse = np.zeros_like(b)
        pulse[-1] = kick
        return pulse
    return clip_vector(-gain * b, umax)


def derivative_i(
    rho_i: np.ndarray,
    controls_i: tuple[np.ndarray, ...],
    u: np.ndarray,
    disorder_i: np.ndarray,
    disorder_strength: float,
) -> np.ndarray:
    h = disorder_strength * disorder_i
    for coeff, hc in zip(u, controls_i):
        h = h + coeff * hc
    return -1.0j * comm(h, rho_i)


def compensation_vector(
    rho_i: np.ndarray,
    target_i: np.ndarray,
    controls_i: tuple[np.ndarray, ...],
    disorder_i: np.ndarray,
    disorder_strength: float,
    eps: float,
) -> np.ndarray:
    a = -float(
        np.real(np.trace(target_i @ (-1.0j * comm(disorder_strength * disorder_i, rho_i))))
    )
    b = np.array(
        [
            -float(np.real(np.trace(target_i @ (-1.0j * comm(hc, rho_i)))))
            for hc in controls_i
        ]
    )
    return -(a * b) / (float(np.dot(b, b)) + eps)


def rk4_i(
    rho_i: np.ndarray,
    t: float,
    dt: float,
    p: Problem,
    u: np.ndarray,
    disorder: np.ndarray,
    disorder_strength: float,
) -> np.ndarray:
    controls_i = interaction_controls(p, t)
    u0 = unitary(p.h0, t)
    disorder_i = dagger(u0) @ disorder @ u0
    k1 = derivative_i(rho_i, controls_i, u, disorder_i, disorder_strength)
    k2 = derivative_i(rho_i + 0.5 * dt * k1, controls_i, u, disorder_i, disorder_strength)
    k3 = derivative_i(rho_i + 0.5 * dt * k2, controls_i, u, disorder_i, disorder_strength)
    k4 = derivative_i(rho_i + dt * k3, controls_i, u, disorder_i, disorder_strength)
    return hermitize_trace_one(rho_i + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def run(
    task: str,
    disorder_strength: float,
    seed: int,
    points: int = 301,
    oracle_compensation: bool = True,
) -> dict[str, float | int | str]:
    p = problem(task)
    rng = np.random.default_rng(seed)
    disorder = random_disorder(rng)
    t_eval = np.linspace(0.0, p.t_final, points)
    rho_i = p.rho_initial.copy()
    fids = []
    energies = []
    for j, t in enumerate(t_eval):
        rho_i = hermitize_trace_one(rho_i)
        fids.append(fidelity(rho_i, p.rho_target_interaction))
        controls_i = interaction_controls(p, float(t))
        u0 = unitary(p.h0, float(t))
        disorder_i = dagger(u0) @ disorder @ u0
        u = control_vector(
            rho_i,
            p.rho_target_interaction,
            controls_i,
            gain=3.0,
            umax=3.0,
            kick=0.2,
        )
        if oracle_compensation and disorder_strength > 0.0:
            u = u + compensation_vector(
                rho_i,
                p.rho_target_interaction,
                controls_i,
                disorder_i,
                disorder_strength,
                eps=1e-2,
            )
            u = clip_vector(u, umax=3.0)
        energies.append(float(np.dot(u, u)))
        if j + 1 < len(t_eval):
            rho_i = rk4_i(
                rho_i,
                float(t),
                float(t_eval[j + 1] - t_eval[j]),
                p,
                u,
                disorder,
                disorder_strength,
            )
    inf = np.maximum(0.0, 1.0 - np.array(fids))
    tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
    threshold = 0.1 * max(float(inf[0]), 1e-12)
    reached = np.where(inf <= threshold)[0]
    return {
        "system": p.task,
        "oracle_compensation": int(oracle_compensation),
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
    summary = []
    for (system, disorder), items in sorted(groups.items()):
        resp = np.array([float(x["response_time"]) for x in items])
        resp = resp[np.isfinite(resp)]
        summary.append(
            {
                "system": system,
                "disorder": f"{disorder:.4g}",
                "n": str(len(items)),
                "tail_infidelity_mean": f"{np.mean([float(x['tail_infidelity_mean']) for x in items]):.6g}",
                "tail_stability_mean": f"{np.mean([float(x['tail_stability_range']) for x in items]):.6g}",
                "final_fidelity_mean": f"{np.mean([float(x['final_fidelity']) for x in items]):.6g}",
                "response_time_mean": f"{np.mean(resp):.6g}" if len(resp) else "nan",
                "control_energy_mean": f"{np.mean([float(x['control_energy']) for x in items]):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("tracking_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("tracking_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Interaction-Frame Tracking Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = []
    for task in ("X", "Z", "H"):
        for disorder in (0.0, 0.05):
            for seed in range(3):
                rows.append(run(task, disorder, seed, oracle_compensation=True))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to results/tracking_results.csv")
    print("wrote aggregate table to results/tracking_summary.md")


if __name__ == "__main__":
    main()
