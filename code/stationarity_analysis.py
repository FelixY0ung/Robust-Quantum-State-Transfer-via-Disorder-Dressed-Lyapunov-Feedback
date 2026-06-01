"""Target-stationarity diagnostics for the two-level transfer tasks.

Fixed-target Lyapunov feedback in the Schrodinger picture implicitly asks the
target projector to be an equilibrium of the uncontrolled generator.  This
script quantifies the Hamiltonian drift residual for the X, Z, and Hadamard
state-transfer tasks used in the paper, and also reports representative
disorder-dressed generator residuals for fixed Pauli dressing channels.
"""

from __future__ import annotations

import csv
from pathlib import Path

from paths import result_path

import numpy as np


def dagger(a: np.ndarray) -> np.ndarray:
    return np.conjugate(a.T)


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def dissipator(a: np.ndarray, rho: np.ndarray) -> np.ndarray:
    a2 = a @ a
    return a @ rho @ a - 0.5 * (a2 @ rho + rho @ a2)


def ket(index: int) -> np.ndarray:
    v = np.zeros((2, 1), dtype=complex)
    v[index, 0] = 1.0
    return v


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def unitary(h: np.ndarray, t: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(h)
    return vecs @ np.diag(np.exp(-1.0j * vals * t)) @ dagger(vecs)


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def task_states(task: str) -> tuple[np.ndarray, np.ndarray]:
    z0 = ket(0)
    z1 = ket(1)
    plus = normalize(z0 + z1)
    minus = normalize(z0 - z1)
    if task == "X":
        return z0, z1
    if task == "Z":
        return plus, minus
    if task == "H":
        return z0, plus
    raise ValueError(task)


def analyze_task(task: str, drift_time: float = 0.1, alpha_design: float = 0.01) -> dict[str, str]:
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    initial, target = task_states(task)
    rho_initial = projector(initial)
    rho_target = projector(target)
    drift_generator = -1.0j * comm(sz, rho_target)
    dressed_x_generator = drift_generator + alpha_design * dissipator(sx, rho_target)
    dressed_xy_generator = drift_generator + alpha_design * (
        dissipator(sx, rho_target) + dissipator(sy, rho_target)
    )
    stationarity_residual = float(np.linalg.norm(comm(sz, rho_target), ord="fro"))
    u0 = unitary(sz, drift_time)
    drifted_target = u0 @ rho_target @ dagger(u0)
    target_self_infidelity = 1.0 - fidelity(drifted_target, rho_target)
    drift_contribution = -float(
        np.real(np.trace(rho_target @ (-1.0j * comm(sz, rho_initial))))
    )
    return {
        "task": task,
        "target_stationarity_residual": f"{stationarity_residual:.6g}",
        "target_drift_speed": f"{np.linalg.norm(drift_generator, ord='fro'):.6g}",
        "dressed_x_residual_alpha0p01": f"{np.linalg.norm(dressed_x_generator, ord='fro'):.6g}",
        "dressed_xy_residual_alpha0p01": f"{np.linalg.norm(dressed_xy_generator, ord='fro'):.6g}",
        "target_self_infidelity_t0p1": f"{target_self_infidelity:.6g}",
        "initial_drift_lyapunov_term": f"{drift_contribution:.6g}",
    }


def write_outputs(rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    with result_path("stationarity_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with result_path("stationarity_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Target Stationarity Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = [analyze_task(task) for task in ("X", "Z", "H")]
    write_outputs(rows)
    print("wrote results/stationarity_results.csv")
    print("wrote results/stationarity_summary.md")


if __name__ == "__main__":
    main()
