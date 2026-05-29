"""Reproducible simulations for disorder-dressed Lyapunov control.

The script implements a density-matrix controller and writes:

  - results/simulation_results.csv
  - results/simulation_summary.md

The default run is deliberately modest. Increase N_SEEDS and N_POINTS for
publication-grade tables.
"""

from __future__ import annotations

import argparse
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


def dissipator(a: np.ndarray, rho: np.ndarray) -> np.ndarray:
    a2 = a @ a
    return a @ rho @ a - 0.5 * (a2 @ rho + rho @ a2)


def ket(index: int, dim: int) -> np.ndarray:
    v = np.zeros((dim, 1), dtype=complex)
    v[index, 0] = 1.0
    return v


def projector(v: np.ndarray) -> np.ndarray:
    return v @ dagger(v)


def normalize_ket(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + dagger(rho))
    tr = np.trace(rho)
    if abs(tr) > 1e-14:
        rho = rho / tr
    return rho


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


def saturation(x: float, umax: float) -> float:
    return float(min(umax, max(-umax, x)))


@dataclass(frozen=True)
class SystemSpec:
    name: str
    h0: np.ndarray
    hc: np.ndarray
    initial: np.ndarray
    target: np.ndarray
    t_final: float


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    alpha_design: float
    gain: float
    eps: float
    umax: float
    kick: float
    kick_threshold: float


def qubit_system(task: str) -> SystemSpec:
    h0 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    hc = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z0 = ket(0, 2)
    z1 = ket(1, 2)
    plus = normalize_ket(z0 + z1)
    minus = normalize_ket(z0 - z1)
    if task == "X":
        initial, target = z0, z1
    elif task == "Z":
        initial, target = plus, minus
    elif task == "H":
        initial, target = z0, plus
    else:
        raise ValueError(f"unknown qubit task {task}")
    return SystemSpec(
        name=f"qubit_{task}",
        h0=h0,
        hc=hc,
        initial=projector(initial),
        target=projector(target),
        t_final=35.0,
    )


def five_level_system() -> SystemSpec:
    h0 = np.diag([1.0, 1.2, 1.3, 2.0, 2.15]).astype(complex)
    hc = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=complex,
    )
    return SystemSpec(
        name="five_level_X",
        h0=h0,
        hc=hc,
        initial=projector(ket(0, 5)),
        target=projector(ket(4, 5)),
        t_final=100.0,
    )


def random_qubit_disorder(rng: np.random.Generator) -> np.ndarray:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    coeff = rng.normal(size=3)
    coeff = coeff / np.linalg.norm(coeff)
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def random_hermitian_disorder(dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim))
    h = 0.5 * (raw + dagger(raw))
    norm = np.linalg.norm(h, ord="fro")
    return h / norm if norm > 0 else h


def lyapunov_scalars(
    rho: np.ndarray,
    system: SystemSpec,
    alpha_design: float,
) -> tuple[float, float]:
    drift = -1.0j * comm(system.h0, rho) + alpha_design * dissipator(system.hc, rho)
    control_direction = -1.0j * comm(system.hc, rho)
    a = -float(np.real(np.trace(system.target @ drift)))
    b = -float(np.real(np.trace(system.target @ control_direction)))
    return a, b


def control_value(
    rho: np.ndarray,
    system: SystemSpec,
    controller: ControllerSpec,
) -> float:
    a, b = lyapunov_scalars(rho, system, controller.alpha_design)
    infidelity = 1.0 - fidelity(rho, system.target)

    if abs(b) < controller.kick_threshold and infidelity > 1e-4:
        return controller.kick

    if controller.name == "nominal":
        return saturation(-controller.gain * b, controller.umax)

    if controller.name == "disorder_dressed":
        raw = -(a * b) / (b * b + controller.eps) - controller.gain * b
        return saturation(raw, controller.umax)

    raise ValueError(f"unknown controller {controller.name}")


def closed_loop_derivative(
    system: SystemSpec,
    controller: ControllerSpec,
    disorder_strength: float,
    disorder: np.ndarray,
    rho: np.ndarray,
    u: float,
    actual_alpha: float = 0.0,
) -> np.ndarray:
    h = system.h0 + u * system.hc + disorder_strength * disorder
    return -1.0j * comm(h, rho) + actual_alpha * dissipator(system.hc, rho)


def rk4_step(
    rho: np.ndarray,
    dt: float,
    system: SystemSpec,
    controller: ControllerSpec,
    disorder_strength: float,
    disorder: np.ndarray,
    u: float,
) -> np.ndarray:
    # Hold the feedback value fixed over one digital control interval.
    k1 = closed_loop_derivative(system, controller, disorder_strength, disorder, rho, u)
    k2 = closed_loop_derivative(
        system, controller, disorder_strength, disorder, rho + 0.5 * dt * k1, u
    )
    k3 = closed_loop_derivative(
        system, controller, disorder_strength, disorder, rho + 0.5 * dt * k2, u
    )
    k4 = closed_loop_derivative(
        system, controller, disorder_strength, disorder, rho + dt * k3, u
    )
    return hermitize_trace_one(rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def run_single(
    system: SystemSpec,
    controller: ControllerSpec,
    disorder_strength: float,
    seed: int,
    n_points: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    dim = system.h0.shape[0]
    disorder = random_qubit_disorder(rng) if dim == 2 else random_hermitian_disorder(dim, rng)
    t_eval = np.linspace(0.0, system.t_final, n_points)
    rho = system.initial.copy()
    fids = []
    controls = []
    for j, _time in enumerate(t_eval):
        rho = hermitize_trace_one(rho)
        fids.append(fidelity(rho, system.target))
        u = control_value(rho, system, controller)
        controls.append(u)
        if j + 1 < len(t_eval):
            rho = rk4_step(
                rho,
                float(t_eval[j + 1] - t_eval[j]),
                system,
                controller,
                disorder_strength,
                disorder,
                u,
            )

    fids_arr = np.array(fids)
    infids = np.maximum(0.0, 1.0 - fids_arr)
    tail_start = int((1.0 - TAIL_FRACTION) * len(infids))
    tail = infids[tail_start:]
    initial_infidelity = max(float(infids[0]), 1e-12)
    threshold = 0.1 * initial_infidelity
    reached = np.where(infids <= threshold)[0]
    response = float(t_eval[reached[0]]) if len(reached) else float("nan")
    controls_arr = np.array(controls)
    energy = float(np.trapezoid(controls_arr * controls_arr, t_eval))

    return {
        "system": system.name,
        "controller": controller.name,
        "disorder_strength": disorder_strength,
        "seed": seed,
        "tail_infidelity_mean": float(np.mean(tail)),
        "tail_infidelity_std": float(np.std(tail)),
        "tail_stability_range": float(np.max(tail) - np.min(tail)),
        "final_fidelity": float(fids_arr[-1]),
        "response_time": response,
        "control_energy": energy,
    }


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["system"]), str(row["controller"]), float(row["disorder_strength"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (system, controller, disorder), items in sorted(groups.items()):
        inf = np.array([float(x["tail_infidelity_mean"]) for x in items])
        stab = np.array([float(x["tail_stability_range"]) for x in items])
        resp = np.array([float(x["response_time"]) for x in items])
        resp = resp[np.isfinite(resp)]
        final_fid = np.array([float(x["final_fidelity"]) for x in items])
        energy = np.array([float(x["control_energy"]) for x in items])
        summary.append(
            {
                "system": system,
                "controller": controller,
                "disorder": f"{disorder:.4g}",
                "n": str(len(items)),
                "tail_infidelity_mean": f"{np.mean(inf):.6g}",
                "tail_infidelity_std_across_seeds": f"{np.std(inf):.6g}",
                "tail_stability_mean": f"{np.mean(stab):.6g}",
                "final_fidelity_mean": f"{np.mean(final_fid):.6g}",
                "response_time_mean": f"{np.mean(resp):.6g}" if len(resp) else "nan",
                "control_energy_mean": f"{np.mean(energy):.6g}",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: list[dict[str, str]]) -> None:
    headers = list(summary[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("# Simulation Summary\n\n")
        f.write("Generated by `python3 code/reproducible_simulation.py`.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the heavier publication-oriented default set",
    )
    parser.add_argument("--seeds", type=int, default=None, help="number of random seeds")
    parser.add_argument("--points", type=int, default=None, help="time grid points")
    parser.add_argument(
        "--include-five-level",
        action="store_true",
        help="include the five-level benchmark in smoke mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_seeds = args.seeds if args.seeds is not None else (6 if args.full else 2)
    n_points = args.points if args.points is not None else (501 if args.full else 201)

    systems = [qubit_system("X"), qubit_system("Z"), qubit_system("H")]
    if args.full or args.include_five_level:
        systems.append(five_level_system())
    controllers = [
        ControllerSpec("nominal", 0.0, 2.0, 1e-4, 2.0, 0.2, 1e-5),
        ControllerSpec("disorder_dressed", 0.01, 2.0, 1e-2, 2.0, 0.2, 1e-5),
    ]
    disorder_grid = [0.0, 0.02, 0.05] if args.full else [0.0, 0.05]

    rows: list[dict[str, float | int | str]] = []
    for system in systems:
        for controller in controllers:
            for disorder_strength in disorder_grid:
                for seed in range(n_seeds):
                    rows.append(
                        run_single(system, controller, disorder_strength, seed, n_points)
                    )

    write_csv(result_path("simulation_results.csv"), rows)
    write_summary(result_path("simulation_summary.md"), aggregate(rows))
    print(f"wrote {len(rows)} rows to results/simulation_results.csv")
    print("wrote aggregate table to results/simulation_summary.md")


if __name__ == "__main__":
    main()
