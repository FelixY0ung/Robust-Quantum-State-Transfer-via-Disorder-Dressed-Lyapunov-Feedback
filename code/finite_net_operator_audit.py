"""Operator-space finite-net robustness audit for two-level disorder.

The scalar finite-net audit covers only strength variation along a few fixed
directions.  This script builds an explicit Cartesian h-net for the compact
Pauli-coefficient ball

    D_R = {a_x sigma_x + a_y sigma_y + a_z sigma_z : ||a||_2 <= R}

and evaluates the terminal-value shifted-horizon pulse on every net point.  The
grid is projected onto the ball, so the net points lie in D_R.  Projection onto
a convex ball is nonexpansive, hence a Cartesian grid with spacing s gives a
coefficient covering radius h <= sqrt(3) s / 2.

For Hamiltonian dynamics with the same controls and two disorder coefficients
a, b, the terminal trace-distance perturbation satisfies

    ||rho_T(a)-rho_T(b)||_1 <= 2 T ||(a-b).sigma||_inf
                             = 2 T ||a-b||_2.

Since the target infidelity is 1 - Tr(P rho) and ||P||_inf = 1, the terminal
infidelity is therefore 2T-Lipschitz in the coefficient metric.  The audit
reports the finite-net worst infidelity and the conservative continuous-ball
bound worst_net_infidelity + 2 T h.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from horizon_lyapunov import (
    default_beam_width,
    design_pulse as design_beam_pulse,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    problem,
    rk4_step,
)
from paths import result_path
from terminal_value_shifted_horizon import (
    TerminalValueHorizonConfig,
    design_pulse as design_terminal_value_pulse,
)


def coefficient_net(radius: float, points_per_axis: int) -> tuple[np.ndarray, float, float]:
    if points_per_axis < 2:
        raise ValueError("points_per_axis must be at least 2")
    axis = np.linspace(-radius, radius, points_per_axis)
    spacing = float(axis[1] - axis[0])
    covering_radius = float(np.sqrt(3.0) * spacing / 2.0)
    raw = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    norms = np.linalg.norm(raw, axis=1)
    projected = raw.copy()
    outside = norms > radius
    projected[outside] *= (radius / norms[outside])[:, None]
    # Round only for duplicate removal after projection; keep enough precision
    # that the covering-radius calculation is unaffected.
    rounded = np.round(projected, decimals=14)
    unique = np.unique(rounded, axis=0)
    return unique, spacing, covering_radius


def pauli_disorder(p, coeff: np.ndarray) -> np.ndarray:
    sx, sy = p.controls
    sz = p.h0
    return coeff[0] * sx + coeff[1] * sy + coeff[2] * sz


def evaluate_coefficients(
    task: str,
    pulse: np.ndarray,
    coeffs: np.ndarray,
    method: str,
    design_seconds: float,
    radius: float,
    points_per_axis: int,
    spacing: float,
    covering_radius: float,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        for t in t_eval[:-1]
    )
    rows: list[dict[str, float | int | str]] = []
    for index, coeff in enumerate(coeffs):
        disorder = pauli_disorder(p, coeff)
        rho = p.initial.copy()
        for j, t in enumerate(t_eval[:-1]):
            disorder_i = interaction_frame_operator(p, disorder, float(t))
            rho = rk4_step(
                rho,
                dt,
                controls_cache[j],
                disorder_i,
                1.0,
                pulse[j],
            )
        rho = hermitize_trace_one(rho)
        final_fidelity = fidelity(rho, p.target)
        rows.append(
            {
                "task": task,
                "method": method,
                "net_index": index,
                "coeff_x": float(coeff[0]),
                "coeff_y": float(coeff[1]),
                "coeff_z": float(coeff[2]),
                "coeff_norm": float(np.linalg.norm(coeff)),
                "ball_radius": radius,
                "points_per_axis": points_per_axis,
                "grid_spacing": spacing,
                "covering_radius": covering_radius,
                "final_fidelity": final_fidelity,
                "final_infidelity": 1.0 - final_fidelity,
                "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
                "design_seconds": design_seconds,
                "time_horizon": p.t_final,
            }
        )
    return rows


def design_task_pulse(task: str, method: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if method == "terminal_value_shifted":
        pulse, _, _ = design_terminal_value_pulse(task, TerminalValueHorizonConfig())
    elif method == "beam_horizon":
        pulse = design_beam_pulse(task, beam_width=default_beam_width(task))
    else:
        raise ValueError(method)
    return pulse, time.perf_counter() - start


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    groups = sorted({(str(row["task"]), str(row["method"])) for row in rows})
    for task, method in groups:
        group = [row for row in rows if row["task"] == task and row["method"] == method]
        infids = np.array([float(row["final_infidelity"]) for row in group])
        fids = 1.0 - infids
        first = group[0]
        time_horizon = float(first["time_horizon"])
        covering_radius = float(first["covering_radius"])
        lipschitz = 2.0 * time_horizon
        penalty = lipschitz * covering_radius
        worst_net_infidelity = float(np.max(infids))
        continuous_bound = min(1.0, worst_net_infidelity + penalty)
        summary.append(
            {
                "task": task,
                "method": method,
                "net_points": str(len(group)),
                "ball_radius": f"{float(first['ball_radius']):.6g}",
                "points_per_axis": str(int(first["points_per_axis"])),
                "grid_spacing": f"{float(first['grid_spacing']):.6g}",
                "covering_radius_h": f"{covering_radius:.6g}",
                "infidelity_lipschitz_2T": f"{lipschitz:.6g}",
                "lipschitz_penalty": f"{penalty:.6g}",
                "worst_net_fidelity": f"{float(np.min(fids)):.6g}",
                "mean_net_fidelity": f"{float(np.mean(fids)):.6g}",
                "worst_net_infidelity": f"{worst_net_infidelity:.6g}",
                "continuous_infidelity_bound": f"{continuous_bound:.6g}",
                "continuous_fidelity_lower_bound": f"{1.0 - continuous_bound:.6g}",
                "pulse_energy": f"{float(first['pulse_energy']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("finite_net_operator_audit_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("finite_net_operator_audit_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Operator-Space Finite-Net Audit\n\n")
        f.write(
            "Explicit Cartesian h-net for the compact Pauli-coefficient disorder "
            "ball. The continuous bound uses the analytic terminal-infidelity "
            "Lipschitz constant 2T in coefficient Euclidean norm. This bound is "
            "conservative but deterministic and covers all disorder directions "
            "inside the stated ball.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")


def run(args: argparse.Namespace) -> None:
    coeffs, spacing, covering_radius = coefficient_net(
        args.ball_radius,
        args.points_per_axis,
    )
    rows: list[dict[str, float | int | str]] = []
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        print(
            f"designing {args.method} pulse for {task}; "
            f"net points={len(coeffs)} h={covering_radius:.6g}",
            flush=True,
        )
        pulse, design_seconds = design_task_pulse(task, args.method)
        rows.extend(
            evaluate_coefficients(
                task,
                pulse,
                coeffs,
                args.method,
                design_seconds,
                args.ball_radius,
                args.points_per_axis,
                spacing,
                covering_radius,
            )
        )
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Z,H")
    parser.add_argument(
        "--method",
        choices=("terminal_value_shifted", "beam_horizon"),
        default="terminal_value_shifted",
    )
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--points-per-axis", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
