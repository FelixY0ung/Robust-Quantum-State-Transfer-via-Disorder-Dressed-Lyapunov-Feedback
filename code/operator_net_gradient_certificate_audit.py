"""Gradient-net sensitivity certificate for the Pauli-ball operator audit.

The operator-space finite-net certificate uses the global terminal-infidelity
Lipschitz constant ``2T``.  This script computes first-order sensitivities of
the terminal infidelity with respect to the Pauli disorder coefficients on the
same explicit coefficient net, then combines the largest audited gradient with
the analytic Hessian envelope ``4T^2``:

    sup ||grad ell(a)|| <= max_net ||grad ell(a_i)|| + 4 T^2 h.

Consequently, if the worst net infidelity is ``eps_h``, then

    sup_B ell(a) <= eps_h + (max_net_grad + 4 T^2 h) h.

For the current net this is still conservative, but it is a genuine
certificate route under the same exact-propagation assumptions as the
Hamiltonian finite-net theorem.  The script also reports the older ``2T h``
bound for comparison.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from finite_net_operator_audit import coefficient_net, design_task_pulse
from horizon_lyapunov import (
    comm,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    problem,
)
from paths import result_path


def combined_derivative(
    rho: np.ndarray,
    sensitivities: np.ndarray,
    hamiltonian: np.ndarray,
    basis_i: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    drho = -1.0j * comm(hamiltonian, rho)
    ds = np.empty_like(sensitivities)
    for k, disorder_basis in enumerate(basis_i):
        ds[k] = -1.0j * comm(hamiltonian, sensitivities[k]) - 1.0j * comm(
            disorder_basis,
            rho,
        )
    return drho, ds


def rk4_state_sensitivity_step(
    rho: np.ndarray,
    sensitivities: np.ndarray,
    dt: float,
    hamiltonian: np.ndarray,
    basis_i: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    k1_rho, k1_s = combined_derivative(rho, sensitivities, hamiltonian, basis_i)
    k2_rho, k2_s = combined_derivative(
        rho + 0.5 * dt * k1_rho,
        sensitivities + 0.5 * dt * k1_s,
        hamiltonian,
        basis_i,
    )
    k3_rho, k3_s = combined_derivative(
        rho + 0.5 * dt * k2_rho,
        sensitivities + 0.5 * dt * k2_s,
        hamiltonian,
        basis_i,
    )
    k4_rho, k4_s = combined_derivative(
        rho + dt * k3_rho,
        sensitivities + dt * k3_s,
        hamiltonian,
        basis_i,
    )
    rho_next = rho + (dt / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho)
    s_next = sensitivities + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    return rho_next, s_next


def evaluate_gradient_net(
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
    basis = (p.controls[0], p.controls[1], p.h0)
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, float(t)) for hc in p.controls)
        for t in t_eval[:-1]
    )
    basis_cache = tuple(
        tuple(interaction_frame_operator(p, op, float(t)) for op in basis)
        for t in t_eval[:-1]
    )

    rows: list[dict[str, float | int | str]] = []
    for index, coeff in enumerate(coeffs):
        rho = p.initial.copy()
        sensitivities = np.zeros((3, *rho.shape), dtype=complex)
        for j in range(len(pulse)):
            hamiltonian = sum(
                float(control_coeff) * controls_cache[j][control_index]
                for control_index, control_coeff in enumerate(pulse[j])
            )
            hamiltonian = hamiltonian + sum(
                float(coeff[k]) * basis_cache[j][k] for k in range(3)
            )
            rho, sensitivities = rk4_state_sensitivity_step(
                rho,
                sensitivities,
                dt,
                hamiltonian,
                basis_cache[j],
            )
        rho = hermitize_trace_one(rho)
        final_fidelity = fidelity(rho, p.target)
        gradient = np.array(
            [-float(np.real(np.trace(p.target @ sensitivities[k]))) for k in range(3)]
        )
        gradient_norm = float(np.linalg.norm(gradient))
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
                "grad_x": float(gradient[0]),
                "grad_y": float(gradient[1]),
                "grad_z": float(gradient[2]),
                "gradient_norm": gradient_norm,
                "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
                "design_seconds": design_seconds,
                "time_horizon": p.t_final,
            }
        )
    return rows


def quantile_text(values: np.ndarray, q: float) -> str:
    return f"{float(np.quantile(values, q)):.6g}"


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    groups = sorted({(str(row["task"]), str(row["method"])) for row in rows})
    for task, method in groups:
        group = [row for row in rows if row["task"] == task and row["method"] == method]
        infids = np.array([float(row["final_infidelity"]) for row in group])
        gradients = np.array([float(row["gradient_norm"]) for row in group])
        fids = 1.0 - infids
        first = group[0]
        time_horizon = float(first["time_horizon"])
        h = float(first["covering_radius"])
        analytic_lipschitz = 2.0 * time_horizon
        hessian_bound = 4.0 * time_horizon * time_horizon
        max_gradient = float(np.max(gradients))
        gradient_lipschitz = max_gradient + hessian_bound * h
        worst_net_infidelity = float(np.max(infids))
        analytic_penalty = analytic_lipschitz * h
        gradient_penalty = gradient_lipschitz * h
        analytic_bound = min(1.0, worst_net_infidelity + analytic_penalty)
        gradient_bound = min(1.0, worst_net_infidelity + gradient_penalty)
        summary.append(
            {
                "task": task,
                "method": method,
                "net_points": str(len(group)),
                "ball_radius": f"{float(first['ball_radius']):.6g}",
                "points_per_axis": str(int(first["points_per_axis"])),
                "covering_radius_h": f"{h:.6g}",
                "worst_net_fidelity": f"{float(np.min(fids)):.6g}",
                "mean_net_fidelity": f"{float(np.mean(fids)):.6g}",
                "worst_net_infidelity": f"{worst_net_infidelity:.6g}",
                "gradient_norm_mean": f"{float(np.mean(gradients)):.6g}",
                "gradient_norm_p95": quantile_text(gradients, 0.95),
                "gradient_norm_p99": quantile_text(gradients, 0.99),
                "gradient_norm_max": f"{max_gradient:.6g}",
                "hessian_bound_4T2": f"{hessian_bound:.6g}",
                "gradient_net_lipschitz_bound": f"{gradient_lipschitz:.6g}",
                "gradient_net_h_penalty": f"{gradient_penalty:.6g}",
                "gradient_net_continuous_infidelity_bound": f"{gradient_bound:.6g}",
                "gradient_net_continuous_fidelity_lower_bound": f"{1.0 - gradient_bound:.6g}",
                "analytic_lipschitz_2T": f"{analytic_lipschitz:.6g}",
                "analytic_h_penalty": f"{analytic_penalty:.6g}",
                "analytic_continuous_fidelity_lower_bound": f"{1.0 - analytic_bound:.6g}",
                "pulse_energy": f"{float(first['pulse_energy']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
            }
        )
    return summary


def write_outputs(
    rows: list[dict[str, float | int | str]],
    write_full_net: bool,
) -> None:
    summary_rows = summarize(rows)
    result_file = result_path("operator_net_gradient_certificate_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if write_full_net:
        net_file = result_path("operator_net_gradient_certificate_net_results.csv")
        with net_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} net rows to {net_file}")

    summary_file = result_path("operator_net_gradient_certificate_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Operator-Net Gradient Certificate Audit\n\n")
        f.write(
            "First-order terminal-infidelity sensitivities are propagated on the "
            "same Pauli-ball h-net used by the operator-space finite-net audit. "
            "The certified sensitivity bound is `max_net ||grad ell|| + 4T^2 h`; "
            "the older `2T` bound is shown for comparison.\n\n"
        )
        headers = list(summary_rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    print(f"wrote summary CSV to {result_file}")
    print(f"wrote summary Markdown to {summary_file}")


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
        started = time.perf_counter()
        print(
            f"designing {args.method} pulse for {task}; "
            f"net points={len(coeffs)} h={covering_radius:.6g}",
            flush=True,
        )
        pulse, design_seconds = design_task_pulse(task, args.method)
        print(
            f"evaluating gradient certificate for {task}; "
            f"design_seconds={design_seconds:.4g}",
            flush=True,
        )
        rows.extend(
            evaluate_gradient_net(
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
        print(f"finished {task} in {time.perf_counter() - started:.4g}s", flush=True)
    write_outputs(rows, args.write_full_net)
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
    parser.add_argument(
        "--write-full-net",
        action="store_true",
        help="Also write per-net-point gradients to a larger CSV.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
