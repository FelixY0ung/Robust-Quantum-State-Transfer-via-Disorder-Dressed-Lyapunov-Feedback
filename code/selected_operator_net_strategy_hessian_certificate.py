"""Hessian-net sensitivity certificate for selected operator-net strategies.

The gradient certificate bounds the continuous Pauli-ball infidelity by using
the global Hessian envelope ``4T^2``.  This audit propagates second-order
terminal-infidelity sensitivities on an explicit coefficient net, then combines
the largest audited Hessian with the analytic third-derivative envelope
``8T^3``:

    sup ||D^2 ell(a)|| <= max_net ||D^2 ell(a_i)|| + 8 T^3 h.

The resulting bound is still deterministic under the same exact-propagation
model assumptions, but it can be much less conservative than using ``4T^2``.
It remains a compact-set robustness certificate for fixed generated pulses,
not a fixed-depth all-time value theorem.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from finite_net_operator_audit import coefficient_net
from horizon_lyapunov import (
    comm,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    problem,
)
from paths import result_path
from selected_operator_net_strategy_gradient_certificate import (
    regenerate_selected_pulse,
    selected_designs,
)


def combined_second_derivative(
    rho: np.ndarray,
    sensitivities: np.ndarray,
    hessians: np.ndarray,
    hamiltonian: np.ndarray,
    basis_i: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drho = -1.0j * comm(hamiltonian, rho)
    ds = np.empty_like(sensitivities)
    dq = np.empty_like(hessians)
    for k, basis_k in enumerate(basis_i):
        ds[k] = -1.0j * comm(hamiltonian, sensitivities[k]) - 1.0j * comm(
            basis_k,
            rho,
        )
        for l, basis_l in enumerate(basis_i):
            dq[k, l] = (
                -1.0j * comm(hamiltonian, hessians[k, l])
                - 1.0j * comm(basis_k, sensitivities[l])
                - 1.0j * comm(basis_l, sensitivities[k])
            )
    return drho, ds, dq


def rk4_state_second_sensitivity_step(
    rho: np.ndarray,
    sensitivities: np.ndarray,
    hessians: np.ndarray,
    dt: float,
    hamiltonian: np.ndarray,
    basis_i: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k1_rho, k1_s, k1_q = combined_second_derivative(
        rho,
        sensitivities,
        hessians,
        hamiltonian,
        basis_i,
    )
    k2_rho, k2_s, k2_q = combined_second_derivative(
        rho + 0.5 * dt * k1_rho,
        sensitivities + 0.5 * dt * k1_s,
        hessians + 0.5 * dt * k1_q,
        hamiltonian,
        basis_i,
    )
    k3_rho, k3_s, k3_q = combined_second_derivative(
        rho + 0.5 * dt * k2_rho,
        sensitivities + 0.5 * dt * k2_s,
        hessians + 0.5 * dt * k2_q,
        hamiltonian,
        basis_i,
    )
    k4_rho, k4_s, k4_q = combined_second_derivative(
        rho + dt * k3_rho,
        sensitivities + dt * k3_s,
        hessians + dt * k3_q,
        hamiltonian,
        basis_i,
    )
    rho_next = rho + (dt / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho)
    s_next = sensitivities + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    q_next = hessians + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    return rho_next, s_next, q_next


def evaluate_hessian_net(
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
        if index and index % 1000 == 0:
            print(f"  {task}/{method}: Hessian net point {index}/{len(coeffs)}", flush=True)
        rho = p.initial.copy()
        sensitivities = np.zeros((3, *rho.shape), dtype=complex)
        hessians = np.zeros((3, 3, *rho.shape), dtype=complex)
        for j in range(len(pulse)):
            hamiltonian = sum(
                float(control_coeff) * controls_cache[j][control_index]
                for control_index, control_coeff in enumerate(pulse[j])
            )
            hamiltonian = hamiltonian + sum(
                float(coeff[k]) * basis_cache[j][k] for k in range(3)
            )
            rho, sensitivities, hessians = rk4_state_second_sensitivity_step(
                rho,
                sensitivities,
                hessians,
                dt,
                hamiltonian,
                basis_cache[j],
            )
        rho = hermitize_trace_one(rho)
        final_fidelity = fidelity(rho, p.target)
        gradient = np.array(
            [-float(np.real(np.trace(p.target @ sensitivities[k]))) for k in range(3)]
        )
        hessian = np.empty((3, 3), dtype=float)
        for k in range(3):
            for l in range(3):
                hessian[k, l] = -float(np.real(np.trace(p.target @ hessians[k, l])))
        hessian = 0.5 * (hessian + hessian.T)
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
                "gradient_norm": float(np.linalg.norm(gradient)),
                "hess_xx": float(hessian[0, 0]),
                "hess_xy": float(hessian[0, 1]),
                "hess_xz": float(hessian[0, 2]),
                "hess_yy": float(hessian[1, 1]),
                "hess_yz": float(hessian[1, 2]),
                "hess_zz": float(hessian[2, 2]),
                "hessian_spectral_norm": float(np.linalg.norm(hessian, 2)),
                "hessian_frobenius_norm": float(np.linalg.norm(hessian, "fro")),
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
        hessians = np.array([float(row["hessian_spectral_norm"]) for row in group])
        fids = 1.0 - infids
        first = group[0]
        time_horizon = float(first["time_horizon"])
        h = float(first["covering_radius"])
        max_gradient = float(np.max(gradients))
        max_hessian = float(np.max(hessians))
        worst_net_infidelity = float(np.max(infids))
        global_hessian = 4.0 * time_horizon * time_horizon
        third_derivative = 8.0 * time_horizon * time_horizon * time_horizon
        hessian_net_envelope = max_hessian + third_derivative * h
        global_gradient_penalty = (max_gradient + global_hessian * h) * h
        hessian_net_penalty = (max_gradient + hessian_net_envelope * h) * h
        global_bound = min(1.0, worst_net_infidelity + global_gradient_penalty)
        hessian_net_bound = min(1.0, worst_net_infidelity + hessian_net_penalty)
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
                "gradient_norm_max": f"{max_gradient:.6g}",
                "hessian_norm_mean": f"{float(np.mean(hessians)):.6g}",
                "hessian_norm_p95": quantile_text(hessians, 0.95),
                "hessian_norm_p99": quantile_text(hessians, 0.99),
                "hessian_norm_max": f"{max_hessian:.6g}",
                "global_hessian_4T2": f"{global_hessian:.6g}",
                "third_derivative_8T3": f"{third_derivative:.6g}",
                "hessian_net_envelope": f"{hessian_net_envelope:.6g}",
                "global_gradient_h_penalty": f"{global_gradient_penalty:.6g}",
                "global_gradient_fidelity_lower_bound": f"{1.0 - global_bound:.6g}",
                "hessian_net_h_penalty": f"{hessian_net_penalty:.6g}",
                "hessian_net_fidelity_lower_bound": f"{1.0 - hessian_net_bound:.6g}",
                "pulse_energy": f"{float(first['pulse_energy']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
            }
        )
    return summary


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    metadata: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item["score_mode"] = metadata["score_mode"]
        item["worst_weight"] = metadata["worst_weight"]
        item["train_net_points"] = metadata["train_net_points"]
        item["base_train_points"] = metadata["base_train_points"]
        item["hard_points_requested"] = metadata["hard_points_requested"]
        item["hard_points_used"] = metadata["hard_points_used"]
        item["design_points_per_axis"] = metadata["design_points_per_axis"]
        item["certificate_points_per_axis"] = metadata["certificate_points_per_axis"]
        annotated.append(item)
    return annotated


def write_markdown_table(f, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_outputs(
    rows: list[dict[str, float | int | str]],
    output_prefix: str,
    write_net: bool,
) -> None:
    summary_rows = summarize(rows)
    result_file = result_path(f"{output_prefix}_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    if write_net:
        net_file = result_path(f"{output_prefix}_net_results.csv")
        with net_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} net rows to {net_file}")

    summary_file = result_path(f"{output_prefix}_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Selected Operator-Net Strategy Hessian Certificate\n\n")
        f.write(
            "Second-order terminal-infidelity sensitivities are propagated on "
            "the selected strategy certificate net. The deterministic envelope "
            "uses `max_net ||D^2 ell|| + 8T^3 h`, then plugs that Hessian "
            "bound into the same gradient-net covering argument. This tightens "
            "the continuous Pauli-ball robustness bound for the fixed selected "
            "pulses; it is not a fixed-depth value-function theorem.\n\n"
        )
        write_markdown_table(f, summary_rows)

    print(f"wrote summary CSV to {result_file}")
    print(f"wrote summary Markdown to {summary_file}")
    for row in summary_rows:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--base-points", type=int, default=7)
    parser.add_argument("--design-points", type=int, default=13)
    parser.add_argument("--certificate-points", type=int, default=25)
    parser.add_argument(
        "--output-prefix",
        default="selected_operator_net_strategy_hessian_certificate_p25",
    )
    parser.add_argument("--no-net-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_coeffs, base_spacing, base_h = coefficient_net(args.ball_radius, args.base_points)
    design_coeffs, design_spacing, design_h = coefficient_net(
        args.ball_radius,
        args.design_points,
    )
    certificate_coeffs, certificate_spacing, certificate_h = coefficient_net(
        args.ball_radius,
        args.certificate_points,
    )
    print(
        "selected-strategy Hessian certificate nets: "
        f"base={len(base_coeffs)} design={len(design_coeffs)} "
        f"certificate={len(certificate_coeffs)} h={certificate_h:.6g}",
        flush=True,
    )
    all_rows: list[dict[str, float | int | str]] = []
    for design in selected_designs():
        pulse, metadata = regenerate_selected_pulse(
            design,
            base_coeffs,
            design_coeffs,
            base_spacing,
            base_h,
            design_spacing,
            design_h,
        )
        metadata["design_points_per_axis"] = args.design_points
        metadata["certificate_points_per_axis"] = args.certificate_points
        print(
            f"evaluating selected Hessian certificate: {metadata['task']} "
            f"{metadata['method']}",
            flush=True,
        )
        started = time.perf_counter()
        rows = evaluate_hessian_net(
            str(metadata["task"]),
            pulse,
            certificate_coeffs,
            str(metadata["method"]),
            float(metadata["design_seconds"]),
            args.ball_radius,
            args.certificate_points,
            certificate_spacing,
            certificate_h,
        )
        print(
            f"  Hessian evaluation seconds: {time.perf_counter() - started:.3f}",
            flush=True,
        )
        all_rows.extend(annotate_rows(rows, metadata))
    write_outputs(all_rows, args.output_prefix, not args.no_net_output)


if __name__ == "__main__":
    main()
