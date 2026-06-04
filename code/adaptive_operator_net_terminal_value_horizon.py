"""Adaptive hard-point operator-net terminal-value horizon diagnostic.

The direct operator-net diagnostic trains on a uniform coarse Pauli-coefficient
net and audits on the denser 2145-point net.  This script tests a richer route:
first train on the coarse net, identify the worst dense-net points under that
coarse pulse, then redesign from scratch on the coarse net augmented by those
hard points.  It keeps the same finite-candidate shifted-fallback architecture
and scheduled terminal-value margin audit.

This is a two-level Bloch-vector diagnostic for the Pauli-ball benchmarks.  It
does not by itself prove a fixed-depth all-time value theorem.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace

import numpy as np

from finite_net_operator_audit import coefficient_net
from fixed_depth_terminal_value_margin_audit import density_states_to_bloch
from horizon_lyapunov import candidate_controls, default_beam_width, problem
from operator_net_trained_terminal_value_horizon import (
    OperatorNetHorizonConfig,
    apply_control,
    evaluate_pulse,
    heldout_coefficients,
    interaction_coeff_cache,
    rollout_sequence,
    score_phi,
    select_sequence,
)
from paths import result_path
from shifted_fallback_horizon import DesignStats


def unique_coefficients(coeffs: np.ndarray) -> np.ndarray:
    rounded = np.round(coeffs, decimals=14)
    return np.unique(rounded, axis=0)


def final_fidelities(task: str, states: np.ndarray) -> np.ndarray:
    p = problem(task)
    target_bloch = density_states_to_bloch((p.target,))[0]
    return 0.5 * (1.0 + states @ target_bloch)


def select_hard_coefficients(
    task: str,
    eval_coeffs: np.ndarray,
    eval_states: np.ndarray,
    hard_points: int,
) -> np.ndarray:
    if hard_points <= 0:
        return np.empty((0, 3), dtype=float)
    fidelities = final_fidelities(task, eval_states)
    order = np.argsort(fidelities)
    return eval_coeffs[order[: min(hard_points, len(order))]]


def run_design_stage(
    task: str,
    design_label: str,
    train_coeffs: np.ndarray,
    eval_coeffs: np.ndarray,
    train_spacing: float,
    train_covering_radius: float,
    eval_spacing: float,
    eval_covering_radius: float,
    base_train_points: int,
    hard_points_requested: int,
    hard_points_used: int,
    config: OperatorNetHorizonConfig,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    np.ndarray,
    np.ndarray,
]:
    started = time.perf_counter()
    p = problem(task)
    target_bloch = density_states_to_bloch((p.target,))[0]
    initial_bloch = density_states_to_bloch((p.initial,))[0]
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    cache_times = tuple(
        float(t_eval[min(j, config.segments)])
        for j in range(config.segments + config.horizon_steps + 2)
    )
    train_control_cache, train_disorder_cache = interaction_coeff_cache(
        p,
        train_coeffs,
        cache_times,
    )
    eval_control_cache, eval_disorder_cache = interaction_coeff_cache(
        p,
        eval_coeffs,
        cache_times,
    )
    candidates = candidate_controls(config.amplitudes)
    beam_width = default_beam_width(task)
    train_strengths = np.ones(len(train_coeffs))
    eval_strengths = np.ones(len(eval_coeffs))
    train_states = np.repeat(initial_bloch[None, :], len(train_coeffs), axis=0)
    eval_states = np.repeat(initial_bloch[None, :], len(eval_coeffs), axis=0)
    previous_sequence: tuple[np.ndarray, ...] | None = None
    shifted_available = 0
    shifted_selected = 0
    pulse: list[np.ndarray] = []
    margin_rows: list[dict[str, float | int | str]] = []

    for j, _ in enumerate(t_eval[:-1]):
        if previous_sequence is not None:
            shifted_tail = previous_sequence[1:]
            train_tail, _ = rollout_sequence(
                train_states,
                train_strengths,
                train_control_cache,
                train_disorder_cache,
                j,
                dt,
                shifted_tail,
            )
            eval_tail, _ = rollout_sequence(
                eval_states,
                eval_strengths,
                eval_control_cache,
                eval_disorder_cache,
                j,
                dt,
                shifted_tail,
            )
            train_phi = float(
                score_phi(
                    train_tail,
                    target_bloch,
                    config.worst_weight,
                    config.score_mode,
                )
            )
            eval_phi = float(
                score_phi(
                    eval_tail,
                    target_bloch,
                    config.worst_weight,
                    config.score_mode,
                )
            )
            margin_rows.append(
                {
                    "task": task,
                    "design_label": design_label,
                    "step": j,
                    "score_mode": config.score_mode,
                    "worst_weight": config.worst_weight,
                    "train_net_points": len(train_coeffs),
                    "base_train_points": base_train_points,
                    "hard_points_requested": hard_points_requested,
                    "hard_points_used": hard_points_used,
                    "eval_net_points": len(eval_coeffs),
                    "train_points_per_axis": config.train_points_per_axis,
                    "eval_points_per_axis": config.eval_points_per_axis,
                    "train_covering_radius": train_covering_radius,
                    "eval_covering_radius": eval_covering_radius,
                    "beam_width": beam_width,
                    "residual_threshold": config.residual_threshold,
                    "train_phi": train_phi,
                    "eval_phi": eval_phi,
                    "train_outside_residual": int(train_phi > config.residual_threshold),
                    "eval_outside_residual": int(eval_phi > config.residual_threshold),
                    "train_scheduled_margin": max(0.0, train_phi - config.residual_threshold),
                    "eval_scheduled_margin": max(0.0, eval_phi - config.residual_threshold),
                    "shifted_selected_fraction": (
                        shifted_selected / shifted_available
                        if shifted_available
                        else 0.0
                    ),
                }
            )

        sequence, selected_shifted, selected_phi, selected_cost = select_sequence(
            train_states,
            train_strengths,
            train_control_cache,
            train_disorder_cache,
            j,
            dt,
            candidates,
            beam_width,
            target_bloch,
            config,
            previous_sequence,
        )
        if previous_sequence is not None:
            shifted_available += 1
            if selected_shifted:
                shifted_selected += 1
        control = sequence[0]
        pulse.append(control)
        train_states = apply_control(
            train_states,
            train_strengths,
            train_control_cache[j],
            train_disorder_cache[j],
            dt,
            control,
        )
        eval_states = apply_control(
            eval_states,
            eval_strengths,
            eval_control_cache[j],
            eval_disorder_cache[j],
            dt,
            control,
        )
        previous_sequence = sequence

        if j % 20 == 0:
            print(
                f"  {task} {design_label} {config.score_mode}: step "
                f"{j}/{config.segments} phi={selected_phi:.6g} "
                f"cost={selected_cost:.6g}",
                flush=True,
            )

    pulse_array = np.asarray(pulse)
    design_seconds = time.perf_counter() - started
    stats = DesignStats(
        shifted_available_steps=shifted_available,
        shifted_selected_steps=shifted_selected,
    )
    pulse_energy = float(np.mean(np.sum(pulse_array * pulse_array, axis=1)))

    net_rows: list[dict[str, float | int | str]] = []
    for label, coeffs, states, spacing, radius_h, points in (
        (
            "train_net",
            train_coeffs,
            train_states,
            train_spacing,
            train_covering_radius,
            config.train_points_per_axis,
        ),
        (
            "eval_net",
            eval_coeffs,
            eval_states,
            eval_spacing,
            eval_covering_radius,
            config.eval_points_per_axis,
        ),
    ):
        fidelities = final_fidelities(task, states)
        for index, (coeff, fidelity) in enumerate(zip(coeffs, fidelities)):
            net_rows.append(
                {
                    "task": task,
                    "design_label": design_label,
                    "net_label": label,
                    "net_index": index,
                    "coeff_x": float(coeff[0]),
                    "coeff_y": float(coeff[1]),
                    "coeff_z": float(coeff[2]),
                    "coeff_norm": float(np.linalg.norm(coeff)),
                    "ball_radius": config.ball_radius,
                    "points_per_axis": points,
                    "grid_spacing": spacing,
                    "covering_radius": radius_h,
                    "score_mode": config.score_mode,
                    "worst_weight": config.worst_weight,
                    "train_net_points": len(train_coeffs),
                    "base_train_points": base_train_points,
                    "hard_points_requested": hard_points_requested,
                    "hard_points_used": hard_points_used,
                    "final_fidelity": float(fidelity),
                    "final_infidelity": float(1.0 - fidelity),
                    "pulse_energy": pulse_energy,
                    "shifted_selected_fraction": stats.shifted_selected_fraction,
                    "design_seconds": design_seconds,
                }
            )

    heldout_rows: list[dict[str, float | int | str]] = []
    for strength in (0.0, 0.02, 0.05, 0.08):
        coeffs = heldout_coefficients(task, strength, range(10, 60))
        heldout_states = evaluate_pulse(task, pulse_array, coeffs)
        fidelities = final_fidelities(task, heldout_states)
        for seed, fidelity in zip(range(10, 60), fidelities):
            heldout_rows.append(
                {
                    "task": task,
                    "design_label": design_label,
                    "disorder_strength": strength,
                    "seed": seed,
                    "score_mode": config.score_mode,
                    "worst_weight": config.worst_weight,
                    "train_points_per_axis": config.train_points_per_axis,
                    "train_net_points": len(train_coeffs),
                    "base_train_points": base_train_points,
                    "hard_points_requested": hard_points_requested,
                    "hard_points_used": hard_points_used,
                    "eval_points_per_axis": config.eval_points_per_axis,
                    "eval_net_points": len(eval_coeffs),
                    "final_fidelity": float(fidelity),
                    "pulse_energy": pulse_energy,
                    "shifted_selected_fraction": stats.shifted_selected_fraction,
                    "design_seconds": design_seconds,
                }
            )

    return heldout_rows, net_rows, margin_rows, pulse_array, eval_states


def summarize_heldout(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted(
        {
            (
                str(row["task"]),
                str(row["design_label"]),
                str(row["score_mode"]),
                float(row["worst_weight"]),
                float(row["disorder_strength"]),
            )
            for row in rows
        }
    )
    for task, design_label, score_mode, worst_weight, strength in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task
            and str(row["design_label"]) == design_label
            and str(row["score_mode"]) == score_mode
            and float(row["worst_weight"]) == worst_weight
            and float(row["disorder_strength"]) == strength
        ]
        fidelities = np.asarray([float(row["final_fidelity"]) for row in group])
        first = group[0]
        summary.append(
            {
                "task": task,
                "design_label": design_label,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(group)),
                "score_mode": score_mode,
                "worst_weight": f"{worst_weight:.6g}",
                "train_net_points": str(int(first["train_net_points"])),
                "hard_points_used": str(int(first["hard_points_used"])),
                "final_fidelity_mean": f"{float(np.mean(fidelities)):.6g}",
                "final_fidelity_min": f"{float(np.min(fidelities)):.6g}",
                "final_fidelity_std": f"{float(np.std(fidelities)):.6g}",
                "pulse_energy_mean": f"{float(first['pulse_energy']):.6g}",
                "shifted_selected_fraction": f"{float(first['shifted_selected_fraction']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
            }
        )
    return summary


def summarize_net(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted(
        {
            (
                str(row["task"]),
                str(row["design_label"]),
                str(row["score_mode"]),
                float(row["worst_weight"]),
                str(row["net_label"]),
            )
            for row in rows
        }
    )
    for task, design_label, score_mode, worst_weight, label in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task
            and str(row["design_label"]) == design_label
            and str(row["score_mode"]) == score_mode
            and float(row["worst_weight"]) == worst_weight
            and str(row["net_label"]) == label
        ]
        fidelities = np.asarray([float(row["final_fidelity"]) for row in group])
        first = group[0]
        summary.append(
            {
                "task": task,
                "design_label": design_label,
                "net_label": label,
                "score_mode": score_mode,
                "worst_weight": f"{worst_weight:.6g}",
                "net_points": str(len(group)),
                "train_net_points": str(int(first["train_net_points"])),
                "hard_points_used": str(int(first["hard_points_used"])),
                "points_per_axis": str(int(first["points_per_axis"])),
                "covering_radius_h": f"{float(first['covering_radius']):.6g}",
                "worst_net_fidelity": f"{float(np.min(fidelities)):.6g}",
                "mean_net_fidelity": f"{float(np.mean(fidelities)):.6g}",
                "worst_net_infidelity": f"{float(np.max(1.0 - fidelities)):.6g}",
                "pulse_energy": f"{float(first['pulse_energy']):.6g}",
                "shifted_selected_fraction": f"{float(first['shifted_selected_fraction']):.6g}",
                "design_seconds": f"{float(first['design_seconds']):.4g}",
            }
        )
    return summary


def summarize_margins(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted(
        {
            (
                str(row["task"]),
                str(row["design_label"]),
                str(row["score_mode"]),
                float(row["worst_weight"]),
            )
            for row in rows
        }
    )
    for task, design_label, score_mode, worst_weight in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task
            and str(row["design_label"]) == design_label
            and str(row["score_mode"]) == score_mode
            and float(row["worst_weight"]) == worst_weight
        ]
        eval_margins = np.asarray([float(row["eval_scheduled_margin"]) for row in group])
        train_margins = np.asarray([float(row["train_scheduled_margin"]) for row in group])
        eval_outside = np.asarray(
            [int(row["eval_outside_residual"]) for row in group],
            dtype=bool,
        )
        train_outside = np.asarray(
            [int(row["train_outside_residual"]) for row in group],
            dtype=bool,
        )
        first = group[0]
        train_positive = (
            float(np.mean(train_margins[train_outside] > 0.0))
            if np.any(train_outside)
            else 1.0
        )
        eval_positive = (
            float(np.mean(eval_margins[eval_outside] > 0.0))
            if np.any(eval_outside)
            else 1.0
        )
        train_min = (
            float(np.min(train_margins[train_outside]))
            if np.any(train_outside)
            else 0.0
        )
        eval_min = (
            float(np.min(eval_margins[eval_outside]))
            if np.any(eval_outside)
            else 0.0
        )
        summary.append(
            {
                "task": task,
                "design_label": design_label,
                "score_mode": score_mode,
                "worst_weight": f"{worst_weight:.6g}",
                "train_net_points": str(int(first["train_net_points"])),
                "eval_net_points": str(int(first["eval_net_points"])),
                "hard_points_used": str(int(first["hard_points_used"])),
                "audited_steps": str(len(group)),
                "train_outside_steps": str(int(np.sum(train_outside))),
                "eval_outside_steps": str(int(np.sum(eval_outside))),
                "train_positive_outside_fraction": f"{train_positive:.6g}",
                "eval_positive_outside_fraction": f"{eval_positive:.6g}",
                "train_min_outside_margin": f"{train_min:.6g}",
                "eval_min_outside_margin": f"{eval_min:.6g}",
                "eval_margin_mean": f"{float(np.mean(eval_margins)):.6g}",
                "eval_margin_min": f"{float(np.min(eval_margins)):.6g}",
            }
        )
    return summary


def write_markdown_table(f, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_outputs(
    heldout_rows: list[dict[str, float | int | str]],
    net_rows: list[dict[str, float | int | str]],
    margin_rows: list[dict[str, float | int | str]],
    output_prefix: str,
) -> None:
    outputs = (
        (f"{output_prefix}_results.csv", heldout_rows),
        (f"{output_prefix}_net_results.csv", net_rows),
        (f"{output_prefix}_margin_results.csv", margin_rows),
    )
    for filename, rows in outputs:
        with result_path(filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    with result_path(f"{output_prefix}_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Adaptive Operator-Net Terminal-Value Horizon Summary\n\n")
        f.write(
            "This diagnostic trains the terminal-value shifted-fallback horizon "
            "on a coarse Pauli coefficient net, identifies hard points on the "
            "2145-point evaluation net, and then redesigns on the coarse net "
            "augmented by those hard points. It tests whether adaptive net "
            "enrichment can improve the direct-net controller; it is not a "
            "fixed-depth all-time theorem.\n\n"
        )
        f.write("## Held-Out Performance\n\n")
        write_markdown_table(f, summarize_heldout(heldout_rows))
        f.write("\n## Operator-Net Performance\n\n")
        write_markdown_table(f, summarize_net(net_rows))
        f.write("\n## Scheduled Margin Audit\n\n")
        write_markdown_table(f, summarize_margins(margin_rows))


def run(args: argparse.Namespace) -> None:
    base_coeffs, base_spacing, base_h = coefficient_net(
        args.ball_radius,
        args.train_points_per_axis,
    )
    eval_coeffs, eval_spacing, eval_h = coefficient_net(
        args.ball_radius,
        args.eval_points_per_axis,
    )
    score_modes = [item.strip() for item in args.score_modes.split(",") if item.strip()]
    heldout_rows: list[dict[str, float | int | str]] = []
    net_rows: list[dict[str, float | int | str]] = []
    margin_rows: list[dict[str, float | int | str]] = []

    for score_mode in score_modes:
        config = OperatorNetHorizonConfig(
            ball_radius=args.ball_radius,
            train_points_per_axis=args.train_points_per_axis,
            eval_points_per_axis=args.eval_points_per_axis,
            segments=args.segments,
            horizon_steps=args.horizon_steps,
            score_mode=score_mode,
            worst_weight=args.worst_weight,
        )
        for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
            print(
                f"designing coarse operator-net horizon for {task} "
                f"score_mode={score_mode}",
                flush=True,
            )
            h_rows, n_rows, m_rows, _, coarse_eval_states = run_design_stage(
                task=task,
                design_label="coarse",
                train_coeffs=base_coeffs,
                eval_coeffs=eval_coeffs,
                train_spacing=base_spacing,
                train_covering_radius=base_h,
                eval_spacing=eval_spacing,
                eval_covering_radius=eval_h,
                base_train_points=len(base_coeffs),
                hard_points_requested=args.hard_points,
                hard_points_used=0,
                config=config,
            )
            heldout_rows.extend(h_rows)
            net_rows.extend(n_rows)
            margin_rows.extend(m_rows)

            hard_coeffs = select_hard_coefficients(
                task,
                eval_coeffs,
                coarse_eval_states,
                args.hard_points,
            )
            adaptive_coeffs = unique_coefficients(np.vstack((base_coeffs, hard_coeffs)))
            hard_used = len(adaptive_coeffs) - len(base_coeffs)
            adaptive_config = replace(config)
            print(
                f"designing hard-augmented horizon for {task} "
                f"score_mode={score_mode}; train points={len(adaptive_coeffs)} "
                f"({hard_used} new hard points)",
                flush=True,
            )
            h_rows, n_rows, m_rows, _, _ = run_design_stage(
                task=task,
                design_label="hard_augmented",
                train_coeffs=adaptive_coeffs,
                eval_coeffs=eval_coeffs,
                train_spacing=base_spacing,
                train_covering_radius=base_h,
                eval_spacing=eval_spacing,
                eval_covering_radius=eval_h,
                base_train_points=len(base_coeffs),
                hard_points_requested=args.hard_points,
                hard_points_used=hard_used,
                config=adaptive_config,
            )
            heldout_rows.extend(h_rows)
            net_rows.extend(n_rows)
            margin_rows.extend(m_rows)

    write_outputs(heldout_rows, net_rows, margin_rows, args.output_prefix)
    for row in summarize_net(net_rows):
        print(row)
    for row in summarize_margins(margin_rows):
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Z,H")
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--train-points-per-axis", type=int, default=7)
    parser.add_argument("--eval-points-per-axis", type=int, default=13)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--hard-points", type=int, default=256)
    parser.add_argument("--score-modes", default="max_net,mean_worst")
    parser.add_argument("--worst-weight", type=float, default=0.25)
    parser.add_argument(
        "--output-prefix",
        default="adaptive_operator_net_terminal_value_horizon",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
