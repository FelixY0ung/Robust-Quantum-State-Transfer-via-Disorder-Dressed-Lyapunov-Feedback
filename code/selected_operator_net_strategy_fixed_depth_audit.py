"""Audit fixed-depth margins for the selected operator-net strategies.

The standard fixed-depth audit is negative for the online terminal-value
shifted controller.  This script tests a materially different controller: the
selected finite-library Pauli-ball strategies used in the p31 Hessian-net
certificate.  It reruns the selected designs from their recorded training
coefficient nets and audits the stricter fixed-depth comparison

    G_{j,L}(z) > min_k G_{j+1,L}(F_k z)

on the 2145-point Pauli evaluation net.  A negative row rules out using these
selected strategies as a fixed-depth all-time value theorem without a further
terminal objective or controller change.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from finite_net_operator_audit import coefficient_net
from fixed_depth_terminal_value_margin_audit import density_states_to_bloch
from horizon_lyapunov import candidate_controls, default_beam_width, problem
from operator_net_trained_terminal_value_horizon import (
    OperatorNetHorizonConfig,
    apply_control,
    interaction_coeff_cache,
    rollout_sequence,
    score_phi,
    select_sequence,
    step_candidates,
)
from paths import result_path


@dataclass(frozen=True)
class SelectedDesign:
    task: str
    method: str
    source_csv: str
    design_label: str
    score_mode: str
    worst_weight: float


SELECTED_DESIGNS = (
    SelectedDesign(
        task="Z",
        method="selected_hard_augmented_mean_worst",
        source_csv="adaptive_operator_net_terminal_value_horizon_net_results.csv",
        design_label="hard_augmented",
        score_mode="mean_worst",
        worst_weight=0.25,
    ),
    SelectedDesign(
        task="H",
        method="selected_full2145_max_net",
        source_csv="operator_net_full_trained_terminal_value_horizon_net_results.csv",
        design_label="",
        score_mode="max_net",
        worst_weight=0.25,
    ),
)


def read_training_coefficients(design: SelectedDesign) -> np.ndarray:
    path = result_path(design.source_csv)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    coeffs = []
    for row in rows:
        if row["task"] != design.task:
            continue
        if row.get("net_label", "") != "train_net":
            continue
        if row.get("score_mode", "") != design.score_mode:
            continue
        if abs(float(row.get("worst_weight", design.worst_weight)) - design.worst_weight) > 1e-12:
            continue
        if design.design_label and row.get("design_label", "") != design.design_label:
            continue
        coeffs.append(
            [
                float(row["coeff_x"]),
                float(row["coeff_y"]),
                float(row["coeff_z"]),
            ]
        )
    if not coeffs:
        raise ValueError(f"no training coefficients found for {design}")
    return np.asarray(coeffs, dtype=float)


def terminal_value_batch(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    target_bloch: np.ndarray,
    config: OperatorNetHorizonConfig,
    depth: int,
) -> np.ndarray:
    phis = score_phi(states, target_bloch, config.worst_weight, config.score_mode)
    if depth == 0:
        return np.asarray(phis, dtype=float)

    cache_index = min(start_index, len(control_cache) - 1)
    children = step_candidates(
        states,
        strengths,
        control_cache[cache_index],
        disorder_cache[cache_index],
        dt,
        candidates,
    )
    flat_children = children.reshape(-1, states.shape[-2], 3)
    child_values = terminal_value_batch(
        flat_children,
        strengths,
        control_cache,
        disorder_cache,
        start_index + 1,
        dt,
        candidates,
        target_bloch,
        config,
        depth - 1,
    ).reshape(states.shape[0], len(candidates))
    stage = np.maximum(0.0, phis - config.residual_threshold)
    return stage + np.min(child_values, axis=1)


def fixed_depth_margin(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    target_bloch: np.ndarray,
    config: OperatorNetHorizonConfig,
    depth: int,
) -> dict[str, float | int]:
    current = float(
        terminal_value_batch(
            states[None, :, :],
            strengths,
            control_cache,
            disorder_cache,
            start_index,
            dt,
            candidates,
            target_bloch,
            config,
            depth,
        )[0]
    )
    cache_index = min(start_index, len(control_cache) - 1)
    successors = step_candidates(
        states[None, :, :],
        strengths,
        control_cache[cache_index],
        disorder_cache[cache_index],
        dt,
        candidates,
    )[0]
    successor_values = terminal_value_batch(
        successors,
        strengths,
        control_cache,
        disorder_cache,
        start_index + 1,
        dt,
        candidates,
        target_bloch,
        config,
        depth,
    )
    best_index = int(np.argmin(successor_values))
    best_successor = float(successor_values[best_index])
    phi = float(score_phi(states, target_bloch, config.worst_weight, config.score_mode))
    return {
        "eval_phi": phi,
        "current_value": current,
        "best_fixed_successor_value": best_successor,
        "best_fixed_depth_margin": current - best_successor,
        "best_fallback_index": best_index,
        "best_fallback_norm": float(np.dot(candidates[best_index], candidates[best_index])),
        "scheduled_stage_margin": max(0.0, phi - config.residual_threshold),
    }


def run_design(
    design: SelectedDesign,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    started = time.perf_counter()
    p = problem(design.task)
    target_bloch = density_states_to_bloch((p.target,))[0]
    initial_bloch = density_states_to_bloch((p.initial,))[0]
    train_coeffs = read_training_coefficients(design)
    eval_coeffs, _, eval_h = coefficient_net(args.ball_radius, args.eval_points_per_axis)
    config = OperatorNetHorizonConfig(
        ball_radius=args.ball_radius,
        train_points_per_axis=args.train_points_per_axis,
        eval_points_per_axis=args.eval_points_per_axis,
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        score_mode=design.score_mode,
        worst_weight=design.worst_weight,
    )
    t_eval = np.linspace(0.0, p.t_final, args.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    max_depth = max(args.value_depths)
    cache_times = tuple(
        float(t_eval[min(j, args.segments)])
        for j in range(args.segments + args.horizon_steps + max_depth + 2)
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
    beam_width = default_beam_width(design.task)
    train_strengths = np.ones(len(train_coeffs))
    eval_strengths = np.ones(len(eval_coeffs))
    train_states = np.repeat(initial_bloch[None, :], len(train_coeffs), axis=0)
    eval_states = np.repeat(initial_bloch[None, :], len(eval_coeffs), axis=0)
    previous_sequence: tuple[np.ndarray, ...] | None = None
    rows: list[dict[str, float | int | str]] = []
    shifted_available = 0
    shifted_selected = 0

    for j, _ in enumerate(t_eval[:-1]):
        if previous_sequence is not None and j % args.audit_stride == 0:
            shifted_tail = previous_sequence[1:]
            eval_tail, _ = rollout_sequence(
                eval_states,
                eval_strengths,
                eval_control_cache,
                eval_disorder_cache,
                j,
                dt,
                shifted_tail,
            )
            start_index = min(j + len(shifted_tail), len(eval_control_cache) - 1)
            print(
                f"  {design.task}/{design.method}: fixed-depth audit step "
                f"{j}/{args.segments}",
                flush=True,
            )
            for depth in args.value_depths:
                record = fixed_depth_margin(
                    eval_tail,
                    eval_strengths,
                    eval_control_cache,
                    eval_disorder_cache,
                    start_index,
                    dt,
                    candidates,
                    target_bloch,
                    config,
                    depth,
                )
                rows.append(
                    {
                        "task": design.task,
                        "method": design.method,
                        "step": j,
                        "value_depth": depth,
                        "audit_stride": args.audit_stride,
                        "score_mode": design.score_mode,
                        "worst_weight": design.worst_weight,
                        "train_net_points": len(train_coeffs),
                        "eval_net_points": len(eval_coeffs),
                        "eval_covering_radius_h": eval_h,
                        "candidate_count": len(candidates),
                        "beam_width": beam_width,
                        "residual_threshold": config.residual_threshold,
                        "eval_outside_residual": int(
                            float(record["eval_phi"]) > config.residual_threshold
                        ),
                        "shifted_selected_fraction": (
                            shifted_selected / shifted_available
                            if shifted_available
                            else 0.0
                        ),
                        **record,
                    }
                )

        sequence, selected_shifted, _, _ = select_sequence(
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

    elapsed = time.perf_counter() - started
    for row in rows:
        row["audit_seconds"] = elapsed
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted(
        {
            (str(row["task"]), str(row["method"]), int(row["value_depth"]))
            for row in rows
        }
    )
    for task, method, depth in keys:
        group = [
            row
            for row in rows
            if row["task"] == task
            and row["method"] == method
            and int(row["value_depth"]) == depth
        ]
        margins = np.asarray([float(row["best_fixed_depth_margin"]) for row in group])
        scheduled = np.asarray([float(row["scheduled_stage_margin"]) for row in group])
        outside = np.asarray([int(row["eval_outside_residual"]) for row in group], dtype=bool)
        outside_margins = margins[outside]
        outside_scheduled = scheduled[outside]
        phis = np.asarray([float(row["eval_phi"]) for row in group])
        first = group[0]
        summary.append(
            {
                "task": task,
                "method": method,
                "value_depth": str(depth),
                "audit_stride": str(int(first["audit_stride"])),
                "audited_steps": str(len(group)),
                "eval_outside_steps": str(int(np.sum(outside))),
                "fixed_positive_outside_fraction": (
                    f"{float(np.mean(outside_margins > 0.0)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "fixed_min_outside": (
                    f"{float(np.min(outside_margins)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "fixed_mean": f"{float(np.mean(margins)):.6g}",
                "fixed_min": f"{float(np.min(margins)):.6g}",
                "scheduled_positive_outside_fraction": (
                    f"{float(np.mean(outside_scheduled > 0.0)):.6g}"
                    if outside_scheduled.size
                    else "nan"
                ),
                "scheduled_min_outside": (
                    f"{float(np.min(outside_scheduled)):.6g}"
                    if outside_scheduled.size
                    else "nan"
                ),
                "eval_phi_mean": f"{float(np.mean(phis)):.6g}",
                "eval_phi_max": f"{float(np.max(phis)):.6g}",
                "audit_seconds": f"{float(first['audit_seconds']):.4g}",
            }
        )
    return summary


def write_markdown_table(f, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_outputs(rows: list[dict[str, float | int | str]], output_prefix: str) -> None:
    result_file = result_path(f"{output_prefix}_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = summarize(rows)
    summary_file = result_path(f"{output_prefix}_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Selected Strategy Fixed-Depth Audit\n\n")
        f.write(
            "This audit reruns the selected finite-library Pauli-ball strategies "
            "and checks the stricter fixed-depth comparison on the 2145-point "
            "evaluation net. It is a theorem-upgrade test: negative fixed-depth "
            "rows show that the selected direct/full-net controllers still do "
            "not support a fixed-depth all-time value theorem.\n\n"
        )
        write_markdown_table(f, summary_rows)
    print(f"wrote {result_file}")
    print(f"wrote {summary_file}")
    for row in summary_rows:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--train-points-per-axis", type=int, default=13)
    parser.add_argument("--eval-points-per-axis", type=int, default=13)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--audit-stride", type=int, default=10)
    parser.add_argument("--value-depths", default="1")
    parser.add_argument(
        "--output-prefix",
        default="selected_operator_net_strategy_fixed_depth_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.value_depths = tuple(
        int(item.strip())
        for item in str(args.value_depths).split(",")
        if item.strip()
    )
    rows: list[dict[str, float | int | str]] = []
    for design in SELECTED_DESIGNS:
        rows.extend(run_design(design, args))
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
