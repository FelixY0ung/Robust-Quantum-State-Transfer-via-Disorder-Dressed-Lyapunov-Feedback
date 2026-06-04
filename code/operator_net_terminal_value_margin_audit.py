"""Audit scheduled terminal-value margins on the Pauli operator net.

The terminal-value shifted-horizon controller is designed on the same finite
training ensemble used in the main two-level experiments.  The operator-space
finite-net audit then evaluates the resulting pulse on an explicit Pauli
coefficient h-net.  This script connects those two checks: it regenerates the
terminal-value shifted-horizon decisions, propagates the full Pauli-ball net in
parallel, and records the scheduled Bellman margin on the shifted-tail terminal
states over that net.

The default rows use the Bellman identity for

    G_{j,1}(z) = [Phi(z)-tau]_+ + min_k Phi(F_{k,j}(z)).

For the energy-free certificate score, the best scheduled margin equals
[Phi(z)-tau]_+.  The existing terminal-value audit verifies this identity
numerically on the controller's training ensemble; this audit reports the same
certificate quantity for the deterministic operator net.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from finite_net_operator_audit import coefficient_net, pauli_disorder
from horizon_lyapunov import (
    candidate_controls,
    default_beam_width,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
    step_precomputed,
)
from paths import result_path
from shifted_fallback_horizon import DesignStats, rollout_sequence
from terminal_value_shifted_horizon import (
    TerminalValueHorizonConfig,
    select_terminal_value_sequence,
)


def infidelity_stats(p, states: tuple[np.ndarray, ...], worst_weight: float) -> dict[str, float]:
    infids = np.array([1.0 - fidelity(rho, p.target) for rho in states])
    mean_infidelity = float(np.mean(infids))
    worst_infidelity = float(np.max(infids))
    return {
        "mean_infidelity": mean_infidelity,
        "worst_infidelity": worst_infidelity,
        "mean_worst_score": mean_infidelity + worst_weight * worst_infidelity,
        "max_score": worst_infidelity,
        "mean_fidelity": 1.0 - mean_infidelity,
        "worst_fidelity": 1.0 - worst_infidelity,
    }


def score_rows(
    task: str,
    step: int,
    shifted_tail_length: int,
    stats: dict[str, float],
    net_points: int,
    ball_radius: float,
    points_per_axis: int,
    grid_spacing: float,
    covering_radius: float,
    config: TerminalValueHorizonConfig,
    beam_width: int,
    shifted_selected_fraction: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    score_modes = (
        ("mean_worst", "mean_worst_score"),
        ("max_net", "max_score"),
    )
    for mode, key in score_modes:
        phi = float(stats[key])
        stage = max(0.0, phi - config.residual_threshold)
        rows.append(
            {
                "task": task,
                "step": step,
                "score_mode": mode,
                "value_depth": config.value_depth,
                "net_points": net_points,
                "ball_radius": ball_radius,
                "points_per_axis": points_per_axis,
                "grid_spacing": grid_spacing,
                "covering_radius": covering_radius,
                "beam_width": beam_width,
                "shifted_tail_length": shifted_tail_length,
                "residual_threshold": config.residual_threshold,
                "terminal_outside_residual": int(phi > config.residual_threshold),
                "phi": phi,
                "best_value_margin": stage,
                "state_stage_cost": stage,
                "bellman_gap": 0.0,
                "mean_infidelity": stats["mean_infidelity"],
                "worst_infidelity": stats["worst_infidelity"],
                "mean_fidelity": stats["mean_fidelity"],
                "worst_fidelity": stats["worst_fidelity"],
                "shifted_selected_fraction": shifted_selected_fraction,
                "margin_source": "bellman_identity",
            }
        )
    return rows


def audit_task(
    task: str,
    coeffs: np.ndarray,
    grid_spacing: float,
    covering_radius: float,
    args: argparse.Namespace,
    config: TerminalValueHorizonConfig,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    started = time.perf_counter()
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    train_strengths = tuple(strength for strength, _ in scenarios)
    train_rhos = tuple(p.initial.copy() for _ in scenarios)
    net_strengths = tuple(1.0 for _ in coeffs)
    net_rhos = tuple(p.initial.copy() for _ in coeffs)
    net_disorders = tuple(pauli_disorder(p, coeff) for coeff in coeffs)
    candidates = candidate_controls(config.amplitudes)
    beam_width = default_beam_width(task)
    cache_times = tuple(
        float(t_eval[min(j, config.segments)])
        for j in range(config.segments + config.horizon_steps + config.value_depth + 1)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    train_disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )
    net_disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for disorder in net_disorders)
        for t in cache_times
    )

    rows: list[dict[str, float | int | str]] = []
    previous_sequence: tuple[np.ndarray, ...] | None = None
    shifted_available_steps = 0
    shifted_selected_steps = 0

    for j, _ in enumerate(t_eval[:-1]):
        if previous_sequence is not None:
            shifted_tail = previous_sequence[1:]
            net_terminal_states, _ = rollout_sequence(
                net_rhos,
                net_strengths,
                controls_cache,
                net_disorder_cache,
                j,
                dt,
                shifted_tail,
            )
            net_stats = infidelity_stats(p, net_terminal_states, config.worst_weight)
            fraction = (
                shifted_selected_steps / shifted_available_steps
                if shifted_available_steps
                else 0.0
            )
            rows.extend(
                score_rows(
                    task,
                    j,
                    len(shifted_tail),
                    net_stats,
                    len(coeffs),
                    args.ball_radius,
                    args.points_per_axis,
                    grid_spacing,
                    covering_radius,
                    config,
                    beam_width,
                    fraction,
                )
            )

        sequence, selected_shifted, selected_phi, selected_cost = select_terminal_value_sequence(
            p,
            train_rhos,
            train_strengths,
            controls_cache,
            train_disorder_cache,
            j,
            dt,
            candidates,
            beam_width,
            config,
            previous_sequence,
        )
        if previous_sequence is not None:
            shifted_available_steps += 1
            if selected_shifted:
                shifted_selected_steps += 1
        control = sequence[0]
        train_rhos = step_precomputed(
            train_rhos,
            train_strengths,
            controls_cache[j],
            train_disorder_cache[j],
            dt,
            control,
        )
        net_rhos = step_precomputed(
            net_rhos,
            net_strengths,
            controls_cache[j],
            net_disorder_cache[j],
            dt,
            control,
        )
        previous_sequence = sequence

        if j % 20 == 0:
            print(
                f"  {task}: operator-net margin step {j}/{config.segments} "
                f"training_phi={selected_phi:.6g} cost={selected_cost:.6g}",
                flush=True,
            )

    final_stats = infidelity_stats(p, net_rhos, config.worst_weight)
    elapsed = time.perf_counter() - started
    final_record: dict[str, float | int | str] = {
        "task": task,
        "net_points": len(coeffs),
        "ball_radius": args.ball_radius,
        "points_per_axis": args.points_per_axis,
        "grid_spacing": grid_spacing,
        "covering_radius": covering_radius,
        "shifted_available_steps": shifted_available_steps,
        "shifted_selected_steps": shifted_selected_steps,
        "shifted_selected_fraction": DesignStats(
            shifted_available_steps=shifted_available_steps,
            shifted_selected_steps=shifted_selected_steps,
        ).shifted_selected_fraction,
        "audit_seconds": elapsed,
        **final_stats,
    }
    return rows, final_record


def summarize(
    rows: list[dict[str, float | int | str]],
    final_records: list[dict[str, float | int | str]],
) -> list[dict[str, str]]:
    finals = {str(row["task"]): row for row in final_records}
    summary: list[dict[str, str]] = []
    keys = sorted({(str(row["task"]), str(row["score_mode"])) for row in rows})
    for task, mode in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and str(row["score_mode"]) == mode
        ]
        margins = np.array([float(row["best_value_margin"]) for row in group])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in group],
            dtype=bool,
        )
        outside_margins = margins[outside]
        phis = np.array([float(row["phi"]) for row in group])
        final = finals[task]
        summary.append(
            {
                "task": task,
                "score_mode": mode,
                "net_points": str(int(final["net_points"])),
                "ball_radius": f"{float(final['ball_radius']):.6g}",
                "points_per_axis": str(int(final["points_per_axis"])),
                "covering_radius_h": f"{float(final['covering_radius']):.6g}",
                "audited_steps": str(len(group)),
                "terminal_outside_steps": str(int(np.sum(outside))),
                "positive_terminal_outside_fraction": (
                    f"{float(np.mean(outside_margins > 0.0)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "certified_epsilon_min_outside": (
                    f"{float(np.min(outside_margins)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "margin_mean": f"{float(np.mean(margins)):.6g}",
                "margin_min": f"{float(np.min(margins)):.6g}",
                "terminal_score_mean": f"{float(np.mean(phis)):.6g}",
                "terminal_score_min": f"{float(np.min(phis)):.6g}",
                "final_worst_net_fidelity": f"{float(final['worst_fidelity']):.6g}",
                "final_mean_net_fidelity": f"{float(final['mean_fidelity']):.6g}",
                "shifted_selected_fraction": (
                    f"{float(final['shifted_selected_fraction']):.6g}"
                ),
                "audit_seconds": f"{float(final['audit_seconds']):.4g}",
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
    rows: list[dict[str, float | int | str]],
    final_records: list[dict[str, float | int | str]],
) -> None:
    with result_path("operator_net_terminal_value_margin_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with result_path("operator_net_terminal_value_margin_final_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(final_records[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(final_records)

    summary = summarize(rows, final_records)
    with result_path("operator_net_terminal_value_margin_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Operator-Net Terminal-Value Margin Audit\n\n")
        f.write(
            "This audit propagates the explicit Pauli-coefficient h-net in "
            "parallel with the terminal-value shifted-horizon controller and "
            "reports the scheduled Bellman margin on the shifted-tail terminal "
            "states. The margin is the energy-free terminal-value identity "
            "`[Phi-tau]_+`; the training-ensemble terminal-value audit verifies "
            "the same identity by explicit successor scoring. The `max_net` "
            "score mode uses worst net infidelity as Phi, while `mean_worst` "
            "uses the controller-style mean-plus-worst score.\n\n"
        )
        write_markdown_table(f, summary)

    print(f"wrote {len(rows)} rows to {result_path('operator_net_terminal_value_margin_results.csv')}")
    print(
        "wrote final net results to "
        f"{result_path('operator_net_terminal_value_margin_final_results.csv')}"
    )
    print(f"wrote summary to {result_path('operator_net_terminal_value_margin_summary.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Z,H")
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--points-per-axis", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TerminalValueHorizonConfig()
    coeffs, grid_spacing, covering_radius = coefficient_net(
        args.ball_radius,
        args.points_per_axis,
    )
    rows: list[dict[str, float | int | str]] = []
    final_records: list[dict[str, float | int | str]] = []
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        print(
            f"auditing operator-net terminal-value margins for {task}; "
            f"net points={len(coeffs)} h={covering_radius:.6g}",
            flush=True,
        )
        task_rows, final_record = audit_task(
            task,
            coeffs,
            grid_spacing,
            covering_radius,
            args,
            config,
        )
        rows.extend(task_rows)
        final_records.append(final_record)
    write_outputs(rows, final_records)
    for row in summarize(rows, final_records):
        print(row)


if __name__ == "__main__":
    main()
