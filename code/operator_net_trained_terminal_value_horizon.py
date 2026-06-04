"""Direct operator-net trained terminal-value horizon diagnostic.

The main terminal-value shifted-horizon controller is trained on 16 sampled
disorder scenarios and then audited on the explicit Pauli-ball operator net.
This script tests the next strengthening route: use an explicit Pauli
coefficient net directly inside the receding-horizon score, while keeping the
same bounded finite-candidate shifted-fallback architecture.

The implementation uses the exact two-level Bloch-vector form of the same
Hamiltonian RK4 dynamics.  It is therefore a diagnostic for the two-level Pauli
benchmarks, not a replacement for the density-matrix code used in the general
model sections.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import numpy as np

from finite_net_operator_audit import coefficient_net, pauli_disorder
from fixed_depth_terminal_value_margin_audit import (
    batch_step_bloch,
    density_states_to_bloch,
    pauli_coefficients,
)
from horizon_lyapunov import (
    candidate_controls,
    default_beam_width,
    interaction_frame_operator,
    problem,
    random_disorder,
)
from paths import result_path
from shifted_fallback_horizon import DesignStats


@dataclass(frozen=True)
class OperatorNetHorizonConfig:
    ball_radius: float = 0.08
    train_points_per_axis: int = 7
    eval_points_per_axis: int = 13
    segments: int = 100
    horizon_steps: int = 6
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    residual_threshold: float = 1e-3
    energy_weight: float = 0.0
    score_mode: str = "max_net"


def score_phi(
    states: np.ndarray,
    target_bloch: np.ndarray,
    worst_weight: float,
    score_mode: str,
) -> np.ndarray:
    fidelities = 0.5 * (1.0 + np.einsum("...j,j->...", states, target_bloch))
    infidelities = 1.0 - fidelities
    mean_infidelity = np.mean(infidelities, axis=-1)
    worst_infidelity = np.max(infidelities, axis=-1)
    if score_mode == "mean_worst":
        return mean_infidelity + worst_weight * worst_infidelity
    if score_mode == "max_net":
        return worst_infidelity
    raise ValueError(score_mode)


def fidelity_stats(states: np.ndarray, target_bloch: np.ndarray) -> dict[str, float]:
    fidelities = 0.5 * (1.0 + states @ target_bloch)
    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "worst_fidelity": float(np.min(fidelities)),
        "std_fidelity": float(np.std(fidelities)),
        "mean_infidelity": float(np.mean(1.0 - fidelities)),
        "worst_infidelity": float(np.max(1.0 - fidelities)),
    }


def interaction_coeff_cache(p, coeffs: np.ndarray, times: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    control_cache = []
    disorder_cache = []
    disorders = tuple(pauli_disorder(p, coeff) for coeff in coeffs)
    for t in times:
        control_cache.append(
            np.asarray(
                [
                    pauli_coefficients(interaction_frame_operator(p, hc, t))
                    for hc in p.controls
                ]
            )
        )
        disorder_cache.append(
            np.asarray(
                [
                    pauli_coefficients(interaction_frame_operator(p, disorder, t))
                    for disorder in disorders
                ]
            )
        )
    return np.asarray(control_cache), np.asarray(disorder_cache)


def step_candidates(
    states: np.ndarray,
    strengths: np.ndarray,
    control_coeffs: np.ndarray,
    disorder_coeffs: np.ndarray,
    dt: float,
    candidates: tuple[np.ndarray, ...],
) -> np.ndarray:
    control_array = np.asarray(candidates, dtype=float)
    control_hamiltonians = np.einsum("mc,cj->mj", control_array, control_coeffs)
    hamiltonians = (
        control_hamiltonians[:, None, :]
        + strengths[None, :, None] * disorder_coeffs[None, :, :]
    )
    return batch_step_bloch(states, hamiltonians, dt)


def apply_control(
    states: np.ndarray,
    strengths: np.ndarray,
    control_coeffs: np.ndarray,
    disorder_coeffs: np.ndarray,
    dt: float,
    control: np.ndarray,
) -> np.ndarray:
    return step_candidates(
        states[None, :, :],
        strengths,
        control_coeffs,
        disorder_coeffs,
        dt,
        (control,),
    )[0, 0]


def rollout_sequence(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    sequence: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, float]:
    out = states
    energy = 0.0
    for offset, control in enumerate(sequence):
        cache_index = min(start_index + offset, len(control_cache) - 1)
        out = apply_control(
            out,
            strengths,
            control_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            control,
        )
        energy += float(np.dot(control, control))
    return out, energy


def terminal_value_score(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    target_bloch: np.ndarray,
    config: OperatorNetHorizonConfig,
) -> tuple[float, float]:
    phi = float(
        score_phi(states, target_bloch, config.worst_weight, config.score_mode)
    )
    cache_index = min(start_index, len(control_cache) - 1)
    children = step_candidates(
        states[None, :, :],
        strengths,
        control_cache[cache_index],
        disorder_cache[cache_index],
        dt,
        candidates,
    )[0]
    child_phi = score_phi(
        children,
        target_bloch,
        config.worst_weight,
        config.score_mode,
    )
    value = max(0.0, phi - config.residual_threshold) + float(np.min(child_phi))
    return phi, value


def sequence_record(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    sequence: tuple[np.ndarray, ...],
    candidates: tuple[np.ndarray, ...],
    target_bloch: np.ndarray,
    config: OperatorNetHorizonConfig,
) -> tuple[float, tuple[np.ndarray, ...], np.ndarray, float, float]:
    terminal_states, energy = rollout_sequence(
        states,
        strengths,
        control_cache,
        disorder_cache,
        start_index,
        dt,
        sequence,
    )
    phi, value = terminal_value_score(
        terminal_states,
        strengths,
        control_cache,
        disorder_cache,
        start_index + len(sequence),
        dt,
        candidates,
        target_bloch,
        config,
    )
    cost = value + config.energy_weight * energy / float(len(sequence))
    return cost, sequence, terminal_states, phi, value


def select_sequence(
    states: np.ndarray,
    strengths: np.ndarray,
    control_cache: np.ndarray,
    disorder_cache: np.ndarray,
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    beam_width: int,
    target_bloch: np.ndarray,
    config: OperatorNetHorizonConfig,
    previous_sequence: tuple[np.ndarray, ...] | None,
) -> tuple[tuple[np.ndarray, ...], bool, float, float]:
    beams: list[tuple[tuple[np.ndarray, ...], np.ndarray, float]] = [((), states, 0.0)]
    candidate_energy = np.asarray([float(np.dot(c, c)) for c in candidates])
    for depth in range(config.horizon_steps):
        beam_states = np.asarray([item[1] for item in beams])
        beam_energies = np.asarray([item[2] for item in beams])
        cache_index = min(start_index + depth, len(control_cache) - 1)
        children = step_candidates(
            beam_states,
            strengths,
            control_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            candidates,
        )
        flat_children = children.reshape(-1, states.shape[0], 3)
        phis = score_phi(
            flat_children,
            target_bloch,
            config.worst_weight,
            config.score_mode,
        )
        energies = (
            beam_energies[:, None] + candidate_energy[None, :]
        ).reshape(-1)
        costs = phis + config.energy_weight * energies / float(depth + 1)
        keep = np.argsort(costs)[:beam_width]
        new_beams: list[tuple[tuple[np.ndarray, ...], np.ndarray, float]] = []
        for index in keep:
            beam_index = int(index // len(candidates))
            candidate_index = int(index % len(candidates))
            new_beams.append(
                (
                    beams[beam_index][0] + (candidates[candidate_index],),
                    flat_children[index],
                    float(energies[index]),
                )
            )
        beams = new_beams

    records: list[tuple[float, bool, tuple[np.ndarray, ...], float]] = []
    for sequence, _, _ in beams:
        cost, _, _, phi, _ = sequence_record(
            states,
            strengths,
            control_cache,
            disorder_cache,
            start_index,
            dt,
            sequence,
            candidates,
            target_bloch,
            config,
        )
        records.append((cost, False, sequence, phi))

    if previous_sequence is not None:
        shifted_tail = previous_sequence[1:]
        for fallback in candidates:
            shifted_sequence = shifted_tail + (fallback,)
            cost, _, _, phi, _ = sequence_record(
                states,
                strengths,
                control_cache,
                disorder_cache,
                start_index,
                dt,
                shifted_sequence,
                candidates,
                target_bloch,
                config,
            )
            records.append((cost, True, shifted_sequence, phi))

    records.sort(key=lambda item: item[0])
    cost, shifted, sequence, phi = records[0]
    return sequence, shifted, phi, cost


def heldout_coefficients(task: str, strength: float, seeds: range) -> np.ndarray:
    p = problem(task)
    return np.asarray(
        [strength * pauli_coefficients(random_disorder(seed)) for seed in seeds]
    )


def evaluate_pulse(
    task: str,
    pulse: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    times = tuple(float(t) for t in t_eval[:-1])
    control_cache, disorder_cache = interaction_coeff_cache(p, coeffs, times)
    strengths = np.ones(len(coeffs))
    states = np.repeat(density_states_to_bloch((p.initial,)), len(coeffs), axis=0)
    for j, control in enumerate(pulse):
        states = apply_control(
            states,
            strengths,
            control_cache[j],
            disorder_cache[j],
            dt,
            control,
        )
    return states


def run_task(
    task: str,
    config: OperatorNetHorizonConfig,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    started = time.perf_counter()
    p = problem(task)
    target_bloch = density_states_to_bloch((p.target,))[0]
    initial_bloch = density_states_to_bloch((p.initial,))[0]
    train_coeffs, train_spacing, train_h = coefficient_net(
        config.ball_radius,
        config.train_points_per_axis,
    )
    eval_coeffs, eval_spacing, eval_h = coefficient_net(
        config.ball_radius,
        config.eval_points_per_axis,
    )
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
                    "step": j,
                    "score_mode": config.score_mode,
                    "train_net_points": len(train_coeffs),
                    "eval_net_points": len(eval_coeffs),
                    "train_points_per_axis": config.train_points_per_axis,
                    "eval_points_per_axis": config.eval_points_per_axis,
                    "train_covering_radius": train_h,
                    "eval_covering_radius": eval_h,
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
                f"  {task}: operator-net trained step {j}/{config.segments} "
                f"phi={selected_phi:.6g} cost={selected_cost:.6g}",
                flush=True,
            )

    pulse_array = np.asarray(pulse)
    design_seconds = time.perf_counter() - started
    stats = DesignStats(
        shifted_available_steps=shifted_available,
        shifted_selected_steps=shifted_selected,
    )
    net_rows: list[dict[str, float | int | str]] = []
    for label, coeffs, states, spacing, radius_h, points in (
        ("train_net", train_coeffs, train_states, train_spacing, train_h, config.train_points_per_axis),
        ("eval_net", eval_coeffs, eval_states, eval_spacing, eval_h, config.eval_points_per_axis),
    ):
        fidelities = 0.5 * (1.0 + states @ target_bloch)
        for index, (coeff, fidelity) in enumerate(zip(coeffs, fidelities)):
            net_rows.append(
                {
                    "task": task,
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
                    "final_fidelity": float(fidelity),
                    "final_infidelity": float(1.0 - fidelity),
                    "pulse_energy": float(np.mean(np.sum(pulse_array * pulse_array, axis=1))),
                    "shifted_selected_fraction": stats.shifted_selected_fraction,
                    "design_seconds": design_seconds,
                }
            )

    heldout_rows: list[dict[str, float | int | str]] = []
    for strength in (0.0, 0.02, 0.05, 0.08):
        coeffs = heldout_coefficients(task, strength, range(10, 60))
        heldout_states = evaluate_pulse(task, pulse_array, coeffs)
        fidelities = 0.5 * (1.0 + heldout_states @ target_bloch)
        for seed, fidelity in zip(range(10, 60), fidelities):
            heldout_rows.append(
                {
                    "task": task,
                    "disorder_strength": strength,
                    "seed": seed,
                    "score_mode": config.score_mode,
                    "train_points_per_axis": config.train_points_per_axis,
                    "train_net_points": len(train_coeffs),
                    "eval_points_per_axis": config.eval_points_per_axis,
                    "eval_net_points": len(eval_coeffs),
                    "final_fidelity": float(fidelity),
                    "pulse_energy": float(np.mean(np.sum(pulse_array * pulse_array, axis=1))),
                    "shifted_selected_fraction": stats.shifted_selected_fraction,
                    "design_seconds": design_seconds,
                }
            )

    return heldout_rows, net_rows, margin_rows


def summarize_heldout(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted({(str(row["task"]), float(row["disorder_strength"])) for row in rows})
    for task, strength in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and float(row["disorder_strength"]) == strength
        ]
        fidelities = np.asarray([float(row["final_fidelity"]) for row in group])
        first = group[0]
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(group)),
                "score_mode": str(first["score_mode"]),
                "train_net_points": str(int(first["train_net_points"])),
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
    keys = sorted({(str(row["task"]), str(row["net_label"])) for row in rows})
    for task, label in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and str(row["net_label"]) == label
        ]
        fidelities = np.asarray([float(row["final_fidelity"]) for row in group])
        first = group[0]
        summary.append(
            {
                "task": task,
                "net_label": label,
                "score_mode": str(first["score_mode"]),
                "net_points": str(len(group)),
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
    for task in sorted({str(row["task"]) for row in rows}):
        group = [row for row in rows if str(row["task"]) == task]
        eval_margins = np.asarray([float(row["eval_scheduled_margin"]) for row in group])
        train_margins = np.asarray([float(row["train_scheduled_margin"]) for row in group])
        eval_outside = np.asarray([int(row["eval_outside_residual"]) for row in group], dtype=bool)
        train_outside = np.asarray([int(row["train_outside_residual"]) for row in group], dtype=bool)
        first = group[0]
        summary.append(
            {
                "task": task,
                "score_mode": str(first["score_mode"]),
                "train_net_points": str(int(first["train_net_points"])),
                "eval_net_points": str(int(first["eval_net_points"])),
                "audited_steps": str(len(group)),
                "train_outside_steps": str(int(np.sum(train_outside))),
                "eval_outside_steps": str(int(np.sum(eval_outside))),
                "train_positive_outside_fraction": f"{float(np.mean(train_margins[train_outside] > 0.0)):.6g}",
                "eval_positive_outside_fraction": f"{float(np.mean(eval_margins[eval_outside] > 0.0)):.6g}",
                "train_min_outside_margin": f"{float(np.min(train_margins[train_outside])):.6g}",
                "eval_min_outside_margin": f"{float(np.min(eval_margins[eval_outside])):.6g}",
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

    with result_path(f"{output_prefix}_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Operator-Net Trained Terminal-Value Horizon Summary\n\n")
        f.write(
            "This diagnostic trains the terminal-value shifted-fallback horizon "
            "directly on an explicit Pauli coefficient net using two-level "
            "Bloch-vector propagation, then evaluates the resulting pulse on "
            "the standard held-out seeds and on a denser operator net. It tests "
            "the direct-net training route; it is not a new fixed-depth theorem.\n\n"
        )
        f.write("## Held-Out Performance\n\n")
        write_markdown_table(f, summarize_heldout(heldout_rows))
        f.write("\n## Operator-Net Performance\n\n")
        write_markdown_table(f, summarize_net(net_rows))
        f.write("\n## Scheduled Margin Audit\n\n")
        write_markdown_table(f, summarize_margins(margin_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Z,H")
    parser.add_argument("--ball-radius", type=float, default=0.08)
    parser.add_argument("--train-points-per-axis", type=int, default=7)
    parser.add_argument("--eval-points-per-axis", type=int, default=13)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--score-mode", choices=("max_net", "mean_worst"), default="max_net")
    parser.add_argument("--output-prefix", default="operator_net_trained_terminal_value_horizon")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OperatorNetHorizonConfig(
        ball_radius=args.ball_radius,
        train_points_per_axis=args.train_points_per_axis,
        eval_points_per_axis=args.eval_points_per_axis,
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        score_mode=args.score_mode,
    )
    heldout_rows: list[dict[str, float | int | str]] = []
    net_rows: list[dict[str, float | int | str]] = []
    margin_rows: list[dict[str, float | int | str]] = []
    for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
        print(
            f"designing operator-net trained terminal-value horizon for {task}",
            flush=True,
        )
        task_heldout, task_net, task_margins = run_task(task, config)
        heldout_rows.extend(task_heldout)
        net_rows.extend(task_net)
        margin_rows.extend(task_margins)
    write_outputs(heldout_rows, net_rows, margin_rows, args.output_prefix)
    for row in summarize_heldout(heldout_rows):
        print(row)
    for row in summarize_net(net_rows):
        print(row)
    for row in summarize_margins(margin_rows):
        print(row)


if __name__ == "__main__":
    main()
