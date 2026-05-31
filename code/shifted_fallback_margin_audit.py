"""Audit the shifted-fallback practical-decrease margin.

The shifted-fallback proposition in the manuscript has two parts:

* candidate inclusion, enforced by explicitly scoring shifted-tail sequences;
* terminal progress, a sufficient condition on the appended fallback.

The main shifted-fallback experiment records final held-out fidelity and how
often shifted candidates are selected. This audit recomputes the training
trajectory and measures the best available shifted-fallback margin at each step
after the first one. The margin is positive when at least one appended fallback
would satisfy the proposition's terminal-progress inequality for that step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from horizon_lyapunov import (
    candidate_controls,
    default_beam_width,
    fidelity,
    interaction_frame_operator,
    problem,
    random_disorder,
    scenario_cost,
    step_precomputed,
)
from paths import result_path
from shifted_fallback_horizon import rollout_sequence


@dataclass(frozen=True)
class MarginConfig:
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    segments: int = 100
    horizon_steps: int = 6
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    energy_weight: float = 0.0
    residual_threshold: float = 1e-3


def terminal_phi(p, states: tuple[np.ndarray, ...], worst_weight: float) -> float:
    infids = np.array([1.0 - fidelity(rho, p.target) for rho in states])
    return float(np.mean(infids) + worst_weight * np.max(infids))


def score_sequence(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    sequence: tuple[np.ndarray, ...],
    worst_weight: float,
    energy_weight: float,
) -> tuple[tuple[np.ndarray, ...], float, float, float]:
    states, energy_sum = rollout_sequence(
        rhos,
        strengths,
        controls_cache,
        disorder_cache,
        start_index,
        dt,
        sequence,
    )
    phi = terminal_phi(p, states, worst_weight)
    energy_mean = energy_sum / float(len(sequence))
    cost = scenario_cost(p, states, energy_mean, worst_weight, energy_weight)
    return states, energy_sum, phi, cost


def select_with_margin_record(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    worst_weight: float,
    energy_weight: float,
    previous_sequence: tuple[np.ndarray, ...] | None,
    previous_phi: float | None,
    previous_first_norm: float | None,
) -> tuple[tuple[np.ndarray, ...], float, float, dict[str, float | int | str]]:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_precomputed(
                    states,
                    strengths,
                    controls_cache[cache_index],
                    disorder_cache[cache_index],
                    dt,
                    candidate,
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = scenario_cost(
                    p,
                    next_states,
                    next_energy / float(depth + 1),
                    worst_weight,
                    energy_weight,
                )
                expanded.append((cost, False, sequence + (candidate,), next_states, next_energy))
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, _, sequence, states, energy_sum in expanded[:beam_width]
        ]

    records: list[tuple[float, bool, tuple[np.ndarray, ...], tuple[np.ndarray, ...], float, float]] = []
    for sequence, states, energy_sum in beams:
        phi = terminal_phi(p, states, worst_weight)
        cost = scenario_cost(
            p,
            states,
            energy_sum / float(horizon_steps),
            worst_weight,
            energy_weight,
        )
        records.append((cost, False, sequence, states, energy_sum, phi))

    shifted_margin_best = float("nan")
    shifted_phi_best = float("nan")
    shifted_cost_best = float("nan")
    shifted_available = 0
    if previous_sequence is not None and previous_phi is not None and previous_first_norm is not None:
        shifted_tail = previous_sequence[1:]
        shifted_margins = []
        for fallback in candidates:
            shifted_sequence = shifted_tail + (fallback,)
            states, energy_sum, phi, cost = score_sequence(
                p,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                start_index,
                dt,
                shifted_sequence,
                worst_weight,
                energy_weight,
            )
            fallback_norm = float(np.dot(fallback, fallback))
            margin = previous_phi - phi - energy_weight * (fallback_norm - previous_first_norm)
            shifted_margins.append((margin, phi, cost))
            records.append((cost, True, shifted_sequence, states, energy_sum, phi))
        shifted_available = 1
        shifted_margin_best, shifted_phi_best, shifted_cost_best = max(
            shifted_margins,
            key=lambda item: item[0],
        )

    records.sort(key=lambda item: item[0])
    selected_cost, selected_shifted, selected_sequence, _, selected_energy, selected_phi = records[0]
    row = {
        "step": start_index,
        "selected_shifted": int(selected_shifted),
        "selected_cost": selected_cost,
        "selected_phi": selected_phi,
        "selected_energy_mean": selected_energy / float(horizon_steps),
        "shifted_available": shifted_available,
        "best_shifted_margin": shifted_margin_best,
        "best_shifted_phi": shifted_phi_best,
        "best_shifted_cost": shifted_cost_best,
    }
    return selected_sequence, selected_phi, float(np.dot(selected_sequence[0], selected_sequence[0])), row


def audit_task(task: str, config: MarginConfig) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in config.train_strengths
        for seed in config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(config.amplitudes)
    cache_times = tuple(
        float(t_eval[min(j, config.segments)])
        for j in range(config.segments + config.horizon_steps)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )

    previous_sequence: tuple[np.ndarray, ...] | None = None
    previous_phi: float | None = None
    previous_first_norm: float | None = None
    beam_width = default_beam_width(task)
    rows: list[dict[str, float | int | str]] = []

    for j, _ in enumerate(t_eval[:-1]):
        sequence, phi, first_norm, row = select_with_margin_record(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            j,
            dt,
            candidates,
            config.horizon_steps,
            beam_width,
            config.worst_weight,
            config.energy_weight,
            previous_sequence,
            previous_phi,
            previous_first_norm,
        )
        control = sequence[0]
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[j],
            disorder_cache[j],
            dt,
            control,
        )
        row.update(
            {
                "task": task,
                "beam_width": beam_width,
                "residual_threshold": config.residual_threshold,
                "outside_residual": int(phi > config.residual_threshold),
            }
        )
        rows.append(row)
        previous_sequence = sequence
        previous_phi = phi
        previous_first_norm = first_norm

    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task and int(row["shifted_available"]) == 1]
        margins = np.array([float(row["best_shifted_margin"]) for row in task_rows])
        outside = np.array([int(row["outside_residual"]) for row in task_rows], dtype=bool)
        outside_margins = margins[outside]
        positive = margins > 0.0
        positive_outside = outside_margins > 0.0
        summary.append(
            {
                "task": task,
                "audited_steps": str(len(task_rows)),
                "positive_margin_fraction": f"{float(np.mean(positive)):.6g}",
                "outside_residual_steps": str(int(np.sum(outside))),
                "positive_outside_residual_fraction": (
                    f"{float(np.mean(positive_outside)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "margin_mean": f"{float(np.mean(margins)):.6g}",
                "margin_median": f"{float(np.median(margins)):.6g}",
                "margin_min": f"{float(np.min(margins)):.6g}",
                "margin_p10": f"{float(np.quantile(margins, 0.1)):.6g}",
                "margin_max": f"{float(np.max(margins)):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("shifted_fallback_margin_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("shifted_fallback_margin_audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Shifted-Fallback Margin Audit Summary\n\n")
        f.write(
            "Margins are computed on the training ensemble along the implemented "
            "shifted-fallback trajectory. Positive margin means that at least one "
            "appended fallback satisfies the proposition's terminal-progress "
            "inequality at that step.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    config = MarginConfig()
    rows = []
    for task in ("Z", "H"):
        print(f"auditing shifted-fallback margins for {task}", flush=True)
        rows.extend(audit_task(task, config))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
