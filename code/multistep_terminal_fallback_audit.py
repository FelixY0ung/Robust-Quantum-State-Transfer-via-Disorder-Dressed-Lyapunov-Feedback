"""Audit multi-step terminal fallbacks for the shifted-fallback certificate.

The one-segment terminal-fallback and adaptive line-search audits show that
local one-step descent rarely activates the practical-decrease margin along the
realized shifted-fallback trajectory.  This script tests the next certificate
design route: whether the previous predicted terminal ensemble state admits a
short bounded fallback block that decreases the terminal Lyapunov score.

The realized shifted-fallback trajectory is kept fixed.  Multi-step fallback
blocks are scored only as terminal-region diagnostics; they are not used to
replace the controller or to claim a one-step shifted-fallback theorem.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from horizon_lyapunov import (
    candidate_controls,
    default_beam_width,
    interaction_frame_operator,
    problem,
    random_disorder,
    step_precomputed,
)
from paths import result_path
from shifted_fallback_horizon import rollout_sequence
from shifted_fallback_margin_audit import (
    MarginConfig,
    select_with_margin_record,
    terminal_phi,
)


@dataclass(frozen=True)
class FallbackBlockSpec:
    name: str
    block_steps: int
    beam_width: int


@dataclass(frozen=True)
class MultiStepConfig:
    specs: tuple[FallbackBlockSpec, ...] = (
        FallbackBlockSpec("one_step_exhaustive", 1, 41),
        FallbackBlockSpec("two_step_beam8", 2, 8),
        FallbackBlockSpec("three_step_beam8", 3, 8),
    )
    audit_stride: int = 2


def best_terminal_block(
    p,
    terminal_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    block_steps: int,
    beam_width: int,
    worst_weight: float,
    energy_weight: float,
) -> tuple[tuple[np.ndarray, ...], float, float, int]:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), terminal_states, 0.0)
    ]
    expanded_count = 0
    for depth in range(block_steps):
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
                phi = terminal_phi(p, next_states, worst_weight)
                cost = phi + energy_weight * next_energy / float(depth + 1)
                expanded.append(
                    (cost, sequence + (candidate,), next_states, next_energy, phi)
                )
        expanded_count += len(expanded)
        expanded.sort(key=lambda item: item[0])
        keep = min(beam_width, len(expanded))
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum, _ in expanded[:keep]
        ]

    scored = []
    for sequence, states, energy_sum in beams:
        phi = terminal_phi(p, states, worst_weight)
        cost = phi + energy_weight * energy_sum / float(block_steps)
        scored.append((cost, sequence, phi, energy_sum))
    scored.sort(key=lambda item: item[0])
    _, sequence, phi, energy_sum = scored[0]
    return sequence, phi, energy_sum, expanded_count


def audit_task(
    task: str,
    margin_config: MarginConfig,
    multistep_config: MultiStepConfig,
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, margin_config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in margin_config.train_strengths
        for seed in margin_config.train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(margin_config.amplitudes)
    beam_width = default_beam_width(task)
    max_block = max(spec.block_steps for spec in multistep_config.specs)
    cache_times = tuple(
        float(t_eval[min(j, margin_config.segments)])
        for j in range(
            margin_config.segments + margin_config.horizon_steps + max_block
        )
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
    rows: list[dict[str, float | int | str]] = []

    for j, _ in enumerate(t_eval[:-1]):
        if j and j % 20 == 0:
            print(f"  {task}: processed step {j}/{margin_config.segments}", flush=True)
        if (
            previous_sequence is not None
            and previous_phi is not None
            and previous_first_norm is not None
            and j % multistep_config.audit_stride == 0
        ):
            shifted_tail = previous_sequence[1:]
            terminal_states, _ = rollout_sequence(
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                j,
                dt,
                shifted_tail,
            )
            tail_phi = terminal_phi(p, terminal_states, margin_config.worst_weight)
            block_start = min(j + len(shifted_tail), len(controls_cache) - 1)
            for spec in multistep_config.specs:
                sequence, phi, energy_sum, expanded_count = best_terminal_block(
                    p,
                    terminal_states,
                    strengths,
                    controls_cache,
                    disorder_cache,
                    block_start,
                    dt,
                    candidates,
                    spec.block_steps,
                    spec.beam_width,
                    margin_config.worst_weight,
                    margin_config.energy_weight,
                )
                raw_margin = previous_phi - phi
                energy_mean = energy_sum / float(spec.block_steps)
                energy_adjusted_margin = raw_margin - margin_config.energy_weight * (
                    energy_mean - previous_first_norm
                )
                rows.append(
                    {
                        "task": task,
                        "step": j,
                        "fallback_spec": spec.name,
                        "block_steps": spec.block_steps,
                        "beam_width": spec.beam_width,
                        "candidate_count": len(candidates),
                        "expanded_count": expanded_count,
                        "previous_terminal_phi": previous_phi,
                        "shift_tail_phi": tail_phi,
                        "tail_phi_abs_error": abs(previous_phi - tail_phi),
                        "best_block_phi": phi,
                        "raw_terminal_margin": raw_margin,
                        "energy_adjusted_margin": energy_adjusted_margin,
                        "best_block_energy_mean": energy_mean,
                        "best_block_first_norm": float(np.dot(sequence[0], sequence[0])),
                        "terminal_outside_residual": int(
                            previous_phi > margin_config.residual_threshold
                        ),
                        "residual_threshold": margin_config.residual_threshold,
                    }
                )

        sequence, phi, first_norm, _ = select_with_margin_record(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            j,
            dt,
            candidates,
            margin_config.horizon_steps,
            beam_width,
            margin_config.worst_weight,
            margin_config.energy_weight,
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
        previous_sequence = sequence
        previous_phi = phi
        previous_first_norm = first_norm

    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary = []
    keys = sorted({(str(row["task"]), str(row["fallback_spec"])) for row in rows})
    for task, spec in keys:
        spec_rows = [
            row
            for row in rows
            if str(row["task"]) == task and str(row["fallback_spec"]) == spec
        ]
        margins = np.array([float(row["energy_adjusted_margin"]) for row in spec_rows])
        raw_margins = np.array([float(row["raw_terminal_margin"]) for row in spec_rows])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in spec_rows],
            dtype=bool,
        )
        outside_margins = margins[outside]
        tail_errors = np.array([float(row["tail_phi_abs_error"]) for row in spec_rows])
        block_steps = int(spec_rows[0]["block_steps"])
        expanded = np.array([int(row["expanded_count"]) for row in spec_rows])
        summary.append(
            {
                "task": task,
                "fallback_spec": spec,
                "block_steps": str(block_steps),
                "audited_steps": str(len(spec_rows)),
                "positive_margin_fraction": f"{float(np.mean(margins > 0.0)):.6g}",
                "terminal_outside_steps": str(int(np.sum(outside))),
                "positive_terminal_outside_fraction": (
                    f"{float(np.mean(outside_margins > 0.0)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "margin_mean": f"{float(np.mean(margins)):.6g}",
                "margin_median": f"{float(np.median(margins)):.6g}",
                "margin_min": f"{float(np.min(margins)):.6g}",
                "margin_max": f"{float(np.max(margins)):.6g}",
                "raw_margin_mean": f"{float(np.mean(raw_margins)):.6g}",
                "best_block_phi_mean": (
                    f"{float(np.mean([float(row['best_block_phi']) for row in spec_rows])):.6g}"
                ),
                "expanded_per_step_mean": f"{float(np.mean(expanded)):.6g}",
                "max_tail_phi_abs_error": f"{float(np.max(tail_errors)):.3g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("multistep_terminal_fallback_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("multistep_terminal_fallback_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Multi-Step Terminal-Fallback Audit Summary\n\n")
        f.write(
            "The realized shifted-fallback trajectory is kept fixed. Each row "
            "evaluates whether a short bounded terminal fallback block can reduce "
            "the previous predicted terminal Lyapunov score. These blocks are "
            "terminal-region diagnostics only; they do not replace the controller "
            "or constitute the one-step shifted-fallback theorem.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    margin_config = MarginConfig()
    multistep_config = MultiStepConfig()
    rows = []
    for task in ("Z", "H"):
        print(f"auditing multi-step terminal fallback for {task}", flush=True)
        rows.extend(audit_task(task, margin_config, multistep_config))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
