"""Audit a certificate-active terminal value score.

The raw shifted-fallback margin audit asks whether the terminal Lyapunov score
Phi decreases after one appended fallback segment.  That condition is too
strong for the current nonstationary Z/H tasks.  This script audits a standard
MPC-style replacement:

    G_{j,L}(z) = min_{kappa_0,...,kappa_{L-1}}
        sum_{r=0}^{L-1} [Phi(z_r) - tau]_+ + Phi(z_L).

For exact finite-set propagation, the Bellman identity gives

    G_{j,L}(z) - G_{j+1,L-1}(F_{kappa^*,j}(z)) = [Phi(z) - tau]_+,

where kappa^* is a minimizing first fallback.  Hence every audited terminal
state outside the residual set Phi <= tau has a positive computable
shifted-fallback margin.  This is a certificate-design audit, not yet a new
closed-loop controller run.
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
class TerminalValueCertificateConfig:
    value_depths: tuple[int, ...] = (1, 2)
    audit_stride: int = 1
    terminal_weight: float = 1.0
    control_stage_weight: float = 0.0


def stage_cost(
    phi_value: float,
    residual_threshold: float,
    control: np.ndarray,
    control_stage_weight: float,
) -> float:
    return (
        max(0.0, phi_value - residual_threshold)
        + control_stage_weight * float(np.dot(control, control))
    )


def terminal_value(
    p,
    states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    depth: int,
    worst_weight: float,
    residual_threshold: float,
    terminal_weight: float,
    control_stage_weight: float,
) -> float:
    phi_now = terminal_phi(p, states, worst_weight)
    if depth == 0:
        return terminal_weight * phi_now

    cache_index = min(start_index, len(controls_cache) - 1)
    values = []
    for candidate in candidates:
        next_states = step_precomputed(
            states,
            strengths,
            controls_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            candidate,
        )
        values.append(
            stage_cost(
                phi_now,
                residual_threshold,
                candidate,
                control_stage_weight,
            )
            + terminal_value(
                p,
                next_states,
                strengths,
                controls_cache,
                disorder_cache,
                start_index + 1,
                dt,
                candidates,
                depth - 1,
                worst_weight,
                residual_threshold,
                terminal_weight,
                control_stage_weight,
            )
        )
    return float(np.min(values))


def best_value_margin(
    p,
    terminal_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    depth: int,
    worst_weight: float,
    residual_threshold: float,
    terminal_weight: float,
    control_stage_weight: float,
) -> dict[str, float | int]:
    phi_now = terminal_phi(p, terminal_states, worst_weight)
    current_value = terminal_value(
        p,
        terminal_states,
        strengths,
        controls_cache,
        disorder_cache,
        start_index,
        dt,
        candidates,
        depth,
        worst_weight,
        residual_threshold,
        terminal_weight,
        control_stage_weight,
    )
    cache_index = min(start_index, len(controls_cache) - 1)
    successor_records = []
    for candidate_index, candidate in enumerate(candidates):
        next_states = step_precomputed(
            terminal_states,
            strengths,
            controls_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            candidate,
        )
        successor_value = terminal_value(
            p,
            next_states,
            strengths,
            controls_cache,
            disorder_cache,
            start_index + 1,
            dt,
            candidates,
            depth - 1,
            worst_weight,
            residual_threshold,
            terminal_weight,
            control_stage_weight,
        )
        margin = current_value - successor_value
        successor_records.append(
            (
                margin,
                successor_value,
                float(np.dot(candidate, candidate)),
                float(np.linalg.norm(candidate)),
                candidate_index,
            )
        )

    margin, successor_value, norm_sq, amp, candidate_index = max(
        successor_records,
        key=lambda item: item[0],
    )
    expected_stage = stage_cost(
        phi_now,
        residual_threshold,
        candidates[candidate_index],
        control_stage_weight,
    )
    return {
        "phi": phi_now,
        "current_value": current_value,
        "best_successor_value": successor_value,
        "best_value_margin": margin,
        "best_fallback_norm": norm_sq,
        "best_fallback_amp": amp,
        "best_fallback_index": candidate_index,
        "state_stage_cost": max(0.0, phi_now - residual_threshold),
        "expected_stage_cost": expected_stage,
        "bellman_gap": abs(margin - expected_stage),
    }


def audit_task(
    task: str,
    margin_config: MarginConfig,
    value_config: TerminalValueCertificateConfig,
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
    max_depth = max(value_config.value_depths)
    cache_times = tuple(
        float(t_eval[min(j, margin_config.segments)])
        for j in range(margin_config.segments + margin_config.horizon_steps + max_depth + 1)
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
        if (
            previous_sequence is not None
            and previous_phi is not None
            and previous_first_norm is not None
            and j % value_config.audit_stride == 0
        ):
            if j % 20 == 0:
                print(f"  {task}: value-certificate step {j}/{margin_config.segments}", flush=True)
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
            terminal_start = min(j + len(shifted_tail), len(controls_cache) - 1)
            tail_phi = terminal_phi(p, terminal_states, margin_config.worst_weight)
            for depth in value_config.value_depths:
                record = best_value_margin(
                    p,
                    terminal_states,
                    strengths,
                    controls_cache,
                    disorder_cache,
                    terminal_start,
                    dt,
                    candidates,
                    depth,
                    margin_config.worst_weight,
                    margin_config.residual_threshold,
                    value_config.terminal_weight,
                    value_config.control_stage_weight,
                )
                rows.append(
                    {
                        "task": task,
                        "step": j,
                        "value_depth": depth,
                        "candidate_count": len(candidates),
                        "previous_terminal_phi": previous_phi,
                        "tail_phi": tail_phi,
                        "tail_phi_abs_error": abs(previous_phi - tail_phi),
                        "terminal_outside_residual": int(
                            previous_phi > margin_config.residual_threshold
                        ),
                        "residual_threshold": margin_config.residual_threshold,
                        **record,
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
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[j],
            disorder_cache[j],
            dt,
            sequence[0],
        )
        previous_sequence = sequence
        previous_phi = phi
        previous_first_norm = first_norm

    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    keys = sorted({(str(row["task"]), int(row["value_depth"])) for row in rows})
    for task, depth in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and int(row["value_depth"]) == depth
        ]
        margins = np.array([float(row["best_value_margin"]) for row in group])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in group],
            dtype=bool,
        )
        outside_margins = margins[outside]
        phis = np.array([float(row["previous_terminal_phi"]) for row in group])
        gaps = np.array([float(row["bellman_gap"]) for row in group])
        summary.append(
            {
                "task": task,
                "value_depth": str(depth),
                "audited_steps": str(len(group)),
                "terminal_outside_steps": str(int(np.sum(outside))),
                "positive_margin_fraction": f"{float(np.mean(margins > 0.0)):.6g}",
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
                "margin_median": f"{float(np.median(margins)):.6g}",
                "margin_min": f"{float(np.min(margins)):.6g}",
                "margin_max": f"{float(np.max(margins)):.6g}",
                "terminal_phi_mean": f"{float(np.mean(phis)):.6g}",
                "terminal_phi_min": f"{float(np.min(phis)):.6g}",
                "max_bellman_gap": f"{float(np.max(gaps)):.3g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("terminal_value_certificate_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("terminal_value_certificate_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Terminal-Value Certificate Audit Summary\n\n")
        f.write(
            "This audit tests a certificate-active terminal score for the "
            "shifted-fallback theorem. The score is a finite-set fallback value "
            "`G_{j,L}` with running cost `[Phi-tau]_+` and terminal cost `Phi`. "
            "For exact propagation, the Bellman identity makes the best "
            "one-segment shifted-fallback margin equal to the positive running "
            "cost whenever the terminal state lies outside `Phi <= tau`. This "
            "is an executable terminal-score certificate audit, not yet a new "
            "closed-loop controller result.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    margin_config = MarginConfig()
    value_config = TerminalValueCertificateConfig()
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        print(f"auditing terminal-value certificate for {task}", flush=True)
        rows.extend(audit_task(task, margin_config, value_config))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
