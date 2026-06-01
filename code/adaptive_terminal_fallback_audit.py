"""Audit adaptive terminal fallbacks for the shifted-fallback certificate.

The fixed fallback alphabet sweep shows that denser in-range quantization does
not make the practical-decrease margin active more often.  This script tests a
different certificate-design idea: after the shifted tail reaches the previous
predicted terminal ensemble state, estimate the local one-segment gradient of
the terminal Lyapunov score and line-search along the local descent direction.

The realized shifted-fallback trajectory is kept fixed.  Therefore this is an
audit of whether an adaptive terminal fallback could improve the certificate,
not a replacement held-out controller result.
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
    score_sequence,
    select_with_margin_record,
    terminal_phi,
)


@dataclass(frozen=True)
class AdaptiveConfig:
    finite_difference_step: float = 1e-3
    line_radii: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    )
    gradient_tol: float = 1e-12


def one_segment_phi(
    p,
    terminal_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    cache_index: int,
    dt: float,
    control: np.ndarray,
    worst_weight: float,
) -> float:
    next_states = step_precomputed(
        terminal_states,
        strengths,
        controls_cache[cache_index],
        disorder_cache[cache_index],
        dt,
        control,
    )
    return terminal_phi(p, next_states, worst_weight)


def adaptive_line_fallback(
    p,
    terminal_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    cache_index: int,
    dt: float,
    worst_weight: float,
    config: AdaptiveConfig,
) -> tuple[np.ndarray, float, float, float]:
    dim = len(p.controls)
    grad = np.zeros(dim)
    h = config.finite_difference_step
    for k in range(dim):
        direction = np.zeros(dim)
        direction[k] = 1.0
        plus = one_segment_phi(
            p,
            terminal_states,
            strengths,
            controls_cache,
            disorder_cache,
            cache_index,
            dt,
            h * direction,
            worst_weight,
        )
        minus = one_segment_phi(
            p,
            terminal_states,
            strengths,
            controls_cache,
            disorder_cache,
            cache_index,
            dt,
            -h * direction,
            worst_weight,
        )
        grad[k] = (plus - minus) / (2.0 * h)

    grad_norm = float(np.linalg.norm(grad))
    if grad_norm <= config.gradient_tol:
        zero = np.zeros(dim)
        phi = one_segment_phi(
            p,
            terminal_states,
            strengths,
            controls_cache,
            disorder_cache,
            cache_index,
            dt,
            zero,
            worst_weight,
        )
        return zero, phi, grad_norm, 1.0

    descent_direction = -grad / grad_norm
    best_control = np.zeros(dim)
    best_phi = np.inf
    best_radius = 0.0
    for radius in config.line_radii:
        control = radius * descent_direction
        phi = one_segment_phi(
            p,
            terminal_states,
            strengths,
            controls_cache,
            disorder_cache,
            cache_index,
            dt,
            control,
            worst_weight,
        )
        if phi < best_phi:
            best_phi = phi
            best_control = control
            best_radius = radius
    return best_control, best_phi, grad_norm, best_radius


def default_best_margin(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    previous_sequence: tuple[np.ndarray, ...],
    previous_phi: float,
    previous_first_norm: float,
    candidates: tuple[np.ndarray, ...],
    config: MarginConfig,
) -> tuple[float, float, float]:
    shifted_tail = previous_sequence[1:]
    best_margin = -np.inf
    best_phi = np.nan
    best_norm = np.nan
    for fallback in candidates:
        shifted_sequence = shifted_tail + (fallback,)
        _, _, phi, _ = score_sequence(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            start_index,
            dt,
            shifted_sequence,
            config.worst_weight,
            config.energy_weight,
        )
        fallback_norm = float(np.dot(fallback, fallback))
        margin = previous_phi - phi - config.energy_weight * (
            fallback_norm - previous_first_norm
        )
        if margin > best_margin:
            best_margin = margin
            best_phi = phi
            best_norm = fallback_norm
    return best_margin, best_phi, best_norm


def audit_task(
    task: str,
    margin_config: MarginConfig,
    adaptive_config: AdaptiveConfig,
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
    design_candidates = candidate_controls(margin_config.amplitudes)
    beam_width = default_beam_width(task)
    cache_times = tuple(
        float(t_eval[min(j, margin_config.segments)])
        for j in range(margin_config.segments + margin_config.horizon_steps)
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
            cache_index = min(
                j + len(shifted_tail),
                len(controls_cache) - 1,
            )
            default_margin, default_phi, default_norm = default_best_margin(
                p,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                j,
                dt,
                previous_sequence,
                previous_phi,
                previous_first_norm,
                design_candidates,
                margin_config,
            )
            adaptive_control, adaptive_phi, grad_norm, radius = adaptive_line_fallback(
                p,
                terminal_states,
                strengths,
                controls_cache,
                disorder_cache,
                cache_index,
                dt,
                margin_config.worst_weight,
                adaptive_config,
            )
            adaptive_norm = float(np.dot(adaptive_control, adaptive_control))
            adaptive_margin = previous_phi - adaptive_phi - margin_config.energy_weight * (
                adaptive_norm - previous_first_norm
            )
            for method, margin, phi, norm, extra in (
                (
                    "default_alphabet",
                    default_margin,
                    default_phi,
                    default_norm,
                    {"gradient_norm": np.nan, "line_radius": np.nan},
                ),
                (
                    "adaptive_gradient_line",
                    adaptive_margin,
                    adaptive_phi,
                    adaptive_norm,
                    {"gradient_norm": grad_norm, "line_radius": radius},
                ),
            ):
                rows.append(
                    {
                        "task": task,
                        "step": j,
                        "method": method,
                        "previous_terminal_phi": previous_phi,
                        "shift_tail_phi": tail_phi,
                        "tail_phi_abs_error": abs(previous_phi - tail_phi),
                        "best_margin": margin,
                        "best_fallback_phi": phi,
                        "best_fallback_norm": norm,
                        "terminal_outside_residual": int(
                            previous_phi > margin_config.residual_threshold
                        ),
                        "residual_threshold": margin_config.residual_threshold,
                        **extra,
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
            design_candidates,
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
    keys = sorted({(str(row["task"]), str(row["method"])) for row in rows})
    for task, method in keys:
        method_rows = [
            row
            for row in rows
            if str(row["task"]) == task and str(row["method"]) == method
        ]
        margins = np.array([float(row["best_margin"]) for row in method_rows])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in method_rows],
            dtype=bool,
        )
        outside_margins = margins[outside]
        tail_errors = np.array([float(row["tail_phi_abs_error"]) for row in method_rows])
        radii = np.array(
            [
                float(row["line_radius"])
                for row in method_rows
                if str(row["line_radius"]) != "nan"
            ]
        )
        summary.append(
            {
                "task": task,
                "method": method,
                "audited_steps": str(len(method_rows)),
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
                "max_tail_phi_abs_error": f"{float(np.max(tail_errors)):.3g}",
                "line_radius_mean": (
                    f"{float(np.mean(radii)):.6g}" if radii.size else "nan"
                ),
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("adaptive_terminal_fallback_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("adaptive_terminal_fallback_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Adaptive Terminal-Fallback Audit Summary\n\n")
        f.write(
            "The realized shifted-fallback trajectory is kept fixed. The adaptive "
            "row estimates the local one-segment terminal-score gradient at the "
            "predicted terminal ensemble state and line-searches along the descent "
            "direction inside the same amplitude cap. This is a certificate-design "
            "audit, not a replacement held-out controller result.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    margin_config = MarginConfig()
    adaptive_config = AdaptiveConfig()
    rows = []
    for task in ("Z", "H"):
        print(f"auditing adaptive terminal fallback for {task}", flush=True)
        rows.extend(audit_task(task, margin_config, adaptive_config))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
