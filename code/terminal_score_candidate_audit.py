"""Audit concrete terminal-score replacements for shifted-fallback descent.

The terminal-score replacement proposition allows the raw terminal Lyapunov
score Phi to be replaced by any nonnegative terminal score G.  This script
tests a simple, computable candidate:

    G_beta(z) = Phi(z) - beta * max(0, Phi(z) - min_k Phi(F_k(z))).

For beta=0 this is the raw terminal Lyapunov score used in the original
terminal-fallback audit.  For beta=1 it is the one-step terminal-value score:
the best score reachable after one admissible fallback segment.  Intermediate
beta values reward terminal states that have immediate controllable descent
without fully replacing Phi by a value function.

The realized shifted-fallback trajectory is kept fixed.  The audit measures
whether there exists an appended fallback that decreases G_beta at the previous
predicted terminal ensemble state.  It is a certificate-design diagnostic, not
a replacement controller result.
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
class TerminalScoreConfig:
    beta_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    audit_stride: int = 4


def one_step_terminal_values(
    p,
    states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    cache_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    worst_weight: float,
) -> tuple[float, float, np.ndarray]:
    phi_now = terminal_phi(p, states, worst_weight)
    phis = []
    for candidate in candidates:
        next_states = step_precomputed(
            states,
            strengths,
            controls_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            candidate,
        )
        phis.append(terminal_phi(p, next_states, worst_weight))
    next_phis = np.array(phis)
    best_next_phi = float(np.min(next_phis))
    return phi_now, best_next_phi, next_phis


def terminal_score(phi_now: float, best_next_phi: float, beta: float) -> float:
    immediate_descent = max(0.0, phi_now - best_next_phi)
    return phi_now - beta * immediate_descent


def score_successors(
    p,
    terminal_states: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    cache_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    worst_weight: float,
) -> tuple[float, float, list[tuple[float, float, float, float]]]:
    phi_z, best_next_phi_z, _ = one_step_terminal_values(
        p,
        terminal_states,
        strengths,
        controls_cache,
        disorder_cache,
        cache_index,
        dt,
        candidates,
        worst_weight,
    )
    successors = []
    next_cache_index = min(cache_index + 1, len(controls_cache) - 1)
    for fallback in candidates:
        next_states = step_precomputed(
            terminal_states,
            strengths,
            controls_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            fallback,
        )
        phi_next, best_after_next_phi, _ = one_step_terminal_values(
            p,
            next_states,
            strengths,
            controls_cache,
            disorder_cache,
            next_cache_index,
            dt,
            candidates,
            worst_weight,
        )
        fallback_norm = float(np.dot(fallback, fallback))
        successors.append(
            (phi_next, best_after_next_phi, fallback_norm, float(np.linalg.norm(fallback)))
        )
    return phi_z, best_next_phi_z, successors


def audit_task(
    task: str,
    margin_config: MarginConfig,
    score_config: TerminalScoreConfig,
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
    cache_times = tuple(
        float(t_eval[min(j, margin_config.segments)])
        for j in range(margin_config.segments + margin_config.horizon_steps + 2)
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
            and j % score_config.audit_stride == 0
        ):
            print(f"  {task}: terminal-score step {j}/{margin_config.segments}", flush=True)
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
            cache_index = min(j + len(shifted_tail), len(controls_cache) - 1)
            phi_z, best_next_phi_z, successors = score_successors(
                p,
                terminal_states,
                strengths,
                controls_cache,
                disorder_cache,
                cache_index,
                dt,
                candidates,
                margin_config.worst_weight,
            )
            tail_error = abs(previous_phi - phi_z)
            terminal_outside = int(previous_phi > margin_config.residual_threshold)
            for beta in score_config.beta_values:
                g_z = terminal_score(phi_z, best_next_phi_z, beta)
                successor_scores = np.array(
                    [
                        terminal_score(phi_next, best_after_next_phi, beta)
                        for phi_next, best_after_next_phi, _, _ in successors
                    ]
                )
                margins = g_z - successor_scores
                best_index = int(np.argmax(margins))
                best_margin = float(margins[best_index])
                phi_next, best_after_next_phi, fallback_norm, fallback_amp = successors[
                    best_index
                ]
                rows.append(
                    {
                        "task": task,
                        "step": j,
                        "beta": beta,
                        "previous_terminal_phi": previous_phi,
                        "tail_phi": phi_z,
                        "tail_phi_abs_error": tail_error,
                        "g_terminal": g_z,
                        "terminal_availability": max(0.0, phi_z - best_next_phi_z),
                        "best_successor_g": float(successor_scores[best_index]),
                        "best_g_margin": best_margin,
                        "best_successor_phi": phi_next,
                        "best_successor_best_next_phi": best_after_next_phi,
                        "best_fallback_norm": fallback_norm,
                        "best_fallback_amp": fallback_amp,
                        "terminal_outside_residual": terminal_outside,
                        "residual_threshold": margin_config.residual_threshold,
                        "g_nonnegative": int(g_z >= -1e-12),
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
    summary = []
    keys = sorted({(str(row["task"]), float(row["beta"])) for row in rows})
    for task, beta in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and float(row["beta"]) == beta
        ]
        margins = np.array([float(row["best_g_margin"]) for row in group])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in group],
            dtype=bool,
        )
        outside_margins = margins[outside]
        g_values = np.array([float(row["g_terminal"]) for row in group])
        availability = np.array([float(row["terminal_availability"]) for row in group])
        tail_errors = np.array([float(row["tail_phi_abs_error"]) for row in group])
        summary.append(
            {
                "task": task,
                "beta": f"{beta:.2f}",
                "audited_steps": str(len(group)),
                "positive_margin_fraction": f"{float(np.mean(margins > 0.0)):.6g}",
                "terminal_outside_steps": str(int(np.sum(outside))),
                "positive_terminal_outside_fraction": (
                    f"{float(np.mean(outside_margins > 0.0)):.6g}"
                    if outside_margins.size
                    else "nan"
                ),
                "g_margin_mean": f"{float(np.mean(margins)):.6g}",
                "g_margin_median": f"{float(np.median(margins)):.6g}",
                "g_margin_min": f"{float(np.min(margins)):.6g}",
                "g_margin_max": f"{float(np.max(margins)):.6g}",
                "g_terminal_mean": f"{float(np.mean(g_values)):.6g}",
                "availability_mean": f"{float(np.mean(availability)):.6g}",
                "nonnegative_fraction": (
                    f"{float(np.mean([int(row['g_nonnegative']) for row in group])):.6g}"
                ),
                "max_tail_phi_abs_error": f"{float(np.max(tail_errors)):.3g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("terminal_score_candidate_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("terminal_score_candidate_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Terminal-Score Candidate Audit Summary\n\n")
        f.write(
            "This audit tests the terminal-score replacement proposition on the "
            "realized shifted-fallback trajectory. The candidate score is "
            "`G_beta(z)=Phi(z)-beta*max(0, Phi(z)-min_k Phi(F_k(z)))`. "
            "`beta=0` recovers the raw terminal Lyapunov score, while `beta=1` "
            "is the one-step terminal-value score. A positive margin means that "
            "some appended fallback decreases this replacement score at the "
            "previous predicted terminal ensemble state. The audit uses every "
            f"{TerminalScoreConfig().audit_stride}th post-initial control step "
            "to keep the nested successor scoring tractable. The result is a "
            "certificate-design diagnostic, not a replacement controller run.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    margin_config = MarginConfig()
    score_config = TerminalScoreConfig()
    rows = []
    for task in ("Z", "H"):
        print(f"auditing terminal-score candidates for {task}", flush=True)
        rows.extend(audit_task(task, margin_config, score_config))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
