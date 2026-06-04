"""Online shifted-fallback horizon with a terminal-value score.

The terminal-value certificate audit verifies a Bellman margin for a fallback
value score after the raw shifted-fallback trajectory is generated.  This
script puts the same score into the online terminal ranking step: ordinary beam
pruning generates a tractable set of horizon candidates, shifted-fallback
candidates are forced into the final scored set, and the selected sequence is
ranked by

    G_{j,L}(z) = min sum_r [Phi(z_r) - tau]_+ + Phi(z_L).

The run is a certification-oriented controller diagnostic.  It reports held-out
performance and audits the scheduled Bellman margin on the terminal states
selected by this terminal-value policy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from horizon_lyapunov import (
    TAIL_FRACTION,
    candidate_controls,
    default_beam_width,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    problem,
    random_disorder,
    rk4_step,
    scenario_cost,
    step_precomputed,
)
from paths import result_path
from shifted_fallback_horizon import DesignStats, rollout_sequence
from shifted_fallback_margin_audit import MarginConfig, terminal_phi
from terminal_value_certificate_audit import best_value_margin, terminal_value


@dataclass(frozen=True)
class TerminalValueHorizonConfig:
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    segments: int = 100
    horizon_steps: int = 6
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    energy_weight: float = 0.0
    residual_threshold: float = 1e-3
    value_depth: int = 1
    terminal_weight: float = 1.0
    control_stage_weight: float = 0.0


def terminal_value_sequence_score(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    sequence: tuple[np.ndarray, ...],
    candidates: tuple[np.ndarray, ...],
    config: TerminalValueHorizonConfig,
) -> tuple[tuple[np.ndarray, ...], float, float, float]:
    terminal_states, energy_sum = rollout_sequence(
        rhos,
        strengths,
        controls_cache,
        disorder_cache,
        start_index,
        dt,
        sequence,
    )
    terminal_start = start_index + len(sequence)
    value_score = terminal_value(
        p,
        terminal_states,
        strengths,
        controls_cache,
        disorder_cache,
        terminal_start,
        dt,
        candidates,
        config.value_depth,
        config.worst_weight,
        config.residual_threshold,
        config.terminal_weight,
        config.control_stage_weight,
    )
    energy_mean = energy_sum / float(len(sequence))
    cost = value_score + config.energy_weight * energy_mean
    phi = terminal_phi(p, terminal_states, config.worst_weight)
    return terminal_states, phi, value_score, cost


def select_terminal_value_sequence(
    p,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    beam_width: int,
    config: TerminalValueHorizonConfig,
    previous_sequence: tuple[np.ndarray, ...] | None,
) -> tuple[tuple[np.ndarray, ...], bool, float, float]:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(config.horizon_steps):
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
                pruning_cost = scenario_cost(
                    p,
                    next_states,
                    next_energy / float(depth + 1),
                    config.worst_weight,
                    config.energy_weight,
                )
                expanded.append(
                    (pruning_cost, sequence + (candidate,), next_states, next_energy)
                )
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]

    records: list[tuple[float, bool, tuple[np.ndarray, ...], float]] = []
    for sequence, _, _ in beams:
        _, phi, value_score, cost = terminal_value_sequence_score(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            start_index,
            dt,
            sequence,
            candidates,
            config,
        )
        records.append((cost, False, sequence, phi))

    if previous_sequence is not None:
        shifted_tail = previous_sequence[1:]
        for fallback in candidates:
            shifted_sequence = shifted_tail + (fallback,)
            _, phi, value_score, cost = terminal_value_sequence_score(
                p,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                start_index,
                dt,
                shifted_sequence,
                candidates,
                config,
            )
            records.append((cost, True, shifted_sequence, phi))

    records.sort(key=lambda item: item[0])
    selected_cost, selected_shifted, selected_sequence, selected_phi = records[0]
    return selected_sequence, selected_shifted, selected_phi, selected_cost


def design_pulse(
    task: str,
    config: TerminalValueHorizonConfig = TerminalValueHorizonConfig(),
) -> tuple[np.ndarray, DesignStats, list[dict[str, float | int | str]]]:
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
    beam_width = default_beam_width(task)
    cache_times = tuple(
        float(t_eval[min(j, config.segments)])
        for j in range(config.segments + config.horizon_steps + config.value_depth + 1)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for _, disorder in scenarios)
        for t in cache_times
    )

    pulse = []
    margin_rows: list[dict[str, float | int | str]] = []
    previous_sequence: tuple[np.ndarray, ...] | None = None
    shifted_available_steps = 0
    shifted_selected_steps = 0

    for j, _ in enumerate(t_eval[:-1]):
        if previous_sequence is not None:
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
            terminal_start = j + len(shifted_tail)
            value_record = best_value_margin(
                p,
                terminal_states,
                strengths,
                controls_cache,
                disorder_cache,
                terminal_start,
                dt,
                candidates,
                config.value_depth,
                config.worst_weight,
                config.residual_threshold,
                config.terminal_weight,
                config.control_stage_weight,
            )
            margin_rows.append(
                {
                    "task": task,
                    "step": j,
                    "value_depth": config.value_depth,
                    "beam_width": beam_width,
                    "residual_threshold": config.residual_threshold,
                    "terminal_outside_residual": int(
                        float(value_record["phi"]) > config.residual_threshold
                    ),
                    **value_record,
                }
            )

        sequence, selected_shifted, selected_phi, selected_cost = (
            select_terminal_value_sequence(
                p,
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                j,
                dt,
                candidates,
                beam_width,
                config,
                previous_sequence,
            )
        )
        if previous_sequence is not None:
            shifted_available_steps += 1
            if selected_shifted:
                shifted_selected_steps += 1
        control = sequence[0]
        pulse.append(control)
        rhos = step_precomputed(
            rhos,
            strengths,
            controls_cache[j],
            disorder_cache[j],
            dt,
            control,
        )
        previous_sequence = sequence

        if j % 20 == 0:
            print(
                f"  {task}: terminal-value horizon step {j}/{config.segments} "
                f"phi={selected_phi:.6g} cost={selected_cost:.6g}",
                flush=True,
            )

    return (
        np.asarray(pulse),
        DesignStats(
            shifted_available_steps=shifted_available_steps,
            shifted_selected_steps=shifted_selected_steps,
        ),
        margin_rows,
    )


def evaluate_pulse(
    task: str,
    pulse: np.ndarray,
    stats: DesignStats,
    disorder_strength: float,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    rows = []
    for seed in test_seeds:
        disorder = random_disorder(seed)
        rho = p.initial.copy()
        fids = []
        for j, t in enumerate(t_eval):
            rho = hermitize_trace_one(rho)
            fids.append(fidelity(rho, p.target))
            if j < len(pulse):
                controls_i = tuple(
                    interaction_frame_operator(p, hc, float(t)) for hc in p.controls
                )
                disorder_i = interaction_frame_operator(p, disorder, float(t))
                rho = rk4_step(
                    rho,
                    float(t_eval[j + 1] - t_eval[j]),
                    controls_i,
                    disorder_i,
                    disorder_strength,
                    pulse[j],
                )
        inf = np.maximum(0.0, 1.0 - np.array(fids))
        tail = inf[int((1.0 - TAIL_FRACTION) * len(inf)) :]
        rows.append(
            {
                "system": f"terminal_value_shifted_{task}",
                "task": task,
                "disorder_strength": disorder_strength,
                "seed": seed,
                "tail_infidelity_mean": float(np.mean(tail)),
                "tail_stability_range": float(np.max(tail) - np.min(tail)),
                "final_fidelity": float(fids[-1]),
                "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
                "shifted_available_steps": stats.shifted_available_steps,
                "shifted_selected_steps": stats.shifted_selected_steps,
                "shifted_selected_fraction": stats.shifted_selected_fraction,
            }
        )
    return rows


def rows_by_key(filename: str) -> dict[tuple[str, float, int], float]:
    path = result_path(filename)
    if not path.exists():
        return {}
    out: dict[tuple[str, float, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = row.get("task")
            if not task:
                task = row["system"].rsplit("_", maxsplit=1)[-1]
            key = (task, float(row["disorder_strength"]), int(row["seed"]))
            out[key] = float(row["final_fidelity"])
    return out


def summarize_performance(
    rows: list[dict[str, float | int | str]]
) -> list[dict[str, str]]:
    plain = rows_by_key("horizon_lyapunov_results.csv")
    shifted = rows_by_key("shifted_fallback_horizon_results.csv")
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), float(row["disorder_strength"])), []).append(row)

    summary = []
    for (task, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        plain_deltas = []
        shifted_deltas = []
        for row in items:
            key = (task, strength, int(row["seed"]))
            if key in plain:
                plain_deltas.append(float(row["final_fidelity"]) - plain[key])
            if key in shifted:
                shifted_deltas.append(float(row["final_fidelity"]) - shifted[key])
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{float(np.mean(fids)):.6g}",
                "final_fidelity_min": f"{float(np.min(fids)):.6g}",
                "final_fidelity_std": f"{float(np.std(fids)):.6g}",
                "paired_delta_plain": (
                    f"{float(np.mean(plain_deltas)):.6g}" if plain_deltas else "NA"
                ),
                "paired_delta_shifted": (
                    f"{float(np.mean(shifted_deltas)):.6g}" if shifted_deltas else "NA"
                ),
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "shifted_selected_fraction": f"{float(items[0]['shifted_selected_fraction']):.6g}",
            }
        )
    return summary


def summarize_margins(
    rows: list[dict[str, float | int | str]]
) -> list[dict[str, str]]:
    summary = []
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if str(row["task"]) == task]
        margins = np.array([float(row["best_value_margin"]) for row in task_rows])
        outside = np.array(
            [int(row["terminal_outside_residual"]) for row in task_rows],
            dtype=bool,
        )
        outside_margins = margins[outside]
        gaps = np.array([float(row["bellman_gap"]) for row in task_rows])
        phis = np.array([float(row["phi"]) for row in task_rows])
        summary.append(
            {
                "task": task,
                "audited_steps": str(len(task_rows)),
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
                "margin_min": f"{float(np.min(margins)):.6g}",
                "terminal_phi_mean": f"{float(np.mean(phis)):.6g}",
                "terminal_phi_min": f"{float(np.min(phis)):.6g}",
                "max_bellman_gap": f"{float(np.max(gaps)):.3g}",
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
    performance_rows: list[dict[str, float | int | str]],
    margin_rows: list[dict[str, float | int | str]],
) -> None:
    with result_path("terminal_value_shifted_horizon_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(performance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(performance_rows)

    with result_path("terminal_value_shifted_horizon_margin_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(margin_rows)

    performance_summary = summarize_performance(performance_rows)
    margin_summary = summarize_margins(margin_rows)
    with result_path("terminal_value_shifted_horizon_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Terminal-Value Shifted-Horizon Summary\n\n")
        f.write(
            "This controller uses the finite-set fallback-value terminal score "
            "inside the online shifted-fallback horizon ranking. Beam pruning "
            "is unchanged, shifted candidates are forced into the final scored "
            "set, and the scheduled Bellman margin is audited along the "
            "resulting training trajectory.\n\n"
        )
        f.write("## Held-Out Performance\n\n")
        write_markdown_table(f, performance_summary)
        f.write("\n## Scheduled Terminal-Value Margins\n\n")
        write_markdown_table(f, margin_summary)


def main() -> None:
    config = TerminalValueHorizonConfig()
    performance_rows: list[dict[str, float | int | str]] = []
    margin_rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        print(f"designing terminal-value shifted horizon for {task}", flush=True)
        pulse, stats, task_margin_rows = design_pulse(task, config)
        margin_rows.extend(task_margin_rows)
        for strength in (0.0, 0.02, 0.05, 0.08):
            performance_rows.extend(evaluate_pulse(task, pulse, stats, strength))
    write_outputs(performance_rows, margin_rows)
    for row in summarize_performance(performance_rows):
        print(row)
    for row in summarize_margins(margin_rows):
        print(row)


if __name__ == "__main__":
    main()
