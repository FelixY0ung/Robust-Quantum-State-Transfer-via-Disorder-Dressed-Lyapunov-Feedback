"""Shifted-fallback beam-horizon Lyapunov controller.

This script implements the certified variant used by the shifted-fallback
decrease proposition in the journal manuscript.  It keeps the ordinary
beam-search candidates, but at each receding-horizon step after the first one
it also scores the shifted tail of the previously selected horizon sequence
with every admissible appended fallback control.
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


@dataclass(frozen=True)
class DesignStats:
    shifted_available_steps: int
    shifted_selected_steps: int

    @property
    def shifted_selected_fraction(self) -> float:
        if self.shifted_available_steps == 0:
            return 0.0
        return self.shifted_selected_steps / float(self.shifted_available_steps)


def rollout_sequence(
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    start_index: int,
    dt: float,
    sequence: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], float]:
    states = rhos
    energy_sum = 0.0
    for depth, control in enumerate(sequence):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        states = step_precomputed(
            states,
            strengths,
            controls_cache[cache_index],
            disorder_cache[cache_index],
            dt,
            control,
        )
        energy_sum += float(np.dot(control, control))
    return states, energy_sum


def select_sequence_with_shifted_fallback(
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
) -> tuple[tuple[np.ndarray, ...], bool]:
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
                expanded.append(
                    (cost, False, sequence + (candidate,), next_states, next_energy)
                )
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, _, sequence, states, energy_sum in expanded[:beam_width]
        ]

    terminal_records: list[
        tuple[float, bool, tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]
    ] = []
    for sequence, states, energy_sum in beams:
        cost = scenario_cost(
            p,
            states,
            energy_sum / float(horizon_steps),
            worst_weight,
            energy_weight,
        )
        terminal_records.append((cost, False, sequence, states, energy_sum))

    if previous_sequence is not None:
        shifted_tail = previous_sequence[1:]
        for fallback in candidates:
            shifted_sequence = shifted_tail + (fallback,)
            states, energy_sum = rollout_sequence(
                rhos,
                strengths,
                controls_cache,
                disorder_cache,
                start_index,
                dt,
                shifted_sequence,
            )
            cost = scenario_cost(
                p,
                states,
                energy_sum / float(horizon_steps),
                worst_weight,
                energy_weight,
            )
            terminal_records.append((cost, True, shifted_sequence, states, energy_sum))

    terminal_records.sort(key=lambda item: item[0])
    _, selected_shifted, selected_sequence, _, _ = terminal_records[0]
    return selected_sequence, selected_shifted


def design_pulse(
    task: str,
    train_strengths: tuple[float, ...] = (0.05, 0.08),
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    segments: int = 100,
    horizon_steps: int = 6,
    beam_width: int = 6,
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0),
    worst_weight: float = 0.25,
    energy_weight: float = 0.0,
) -> tuple[np.ndarray, DesignStats]:
    p = problem(task)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    scenarios = tuple(
        (strength, random_disorder(seed))
        for strength in train_strengths
        for seed in train_seeds
    )
    strengths = tuple(strength for strength, _ in scenarios)
    rhos = tuple(p.initial.copy() for _ in scenarios)
    candidates = candidate_controls(amplitudes)
    cache_times = tuple(
        float(t_eval[min(j, segments)]) for j in range(segments + horizon_steps)
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
    previous_sequence: tuple[np.ndarray, ...] | None = None
    shifted_available_steps = 0
    shifted_selected_steps = 0

    for j, _ in enumerate(t_eval[:-1]):
        sequence, selected_shifted = select_sequence_with_shifted_fallback(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            j,
            dt,
            candidates,
            horizon_steps,
            beam_width,
            worst_weight,
            energy_weight,
            previous_sequence,
        )
        if previous_sequence is not None:
            shifted_available_steps += 1
            if selected_shifted:
                shifted_selected_steps += 1
        control = sequence[0]
        pulse.append(control)
        rhos = step_precomputed(
            rhos, strengths, controls_cache[j], disorder_cache[j], dt, control
        )
        previous_sequence = sequence

    return np.asarray(pulse), DesignStats(
        shifted_available_steps=shifted_available_steps,
        shifted_selected_steps=shifted_selected_steps,
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
                "system": f"shifted_fallback_{task}",
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


def plain_rows_by_key() -> dict[tuple[str, float, int], float]:
    path = result_path("horizon_lyapunov_results.csv")
    if not path.exists():
        return {}
    out: dict[tuple[str, float, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            system = row["system"]
            task = system.rsplit("_", maxsplit=1)[-1]
            key = (task, float(row["disorder_strength"]), int(row["seed"]))
            out[key] = float(row["final_fidelity"])
    return out


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    plain = plain_rows_by_key()
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), float(row["disorder_strength"])), []).append(row)

    summary = []
    for (task, strength), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        plain_pairs = []
        deltas = []
        for row in items:
            key = (task, strength, int(row["seed"]))
            if key in plain:
                plain_pairs.append(plain[key])
                deltas.append(float(row["final_fidelity"]) - plain[key])
        summary.append(
            {
                "task": task,
                "eval_strength": f"{strength:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "plain_mean": f"{np.mean(plain_pairs):.6g}" if plain_pairs else "NA",
                "paired_delta_mean": f"{np.mean(deltas):.6g}" if deltas else "NA",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
                "shifted_selected_fraction": f"{float(items[0]['shifted_selected_fraction']):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("shifted_fallback_horizon_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("shifted_fallback_horizon_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Shifted-Fallback Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        pulse, stats = design_pulse(task, beam_width=default_beam_width(task))
        for strength in (0.0, 0.02, 0.05, 0.08):
            rows.extend(evaluate_pulse(task, pulse, stats, disorder_strength=strength))
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('shifted_fallback_horizon_results.csv')}")
    print(f"wrote aggregate table to {result_path('shifted_fallback_horizon_summary.md')}")


if __name__ == "__main__":
    main()
