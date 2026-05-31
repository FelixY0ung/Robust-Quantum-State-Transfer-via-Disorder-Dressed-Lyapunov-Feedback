"""Slew-constrained beam-horizon diagnostic.

This script adds a physical-control stress test for the journal manuscript.  It
keeps the same two-level disorder-dressed model as ``horizon_lyapunov.py`` but
adds a quadratic step-to-step slew penalty inside the finite beam search.  The
goal is not to replace the main high-fidelity controller; it is to quantify how
much held-out robustness survives when abrupt changes in the piecewise-constant
pulse are discouraged.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import numpy as np

from horizon_lyapunov import (
    candidate_controls,
    default_beam_width,
    evaluate_pulse,
    interaction_frame_operator,
    problem,
    random_disorder,
    scenario_cost,
    step_precomputed,
)
from paths import result_path


@dataclass(frozen=True)
class SlewConfig:
    segments: int = 60
    horizon_steps: int = 4
    train_strengths: tuple[float, ...] = (0.05, 0.08)
    train_seeds: tuple[int, ...] = tuple(range(8))
    eval_strength: float = 0.08
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)
    worst_weight: float = 0.25
    energy_weight: float = 0.0


def select_control_slew(
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
    slew_weight: float,
    previous_control: np.ndarray,
) -> np.ndarray:
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float, float, np.ndarray]] = [
        ((), rhos, 0.0, 0.0, previous_control)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        controls_i = controls_cache[cache_index]
        disorders_i = disorder_cache[cache_index]
        expanded = []
        for sequence, states, energy_sum, slew_sum, last_control in beams:
            for candidate in candidates:
                next_states = step_precomputed(
                    states, strengths, controls_i, disorders_i, dt, candidate
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                step_slew = float(np.sum(np.square(candidate - last_control)))
                next_slew = slew_sum + step_slew
                depth_count = float(depth + 1)
                cost = scenario_cost(
                    p,
                    next_states,
                    next_energy / depth_count,
                    worst_weight,
                    energy_weight,
                )
                cost += slew_weight * next_slew / depth_count
                expanded.append(
                    (
                        cost,
                        sequence + (candidate,),
                        next_states,
                        next_energy,
                        next_slew,
                        candidate,
                    )
                )
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum, slew_sum, last_control)
            for _, sequence, states, energy_sum, slew_sum, last_control in expanded[:beam_width]
        ]
    return beams[0][0][0]


def design_slew_pulse(
    task: str,
    slew_weight: float,
    config: SlewConfig,
    beam_width: int | None = None,
) -> tuple[np.ndarray, float]:
    p = problem(task)
    width = default_beam_width(task) if beam_width is None else beam_width
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
    pulse = []
    previous_control = np.zeros(len(p.controls), dtype=float)

    start = time.perf_counter()
    for j, _ in enumerate(t_eval[:-1]):
        control = select_control_slew(
            p,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            j,
            dt,
            candidates,
            config.horizon_steps,
            width,
            config.worst_weight,
            config.energy_weight,
            slew_weight,
            previous_control,
        )
        pulse.append(control)
        rhos = step_precomputed(
            rhos, strengths, controls_cache[j], disorder_cache[j], dt, control
        )
        previous_control = control

    return np.asarray(pulse), time.perf_counter() - start


def smoothness_metrics(pulse: np.ndarray, t_final: float) -> dict[str, float]:
    initial = np.zeros((1, pulse.shape[1]), dtype=float)
    diffs = np.diff(np.vstack([initial, pulse]), axis=0)
    step_norms = np.linalg.norm(diffs, axis=1)
    energy = float(np.mean(np.sum(np.square(pulse), axis=1)))
    rms_slew = float(np.sqrt(np.mean(np.sum(np.square(diffs), axis=1))))
    max_step = float(np.max(step_norms))
    total_variation = float(np.sum(step_norms))

    spectrum = np.fft.rfft(pulse, axis=0)
    power = np.sum(np.square(np.abs(spectrum)), axis=1)
    freqs = np.fft.rfftfreq(len(pulse), d=t_final / len(pulse))
    cutoff = np.quantile(freqs, 2.0 / 3.0)
    total_power = float(np.sum(power[1:]))
    if total_power <= 0.0:
        high_frequency_fraction = 0.0
    else:
        high_frequency_fraction = float(np.sum(power[freqs >= cutoff]) / total_power)

    return {
        "pulse_energy": energy,
        "rms_slew": rms_slew,
        "max_step": max_step,
        "total_variation": total_variation,
        "high_frequency_fraction": high_frequency_fraction,
    }


def run_audit(weights: tuple[float, ...], config: SlewConfig) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        p = problem(task)
        for slew_weight in weights:
            pulse, design_seconds = design_slew_pulse(task, slew_weight, config)
            metrics = smoothness_metrics(pulse, p.t_final)
            eval_rows = evaluate_pulse(
                task,
                pulse,
                disorder_strength=config.eval_strength,
                test_seeds=range(config.eval_seeds[0], config.eval_seeds[-1] + 1),
            )
            for eval_row in eval_rows:
                rows.append(
                    {
                        "task": task,
                        "controller": "slew_constrained_beam_horizon",
                        "slew_weight": slew_weight,
                        "eval_strength": config.eval_strength,
                        "seed": int(eval_row["seed"]),
                        "final_fidelity": float(eval_row["final_fidelity"]),
                        "tail_infidelity_mean": float(eval_row["tail_infidelity_mean"]),
                        "tail_stability_range": float(eval_row["tail_stability_range"]),
                        "segments": config.segments,
                        "horizon_steps": config.horizon_steps,
                        "beam_width": default_beam_width(task),
                        "design_seconds": design_seconds,
                        **metrics,
                    }
                )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), float(row["slew_weight"])), []).append(row)

    summary = []
    for (task, slew_weight), items in sorted(grouped.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "slew_weight": f"{slew_weight:.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "pulse_energy": f"{float(items[0]['pulse_energy']):.6g}",
                "rms_slew": f"{float(items[0]['rms_slew']):.6g}",
                "max_step": f"{float(items[0]['max_step']):.6g}",
                "total_variation": f"{float(items[0]['total_variation']):.6g}",
                "high_frequency_fraction": f"{float(items[0]['high_frequency_fraction']):.6g}",
                "design_seconds": f"{float(items[0]['design_seconds']):.3f}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("slew_constrained_horizon_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("slew_constrained_horizon_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Slew-Constrained Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="0,0.001,0.005",
        help="Comma-separated slew weights to audit.",
    )
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = tuple(float(item) for item in args.weights.split(",") if item)
    config = SlewConfig(segments=args.segments, horizon_steps=args.horizon_steps)
    rows = run_audit(weights, config)
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('slew_constrained_horizon_results.csv')}")
    print(f"wrote summary to {result_path('slew_constrained_horizon_summary.md')}")


if __name__ == "__main__":
    main()
