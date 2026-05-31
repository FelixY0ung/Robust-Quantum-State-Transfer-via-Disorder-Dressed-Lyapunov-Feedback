"""Low-pass filter robustness audit for beam-horizon pulses.

The slew-constrained audit shows that the finite-candidate horizon can be
optimized with a first-difference penalty.  This script adds a second,
post-design diagnostic: it applies simple FIR smoothing filters to the compact
beam-horizon pulses and evaluates whether held-out robustness survives after
bandwidth-like smoothing.  The filters are deliberately transparent and
deterministic; they are not a calibrated hardware transfer function.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

from horizon_lyapunov import default_beam_width, evaluate_pulse, problem
from paths import result_path
from slew_constrained_horizon import SlewConfig, design_slew_pulse, smoothness_metrics


@dataclass(frozen=True)
class FilterSpec:
    name: str
    kind: str
    width: int
    sigma: float = 0.0


DEFAULT_FILTERS = (
    FilterSpec("none", "none", 1),
    FilterSpec("boxcar3", "boxcar", 3),
    FilterSpec("boxcar7", "boxcar", 7),
    FilterSpec("gaussian7", "gaussian", 7, 1.5),
)


def kernel(spec: FilterSpec) -> np.ndarray:
    if spec.kind == "none":
        return np.array([1.0], dtype=float)
    if spec.width < 1 or spec.width % 2 == 0:
        raise ValueError(f"filter width must be a positive odd integer: {spec}")
    if spec.kind == "boxcar":
        values = np.ones(spec.width, dtype=float)
    elif spec.kind == "gaussian":
        offsets = np.arange(spec.width, dtype=float) - (spec.width - 1.0) / 2.0
        sigma = spec.sigma if spec.sigma > 0.0 else max(spec.width / 6.0, 1e-6)
        values = np.exp(-0.5 * np.square(offsets / sigma))
    else:
        raise ValueError(spec.kind)
    return values / float(np.sum(values))


def apply_filter(pulse: np.ndarray, spec: FilterSpec) -> np.ndarray:
    weights = kernel(spec)
    if len(weights) == 1:
        return pulse.copy()
    pad = len(weights) // 2
    filtered = np.zeros_like(pulse)
    for channel in range(pulse.shape[1]):
        padded = np.pad(pulse[:, channel], pad_width=pad, mode="edge")
        filtered[:, channel] = np.convolve(padded, weights, mode="valid")
    return filtered


def parse_filter_specs(text: str) -> tuple[FilterSpec, ...]:
    known = {spec.name: spec for spec in DEFAULT_FILTERS}
    specs = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        if item not in known:
            raise ValueError(f"unknown filter {item}; choices: {', '.join(known)}")
        specs.append(known[item])
    return tuple(specs)


def run_audit(
    base_weights: tuple[float, ...],
    filters: tuple[FilterSpec, ...],
    config: SlewConfig,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for task in ("Z", "H"):
        p = problem(task)
        for base_weight in base_weights:
            base_pulse, design_seconds = design_slew_pulse(task, base_weight, config)
            for spec in filters:
                pulse = apply_filter(base_pulse, spec)
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
                            "controller": "bandwidth_filtered_beam_horizon",
                            "base_slew_weight": base_weight,
                            "filter": spec.name,
                            "filter_kind": spec.kind,
                            "filter_width": spec.width,
                            "filter_sigma": spec.sigma,
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
    grouped: dict[tuple[str, float, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["task"]), float(row["base_slew_weight"]), str(row["filter"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (task, base_weight, filter_name), items in sorted(grouped.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        summary.append(
            {
                "task": task,
                "base_slew_weight": f"{base_weight:.4g}",
                "filter": filter_name,
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
    with result_path("bandwidth_filter_audit_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("bandwidth_filter_audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Bandwidth Filter Audit Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-weights",
        default="0,0.005",
        help="Comma-separated slew weights used to design the base pulses.",
    )
    parser.add_argument(
        "--filters",
        default="none,boxcar3,boxcar7,gaussian7",
        help="Comma-separated filter names: none, boxcar3, boxcar7, gaussian7.",
    )
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--horizon-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_weights = tuple(float(item) for item in args.base_weights.split(",") if item)
    filters = parse_filter_specs(args.filters)
    config = SlewConfig(segments=args.segments, horizon_steps=args.horizon_steps)
    rows = run_audit(base_weights, filters, config)
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('bandwidth_filter_audit_results.csv')}")
    print(f"wrote summary to {result_path('bandwidth_filter_audit_summary.md')}")


if __name__ == "__main__":
    main()
