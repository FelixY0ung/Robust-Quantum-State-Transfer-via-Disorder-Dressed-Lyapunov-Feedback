"""Seeded no-reference open-leakage continuation sweep.

This exploratory audit tests whether the five-level leakage-plus-Lindblad
frontier improves when the continuation starts from a stronger no-reference
target/leakage seed instead of the conservative low-leakage seed used in
``open_leakage_continuation_sweep.py``.  The script writes separate outputs so
the established continuation audit is not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from open_leakage_continuation_sweep import (
    StageSpec,
    annotate_rows,
    combined_points,
    config_from_stage,
    load_reference_points as load_base_reference_points,
    stage_description,
    summarize,
)
from paths import figure_path, result_path
from transmon_leakage_horizon import design_pulse
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
    load_result_rows,
)
from transmon_open_system_leakage import evaluate_pulse


@dataclass(frozen=True)
class SeededContinuationVariant:
    controller: str
    label: str
    seed: StageSpec
    stages: tuple[StageSpec, ...]


STRONG_TARGET_SEED = StageSpec(
    terminal_target_weight=1.0,
    leakage_weight=1.5,
    trust_radius=0.04,
    horizon_maxiter=6,
)


VARIANTS = (
    SeededContinuationVariant(
        "seed_target10_leak15_target08_leak12",
        "target-1.0 seed, target 0.8 then leak 1.2",
        STRONG_TARGET_SEED,
        (
            StageSpec(1.0, 0.8, 0.02, 6),
            StageSpec(1.0, 1.2, 0.015, 6),
        ),
    ),
    SeededContinuationVariant(
        "seed_target10_leak15_target08_leak10",
        "target-1.0 seed, target 0.8 then leak 1.0",
        STRONG_TARGET_SEED,
        (
            StageSpec(1.0, 0.8, 0.02, 6),
            StageSpec(1.0, 1.0, 0.015, 6),
        ),
    ),
    SeededContinuationVariant(
        "seed_target10_leak15_target08_leak08",
        "target-1.0 seed, target 0.8 then leak 0.8",
        STRONG_TARGET_SEED,
        (
            StageSpec(1.0, 0.8, 0.02, 6),
            StageSpec(1.0, 0.8, 0.015, 6),
        ),
    ),
)


def config_from_args(args: argparse.Namespace) -> OpenLeakageAdjointConfig:
    if args.quick:
        return OpenLeakageAdjointConfig(
            segments=30,
            horizon_steps=3,
            horizon_maxiter=2,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 15)),
        )
    return OpenLeakageAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
    )


def load_context_points() -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, float | int | str]] = []
    rows.extend(load_result_rows("open_leakage_continuation_sweep_results.csv"))
    context = load_base_reference_points()
    context.update(combined_points(rows))
    return context


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    context_points = load_context_points()
    new_points = combined_points(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for controller, (mean_fid, worst_fid, leak) in context_points.items():
        ax.scatter(leak, mean_fid, marker="o", s=20, color="0.70", alpha=0.65)
        ax.scatter(leak, worst_fid, marker="x", s=16, color="0.70", alpha=0.65)
    for controller, (mean_fid, worst_fid, leak) in new_points.items():
        if controller == "seed_continuation_path_seed":
            continue
        marker = "D" if controller == "seed_target10_leak15" else "s"
        ax.scatter(leak, mean_fid, marker=marker, s=42, color="#9467bd")
        ax.scatter(leak, worst_fid, marker="x", s=26, color="#9467bd")
        ax.annotate(
            controller.replace("seed_target10_leak15_", ""),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Seeded no-reference open-leakage continuation")
    ax.set_xlim(0.0, 0.08)
    ax.set_ylim(0.50, 0.96)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_seed_continuation_sweep.pdf"))
    fig.savefig(figure_path("open_leakage_seed_continuation_sweep.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_seed_continuation_sweep_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_seed_continuation_sweep_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Seeded Open-Leakage Continuation Sweep\n\n")
        f.write(
            "No-reference continuation horizons initialized from a stronger "
            "target/leakage seed and trained through the five-level Lindblad "
            "leakage model.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_seed_continuation_sweep.pdf')}")


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no seeded continuation variants matched --only={args.only!r}")

    if args.quick:
        path_kwargs = {
            "train_strengths": (base_config.train_strength,),
            "train_seeds": base_config.train_seeds,
            "segments": base_config.segments,
            "horizon_steps": base_config.horizon_steps,
            "beam_width": 3,
            "amplitudes": (0.015, 0.035, 0.06),
            "leakage_weight": 0.5,
        }
    else:
        path_kwargs = {"segments": base_config.segments}

    print("designing finite-candidate path horizon seed", flush=True)
    start = time.perf_counter()
    path_pulse = design_pulse(**path_kwargs)
    path_seconds = time.perf_counter() - start
    eval_seeds = range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1)
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        annotate_rows(
            evaluate_pulse(
                path_pulse,
                "seed_continuation_path_seed",
                path_seconds,
                None,
                None,
                None,
                disorder_strength=base_config.eval_strength,
                test_seeds=eval_seeds,
            ),
            "path seed",
            "",
        )
    )

    seed_cache: dict[StageSpec, tuple[np.ndarray, float, int, bool, float]] = {}
    for variant in variants:
        if variant.seed not in seed_cache:
            print(f"running seed for {variant.label}", flush=True)
            seed_config = config_from_stage(base_config, variant.seed)
            seed_cache[variant.seed] = design_open_leakage_adjoint_horizon(
                path_pulse,
                seed_config,
            )
            seed_pulse, seed_objective, seed_iterations, seed_success, seed_seconds = (
                seed_cache[variant.seed]
            )
            rows.extend(
                annotate_rows(
                    evaluate_pulse(
                        seed_pulse,
                        "seed_target10_leak15",
                        path_seconds + seed_seconds,
                        seed_objective,
                        seed_iterations,
                        seed_success,
                        disorder_strength=base_config.eval_strength,
                        test_seeds=eval_seeds,
                    ),
                    "target-1.0 leakage-1.5 seed",
                    stage_description((variant.seed,)),
                )
            )

        print(f"running {variant.controller}", flush=True)
        seed_pulse, seed_objective, seed_iterations, seed_success, seed_seconds = (
            seed_cache[variant.seed]
        )
        pulse = np.array(seed_pulse, copy=True)
        total_seconds = path_seconds + seed_seconds
        total_iterations = seed_iterations
        success = seed_success
        objective = seed_objective
        for stage in variant.stages:
            config = config_from_stage(base_config, stage)
            pulse, objective, iterations, stage_success, seconds = (
                design_open_leakage_adjoint_horizon(pulse, config)
            )
            total_seconds += seconds
            total_iterations += iterations
            success = success and stage_success
        rows.extend(
            annotate_rows(
                evaluate_pulse(
                    pulse,
                    variant.controller,
                    total_seconds,
                    objective,
                    total_iterations,
                    success,
                    disorder_strength=base_config.eval_strength,
                    test_seeds=eval_seeds,
                ),
                variant.label,
                stage_description((variant.seed,) + variant.stages),
            )
        )

    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", help="comma-separated controller names")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
