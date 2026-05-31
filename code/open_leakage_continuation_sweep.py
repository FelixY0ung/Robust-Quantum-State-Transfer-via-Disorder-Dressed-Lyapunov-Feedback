"""Continuation no-reference open-leakage horizon sweep.

This script tests whether the combined five-level leakage-plus-Lindblad gap can
be reduced without using a terminal GRAPE reference.  It first constructs a
low-leakage target-balanced direct horizon, then applies target/leakage
continuation stages with tighter trust regions.  The result is a reproducible
no-reference stress test that complements the single-stage Pareto refinement.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from paths import figure_path, result_path
from transmon_leakage_horizon import design_pulse
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
    load_result_rows,
)
from transmon_open_system_leakage import evaluate_pulse


@dataclass(frozen=True)
class StageSpec:
    terminal_target_weight: float
    leakage_weight: float
    trust_radius: float
    horizon_maxiter: int
    worst_weight: float = 0.25


@dataclass(frozen=True)
class ContinuationVariant:
    controller: str
    label: str
    stages: tuple[StageSpec, ...]


LOW_LEAKAGE_SEED = StageSpec(
    terminal_target_weight=0.8,
    leakage_weight=1.5,
    trust_radius=0.04,
    horizon_maxiter=6,
)


VARIANTS = (
    ContinuationVariant(
        "continuation_target08_leak12",
        "low-leak seed, target 0.8 then leak 1.2",
        (
            StageSpec(1.0, 0.8, 0.02, 6),
            StageSpec(1.0, 1.2, 0.015, 6),
        ),
    ),
    ContinuationVariant(
        "continuation_target05_leak15",
        "low-leak seed, target 0.5 then leak 1.5",
        (
            StageSpec(1.0, 0.5, 0.025, 6),
            StageSpec(1.0, 1.5, 0.015, 6),
        ),
    ),
    ContinuationVariant(
        "continuation_robust_leak12",
        "low-leak seed, robust target 0.8 then leak 1.2",
        (
            StageSpec(1.0, 0.8, 0.025, 6, 0.5),
            StageSpec(1.0, 1.2, 0.015, 6, 0.5),
        ),
    ),
)


def config_from_stage(
    base: OpenLeakageAdjointConfig,
    stage: StageSpec,
) -> OpenLeakageAdjointConfig:
    return OpenLeakageAdjointConfig(
        segments=base.segments,
        horizon_steps=base.horizon_steps,
        horizon_maxiter=stage.horizon_maxiter,
        train_strength=base.train_strength,
        train_seeds=base.train_seeds,
        eval_strength=base.eval_strength,
        eval_seeds=base.eval_seeds,
        umax=base.umax,
        trust_radius=stage.trust_radius,
        worst_weight=stage.worst_weight,
        leakage_weight=stage.leakage_weight,
        energy_weight=base.energy_weight,
        trust_weight=base.trust_weight,
        terminal_target_weight=stage.terminal_target_weight,
    )


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    label: str,
    stage_description: str,
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "continuation_label": label,
                "stage_description": stage_description,
            }
        )
        annotated.append(item)
    return annotated


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), str(row["noise_case"])), []).append(row)

    summaries: list[dict[str, str]] = []
    for (controller, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        final_leaks = np.array([float(row["final_leakage"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        first = items[0]
        ci95 = 1.96 * float(np.std(fids)) / np.sqrt(len(fids))
        summaries.append(
            {
                "controller": controller,
                "label": str(first["continuation_label"]),
                "noise_case": noise_case,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{float(first['pulse_energy']):.6g}",
                "segments": str(first["segments"]),
                "training_seconds": f"{float(first['training_seconds']):.4g}",
                "stage_description": str(first["stage_description"]),
            }
        )
    return summaries


def combined_points(rows: list[dict[str, float | int | str]]) -> dict[str, tuple[float, float, float]]:
    points: dict[str, tuple[float, float, float]] = {}
    for controller in sorted({str(row["controller"]) for row in rows}):
        items = [
            row
            for row in rows
            if str(row["controller"]) == controller
            and str(row["noise_case"]) == "combined"
        ]
        if not items:
            continue
        fids = np.array([float(row["final_fidelity"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        points[controller] = (
            float(np.mean(fids)),
            float(np.min(fids)),
            float(np.mean(max_leaks)),
        )
    return points


def load_reference_points() -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, float | int | str]] = []
    rows.extend(load_result_rows("transmon_open_leakage_adjoint_results.csv"))
    rows.extend(load_result_rows("open_leakage_pareto_refinement_results.csv"))
    rows.extend(load_result_rows("open_leakage_integrated_sweep_results.csv"))
    rows.extend(
        row
        for row in load_result_rows("transmon_open_system_leakage_results.csv")
        if str(row.get("controller")) in {"adjoint_horizon", "leakage_penalized_grape"}
    )
    return combined_points(rows)


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    reference_points = load_reference_points()
    new_points = combined_points(rows)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for controller, (mean_fid, worst_fid, leak) in reference_points.items():
        ax.scatter(leak, mean_fid, marker="o", s=22, color="0.68", alpha=0.6)
        ax.scatter(leak, worst_fid, marker="x", s=18, color="0.68", alpha=0.6)
    for controller, (mean_fid, worst_fid, leak) in new_points.items():
        if controller == "continuation_path_seed":
            continue
        marker = "D" if controller.startswith("continuation_stageA") else "s"
        ax.scatter(leak, mean_fid, marker=marker, s=44, color="#2ca02c")
        ax.annotate(
            controller.replace("continuation_", ""),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
        ax.scatter(leak, worst_fid, marker="x", s=28, color="#2ca02c")
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("No-reference continuation open-leakage sweep")
    ax.set_xlim(0.0, 0.07)
    ax.set_ylim(0.50, 0.96)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_continuation_sweep.pdf"))
    fig.savefig(figure_path("open_leakage_continuation_sweep.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_continuation_sweep_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_continuation_sweep_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Open-Leakage Continuation Sweep\n\n")
        f.write(
            "No-reference continuation horizons trained through the five-level "
            "Lindblad leakage model.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_continuation_sweep.pdf')}")


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


def stage_description(stages: tuple[StageSpec, ...]) -> str:
    return "; ".join(
        f"a={stage.terminal_target_weight:g},lw={stage.leakage_weight:g},"
        f"r={stage.trust_radius:g},w={stage.worst_weight:g}"
        for stage in stages
    )


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no continuation variants matched --only={args.only!r}")

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
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        annotate_rows(
            evaluate_pulse(
                path_pulse,
                "continuation_path_seed",
                path_seconds,
                None,
                None,
                None,
                disorder_strength=base_config.eval_strength,
                test_seeds=range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1),
            ),
            "path seed",
            "",
        )
    )

    print("running low-leakage continuation seed", flush=True)
    seed_config = config_from_stage(base_config, LOW_LEAKAGE_SEED)
    seed_pulse, seed_objective, seed_iterations, seed_success, seed_seconds = (
        design_open_leakage_adjoint_horizon(path_pulse, seed_config)
    )
    rows.extend(
        annotate_rows(
            evaluate_pulse(
                seed_pulse,
                "continuation_stageA_alpha0p8_lw1p5",
                path_seconds + seed_seconds,
                seed_objective,
                seed_iterations,
                seed_success,
                disorder_strength=base_config.eval_strength,
                test_seeds=range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1),
            ),
            "low-leakage seed",
            stage_description((LOW_LEAKAGE_SEED,)),
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        pulse = seed_pulse
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
                    test_seeds=range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1),
                ),
                variant.label,
                stage_description((LOW_LEAKAGE_SEED,) + variant.stages),
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
