"""Long-horizon no-reference open-leakage continuation sweep.

The strongest no-reference open-leakage continuation row in the manuscript uses
five-step short horizons.  This audit tests whether the remaining gap to the
reference-assisted horizon is mainly caused by too short a local lookahead.  It
keeps the same finite-candidate path seed and the same Liouville-space
Frechet-gradient objective, then reruns continuation with longer direct
open-system leakage-aware horizons and slightly wider trust regions.

The sweep writes separate result files so the manuscript's established
continuation frontier is not overwritten.  The rows remain GRAPE-free: no
terminal optimal-control reference pulse is used.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from open_leakage_continuation_sweep import (
    LOW_LEAKAGE_SEED,
    StageSpec,
    stage_description,
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
class LongHorizonVariant:
    controller: str
    label: str
    horizon_steps: int
    seed_stage: StageSpec
    stages: tuple[StageSpec, ...]


VARIANTS = (
    LongHorizonVariant(
        "long_q8_leak08",
        "q=8 continuation, leak 0.8",
        8,
        LOW_LEAKAGE_SEED,
        (
            StageSpec(1.0, 0.8, 0.025, 8),
            StageSpec(1.0, 0.8, 0.020, 8),
        ),
    ),
    LongHorizonVariant(
        "long_q8_leak10",
        "q=8 continuation, leak 1.0",
        8,
        LOW_LEAKAGE_SEED,
        (
            StageSpec(1.0, 0.8, 0.025, 8),
            StageSpec(1.0, 1.0, 0.020, 8),
        ),
    ),
    LongHorizonVariant(
        "long_q8_robust_leak08",
        "q=8 robust continuation, leak 0.8",
        8,
        LOW_LEAKAGE_SEED,
        (
            StageSpec(1.0, 0.8, 0.025, 8, 0.5),
            StageSpec(1.0, 0.8, 0.020, 8, 0.5),
        ),
    ),
    LongHorizonVariant(
        "long_q10_leak10",
        "q=10 continuation, leak 1.0",
        10,
        LOW_LEAKAGE_SEED,
        (
            StageSpec(1.0, 0.8, 0.025, 6),
            StageSpec(1.0, 1.0, 0.020, 6),
        ),
    ),
)


def config_from_stage(
    base: OpenLeakageAdjointConfig,
    stage: StageSpec,
    horizon_steps: int,
) -> OpenLeakageAdjointConfig:
    return OpenLeakageAdjointConfig(
        segments=base.segments,
        horizon_steps=horizon_steps,
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
    horizon_steps: int | str,
    stage_text: str,
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "sweep_label": label,
                "horizon_steps": horizon_steps,
                "stage_description": stage_text,
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
                "label": str(first["sweep_label"]),
                "noise_case": noise_case,
                "n": str(len(items)),
                "horizon_steps": str(first["horizon_steps"]),
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
        leaks = np.array([float(row["max_leakage"]) for row in items])
        points[controller] = (
            float(np.mean(fids)),
            float(np.min(fids)),
            float(np.mean(leaks)),
        )
    return points


def load_reference_points() -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, float | int | str]] = []
    rows.extend(load_result_rows("open_leakage_continuation_sweep_results.csv"))
    rows.extend(
        row
        for row in load_result_rows("transmon_open_system_leakage_results.csv")
        if str(row.get("controller")) in {"adjoint_horizon", "leakage_penalized_grape"}
    )
    wanted = {
        "continuation_target08_leak12",
        "continuation_target08_leak10",
        "continuation_target08_leak08",
        "continuation_robust_leak08",
        "adjoint_horizon",
        "leakage_penalized_grape",
    }
    return {
        key: value
        for key, value in combined_points(rows).items()
        if key in wanted
    }


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    reference_points = load_reference_points()
    new_points = combined_points(rows)
    fig, ax = plt.subplots(figsize=(5.4, 3.35))
    for controller, (mean_fid, worst_fid, leak) in reference_points.items():
        ax.scatter(leak, mean_fid, marker="o", s=26, color="0.65", alpha=0.75)
        ax.scatter(leak, worst_fid, marker="x", s=18, color="0.65", alpha=0.75)
    for controller, (mean_fid, worst_fid, leak) in new_points.items():
        if controller == "long_horizon_path_seed":
            continue
        ax.scatter(leak, mean_fid, marker="D", s=44, color="#d62728")
        ax.scatter(leak, worst_fid, marker="x", s=30, color="#d62728")
        ax.annotate(
            controller.replace("long_", ""),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Long-horizon open-leakage continuation")
    ax.set_xlim(0.0, 0.085)
    ax.set_ylim(0.50, 0.98)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_long_horizon_sweep.pdf"))
    fig.savefig(figure_path("open_leakage_long_horizon_sweep.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_long_horizon_sweep_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_long_horizon_sweep_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Long-Horizon Open-Leakage Continuation Sweep\n\n")
        f.write(
            "No-reference continuation horizons with longer direct Lindblad leakage-aware "
            "lookahead and wider trust regions.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_long_horizon_sweep.pdf')}")


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


def selected_variants(args: argparse.Namespace) -> tuple[LongHorizonVariant, ...]:
    if not args.only:
        return VARIANTS
    wanted = set(args.only.split(","))
    selected = tuple(variant for variant in VARIANTS if variant.controller in wanted)
    if not selected:
        raise ValueError(f"no long-horizon variants matched --only={args.only!r}")
    return selected


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = selected_variants(args)

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

    eval_seed_range = range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1)
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        annotate_rows(
            evaluate_pulse(
                path_pulse,
                "long_horizon_path_seed",
                path_seconds,
                None,
                None,
                None,
                disorder_strength=base_config.eval_strength,
                test_seeds=eval_seed_range,
            ),
            "path seed",
            "",
            "",
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        seed_config = config_from_stage(
            base_config,
            variant.seed_stage,
            variant.horizon_steps,
        )
        pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
            path_pulse,
            seed_config,
        )
        total_seconds = path_seconds + seconds
        total_iterations = iterations
        total_success = success
        for stage in variant.stages:
            config = config_from_stage(base_config, stage, variant.horizon_steps)
            pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
                pulse,
                config,
            )
            total_seconds += seconds
            total_iterations += iterations
            total_success = total_success and success

        rows.extend(
            annotate_rows(
                evaluate_pulse(
                    pulse,
                    variant.controller,
                    total_seconds,
                    objective,
                    total_iterations,
                    total_success,
                    disorder_strength=base_config.eval_strength,
                    test_seeds=eval_seed_range,
                ),
                variant.label,
                variant.horizon_steps,
                stage_description((variant.seed_stage,) + variant.stages),
            )
        )

    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", help="comma-separated controller names from the sweep")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
