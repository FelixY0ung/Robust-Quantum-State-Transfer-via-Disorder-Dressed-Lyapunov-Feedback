"""Worst-seed-preserving leakage-cap refinement audit.

The existing cap-refinement audit starts from the high-mean target-push
continuation pulse.  This focused companion starts from the worst-weighted
target-push continuation pulse and then applies the same no-reference
leakage-cap polish.  The goal is to test whether the five-level
leakage-plus-Lindblad frontier can preserve the stronger worst-seed behavior
while reducing transient leakage, without introducing a terminal GRAPE
reference.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from open_leakage_cap_refinement import (
    LOW_LEAKAGE_SEED,
    CapVariant,
    StageSpec,
    adjoint_config_from_stage,
    annotate_rows,
    cap_config_from_variant,
    combined_points,
    design_cap_refinement,
    evaluate_standard,
    load_reference_points,
    stage_text,
    summarize,
)
from paths import figure_path, result_path
from transmon_leakage_horizon import design_pulse
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
)


ROBUST_TARGET_PUSH_STAGES = (
    StageSpec(1.0, 0.8, 0.025, 6, 0.5),
    StageSpec(1.0, 0.5, 0.015, 6, 0.5),
)

WORST_CAP_VARIANTS = (
    CapVariant(
        "worstcap050_w120_mean02_worst05",
        "worst-seed target push, cap 0.050, weight 120, mean leak 0.2, worst 0.5",
        leakage_cap=0.050,
        cap_weight=120.0,
        trust_radius=0.010,
        horizon_maxiter=6,
        mean_leakage_weight=0.2,
        worst_weight=0.5,
    ),
    CapVariant(
        "worstcap055_w80_mean02_worst05",
        "worst-seed target push, cap 0.055, weight 80, mean leak 0.2, worst 0.5",
        leakage_cap=0.055,
        cap_weight=80.0,
        trust_radius=0.012,
        horizon_maxiter=6,
        mean_leakage_weight=0.2,
        worst_weight=0.5,
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


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    reference_points = load_reference_points()
    reference_points.update(combined_points(rows))
    labels = {
        "worstcap_path_seed": "Path",
        "worstcap_low_leak_seed": "Low-leakage seed",
        "worstcap_target_push_reference": "Worst-weighted target push",
        "worstcap050_w120_mean02_worst05": "Worst-cap 0.050",
        "worstcap055_w80_mean02_worst05": "Worst-cap 0.055",
        "adjoint_horizon": "Ref.-adjoint",
        "leakage_penalized_grape": "Leakage-GRAPE",
        "hf_leak05_worst05": "HF worst push",
        "cap050_w120_mean02_worst05": "Robust cap",
        "cap050_w120": "High-mean cap",
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for controller, (mean_fid, worst_fid, leak) in reference_points.items():
        is_new = controller.startswith("worstcap")
        if not is_new and controller not in {
            "adjoint_horizon",
            "leakage_penalized_grape",
            "hf_leak05_worst05",
            "cap050_w120",
            "cap050_w120_mean02_worst05",
        }:
            continue
        color = "#9467bd" if is_new else "0.65"
        marker = "D" if is_new else "o"
        alpha = 0.95 if is_new else 0.60
        ax.scatter(leak, mean_fid, marker=marker, s=42, color=color, alpha=alpha)
        ax.scatter(leak, worst_fid, marker="x", s=28, color=color, alpha=alpha)
        ax.annotate(
            labels.get(controller, controller),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Worst-seed leakage-cap refinement")
    ax.set_xlim(0.045, 0.095)
    ax.set_ylim(0.74, 0.96)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_worst_cap_refinement.pdf"))
    fig.savefig(figure_path("open_leakage_worst_cap_refinement.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_worst_cap_refinement_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    headers = list(summary_rows[0].keys())
    summary_file = result_path("open_leakage_worst_cap_refinement_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Worst-Seed Leakage-Cap Refinement\n\n")
        f.write(
            "No-reference leakage-cap refinements initialized from the "
            "worst-seed-weighted target-push continuation pulse and trained "
            "through the five-level Lindblad leakage model.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_worst_cap_refinement.pdf')}")


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = WORST_CAP_VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no worst-cap variants matched --only={args.only!r}")

    if args.quick:
        path_kwargs: dict[str, object] = {
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
    import time

    start = time.perf_counter()
    path_pulse = design_pulse(**path_kwargs)
    path_seconds = time.perf_counter() - start

    reference_text = stage_text((LOW_LEAKAGE_SEED,) + ROBUST_TARGET_PUSH_STAGES)
    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        annotate_rows(
            evaluate_standard(
                path_pulse,
                "worstcap_path_seed",
                path_seconds,
                None,
                None,
                None,
                base_config,
            ),
            "path seed",
            reference_text,
            None,
        )
    )

    print("building low-leakage continuation seed", flush=True)
    low_config = adjoint_config_from_stage(base_config, LOW_LEAKAGE_SEED)
    low_pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
        path_pulse,
        low_config,
    )
    low_total_seconds = path_seconds + seconds
    rows.extend(
        annotate_rows(
            evaluate_standard(
                low_pulse,
                "worstcap_low_leak_seed",
                low_total_seconds,
                objective,
                iterations,
                success,
                base_config,
            ),
            "low-leakage seed",
            reference_text,
            None,
        )
    )

    reference_pulse = low_pulse
    total_stage_seconds = low_total_seconds
    reference_objective = objective
    reference_iterations = iterations
    reference_success = success
    for stage in ROBUST_TARGET_PUSH_STAGES:
        print(
            "running robust target-push stage "
            f"a={stage.terminal_target_weight:g}, lw={stage.leakage_weight:g}, "
            f"w={stage.worst_weight:g}",
            flush=True,
        )
        stage_config = adjoint_config_from_stage(base_config, stage)
        reference_pulse, reference_objective, iters, stage_success, seconds = (
            design_open_leakage_adjoint_horizon(reference_pulse, stage_config)
        )
        total_stage_seconds += seconds
        reference_iterations += iters
        reference_success = reference_success and stage_success

    rows.extend(
        annotate_rows(
            evaluate_standard(
                reference_pulse,
                "worstcap_target_push_reference",
                total_stage_seconds,
                reference_objective,
                reference_iterations,
                reference_success,
                base_config,
            ),
            "worst-seed target-push reference",
            reference_text,
            None,
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        cap_config = cap_config_from_variant(base_config, variant)
        pulse, objective, iterations, success, seconds = design_cap_refinement(
            reference_pulse,
            cap_config,
        )
        rows.extend(
            annotate_rows(
                evaluate_standard(
                    pulse,
                    variant.controller,
                    total_stage_seconds + seconds,
                    objective,
                    iterations,
                    success,
                    base_config,
                ),
                variant.label,
                reference_text,
                cap_config,
            )
        )

    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", help="comma-separated controller names from the audit")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
