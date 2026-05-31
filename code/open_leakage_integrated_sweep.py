"""Integrated no-reference open-leakage horizon sweep.

This script extends the five-level leakage-plus-Lindblad diagnostic by testing
single-stage direct horizon objectives with stronger terminal target pressure
and leakage penalties.  It reuses the exact Liouville-space Frechet-gradient
machinery from ``transmon_open_leakage_adjoint_horizon.py`` and writes separate
result files so the published baseline rows are not overwritten.

The sweep is intentionally modest: it is meant to identify whether a single
integrated no-reference objective can move the direct horizon closer to the
reference-assisted Pareto front, not to perform an exhaustive hyperparameter
search.
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
class SweepVariant:
    controller: str
    label: str
    terminal_target_weight: float
    leakage_weight: float
    trust_radius: float
    horizon_maxiter: int


COMPACT_VARIANTS = (
    SweepVariant("integrated_alpha0p9_lw1p0", "alpha 0.9, leak 1.0", 0.9, 1.0, 0.025, 4),
    SweepVariant("integrated_alpha1p0_lw1p2", "alpha 1.0, leak 1.2", 1.0, 1.2, 0.025, 4),
    SweepVariant("integrated_alpha1p0_lw1p5", "alpha 1.0, leak 1.5", 1.0, 1.5, 0.025, 4),
)

EXTENDED_VARIANTS = COMPACT_VARIANTS + (
    SweepVariant("integrated_alpha1p0_lw1p2_trust004", "alpha 1.0, leak 1.2, trust 0.04", 1.0, 1.2, 0.04, 6),
    SweepVariant("integrated_alpha1p0_lw1p5_trust004", "alpha 1.0, leak 1.5, trust 0.04", 1.0, 1.5, 0.04, 6),
)


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    variant: SweepVariant | None,
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        if variant is None:
            item.update(
                {
                    "sweep_label": "path seed",
                    "terminal_target_weight": "",
                    "leakage_weight": "",
                    "trust_radius": "",
                    "horizon_maxiter": "",
                }
            )
        else:
            item.update(
                {
                    "sweep_label": variant.label,
                    "terminal_target_weight": variant.terminal_target_weight,
                    "leakage_weight": variant.leakage_weight,
                    "trust_radius": variant.trust_radius,
                    "horizon_maxiter": variant.horizon_maxiter,
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
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{float(first['pulse_energy']):.6g}",
                "segments": str(first["segments"]),
                "training_seconds": f"{float(first['training_seconds']):.4g}",
                "terminal_target_weight": str(first["terminal_target_weight"]),
                "leakage_weight": str(first["leakage_weight"]),
                "trust_radius": str(first["trust_radius"]),
                "horizon_maxiter": str(first["horizon_maxiter"]),
            }
        )
    return summaries


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_integrated_sweep_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_integrated_sweep_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Integrated Open-Leakage Horizon Sweep\n\n")
        f.write("Single-stage no-reference horizons trained through the five-level Lindblad leakage model.\n\n")
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_integrated_sweep.pdf')}")


def combined_summary_points(rows: list[dict[str, float | int | str]]) -> dict[str, tuple[float, float, float]]:
    points: dict[str, tuple[float, float, float]] = {}
    by_controller: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        if str(row["noise_case"]) == "combined":
            by_controller.setdefault(str(row["controller"]), []).append(row)
    for controller, items in by_controller.items():
        fids = np.array([float(row["final_fidelity"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        points[controller] = (float(np.mean(fids)), float(np.min(fids)), float(np.mean(max_leaks)))
    return points


def load_reference_points() -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, float | int | str]] = []
    rows.extend(load_result_rows("transmon_open_leakage_adjoint_results.csv"))
    rows.extend(
        row
        for row in load_result_rows("transmon_open_system_leakage_results.csv")
        if str(row.get("controller")) in {"adjoint_horizon", "leakage_penalized_grape"}
    )
    return combined_summary_points(rows)


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    new_points = combined_summary_points(rows)
    reference_points = load_reference_points()
    labels = {
        "open_leakage_path_seed": "Path",
        "standalone_open_leakage_adjoint": "Direct",
        "target_biased_open_leakage_adjoint": "Target-biased",
        "two_stage_target_biased_open_leakage_adjoint": "Two-stage",
        "adjoint_horizon": "Ref.-adjoint",
        "leakage_penalized_grape": "Leakage-GRAPE",
        "integrated_alpha0p9_lw1p0": "alpha0.9/lw1.0",
        "integrated_alpha1p0_lw1p2": "alpha1.0/lw1.2",
        "integrated_alpha1p0_lw1p5": "alpha1.0/lw1.5",
        "integrated_alpha1p0_lw1p2_trust004": "alpha1.0/lw1.2/t0.04",
        "integrated_alpha1p0_lw1p5_trust004": "alpha1.0/lw1.5/t0.04",
    }

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for controller, (mean_fid, worst_fid, leak) in reference_points.items():
        ax.scatter(leak, mean_fid, marker="o", s=38, color="0.55", alpha=0.75)
        ax.annotate(labels.get(controller, controller), (leak, mean_fid), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.scatter(leak, worst_fid, marker="x", s=24, color="0.55", alpha=0.75)
    for controller, (mean_fid, worst_fid, leak) in new_points.items():
        if controller == "open_leakage_path_seed":
            continue
        ax.scatter(leak, mean_fid, marker="D", s=42, color="#1f77b4")
        ax.annotate(labels.get(controller, controller), (leak, mean_fid), xytext=(4, -9), textcoords="offset points", fontsize=7)
        ax.scatter(leak, worst_fid, marker="x", s=28, color="#1f77b4")

    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Integrated open-leakage horizon sweep")
    ax.set_xlim(0.0, 0.08)
    ax.set_ylim(0.50, 0.98)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_integrated_sweep.pdf"))
    fig.savefig(figure_path("open_leakage_integrated_sweep.png"), dpi=220)
    plt.close(fig)


def config_from_args(args: argparse.Namespace) -> OpenLeakageAdjointConfig:
    if args.quick:
        return OpenLeakageAdjointConfig(
            segments=30,
            horizon_steps=3,
            horizon_maxiter=2,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 15)),
            trust_radius=0.025,
            terminal_target_weight=1.0,
        )
    return OpenLeakageAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        trust_radius=args.trust_radius,
        terminal_target_weight=1.0,
    )


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = EXTENDED_VARIANTS if args.extended else COMPACT_VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no sweep variants matched --only={args.only!r}")

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
                "open_leakage_path_seed",
                path_seconds,
                None,
                None,
                None,
                disorder_strength=base_config.eval_strength,
                test_seeds=range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1),
            ),
            None,
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        config = OpenLeakageAdjointConfig(
            segments=base_config.segments,
            horizon_steps=base_config.horizon_steps,
            horizon_maxiter=variant.horizon_maxiter if not args.quick else base_config.horizon_maxiter,
            train_strength=base_config.train_strength,
            train_seeds=base_config.train_seeds,
            eval_strength=base_config.eval_strength,
            eval_seeds=base_config.eval_seeds,
            umax=base_config.umax,
            trust_radius=variant.trust_radius if not args.quick else base_config.trust_radius,
            worst_weight=base_config.worst_weight,
            leakage_weight=variant.leakage_weight,
            energy_weight=base_config.energy_weight,
            trust_weight=base_config.trust_weight,
            terminal_target_weight=variant.terminal_target_weight,
        )
        pulse, objective, iterations, success, horizon_seconds = design_open_leakage_adjoint_horizon(
            path_pulse,
            config,
        )
        rows.extend(
            annotate_rows(
                evaluate_pulse(
                    pulse,
                    variant.controller,
                    path_seconds + horizon_seconds,
                    objective,
                    iterations,
                    success,
                    disorder_strength=config.eval_strength,
                    test_seeds=range(min(config.eval_seeds), max(config.eval_seeds) + 1),
                ),
                variant,
            )
        )

    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--only", help="comma-separated controller names from the sweep")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    parser.add_argument("--trust-radius", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
