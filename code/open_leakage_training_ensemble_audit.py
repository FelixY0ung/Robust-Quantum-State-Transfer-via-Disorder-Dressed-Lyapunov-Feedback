"""Training-ensemble audit for no-reference open-leakage continuation.

The continuation controller in ``open_leakage_continuation_sweep.py`` uses four
training disorder seeds in the five-level leakage-plus-Lindblad model.  This
audit asks whether the remaining no-reference fidelity gap is mainly due to the
small training ensemble.  It reruns the same low-leakage-seed plus
target/leakage continuation recipe with three training sets:

* the original four seeds;
* a larger eight-seed ensemble;
* a hard-sample-enriched ensemble that adds the two hardest seeds observed in
  the earlier held-out audit, then evaluates on a fresh disjoint seed range.

The enriched row is an adversarial-sample diagnostic, not a claim of continuum
robust optimization.
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
    config_from_stage,
    stage_description,
)
from paths import figure_path, result_path
from transmon_leakage_horizon import design_pulse
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
)
from transmon_open_system_leakage import evaluate_pulse


CONTINUATION_STAGES = (
    StageSpec(1.0, 0.8, 0.02, 6),
    StageSpec(1.0, 0.8, 0.015, 6),
)


@dataclass(frozen=True)
class TrainingSet:
    controller: str
    label: str
    seeds: tuple[int, ...]


TRAINING_SETS = (
    TrainingSet("train4_continuation_leak08", "original four training seeds", (0, 1, 2, 3)),
    TrainingSet("train8_continuation_leak08", "eight training seeds", tuple(range(8))),
    TrainingSet(
        "hard_enriched_continuation_leak08",
        "four seeds plus hard samples 13 and 37",
        (0, 1, 2, 3, 13, 37),
    ),
)


def config_with_train_seeds(
    base: OpenLeakageAdjointConfig,
    seeds: tuple[int, ...],
) -> OpenLeakageAdjointConfig:
    return OpenLeakageAdjointConfig(
        segments=base.segments,
        horizon_steps=base.horizon_steps,
        horizon_maxiter=base.horizon_maxiter,
        train_strength=base.train_strength,
        train_seeds=seeds,
        eval_strength=base.eval_strength,
        eval_seeds=base.eval_seeds,
        umax=base.umax,
        trust_radius=base.trust_radius,
        worst_weight=base.worst_weight,
        leakage_weight=base.leakage_weight,
        energy_weight=base.energy_weight,
        trust_weight=base.trust_weight,
        terminal_target_weight=base.terminal_target_weight,
    )


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
                "label": str(first["training_set_label"]),
                "noise_case": noise_case,
                "n": str(len(items)),
                "train_seeds": str(first["train_seeds"]),
                "eval_seeds": str(first["eval_seeds"]),
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


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    training_set: TrainingSet | None,
    eval_seeds: tuple[int, ...],
    stage_text: str,
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "training_set_label": "path seed" if training_set is None else training_set.label,
                "train_seeds": "" if training_set is None else ",".join(str(seed) for seed in training_set.seeds),
                "eval_seeds": f"{min(eval_seeds)}-{max(eval_seeds)}",
                "stage_description": stage_text,
            }
        )
        annotated.append(item)
    return annotated


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


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    points = combined_points(rows)
    labels = {
        "training_ensemble_path_seed": "Path",
        "train4_continuation_leak08": "Train-4",
        "train8_continuation_leak08": "Train-8",
        "hard_enriched_continuation_leak08": "Hard-enriched",
    }

    fig, ax = plt.subplots(figsize=(5.4, 3.35))
    for controller, (mean_fid, worst_fid, leak) in points.items():
        marker = "o" if controller == "training_ensemble_path_seed" else "D"
        color = "0.55" if controller == "training_ensemble_path_seed" else "#9467bd"
        ax.scatter(leak, mean_fid, marker=marker, s=42, color=color)
        ax.scatter(leak, worst_fid, marker="x", s=28, color=color)
        ax.annotate(
            labels.get(controller, controller),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Training-ensemble audit on fresh held-out seeds")
    ax.set_xlim(0.0, 0.08)
    ax.set_ylim(0.50, 0.97)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_training_ensemble_audit.pdf"))
    fig.savefig(figure_path("open_leakage_training_ensemble_audit.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_training_ensemble_audit_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_training_ensemble_audit_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Open-Leakage Training-Ensemble Audit\n\n")
        f.write(
            "No-reference continuation horizons retrained with larger or hard-sample-enriched "
            "training ensembles and evaluated on a fresh disjoint held-out seed range.\n\n"
        )
        headers = list(summary[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_training_ensemble_audit.pdf')}")


def config_from_args(args: argparse.Namespace) -> OpenLeakageAdjointConfig:
    if args.quick:
        return OpenLeakageAdjointConfig(
            segments=30,
            horizon_steps=3,
            horizon_maxiter=2,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(60, 65)),
        )
    return OpenLeakageAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
        eval_seeds=tuple(range(args.eval_seed_start, args.eval_seed_stop)),
    )


def selected_training_sets(args: argparse.Namespace) -> tuple[TrainingSet, ...]:
    if not args.only:
        return TRAINING_SETS
    wanted = set(args.only.split(","))
    selected = tuple(item for item in TRAINING_SETS if item.controller in wanted)
    if not selected:
        raise ValueError(f"no training-set variants matched --only={args.only!r}")
    return selected


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    eval_seeds = tuple(base_config.eval_seeds)
    training_sets = selected_training_sets(args)

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
                "training_ensemble_path_seed",
                path_seconds,
                None,
                None,
                None,
                disorder_strength=base_config.eval_strength,
                test_seeds=range(min(eval_seeds), max(eval_seeds) + 1),
            ),
            None,
            eval_seeds,
            "",
        )
    )

    for training_set in training_sets:
        print(f"running {training_set.controller}", flush=True)
        train_base = config_with_train_seeds(base_config, training_set.seeds)
        seed_config = config_from_stage(train_base, LOW_LEAKAGE_SEED)
        pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
            path_pulse,
            seed_config,
        )
        total_seconds = path_seconds + seconds
        total_iterations = iterations
        total_success = success
        for stage in CONTINUATION_STAGES:
            config = config_from_stage(train_base, stage)
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
                    training_set.controller,
                    total_seconds,
                    objective,
                    total_iterations,
                    total_success,
                    disorder_strength=base_config.eval_strength,
                    test_seeds=range(min(eval_seeds), max(eval_seeds) + 1),
                ),
                training_set,
                eval_seeds,
                stage_description((LOW_LEAKAGE_SEED,) + CONTINUATION_STAGES),
            )
        )

    write_outputs(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", help="comma-separated controller names from the audit")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    parser.add_argument("--eval-seed-start", type=int, default=60)
    parser.add_argument("--eval-seed-stop", type=int, default=110)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
