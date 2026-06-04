"""Gradient certificate for selected operator-net training strategies.

The strategy comparison identifies the best deterministic 2145-point net rows:

* Z: hard-augmented mean-plus-worst terminal-value score.
* H: full 2145-point max-net terminal-value score.

This script regenerates those two pulses, evaluates terminal-infidelity
gradients on the same Pauli-ball coefficient net, and combines the worst net
infidelity with the audited-gradient-plus-Hessian envelope
``max ||grad ell|| + 4T^2 h``.  It is a sensitivity certificate for the selected
finite-run strategies, not a fixed-depth all-time value theorem.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from adaptive_operator_net_terminal_value_horizon import (
    run_design_stage,
    select_hard_coefficients,
    unique_coefficients,
)
from finite_net_operator_audit import coefficient_net
from operator_net_gradient_certificate_audit import evaluate_gradient_net, summarize
from operator_net_trained_terminal_value_horizon import OperatorNetHorizonConfig
from paths import result_path


def selected_designs() -> tuple[dict[str, object], ...]:
    return (
        {
            "task": "Z",
            "method": "selected_hard_augmented_mean_worst",
            "score_mode": "mean_worst",
            "worst_weight": 0.25,
            "hard_points": 256,
            "full_train": False,
        },
        {
            "task": "H",
            "method": "selected_full2145_max_net",
            "score_mode": "max_net",
            "worst_weight": 0.25,
            "hard_points": 0,
            "full_train": True,
        },
    )


def regenerate_selected_pulse(
    design: dict[str, object],
    base_coeffs: np.ndarray,
    eval_coeffs: np.ndarray,
    base_spacing: float,
    base_h: float,
    eval_spacing: float,
    eval_h: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    task = str(design["task"])
    method = str(design["method"])
    score_mode = str(design["score_mode"])
    worst_weight = float(design["worst_weight"])
    hard_points = int(design["hard_points"])
    config = OperatorNetHorizonConfig(
        ball_radius=0.08,
        train_points_per_axis=7,
        eval_points_per_axis=13,
        score_mode=score_mode,
        worst_weight=worst_weight,
    )

    if bool(design["full_train"]):
        train_coeffs = eval_coeffs
        train_spacing = eval_spacing
        train_h = eval_h
        hard_used = 0
    else:
        print(
            f"pretraining coarse net for hard-point selection: {task} {method}",
            flush=True,
        )
        _, _, _, _, coarse_eval_states = run_design_stage(
            task=task,
            design_label=f"{method}_coarse_selector",
            train_coeffs=base_coeffs,
            eval_coeffs=eval_coeffs,
            train_spacing=base_spacing,
            train_covering_radius=base_h,
            eval_spacing=eval_spacing,
            eval_covering_radius=eval_h,
            base_train_points=len(base_coeffs),
            hard_points_requested=hard_points,
            hard_points_used=0,
            config=config,
        )
        hard_coeffs = select_hard_coefficients(
            task,
            eval_coeffs,
            coarse_eval_states,
            hard_points,
        )
        train_coeffs = unique_coefficients(np.vstack((base_coeffs, hard_coeffs)))
        train_spacing = base_spacing
        train_h = base_h
        hard_used = len(train_coeffs) - len(base_coeffs)

    print(
        f"regenerating selected pulse: {task} {method}; "
        f"train points={len(train_coeffs)}",
        flush=True,
    )
    started = time.perf_counter()
    _, _, _, pulse, _ = run_design_stage(
        task=task,
        design_label=method,
        train_coeffs=train_coeffs,
        eval_coeffs=eval_coeffs,
        train_spacing=train_spacing,
        train_covering_radius=train_h,
        eval_spacing=eval_spacing,
        eval_covering_radius=eval_h,
        base_train_points=len(base_coeffs),
        hard_points_requested=hard_points,
        hard_points_used=hard_used,
        config=config,
    )
    metadata: dict[str, float | int | str] = {
        "task": task,
        "method": method,
        "score_mode": score_mode,
        "worst_weight": worst_weight,
        "train_net_points": len(train_coeffs),
        "base_train_points": len(base_coeffs),
        "hard_points_requested": hard_points,
        "hard_points_used": hard_used,
        "design_seconds": time.perf_counter() - started,
        "pulse_energy": float(np.mean(np.sum(pulse * pulse, axis=1))),
    }
    return pulse, metadata


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    metadata: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    annotated = []
    for row in rows:
        item = dict(row)
        item["score_mode"] = metadata["score_mode"]
        item["worst_weight"] = metadata["worst_weight"]
        item["train_net_points"] = metadata["train_net_points"]
        item["base_train_points"] = metadata["base_train_points"]
        item["hard_points_requested"] = metadata["hard_points_requested"]
        item["hard_points_used"] = metadata["hard_points_used"]
        annotated.append(item)
    return annotated


def summary_with_metadata(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, str]]:
    summary_rows = summarize(rows)
    metadata_by_key: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["method"]))
        metadata_by_key[key] = row
    for summary_row in summary_rows:
        key = (summary_row["task"], summary_row["method"])
        metadata = metadata_by_key[key]
        summary_row["score_mode"] = str(metadata["score_mode"])
        summary_row["worst_weight"] = f"{float(metadata['worst_weight']):.6g}"
        summary_row["train_net_points"] = str(int(metadata["train_net_points"]))
        summary_row["hard_points_used"] = str(int(metadata["hard_points_used"]))
    return summary_rows


def write_markdown_table(f, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    summary_rows = summary_with_metadata(rows)
    result_file = result_path("selected_operator_net_strategy_gradient_certificate_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    net_file = result_path("selected_operator_net_strategy_gradient_certificate_net_results.csv")
    with net_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_file = result_path("selected_operator_net_strategy_gradient_certificate_summary.md")
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Selected Operator-Net Strategy Gradient Certificate\n\n")
        f.write(
            "Gradient-net sensitivity certificate for the two strategy rows "
            "selected by deterministic worst-net fidelity in the operator-net "
            "training comparison. The continuous lower bound uses the same "
            "`max ||grad ell|| + 4T^2 h` certificate as the default "
            "operator-net gradient audit.\n\n"
        )
        write_markdown_table(f, summary_rows)

    print(f"wrote summary CSV to {result_file}")
    print(f"wrote {len(rows)} net rows to {net_file}")
    print(f"wrote summary Markdown to {summary_file}")
    for row in summary_rows:
        print(row)


def main() -> None:
    base_coeffs, base_spacing, base_h = coefficient_net(0.08, 7)
    eval_coeffs, eval_spacing, eval_h = coefficient_net(0.08, 13)
    all_rows: list[dict[str, float | int | str]] = []
    for design in selected_designs():
        pulse, metadata = regenerate_selected_pulse(
            design,
            base_coeffs,
            eval_coeffs,
            base_spacing,
            base_h,
            eval_spacing,
            eval_h,
        )
        print(
            f"evaluating selected gradient certificate: {metadata['task']} "
            f"{metadata['method']}",
            flush=True,
        )
        rows = evaluate_gradient_net(
            str(metadata["task"]),
            pulse,
            eval_coeffs,
            str(metadata["method"]),
            float(metadata["design_seconds"]),
            0.08,
            13,
            eval_spacing,
            eval_h,
        )
        all_rows.extend(annotate_rows(rows, metadata))
    write_outputs(all_rows)


if __name__ == "__main__":
    main()
