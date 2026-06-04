"""Compare operator-net terminal-value training strategies.

This postprocessor consolidates the coarse, hard-augmented, and full-net
two-level Pauli-ball diagnostics.  It does not rerun controllers; it reads the
generated CSV artifacts and reports the dense-net/held-out/margin tradeoffs so
the manuscript can cite one reproducible comparison table.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import result_path


INPUTS = (
    {
        "prefix": "adaptive_operator_net_terminal_value_horizon",
        "family": "adaptive_w0.25",
        "default_worst_weight": 0.25,
    },
    {
        "prefix": "adaptive_operator_net_terminal_value_horizon_w05",
        "family": "adaptive_w0.5",
        "default_worst_weight": 0.5,
    },
    {
        "prefix": "operator_net_full_trained_terminal_value_horizon",
        "family": "full2145",
        "default_design_label": "full_net",
        "default_worst_weight": 0.25,
    },
    {
        "prefix": "operator_net_full_trained_terminal_value_horizon_mean_worst",
        "family": "full2145",
        "default_design_label": "full_net",
        "default_worst_weight": 0.25,
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, str], key: str, default: str = "") -> str:
    item = row.get(key, default)
    return default if item is None or item == "" else item


def method_key(row: dict[str, str], source: dict[str, object]) -> tuple[str, str, str, str]:
    task = value(row, "task")
    design_label = value(
        row,
        "design_label",
        str(source.get("default_design_label", "coarse")),
    )
    score_mode = value(row, "score_mode")
    worst_weight = value(
        row,
        "worst_weight",
        str(source.get("default_worst_weight", 0.25)),
    )
    return task, design_label, score_mode, worst_weight


def method_label(
    family: str,
    design_label: str,
    score_mode: str,
    worst_weight: str,
    train_points: int,
    hard_points: int,
) -> str:
    if design_label == "full_net":
        return f"full-{train_points}-{score_mode}"
    if hard_points:
        return f"{family}-{design_label}-{score_mode}"
    return f"{family}-{design_label}-{score_mode}"


def collect() -> list[dict[str, float | int | str]]:
    net_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    heldout_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    margin_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}

    for source in INPUTS:
        prefix = str(source["prefix"])
        family = str(source["family"])
        net_path = result_path(f"{prefix}_net_results.csv")
        heldout_path = result_path(f"{prefix}_results.csv")
        margin_path = result_path(f"{prefix}_margin_results.csv")
        if not (net_path.exists() and heldout_path.exists() and margin_path.exists()):
            missing = [
                path.name
                for path in (net_path, heldout_path, margin_path)
                if not path.exists()
            ]
            raise FileNotFoundError(", ".join(missing))
        for row in read_csv(net_path):
            task, design_label, score_mode, worst_weight = method_key(row, source)
            key = (family, task, design_label, score_mode, worst_weight)
            net_groups.setdefault(key, []).append(row)
        for row in read_csv(heldout_path):
            task, design_label, score_mode, worst_weight = method_key(row, source)
            key = (family, task, design_label, score_mode, worst_weight)
            heldout_groups.setdefault(key, []).append(row)
        for row in read_csv(margin_path):
            task, design_label, score_mode, worst_weight = method_key(row, source)
            key = (family, task, design_label, score_mode, worst_weight)
            margin_groups.setdefault(key, []).append(row)

    rows: list[dict[str, float | int | str]] = []
    for key in sorted(net_groups):
        family, task, design_label, score_mode, worst_weight = key
        eval_rows = [
            row for row in net_groups[key] if value(row, "net_label", "eval_net") == "eval_net"
        ]
        if not eval_rows:
            continue
        fidelities = np.asarray([float(row["final_fidelity"]) for row in eval_rows])
        first = eval_rows[0]
        train_points = int(value(first, "train_net_points", str(len(eval_rows))))
        hard_points = int(value(first, "hard_points_used", "0"))

        heldout_rows = [
            row
            for row in heldout_groups[key]
            if abs(float(row["disorder_strength"]) - 0.08) < 1e-12
        ]
        heldout_fids = np.asarray(
            [float(row["final_fidelity"]) for row in heldout_rows],
            dtype=float,
        )
        margins = margin_groups[key]
        if margins:
            eval_outside = np.asarray(
                [int(value(row, "eval_outside_residual", "1")) for row in margins],
                dtype=bool,
            )
            eval_margin_values = np.asarray(
                [float(value(row, "eval_scheduled_margin", "0")) for row in margins],
                dtype=float,
            )
            outside_values = eval_margin_values[eval_outside]
            min_outside_margin = (
                float(np.min(outside_values)) if len(outside_values) else 0.0
            )
            positive_fraction = (
                float(np.mean(outside_values > 0.0)) if len(outside_values) else 1.0
            )
            outside_steps = int(np.sum(eval_outside))
        else:
            min_outside_margin = float("nan")
            positive_fraction = float("nan")
            outside_steps = 0

        rows.append(
            {
                "task": task,
                "method": method_label(
                    family,
                    design_label,
                    score_mode,
                    worst_weight,
                    train_points,
                    hard_points,
                ),
                "family": family,
                "design_label": design_label,
                "score_mode": score_mode,
                "worst_weight": float(worst_weight),
                "train_net_points": train_points,
                "hard_points_used": hard_points,
                "eval_net_points": len(eval_rows),
                "covering_radius_h": float(first["covering_radius"]),
                "worst_net_fidelity": float(np.min(fidelities)),
                "mean_net_fidelity": float(np.mean(fidelities)),
                "heldout_delta_0_08_mean": float(np.mean(heldout_fids)),
                "heldout_delta_0_08_min": float(np.min(heldout_fids)),
                "heldout_delta_0_08_std": float(np.std(heldout_fids)),
                "pulse_energy": float(first["pulse_energy"]),
                "shifted_selected_fraction": float(first["shifted_selected_fraction"]),
                "design_seconds": float(first["design_seconds"]),
                "eval_outside_steps": outside_steps,
                "eval_positive_outside_fraction": positive_fraction,
                "eval_min_outside_margin": min_outside_margin,
            }
        )

    best_by_task: dict[str, float] = defaultdict(lambda: -np.inf)
    for row in rows:
        task = str(row["task"])
        best_by_task[task] = max(best_by_task[task], float(row["worst_net_fidelity"]))
    for row in rows:
        row["best_worst_net_for_task"] = int(
            abs(float(row["worst_net_fidelity"]) - best_by_task[str(row["task"])])
            < 1e-12
        )
    return rows


def write_csv_rows(rows: list[dict[str, float | int | str]]) -> None:
    path = result_path("operator_net_training_strategy_comparison_results.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def fmt(value: float | int | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def table(f, rows: list[dict[str, float | int | str]], columns: list[str]) -> None:
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(fmt(row[column]) for column in columns) + " |\n")


def write_summary(rows: list[dict[str, float | int | str]]) -> None:
    path = result_path("operator_net_training_strategy_comparison_summary.md")
    best_rows = sorted(
        [row for row in rows if int(row["best_worst_net_for_task"])],
        key=lambda row: str(row["task"]),
    )
    all_rows = sorted(
        rows,
        key=lambda row: (
            str(row["task"]),
            -float(row["worst_net_fidelity"]),
            str(row["method"]),
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        f.write("# Operator-Net Training Strategy Comparison\n\n")
        f.write(
            "Postprocessed comparison of direct coarse-net, adaptive hard-point, "
            "and full 2145-point Pauli-ball terminal-value horizon designs. "
            "Best rows are selected by worst fidelity on the deterministic "
            "2145-point certificate net. This is a design-selection diagnostic, "
            "not an independent distribution-free generalization claim.\n\n"
        )
        f.write("## Best Worst-Net Rows\n\n")
        table(
            f,
            best_rows,
            [
                "task",
                "method",
                "train_net_points",
                "hard_points_used",
                "worst_net_fidelity",
                "mean_net_fidelity",
                "heldout_delta_0_08_mean",
                "heldout_delta_0_08_min",
                "eval_min_outside_margin",
            ],
        )
        f.write("\n## All Strategy Rows\n\n")
        table(
            f,
            all_rows,
            [
                "task",
                "method",
                "worst_net_fidelity",
                "mean_net_fidelity",
                "heldout_delta_0_08_mean",
                "heldout_delta_0_08_min",
                "eval_positive_outside_fraction",
                "eval_min_outside_margin",
                "pulse_energy",
                "design_seconds",
            ],
        )
    print(f"wrote {path}")


def main() -> None:
    rows = collect()
    write_csv_rows(rows)
    write_summary(rows)
    for row in sorted(rows, key=lambda item: (str(item["task"]), -float(item["worst_net_fidelity"]))):
        print(
            row["task"],
            row["method"],
            f"worst_net={float(row['worst_net_fidelity']):.6g}",
            f"heldout_mean={float(row['heldout_delta_0_08_mean']):.6g}",
            f"margin={float(row['eval_min_outside_margin']):.6g}",
        )


if __name__ == "__main__":
    main()
