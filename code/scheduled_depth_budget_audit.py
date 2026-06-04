"""Postprocess scheduled terminal-value margins into depth-budget evidence.

The main theorem has two separate branches:

* the depth-one online diagnostics verify scheduled Bellman margins;
* finite entry additionally requires a coherent terminal-value schedule
  ``L_{j+1}=L_j-1`` with enough initial depth.

This audit computes the sufficient initial depth budget implied by the
recorded margin CSVs.  It does not turn the depth-one run into a fixed-depth
or all-time theorem; it quantifies the extra coherent-schedule requirement.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from paths import result_path


@dataclass(frozen=True)
class SourceSpec:
    label: str
    filename: str


SOURCES = (
    SourceSpec("online_terminal_value_shifted", "terminal_value_shifted_horizon_margin_results.csv"),
    SourceSpec("offline_shifted_fallback", "terminal_value_certificate_audit_results.csv"),
)


def read_rows(source: SourceSpec) -> dict[tuple[str, int], list[dict[str, str]]]:
    rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with result_path(source.filename).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["task"], int(row["value_depth"]))].append(row)
    return rows


def leading_outside_steps(rows: Iterable[dict[str, str]]) -> int:
    count = 0
    for row in sorted(rows, key=lambda item: int(item["step"])):
        if int(row["terminal_outside_residual"]):
            count += 1
        else:
            break
    return count


def first_entry_step(rows: Iterable[dict[str, str]]) -> int | None:
    for row in sorted(rows, key=lambda item: int(item["step"])):
        if not int(row["terminal_outside_residual"]):
            return int(row["step"])
    return None


def summarize_group(
    source: SourceSpec,
    task: str,
    value_depth: int,
    rows: list[dict[str, str]],
) -> dict[str, str | int | float]:
    ordered = sorted(rows, key=lambda item: int(item["step"]))
    outside = [row for row in ordered if int(row["terminal_outside_residual"])]
    if not outside:
        raise ValueError(f"{source.filename} {task} depth {value_depth}: no outside rows")

    epsilon_min = min(float(row["best_value_margin"]) for row in outside)
    first = ordered[0]
    first_value = float(first["current_value"])
    sufficient_depth = math.floor(first_value / epsilon_min) + 1
    max_value = max(float(row["current_value"]) for row in outside)
    worst_case_depth = math.floor(max_value / epsilon_min) + 1
    entry_step = first_entry_step(ordered)
    leading_outside = leading_outside_steps(ordered)
    max_bellman_gap = max(abs(float(row["bellman_gap"])) for row in ordered)

    return {
        "source": source.label,
        "source_file": source.filename,
        "task": task,
        "value_depth": value_depth,
        "audited_steps": len(ordered),
        "outside_steps": len(outside),
        "leading_outside_steps": leading_outside,
        "first_entry_step": "" if entry_step is None else entry_step,
        "epsilon_min_outside": epsilon_min,
        "first_current_value": first_value,
        "sufficient_initial_depth_from_first_value": sufficient_depth,
        "max_current_value_outside": max_value,
        "worst_case_depth_from_max_value": worst_case_depth,
        "audited_steps_cover_sufficient_depth": int(len(ordered) >= sufficient_depth),
        "entry_observed_before_or_at_sufficient_depth": int(
            entry_step is not None and entry_step <= sufficient_depth
        ),
        "max_bellman_gap": max_bellman_gap,
    }


def write_summary(rows: list[dict[str, str | int | float]]) -> None:
    out = result_path("scheduled_depth_budget_audit_summary.md")
    with out.open("w", encoding="utf-8") as f:
        f.write("# Scheduled Depth-Budget Audit\n\n")
        f.write(
            "This postprocesses the terminal-value margin CSVs into the depth "
            "budget required by the coherent scheduled finite-entry clause. "
            "It is not a fixed-depth or all-time certificate: the online "
            "depth-one run verifies one-step scheduled Bellman margins, while "
            "finite entry requires a preallocated schedule with "
            "`L_{j+1}=L_j-1`.\n\n"
        )
        f.write(
            "| source | task | depth | outside | first entry | eps min | first value | "
            "needed L0 | audited covers L0 | entry <= L0 | max gap |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            first_entry = row["first_entry_step"] if row["first_entry_step"] != "" else "--"
            f.write(
                f"| {row['source']} | {row['task']} | {row['value_depth']} | "
                f"{row['outside_steps']}/{row['audited_steps']} | {first_entry} | "
                f"{float(row['epsilon_min_outside']):.6g} | "
                f"{float(row['first_current_value']):.6g} | "
                f"{row['sufficient_initial_depth_from_first_value']} | "
                f"{row['audited_steps_cover_sufficient_depth']} | "
                f"{row['entry_observed_before_or_at_sufficient_depth']} | "
                f"{float(row['max_bellman_gap']):.3g} |\n"
            )
        f.write("\n")
        f.write(
            "The `needed L0` column is `floor(first_current_value / eps_min) + 1`. "
            "For H, the recorded terminal score can enter the residual set before "
            "this conservative bound; for Z, the depth-one recorded run remains "
            "outside, so the row should be read only as the coherent-schedule "
            "budget required by the theorem.\n"
        )


def main() -> None:
    summary_rows: list[dict[str, str | int | float]] = []
    for source in SOURCES:
        grouped = read_rows(source)
        for (task, value_depth), rows in sorted(grouped.items()):
            summary_rows.append(summarize_group(source, task, value_depth, rows))

    fieldnames = list(summary_rows[0].keys())
    with result_path("scheduled_depth_budget_audit_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_summary(summary_rows)


if __name__ == "__main__":
    main()
