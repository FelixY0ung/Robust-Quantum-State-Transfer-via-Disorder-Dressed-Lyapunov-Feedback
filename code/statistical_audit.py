"""Statistical audit for key held-out comparisons.

The manuscript reports many 50-seed held-out means.  This script computes
confidence intervals and paired effect sizes for the comparisons most relevant
to a journal submission:

* one-step versus beam-horizon Lyapunov transfer at delta = 0.08;
* beam-horizon transfer versus polished terminal open-loop ceiling;
* process-horizon versus GRAPE process average-gate fidelity;
* transfer-horizon gate diagnostic versus process-oriented alternatives.

All comparisons use the existing held-out CSV files and shared seed indices
whenever possible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paths import result_path


@dataclass(frozen=True)
class Series:
    label: str
    values: pd.DataFrame
    metric: str


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return float(np.mean(values)), 0.0
    std = float(np.std(values, ddof=1))
    # 50 held-out samples dominate here; the normal approximation is adequate
    # for concise reporting and avoids adding another dependency.
    return float(np.mean(values)), 1.96 * std / np.sqrt(values.size)


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> tuple[int, float, float, float]:
    merged = a[["seed", metric]].merge(
        b[["seed", metric]],
        on="seed",
        suffixes=("_a", "_b"),
    )
    diff = merged[f"{metric}_a"].to_numpy(dtype=float) - merged[f"{metric}_b"].to_numpy(dtype=float)
    mean, ci = mean_ci(diff)
    std = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    effect = mean / std if std > 0 else float("inf")
    return int(diff.size), mean, ci, effect


def horizon_series(task: str, config: str = "q6_b6") -> pd.DataFrame:
    df = pd.read_csv(result_path("horizon_ablation_results.csv"))
    return df[(df["task"] == task) & (df["config"] == config)].copy()


def polished_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("polished_openloop_results.csv"))
    subset = df[(df["task"] == task) & (df["eval_strength"] == 0.08)].copy()
    return subset.rename(columns={"eval_strength": "disorder_strength"})


def dcrab_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("dcrab_baseline_results.csv"))
    subset = df[(df["task"] == task) & (df["eval_strength"] == 0.08)].copy()
    return subset.rename(columns={"eval_strength": "disorder_strength"})


def gate_probe_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("gate_fidelity_probe_results.csv"))
    return df[df["task"] == task].copy()


def process_horizon_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_horizon_results.csv"))
    return df[df["task"] == task].copy()


def grape_process_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("ensemble_grape_baseline_results.csv"))
    return df[(df["task"] == task) & (df["objective_kind"] == "process")].copy()


def grape_state_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("ensemble_grape_baseline_results.csv"))
    return df[(df["task"] == task) & (df["objective_kind"] == "state")].copy()


def open_noise_series(task: str, noise_case: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_system_noise_results.csv"))
    return df[(df["task"] == task) & (df["noise_case"] == noise_case)].copy()


def summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    series_specs = [
        ("Z", "beam horizon transfer", horizon_series("Z", "q6_b6"), "final_fidelity"),
        ("H", "beam horizon transfer", horizon_series("H", "q6_b8"), "final_fidelity"),
        ("Z", "dCRAB transfer ceiling", dcrab_series("Z"), "final_fidelity"),
        ("H", "dCRAB transfer ceiling", dcrab_series("H"), "final_fidelity"),
        ("Z", "polished transfer ceiling", polished_series("Z"), "final_fidelity"),
        ("H", "polished transfer ceiling", polished_series("H"), "final_fidelity"),
        ("Z", "process horizon gate", process_horizon_series("Z"), "average_gate_fidelity"),
        ("H", "process horizon gate", process_horizon_series("H"), "average_gate_fidelity"),
        ("Z", "GRAPE process gate", grape_process_series("Z"), "average_gate_fidelity"),
        ("H", "GRAPE process gate", grape_process_series("H"), "average_gate_fidelity"),
        ("Z", "GRAPE state gate", grape_state_series("Z"), "average_gate_fidelity"),
        ("H", "GRAPE state gate", grape_state_series("H"), "average_gate_fidelity"),
        ("Z", "transfer horizon gate", gate_probe_series("Z"), "average_gate_fidelity"),
        ("H", "transfer horizon gate", gate_probe_series("H"), "average_gate_fidelity"),
        ("Z", "open noise combined", open_noise_series("Z", "combined"), "final_fidelity"),
        ("H", "open noise combined", open_noise_series("H", "combined"), "final_fidelity"),
    ]

    for task, label, frame, metric in series_specs:
        values = frame[metric].to_numpy(dtype=float)
        mean, ci = mean_ci(values)
        rows.append(
            {
                "kind": "summary",
                "task": task,
                "comparison": label,
                "metric": metric,
                "n": str(values.size),
                "mean": f"{mean:.9g}",
                "ci95_halfwidth": f"{ci:.3g}",
                "paired_delta": "",
                "paired_delta_ci95": "",
                "paired_effect_dz": "",
            }
        )

    comparisons = [
        (
            "Z",
            "beam horizon minus one-step horizon",
            horizon_series("Z", "q6_b6"),
            horizon_series("Z", "q1_b1"),
            "final_fidelity",
        ),
        (
            "H",
            "beam horizon minus one-step horizon",
            horizon_series("H", "q6_b8"),
            horizon_series("H", "q1_b1"),
            "final_fidelity",
        ),
        (
            "Z",
            "dCRAB ceiling minus beam horizon",
            dcrab_series("Z"),
            horizon_series("Z", "q6_b6"),
            "final_fidelity",
        ),
        (
            "H",
            "dCRAB ceiling minus beam horizon",
            dcrab_series("H"),
            horizon_series("H", "q6_b8"),
            "final_fidelity",
        ),
        (
            "Z",
            "polished ceiling minus dCRAB ceiling",
            polished_series("Z"),
            dcrab_series("Z"),
            "final_fidelity",
        ),
        (
            "H",
            "polished ceiling minus dCRAB ceiling",
            polished_series("H"),
            dcrab_series("H"),
            "final_fidelity",
        ),
        (
            "Z",
            "polished ceiling minus beam horizon",
            polished_series("Z"),
            horizon_series("Z", "q6_b6"),
            "final_fidelity",
        ),
        (
            "H",
            "polished ceiling minus beam horizon",
            polished_series("H"),
            horizon_series("H", "q6_b8"),
            "final_fidelity",
        ),
        (
            "Z",
            "GRAPE process gate minus process horizon",
            grape_process_series("Z"),
            process_horizon_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "GRAPE process gate minus process horizon",
            grape_process_series("H"),
            process_horizon_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "process horizon gate minus transfer horizon gate",
            process_horizon_series("Z"),
            gate_probe_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "process horizon gate minus transfer horizon gate",
            process_horizon_series("H"),
            gate_probe_series("H"),
            "average_gate_fidelity",
        ),
    ]

    for task, label, a, b, metric in comparisons:
        n, delta, ci, effect = paired_delta(a, b, metric)
        rows.append(
            {
                "kind": "paired",
                "task": task,
                "comparison": label,
                "metric": metric,
                "n": str(n),
                "mean": "",
                "ci95_halfwidth": "",
                "paired_delta": f"{delta:.9g}",
                "paired_delta_ci95": f"{ci:.3g}",
                "paired_effect_dz": f"{effect:.3g}",
            }
        )

    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    headers = [
        "kind",
        "task",
        "comparison",
        "metric",
        "n",
        "mean",
        "ci95_halfwidth",
        "paired_delta",
        "paired_delta_ci95",
        "paired_effect_dz",
    ]
    with result_path("statistical_audit_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with result_path("statistical_audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Statistical Audit Summary\n\n")
        f.write("Confidence intervals use 1.96 standard errors over held-out seeds. ")
        f.write("Paired deltas are computed by matching the same held-out seed indices.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = summary_rows()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('statistical_audit_results.csv')}")
    print(f"wrote summary to {result_path('statistical_audit_summary.md')}")


if __name__ == "__main__":
    main()
