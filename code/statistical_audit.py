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


def dcrab_train8_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("dcrab_train8_baseline_results.csv"))
    subset = df[(df["task"] == task) & (df["eval_strength"] == 0.08)].copy()
    return subset.rename(columns={"eval_strength": "disorder_strength"})


def krotov_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("krotov_baseline_results.csv"))
    subset = df[(df["task"] == task) & (df["eval_strength"] == 0.08)].copy()
    return subset.rename(columns={"eval_strength": "disorder_strength"})


def gate_probe_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("gate_fidelity_probe_results.csv"))
    return df[df["task"] == task].copy()


def process_horizon_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_horizon_results.csv"))
    return df[df["task"] == task].copy()


def process_seeded_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_seeded_horizon_results.csv"))
    return df[(df["task"] == task) & (df["controller"] == "seeded_process_horizon")].copy()


def process_dcrab_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_dcrab_baseline_results.csv"))
    subset = df[(df["task"] == task) & (df["eval_strength"] == 0.08)].copy()
    return subset


def process_dcrab_seeded_series(task: str, controller: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_dcrab_seeded_horizon_results.csv"))
    return df[(df["task"] == task) & (df["controller"] == controller)].copy()


def process_adjoint_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_adjoint_horizon_results.csv"))
    return df[(df["task"] == task) & (df["controller"] == "adjoint_process_horizon")].copy()


def standalone_process_adjoint_series(task: str, controller: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("process_standalone_adjoint_results.csv"))
    subset = df[(df["task"] == task) & (df["controller"] == controller)].copy()
    return subset.rename(columns={"avg_gate_fidelity": "average_gate_fidelity"})


def grape_process_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("ensemble_grape_baseline_results.csv"))
    return df[(df["task"] == task) & (df["objective_kind"] == "process")].copy()


def grape_state_series(task: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("ensemble_grape_baseline_results.csv"))
    return df[(df["task"] == task) & (df["objective_kind"] == "state")].copy()


def open_noise_series(task: str, noise_case: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_system_noise_results.csv"))
    return df[(df["task"] == task) & (df["noise_case"] == noise_case)].copy()


def open_adjoint_series(task: str, controller: str, noise_case: str = "combined") -> pd.DataFrame:
    df = pd.read_csv(result_path("open_system_adjoint_horizon_results.csv"))
    return df[
        (df["task"] == task)
        & (df["controller"] == controller)
        & (df["eval_noise_case"] == noise_case)
    ].copy()


def standalone_open_series(task: str, controller: str, noise_case: str = "combined") -> pd.DataFrame:
    df = pd.read_csv(result_path("open_system_standalone_adjoint_results.csv"))
    return df[
        (df["task"] == task)
        & (df["controller"] == controller)
        & (df["eval_noise_case"] == noise_case)
    ].copy()


def standalone_leakage_series(controller: str, strength: float = 0.03) -> pd.DataFrame:
    df = pd.read_csv(result_path("transmon_standalone_adjoint_results.csv"))
    return df[(df["controller"] == controller) & (df["eval_strength"] == strength)].copy()


def leakage_series(controller: str, strength: float = 0.03) -> pd.DataFrame:
    df = pd.read_csv(result_path("transmon_leakage_results.csv"))
    return df[(df["controller"] == controller) & (df["eval_strength"] == strength)].copy()


def open_leakage_series(
    controller: str,
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("transmon_open_system_leakage_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def standalone_open_leakage_series(
    controller: str,
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("transmon_open_leakage_adjoint_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def integrated_open_leakage_series(
    controller: str = "integrated_alpha1p0_lw1p5_trust004",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_integrated_sweep_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def continuation_open_leakage_series(
    controller: str = "continuation_target08_leak08",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_continuation_sweep_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def high_fidelity_open_leakage_series(
    controller: str = "hf_leak05",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_high_fidelity_sweep_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def cap_open_leakage_series(
    controller: str = "cap050_w120",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_cap_refinement_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def worst_cap_open_leakage_series(
    controller: str = "worstcap050_w120_mean02_worst05",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_worst_cap_refinement_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def smooth_cap_open_leakage_series(
    controller: str = "smoothcap050_w120_slew10",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_smooth_cap_refinement_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def full_pulse_open_leakage_series(
    controller: str = "fullpulse050_w120_tail02_worst05_slew10",
    noise_case: str = "combined",
    strength: float = 0.03,
) -> pd.DataFrame:
    df = pd.read_csv(result_path("open_leakage_full_pulse_refinement_results.csv"))
    return df[
        (df["controller"] == controller)
        & (df["noise_case"] == noise_case)
        & (df["eval_strength"] == strength)
    ].copy()


def slew_series(task: str, weight: float) -> pd.DataFrame:
    df = pd.read_csv(result_path("slew_constrained_horizon_results.csv"))
    return df[
        (df["task"] == task)
        & (df["slew_weight"].astype(float) == weight)
    ].copy()


def bandwidth_series(task: str, base_weight: float, filter_name: str) -> pd.DataFrame:
    df = pd.read_csv(result_path("bandwidth_filter_audit_results.csv"))
    return df[
        (df["task"] == task)
        & (df["base_slew_weight"].astype(float) == base_weight)
        & (df["filter"] == filter_name)
    ].copy()


def summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    series_specs = [
        ("Z", "beam horizon transfer", horizon_series("Z", "q6_b6"), "final_fidelity"),
        ("H", "beam horizon transfer", horizon_series("H", "q6_b8"), "final_fidelity"),
        ("Z", "Krotov-package transfer comparator", krotov_series("Z"), "state_transfer_fidelity"),
        ("H", "Krotov-package transfer comparator", krotov_series("H"), "state_transfer_fidelity"),
        ("Z", "dCRAB transfer ceiling", dcrab_series("Z"), "final_fidelity"),
        ("H", "dCRAB transfer ceiling", dcrab_series("H"), "final_fidelity"),
        ("Z", "dCRAB train-8 transfer ceiling", dcrab_train8_series("Z"), "final_fidelity"),
        ("H", "dCRAB train-8 transfer ceiling", dcrab_train8_series("H"), "final_fidelity"),
        ("Z", "polished transfer ceiling", polished_series("Z"), "final_fidelity"),
        ("H", "polished transfer ceiling", polished_series("H"), "final_fidelity"),
        ("Z", "process horizon gate", process_horizon_series("Z"), "average_gate_fidelity"),
        ("H", "process horizon gate", process_horizon_series("H"), "average_gate_fidelity"),
        ("Z", "process dCRAB gate", process_dcrab_series("Z"), "average_gate_fidelity"),
        ("H", "process dCRAB gate", process_dcrab_series("H"), "average_gate_fidelity"),
        ("Z", "dCRAB-seeded process horizon gate", process_dcrab_seeded_series("Z", "dcrab_seeded_process_horizon"), "average_gate_fidelity"),
        ("H", "dCRAB-seeded process horizon gate", process_dcrab_seeded_series("H", "dcrab_seeded_process_horizon"), "average_gate_fidelity"),
        ("Z", "seeded process horizon gate", process_seeded_series("Z"), "average_gate_fidelity"),
        ("H", "seeded process horizon gate", process_seeded_series("H"), "average_gate_fidelity"),
        ("Z", "adjoint process horizon gate", process_adjoint_series("Z"), "average_gate_fidelity"),
        ("H", "adjoint process horizon gate", process_adjoint_series("H"), "average_gate_fidelity"),
        ("Z", "standalone process-adjoint gate", standalone_process_adjoint_series("Z", "standalone_process_adjoint"), "average_gate_fidelity"),
        ("H", "standalone process-adjoint gate", standalone_process_adjoint_series("H", "standalone_process_adjoint"), "average_gate_fidelity"),
        ("Z", "GRAPE process gate", grape_process_series("Z"), "average_gate_fidelity"),
        ("H", "GRAPE process gate", grape_process_series("H"), "average_gate_fidelity"),
        ("Z", "GRAPE state gate", grape_state_series("Z"), "average_gate_fidelity"),
        ("H", "GRAPE state gate", grape_state_series("H"), "average_gate_fidelity"),
        ("Z", "transfer horizon gate", gate_probe_series("Z"), "average_gate_fidelity"),
        ("H", "transfer horizon gate", gate_probe_series("H"), "average_gate_fidelity"),
        ("Z", "open noise combined", open_noise_series("Z", "combined"), "final_fidelity"),
        ("H", "open noise combined", open_noise_series("H", "combined"), "final_fidelity"),
        ("Z", "standalone Lindblad seed combined", standalone_open_series("Z", "standalone_open_seed"), "final_fidelity"),
        ("H", "standalone Lindblad seed combined", standalone_open_series("H", "standalone_open_seed"), "final_fidelity"),
        ("Z", "standalone Lindblad adjoint combined", standalone_open_series("Z", "standalone_open_adjoint"), "final_fidelity"),
        ("H", "standalone Lindblad adjoint combined", standalone_open_series("H", "standalone_open_adjoint"), "final_fidelity"),
        ("Z", "adjoint Lindblad horizon combined", open_adjoint_series("Z", "adjoint_open_horizon"), "final_fidelity"),
        ("H", "adjoint Lindblad horizon combined", open_adjoint_series("H", "adjoint_open_horizon"), "final_fidelity"),
        ("Z", "open-system GRAPE combined", open_adjoint_series("Z", "open_grape_reference"), "final_fidelity"),
        ("H", "open-system GRAPE combined", open_adjoint_series("H", "open_grape_reference"), "final_fidelity"),
        ("L", "leakage path seed fidelity", standalone_leakage_series("standalone_path_seed"), "final_fidelity"),
        ("L", "standalone leakage adjoint fidelity", standalone_leakage_series("standalone_adjoint_horizon"), "final_fidelity"),
        ("L", "leakage-GRAPE fidelity", leakage_series("leakage_penalized_grape"), "final_fidelity"),
        ("L", "leakage path seed max leakage", standalone_leakage_series("standalone_path_seed"), "max_leakage"),
        ("L", "standalone leakage adjoint max leakage", standalone_leakage_series("standalone_adjoint_horizon"), "max_leakage"),
        ("L", "leakage-GRAPE max leakage", leakage_series("leakage_penalized_grape"), "max_leakage"),
        ("OL", "open leakage path combined fidelity", open_leakage_series("path_horizon"), "final_fidelity"),
        ("OL", "standalone open leakage adjoint combined fidelity", standalone_open_leakage_series("standalone_open_leakage_adjoint"), "final_fidelity"),
        ("OL", "target-biased open leakage adjoint combined fidelity", standalone_open_leakage_series("target_biased_open_leakage_adjoint"), "final_fidelity"),
        ("OL", "integrated open leakage adjoint combined fidelity", integrated_open_leakage_series(), "final_fidelity"),
        ("OL", "two-stage target-biased open leakage adjoint combined fidelity", standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"), "final_fidelity"),
        ("OL", "continuation open leakage adjoint combined fidelity", continuation_open_leakage_series("continuation_target08_leak12"), "final_fidelity"),
        ("OL", "fidelity-favoring continuation combined fidelity", continuation_open_leakage_series("continuation_target08_leak08"), "final_fidelity"),
        ("OL", "target-push high-fidelity continuation combined fidelity", high_fidelity_open_leakage_series("hf_leak05"), "final_fidelity"),
        ("OL", "leakage-cap target-push combined fidelity", cap_open_leakage_series("cap050_w120"), "final_fidelity"),
        ("OL", "robust leakage-cap target-push combined fidelity", cap_open_leakage_series("cap050_w120_mean02_worst05"), "final_fidelity"),
        ("OL", "worst-seed leakage-cap target-push combined fidelity", worst_cap_open_leakage_series(), "final_fidelity"),
        ("OL", "slew-aware leakage-cap target-push combined fidelity", smooth_cap_open_leakage_series(), "final_fidelity"),
        ("OL", "full-pulse capped target-push combined fidelity", full_pulse_open_leakage_series(), "final_fidelity"),
        ("OL", "open leakage adjoint combined fidelity", open_leakage_series("adjoint_horizon"), "final_fidelity"),
        ("OL", "open leakage-GRAPE combined fidelity", open_leakage_series("leakage_penalized_grape"), "final_fidelity"),
        ("OL", "open leakage path combined max leakage", open_leakage_series("path_horizon"), "max_leakage"),
        ("OL", "standalone open leakage adjoint combined max leakage", standalone_open_leakage_series("standalone_open_leakage_adjoint"), "max_leakage"),
        ("OL", "target-biased open leakage adjoint combined max leakage", standalone_open_leakage_series("target_biased_open_leakage_adjoint"), "max_leakage"),
        ("OL", "integrated open leakage adjoint combined max leakage", integrated_open_leakage_series(), "max_leakage"),
        ("OL", "two-stage target-biased open leakage adjoint combined max leakage", standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"), "max_leakage"),
        ("OL", "continuation open leakage adjoint combined max leakage", continuation_open_leakage_series("continuation_target08_leak12"), "max_leakage"),
        ("OL", "fidelity-favoring continuation combined max leakage", continuation_open_leakage_series("continuation_target08_leak08"), "max_leakage"),
        ("OL", "target-push high-fidelity continuation combined max leakage", high_fidelity_open_leakage_series("hf_leak05"), "max_leakage"),
        ("OL", "leakage-cap target-push combined max leakage", cap_open_leakage_series("cap050_w120"), "max_leakage"),
        ("OL", "robust leakage-cap target-push combined max leakage", cap_open_leakage_series("cap050_w120_mean02_worst05"), "max_leakage"),
        ("OL", "worst-seed leakage-cap target-push combined max leakage", worst_cap_open_leakage_series(), "max_leakage"),
        ("OL", "slew-aware leakage-cap target-push combined max leakage", smooth_cap_open_leakage_series(), "max_leakage"),
        ("OL", "full-pulse capped target-push combined max leakage", full_pulse_open_leakage_series(), "max_leakage"),
        ("OL", "open leakage adjoint combined max leakage", open_leakage_series("adjoint_horizon"), "max_leakage"),
        ("OL", "open leakage-GRAPE combined max leakage", open_leakage_series("leakage_penalized_grape"), "max_leakage"),
        ("Z", "compact beam no-slew transfer", slew_series("Z", 0.0), "final_fidelity"),
        ("H", "compact beam no-slew transfer", slew_series("H", 0.0), "final_fidelity"),
        ("Z", "compact slew-constrained transfer", slew_series("Z", 0.005), "final_fidelity"),
        ("H", "compact slew-constrained transfer", slew_series("H", 0.005), "final_fidelity"),
        ("Z", "filtered no-slew boxcar3 transfer", bandwidth_series("Z", 0.0, "boxcar3"), "final_fidelity"),
        ("H", "filtered no-slew boxcar3 transfer", bandwidth_series("H", 0.0, "boxcar3"), "final_fidelity"),
        ("Z", "filtered slew boxcar3 transfer", bandwidth_series("Z", 0.005, "boxcar3"), "final_fidelity"),
        ("H", "filtered slew boxcar3 transfer", bandwidth_series("H", 0.005, "boxcar3"), "final_fidelity"),
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
            "dCRAB train-8 ceiling minus beam horizon",
            dcrab_train8_series("Z"),
            horizon_series("Z", "q6_b6"),
            "final_fidelity",
        ),
        (
            "H",
            "dCRAB train-8 ceiling minus beam horizon",
            dcrab_train8_series("H"),
            horizon_series("H", "q6_b8"),
            "final_fidelity",
        ),
        (
            "Z",
            "dCRAB train-8 minus dCRAB train-4",
            dcrab_train8_series("Z"),
            dcrab_series("Z"),
            "final_fidelity",
        ),
        (
            "H",
            "dCRAB train-8 minus dCRAB train-4",
            dcrab_train8_series("H"),
            dcrab_series("H"),
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
            "adjoint Lindblad horizon minus closed horizon",
            open_adjoint_series("Z", "adjoint_open_horizon"),
            open_noise_series("Z", "combined"),
            "final_fidelity",
        ),
        (
            "Z",
            "standalone Lindblad adjoint minus finite Lindblad seed",
            standalone_open_series("Z", "standalone_open_adjoint"),
            standalone_open_series("Z", "standalone_open_seed"),
            "final_fidelity",
        ),
        (
            "H",
            "standalone Lindblad adjoint minus finite Lindblad seed",
            standalone_open_series("H", "standalone_open_adjoint"),
            standalone_open_series("H", "standalone_open_seed"),
            "final_fidelity",
        ),
        (
            "Z",
            "closed horizon minus standalone Lindblad adjoint",
            open_noise_series("Z", "combined"),
            standalone_open_series("Z", "standalone_open_adjoint"),
            "final_fidelity",
        ),
        (
            "H",
            "closed horizon minus standalone Lindblad adjoint",
            open_noise_series("H", "combined"),
            standalone_open_series("H", "standalone_open_adjoint"),
            "final_fidelity",
        ),
        (
            "Z",
            "adjoint Lindblad horizon minus standalone Lindblad adjoint",
            open_adjoint_series("Z", "adjoint_open_horizon"),
            standalone_open_series("Z", "standalone_open_adjoint"),
            "final_fidelity",
        ),
        (
            "H",
            "adjoint Lindblad horizon minus closed horizon",
            open_adjoint_series("H", "adjoint_open_horizon"),
            open_noise_series("H", "combined"),
            "final_fidelity",
        ),
        (
            "H",
            "adjoint Lindblad horizon minus standalone Lindblad adjoint",
            open_adjoint_series("H", "adjoint_open_horizon"),
            standalone_open_series("H", "standalone_open_adjoint"),
            "final_fidelity",
        ),
        (
            "Z",
            "open-system GRAPE minus closed horizon",
            open_adjoint_series("Z", "open_grape_reference"),
            open_noise_series("Z", "combined"),
            "final_fidelity",
        ),
        (
            "H",
            "open-system GRAPE minus closed horizon",
            open_adjoint_series("H", "open_grape_reference"),
            open_noise_series("H", "combined"),
            "final_fidelity",
        ),
        (
            "Z",
            "open-system GRAPE minus adjoint Lindblad horizon",
            open_adjoint_series("Z", "open_grape_reference"),
            open_adjoint_series("Z", "adjoint_open_horizon"),
            "final_fidelity",
        ),
        (
            "H",
            "open-system GRAPE minus adjoint Lindblad horizon",
            open_adjoint_series("H", "open_grape_reference"),
            open_adjoint_series("H", "adjoint_open_horizon"),
            "final_fidelity",
        ),
        (
            "Z",
            "seeded process horizon minus process horizon",
            process_seeded_series("Z"),
            process_horizon_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "seeded process horizon minus process horizon",
            process_seeded_series("H"),
            process_horizon_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "process dCRAB minus process horizon",
            process_dcrab_series("Z"),
            process_horizon_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "process dCRAB minus process horizon",
            process_dcrab_series("H"),
            process_horizon_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "dCRAB-seeded process horizon minus process dCRAB reference",
            process_dcrab_seeded_series("Z", "dcrab_seeded_process_horizon"),
            process_dcrab_seeded_series("Z", "process_dcrab_reference"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "dCRAB-seeded process horizon minus process dCRAB reference",
            process_dcrab_seeded_series("H", "dcrab_seeded_process_horizon"),
            process_dcrab_seeded_series("H", "process_dcrab_reference"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "dCRAB-seeded process horizon minus process horizon",
            process_dcrab_seeded_series("Z", "dcrab_seeded_process_horizon"),
            process_horizon_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "dCRAB-seeded process horizon minus process horizon",
            process_dcrab_seeded_series("H", "dcrab_seeded_process_horizon"),
            process_horizon_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "adjoint process horizon minus process horizon",
            process_adjoint_series("Z"),
            process_horizon_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "adjoint process horizon minus process horizon",
            process_adjoint_series("H"),
            process_horizon_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "GRAPE process gate minus adjoint process horizon",
            grape_process_series("Z"),
            process_adjoint_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "GRAPE process gate minus adjoint process horizon",
            grape_process_series("H"),
            process_adjoint_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "GRAPE process gate minus process dCRAB",
            grape_process_series("Z"),
            process_dcrab_series("Z"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "GRAPE process gate minus process dCRAB",
            grape_process_series("H"),
            process_dcrab_series("H"),
            "average_gate_fidelity",
        ),
        (
            "Z",
            "standalone process adjoint minus finite process seed",
            standalone_process_adjoint_series("Z", "standalone_process_adjoint"),
            standalone_process_adjoint_series("Z", "finite_process_seed"),
            "average_gate_fidelity",
        ),
        (
            "H",
            "standalone process adjoint minus finite process seed",
            standalone_process_adjoint_series("H", "standalone_process_adjoint"),
            standalone_process_adjoint_series("H", "finite_process_seed"),
            "average_gate_fidelity",
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
        (
            "L",
            "standalone leakage adjoint minus path seed",
            standalone_leakage_series("standalone_adjoint_horizon"),
            standalone_leakage_series("standalone_path_seed"),
            "final_fidelity",
        ),
        (
            "L",
            "leakage-GRAPE minus standalone leakage adjoint",
            leakage_series("leakage_penalized_grape"),
            standalone_leakage_series("standalone_adjoint_horizon"),
            "final_fidelity",
        ),
        (
            "L",
            "path seed max leakage minus standalone leakage adjoint",
            standalone_leakage_series("standalone_path_seed"),
            standalone_leakage_series("standalone_adjoint_horizon"),
            "max_leakage",
        ),
        (
            "L",
            "leakage-GRAPE max leakage minus standalone leakage adjoint",
            leakage_series("leakage_penalized_grape"),
            standalone_leakage_series("standalone_adjoint_horizon"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage adjoint minus path horizon",
            open_leakage_series("adjoint_horizon"),
            open_leakage_series("path_horizon"),
            "final_fidelity",
        ),
        (
            "OL",
            "standalone open leakage adjoint minus path seed",
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            standalone_open_leakage_series("open_leakage_path_seed"),
            "final_fidelity",
        ),
        (
            "OL",
            "target-biased open leakage adjoint minus path seed",
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("open_leakage_path_seed"),
            "final_fidelity",
        ),
        (
            "OL",
            "integrated open leakage adjoint minus path seed",
            integrated_open_leakage_series(),
            standalone_open_leakage_series("open_leakage_path_seed"),
            "final_fidelity",
        ),
        (
            "OL",
            "target-biased open leakage adjoint minus direct standalone open leakage adjoint",
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "target-biased open leakage adjoint minus integrated open leakage adjoint",
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            integrated_open_leakage_series(),
            "final_fidelity",
        ),
        (
            "OL",
            "two-stage target-biased open leakage adjoint minus path seed",
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("open_leakage_path_seed"),
            "final_fidelity",
        ),
        (
            "OL",
            "two-stage target-biased open leakage adjoint minus target-biased open leakage adjoint",
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "reference-assisted open leakage adjoint minus standalone open leakage adjoint",
            open_leakage_series("adjoint_horizon"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "reference-assisted open leakage adjoint minus target-biased open leakage adjoint",
            open_leakage_series("adjoint_horizon"),
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "reference-assisted open leakage adjoint minus integrated open leakage adjoint",
            open_leakage_series("adjoint_horizon"),
            integrated_open_leakage_series(),
            "final_fidelity",
        ),
        (
            "OL",
            "reference-assisted open leakage adjoint minus two-stage target-biased open leakage adjoint",
            open_leakage_series("adjoint_horizon"),
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "target-push high-fidelity continuation minus fidelity-favoring continuation",
            high_fidelity_open_leakage_series("hf_leak05"),
            continuation_open_leakage_series("continuation_target08_leak08"),
            "final_fidelity",
        ),
        (
            "OL",
            "target-push high-fidelity continuation minus reference-assisted open leakage adjoint",
            high_fidelity_open_leakage_series("hf_leak05"),
            open_leakage_series("adjoint_horizon"),
            "final_fidelity",
        ),
        (
            "OL",
            "leakage-cap target-push minus target-push high-fidelity continuation",
            cap_open_leakage_series("cap050_w120"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "final_fidelity",
        ),
        (
            "OL",
            "robust leakage-cap target-push minus target-push high-fidelity continuation",
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "final_fidelity",
        ),
        (
            "OL",
            "worst-seed leakage-cap target-push minus worst-seed target-push reference",
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            worst_cap_open_leakage_series("worstcap_target_push_reference"),
            "final_fidelity",
        ),
        (
            "OL",
            "worst-seed leakage-cap target-push minus robust leakage-cap target-push",
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "final_fidelity",
        ),
        (
            "OL",
            "slew-aware leakage-cap target-push minus target-push high-fidelity continuation",
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "final_fidelity",
        ),
        (
            "OL",
            "slew-aware leakage-cap target-push minus robust leakage-cap target-push",
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "final_fidelity",
        ),
        (
            "OL",
            "full-pulse capped target-push minus target-push high-fidelity continuation",
            full_pulse_open_leakage_series(),
            high_fidelity_open_leakage_series("hf_leak05"),
            "final_fidelity",
        ),
        (
            "OL",
            "full-pulse capped target-push minus leakage-cap target-push",
            full_pulse_open_leakage_series(),
            cap_open_leakage_series("cap050_w120"),
            "final_fidelity",
        ),
        (
            "OL",
            "full-pulse capped target-push minus robust leakage-cap target-push",
            full_pulse_open_leakage_series(),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "final_fidelity",
        ),
        (
            "OL",
            "robust leakage-cap target-push minus leakage-cap target-push",
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            cap_open_leakage_series("cap050_w120"),
            "final_fidelity",
        ),
        (
            "OL",
            "leakage-cap target-push minus reference-assisted open leakage adjoint",
            cap_open_leakage_series("cap050_w120"),
            open_leakage_series("adjoint_horizon"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus adjoint horizon",
            open_leakage_series("leakage_penalized_grape"),
            open_leakage_series("adjoint_horizon"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus standalone open leakage adjoint",
            open_leakage_series("leakage_penalized_grape"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus target-biased open leakage adjoint",
            open_leakage_series("leakage_penalized_grape"),
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus integrated open leakage adjoint",
            open_leakage_series("leakage_penalized_grape"),
            integrated_open_leakage_series(),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus two-stage target-biased open leakage adjoint",
            open_leakage_series("leakage_penalized_grape"),
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus target-push high-fidelity continuation",
            open_leakage_series("leakage_penalized_grape"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus leakage-cap target-push",
            open_leakage_series("leakage_penalized_grape"),
            cap_open_leakage_series("cap050_w120"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus robust leakage-cap target-push",
            open_leakage_series("leakage_penalized_grape"),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus worst-seed leakage-cap target-push",
            open_leakage_series("leakage_penalized_grape"),
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus slew-aware leakage-cap target-push",
            open_leakage_series("leakage_penalized_grape"),
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage-GRAPE minus full-pulse capped target-push",
            open_leakage_series("leakage_penalized_grape"),
            full_pulse_open_leakage_series(),
            "final_fidelity",
        ),
        (
            "OL",
            "open leakage path max leakage minus adjoint horizon",
            open_leakage_series("path_horizon"),
            open_leakage_series("adjoint_horizon"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage path seed max leakage minus standalone open leakage adjoint",
            standalone_open_leakage_series("open_leakage_path_seed"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage path seed max leakage minus target-biased open leakage adjoint",
            standalone_open_leakage_series("open_leakage_path_seed"),
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage path seed max leakage minus integrated open leakage adjoint",
            standalone_open_leakage_series("open_leakage_path_seed"),
            integrated_open_leakage_series(),
            "max_leakage",
        ),
        (
            "OL",
            "target-biased open leakage adjoint max leakage minus direct standalone open leakage adjoint",
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "target-biased open leakage adjoint max leakage minus integrated open leakage adjoint",
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            integrated_open_leakage_series(),
            "max_leakage",
        ),
        (
            "OL",
            "two-stage target-biased open leakage adjoint max leakage minus target-biased open leakage adjoint",
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            standalone_open_leakage_series("target_biased_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "target-push high-fidelity continuation max leakage minus fidelity-favoring continuation",
            high_fidelity_open_leakage_series("hf_leak05"),
            continuation_open_leakage_series("continuation_target08_leak08"),
            "max_leakage",
        ),
        (
            "OL",
            "target-push high-fidelity continuation max leakage minus reference-assisted open leakage adjoint",
            high_fidelity_open_leakage_series("hf_leak05"),
            open_leakage_series("adjoint_horizon"),
            "max_leakage",
        ),
        (
            "OL",
            "reference-assisted open leakage adjoint max leakage minus standalone open leakage adjoint",
            open_leakage_series("adjoint_horizon"),
            standalone_open_leakage_series("standalone_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "target-push high-fidelity continuation max leakage minus open leakage-GRAPE",
            high_fidelity_open_leakage_series("hf_leak05"),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "leakage-cap target-push max leakage minus target-push high-fidelity continuation",
            cap_open_leakage_series("cap050_w120"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "max_leakage",
        ),
        (
            "OL",
            "robust leakage-cap target-push max leakage minus target-push high-fidelity continuation",
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "max_leakage",
        ),
        (
            "OL",
            "worst-seed leakage-cap target-push max leakage minus worst-seed target-push reference",
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            worst_cap_open_leakage_series("worstcap_target_push_reference"),
            "max_leakage",
        ),
        (
            "OL",
            "worst-seed leakage-cap target-push max leakage minus robust leakage-cap target-push",
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "max_leakage",
        ),
        (
            "OL",
            "slew-aware leakage-cap target-push max leakage minus target-push high-fidelity continuation",
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            high_fidelity_open_leakage_series("hf_leak05"),
            "max_leakage",
        ),
        (
            "OL",
            "slew-aware leakage-cap target-push max leakage minus robust leakage-cap target-push",
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "max_leakage",
        ),
        (
            "OL",
            "full-pulse capped target-push max leakage minus target-push high-fidelity continuation",
            full_pulse_open_leakage_series(),
            high_fidelity_open_leakage_series("hf_leak05"),
            "max_leakage",
        ),
        (
            "OL",
            "full-pulse capped target-push max leakage minus leakage-cap target-push",
            full_pulse_open_leakage_series(),
            cap_open_leakage_series("cap050_w120"),
            "max_leakage",
        ),
        (
            "OL",
            "full-pulse capped target-push max leakage minus robust leakage-cap target-push",
            full_pulse_open_leakage_series(),
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            "max_leakage",
        ),
        (
            "OL",
            "robust leakage-cap target-push max leakage minus leakage-cap target-push",
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            cap_open_leakage_series("cap050_w120"),
            "max_leakage",
        ),
        (
            "OL",
            "leakage-cap target-push max leakage minus open leakage-GRAPE",
            cap_open_leakage_series("cap050_w120"),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "robust leakage-cap target-push max leakage minus open leakage-GRAPE",
            cap_open_leakage_series("cap050_w120_mean02_worst05"),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "worst-seed leakage-cap target-push max leakage minus open leakage-GRAPE",
            worst_cap_open_leakage_series("worstcap050_w120_mean02_worst05"),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "slew-aware leakage-cap target-push max leakage minus open leakage-GRAPE",
            smooth_cap_open_leakage_series("smoothcap050_w120_slew10"),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "full-pulse capped target-push max leakage minus open leakage-GRAPE",
            full_pulse_open_leakage_series(),
            open_leakage_series("leakage_penalized_grape"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage-GRAPE max leakage minus two-stage target-biased open leakage adjoint",
            open_leakage_series("leakage_penalized_grape"),
            standalone_open_leakage_series("two_stage_target_biased_open_leakage_adjoint"),
            "max_leakage",
        ),
        (
            "OL",
            "open leakage-GRAPE max leakage minus adjoint horizon",
            open_leakage_series("leakage_penalized_grape"),
            open_leakage_series("adjoint_horizon"),
            "max_leakage",
        ),
        (
            "Z",
            "compact slew-constrained minus no-slew beam",
            slew_series("Z", 0.005),
            slew_series("Z", 0.0),
            "final_fidelity",
        ),
        (
            "H",
            "compact slew-constrained minus no-slew beam",
            slew_series("H", 0.005),
            slew_series("H", 0.0),
            "final_fidelity",
        ),
        (
            "Z",
            "boxcar3 no-slew filtered minus no-slew beam",
            bandwidth_series("Z", 0.0, "boxcar3"),
            bandwidth_series("Z", 0.0, "none"),
            "final_fidelity",
        ),
        (
            "H",
            "boxcar3 no-slew filtered minus no-slew beam",
            bandwidth_series("H", 0.0, "boxcar3"),
            bandwidth_series("H", 0.0, "none"),
            "final_fidelity",
        ),
        (
            "Z",
            "boxcar3 filtered slew minus slew beam",
            bandwidth_series("Z", 0.005, "boxcar3"),
            bandwidth_series("Z", 0.005, "none"),
            "final_fidelity",
        ),
        (
            "H",
            "boxcar3 filtered slew minus slew beam",
            bandwidth_series("H", 0.005, "boxcar3"),
            bandwidth_series("H", 0.005, "none"),
            "final_fidelity",
        ),
        (
            "Z",
            "boxcar3 filtered slew minus filtered no-slew",
            bandwidth_series("Z", 0.005, "boxcar3"),
            bandwidth_series("Z", 0.0, "boxcar3"),
            "final_fidelity",
        ),
        (
            "H",
            "boxcar3 filtered slew minus filtered no-slew",
            bandwidth_series("H", 0.005, "boxcar3"),
            bandwidth_series("H", 0.0, "boxcar3"),
            "final_fidelity",
        ),
        (
            "Z",
            "gaussian7 filtered slew minus slew beam",
            bandwidth_series("Z", 0.005, "gaussian7"),
            bandwidth_series("Z", 0.005, "none"),
            "final_fidelity",
        ),
        (
            "H",
            "gaussian7 filtered slew minus slew beam",
            bandwidth_series("H", 0.005, "gaussian7"),
            bandwidth_series("H", 0.005, "none"),
            "final_fidelity",
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
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
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
