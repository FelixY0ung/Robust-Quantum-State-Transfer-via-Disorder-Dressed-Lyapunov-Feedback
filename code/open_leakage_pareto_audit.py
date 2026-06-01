"""Pareto audit for the five-level leakage-plus-Lindblad stress test.

This script does not rerun pulse design.  It reads the existing held-out CSV
files, combines the GRAPE-free direct horizons, integrated sweep, Pareto
refinement and continuation rows, reference-assisted horizon, and terminal
baselines, and reports the fidelity/leakage tradeoff at the hardest
combined-noise setting.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from paths import figure_path, result_path


@dataclass(frozen=True)
class ControllerSpec:
    source: str
    controller: str
    label: str
    family: str
    role: str


SPECS = (
    ControllerSpec(
        "transmon_open_leakage_adjoint_results.csv",
        "open_leakage_path_seed",
        "Path horizon",
        "GRAPE-free horizon",
        "dominated seed",
    ),
    ControllerSpec(
        "transmon_open_leakage_adjoint_results.csv",
        "standalone_open_leakage_adjoint",
        "Direct adjoint",
        "GRAPE-free horizon",
        "low-leakage point",
    ),
    ControllerSpec(
        "transmon_open_leakage_adjoint_results.csv",
        "target_biased_open_leakage_adjoint",
        "Target-biased direct",
        "GRAPE-free horizon",
        "no-reference tradeoff",
    ),
    ControllerSpec(
        "open_leakage_integrated_sweep_results.csv",
        "integrated_alpha1p0_lw1p5_trust004",
        "Integrated direct",
        "GRAPE-free horizon",
        "single-stage integrated tradeoff",
    ),
    ControllerSpec(
        "open_leakage_pareto_refinement_results.csv",
        "pareto_alpha0p8_lw1p5_trust004",
        "Pareto alpha0.8 leak1.5",
        "GRAPE-free horizon",
        "low-leakage refinement",
    ),
    ControllerSpec(
        "open_leakage_pareto_refinement_results.csv",
        "pareto_alpha0p8_lw1p2_trust004",
        "Pareto alpha0.8 leak1.2",
        "GRAPE-free horizon",
        "low-leakage refinement",
    ),
    ControllerSpec(
        "open_leakage_pareto_refinement_results.csv",
        "pareto_alpha0p8_lw1p0_trust004",
        "Pareto alpha0.8 leak1.0",
        "GRAPE-free horizon",
        "low-leakage refinement",
    ),
    ControllerSpec(
        "open_leakage_pareto_refinement_results.csv",
        "pareto_alpha0p8_lw0p8_trust004",
        "Pareto alpha0.8 leak0.8",
        "GRAPE-free horizon",
        "target-balanced refinement",
    ),
    ControllerSpec(
        "transmon_open_leakage_adjoint_results.csv",
        "two_stage_target_biased_open_leakage_adjoint",
        "Two-stage direct",
        "GRAPE-free horizon",
        "best no-reference worst seed",
    ),
    ControllerSpec(
        "open_leakage_continuation_sweep_results.csv",
        "continuation_target08_leak12",
        "Continuation controlled",
        "GRAPE-free horizon",
        "leakage-controlled continuation point",
    ),
    ControllerSpec(
        "open_leakage_continuation_sweep_results.csv",
        "continuation_target08_leak10",
        "Continuation balanced",
        "GRAPE-free horizon",
        "balanced continuation point",
    ),
    ControllerSpec(
        "open_leakage_continuation_sweep_results.csv",
        "continuation_target08_leak08",
        "Continuation high-fidelity",
        "GRAPE-free horizon",
        "fidelity-favoring continuation point",
    ),
    ControllerSpec(
        "open_leakage_high_fidelity_sweep_results.csv",
        "hf_leak06",
        "HF continuation leak0.6",
        "GRAPE-free horizon",
        "high-fidelity target-push point",
    ),
    ControllerSpec(
        "open_leakage_high_fidelity_sweep_results.csv",
        "hf_leak05",
        "HF continuation leak0.5",
        "GRAPE-free horizon",
        "high-fidelity target-push point",
    ),
    ControllerSpec(
        "open_leakage_cap_refinement_results.csv",
        "cap050_w120",
        "Leakage-cap target-push",
        "GRAPE-free horizon",
        "cap-refined target-push point",
    ),
    ControllerSpec(
        "open_leakage_cap_refinement_results.csv",
        "cap050_w120_mean02_worst05",
        "Robust leakage-cap target-push",
        "GRAPE-free horizon",
        "balanced cap-refined target-push point",
    ),
    ControllerSpec(
        "open_leakage_worst_cap_refinement_results.csv",
        "worstcap050_w120_mean02_worst05",
        "Worst-seed leakage-cap target-push",
        "GRAPE-free horizon",
        "worst-seed-preserving cap-refined target-push point",
    ),
    ControllerSpec(
        "open_leakage_smooth_cap_refinement_results.csv",
        "smoothcap050_w120_slew10",
        "Slew-aware leakage-cap target-push",
        "GRAPE-free horizon",
        "physical-regularity cap-refined target-push point",
    ),
    ControllerSpec(
        "open_leakage_full_pulse_refinement_results.csv",
        "fullpulse050_w120_tail02_worst05_slew10",
        "Full-pulse capped target-push",
        "GRAPE-free horizon",
        "full-pulse adjoint cap diagnostic",
    ),
    ControllerSpec(
        "open_leakage_high_fidelity_sweep_results.csv",
        "hf_leak04",
        "HF continuation leak0.4",
        "GRAPE-free horizon",
        "highest-mean no-reference point",
    ),
    ControllerSpec(
        "transmon_open_system_leakage_results.csv",
        "adjoint_horizon",
        "Reference-assisted horizon",
        "reference-assisted horizon",
        "reference-assisted Pareto point",
    ),
    ControllerSpec(
        "transmon_open_system_leakage_results.csv",
        "leakage_penalized_grape",
        "Leakage-GRAPE",
        "terminal optimizer",
        "terminal ceiling",
    ),
)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def load_controller(spec: ControllerSpec) -> pd.DataFrame:
    df = pd.read_csv(result_path(spec.source))
    subset = df[
        (df["controller"] == spec.controller)
        & (df["eval_strength"] == 0.03)
        & (df["noise_case"] == "combined")
    ].copy()
    if subset.empty:
        raise ValueError(f"empty subset for {spec.source}/{spec.controller}")
    return subset


def summarize() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw: list[dict[str, float | str]] = []
    for spec in SPECS:
        subset = load_controller(spec)
        raw.append(
            {
                "controller": spec.controller,
                "label": spec.label,
                "family": spec.family,
                "role": spec.role,
                "n": float(len(subset)),
                "mean_fidelity": float(subset["final_fidelity"].mean()),
                "worst_fidelity": float(subset["final_fidelity"].min()),
                "ci95_fidelity": 1.96
                * float(subset["final_fidelity"].std(ddof=1))
                / float(len(subset) ** 0.5),
                "mean_max_leakage": float(subset["max_leakage"].mean()),
                "mean_final_leakage": float(subset["final_leakage"].mean()),
                "pulse_energy": float(subset["pulse_energy"].mean()),
                "training_seconds": float(subset["training_seconds"].mean()),
            }
        )

    for item in raw:
        dominated_by: list[str] = []
        for other in raw:
            if item is other:
                continue
            better_or_equal_fidelity = other["mean_fidelity"] >= item["mean_fidelity"]
            lower_or_equal_leakage = other["mean_max_leakage"] <= item["mean_max_leakage"]
            strict = (
                other["mean_fidelity"] > item["mean_fidelity"]
                or other["mean_max_leakage"] < item["mean_max_leakage"]
            )
            if better_or_equal_fidelity and lower_or_equal_leakage and strict:
                dominated_by.append(str(other["label"]))
        item["pareto_status"] = "Pareto" if not dominated_by else "dominated by " + "; ".join(dominated_by)

    for item in raw:
        rows.append(
            {
                "controller": str(item["controller"]),
                "label": str(item["label"]),
                "family": str(item["family"]),
                "role": str(item["role"]),
                "n": str(int(item["n"])),
                "mean_fidelity": fmt(float(item["mean_fidelity"])),
                "worst_fidelity": fmt(float(item["worst_fidelity"])),
                "ci95_fidelity": fmt(float(item["ci95_fidelity"])),
                "mean_max_leakage": fmt(float(item["mean_max_leakage"])),
                "mean_final_leakage": fmt(float(item["mean_final_leakage"])),
                "pulse_energy": f"{float(item['pulse_energy']):.4g}",
                "training_seconds": f"{float(item['training_seconds']):.1f}",
                "pareto_status": str(item["pareto_status"]),
            }
        )
    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    csv_path = result_path("open_leakage_pareto_audit_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = result_path("open_leakage_pareto_audit_summary.md")
    with md_path.open("w") as f:
        f.write("# Open Leakage Pareto Audit\n\n")
        f.write(
            "Combined five-level leakage-plus-Lindblad evaluation at "
            "$\\delta=0.03$, $\\gamma_\\phi=0.001$, and $\\gamma_1=0.0005$.\n\n"
        )
        f.write("| label | family | mean fidelity | worst fidelity | mean max leakage | Pareto status |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {label} | {family} | {mean_fidelity} | {worst_fidelity} | "
                "{mean_max_leakage} | {pareto_status} |\n".format(**row)
            )


def plot(rows: list[dict[str, str]]) -> None:
    colors = {
        "GRAPE-free horizon": "#1f77b4",
        "reference-assisted horizon": "#2ca02c",
        "terminal optimizer": "#d62728",
    }
    markers = {
        "GRAPE-free horizon": "o",
        "reference-assisted horizon": "s",
        "terminal optimizer": "^",
    }
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for row in rows:
        x = float(row["mean_max_leakage"])
        y = float(row["mean_fidelity"])
        family = row["family"]
        ax.scatter(
            x,
            y,
            s=62,
            color=colors[family],
            marker=markers[family],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (x, y),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Mean final fidelity")
    ax.set_title("Open-leakage fidelity/leakage Pareto audit", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(left=0.0)
    ax.set_xlim(right=0.095)
    ax.set_ylim(0.81, 0.965)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_pareto_audit.pdf"))
    fig.savefig(figure_path("open_leakage_pareto_audit.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    rows = summarize()
    write_outputs(rows)
    plot(rows)
    print(f"wrote {result_path('open_leakage_pareto_audit_results.csv')}")
    print(f"wrote {result_path('open_leakage_pareto_audit_summary.md')}")
    print(f"wrote {figure_path('open_leakage_pareto_audit.pdf')}")


if __name__ == "__main__":
    main()
