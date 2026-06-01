"""Resource and performance audit for the journal manuscript.

The manuscript contains several controllers that solve different objectives.
This script gathers the representative held-out performance, pulse energy, and
available resource metadata into one reproducible table.  The audit is not a
speed benchmark: older finite-candidate runs did not record wall-clock design
time, so the table reports logged design seconds only where the generating
script wrote them to disk.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from paths import result_path


TaskFilter = Callable[[pd.DataFrame, str], pd.DataFrame]


@dataclass(frozen=True)
class PairSummary:
    n: str
    mean: str
    worst: str
    energy: str
    seconds: str


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def load(filename: str) -> pd.DataFrame:
    return pd.read_csv(result_path(filename))


def pair_summary(
    df: pd.DataFrame,
    selector: TaskFilter,
    metric: str,
    energy: str = "pulse_energy",
    seconds_columns: tuple[str, ...] = (),
) -> PairSummary:
    means: list[str] = []
    worsts: list[str] = []
    energies: list[str] = []
    seconds: list[str] = []
    ns: list[str] = []
    for task in ("Z", "H"):
        subset = selector(df, task)
        if subset.empty:
            raise ValueError(f"empty subset for task {task} and metric {metric}")
        values = subset[metric].astype(float)
        means.append(fmt(float(values.mean())))
        worsts.append(fmt(float(values.min())))
        energies.append(f"{float(subset[energy].astype(float).mean()):.4g}")
        ns.append(str(len(subset)))
        if seconds_columns:
            total = 0.0
            for column in seconds_columns:
                total += float(subset[column].astype(float).mean())
            seconds.append(fmt_seconds(total))
    seconds_text = "/".join(seconds) if seconds else ""
    return PairSummary(
        n="/".join(ns),
        mean="/".join(means),
        worst="/".join(worsts),
        energy="/".join(energies),
        seconds=seconds_text,
    )


def leakage_summary(df: pd.DataFrame, controller: str) -> PairSummary:
    subset = df[(df["controller"] == controller) & (df["eval_strength"] == 0.03)]
    if subset.empty:
        raise ValueError(f"empty leakage subset for {controller}")
    fids = subset["final_fidelity"].astype(float)
    max_leak = subset["max_leakage"].astype(float)
    seconds = float(subset["training_seconds"].astype(float).mean())
    return PairSummary(
        n=str(len(subset)),
        mean=fmt(float(fids.mean())),
        worst=f"{fmt(float(fids.min()))}; leak {fmt(float(max_leak.mean()), 4)}",
        energy=f"{float(subset['pulse_energy'].astype(float).mean()):.4g}",
        seconds=fmt_seconds(seconds),
    )


def open_leakage_summary(
    df: pd.DataFrame,
    controller: str,
    noise_case: str = "combined",
) -> PairSummary:
    subset = df[
        (df["controller"] == controller)
        & (df["eval_strength"] == 0.03)
        & (df["noise_case"] == noise_case)
    ]
    if subset.empty:
        raise ValueError(f"empty open leakage subset for {controller}/{noise_case}")
    fids = subset["final_fidelity"].astype(float)
    max_leak = subset["max_leakage"].astype(float)
    seconds = float(subset["training_seconds"].astype(float).mean())
    return PairSummary(
        n=str(len(subset)),
        mean=fmt(float(fids.mean())),
        worst=f"{fmt(float(fids.min()))}; leak {fmt(float(max_leak.mean()), 4)}",
        energy=f"{float(subset['pulse_energy'].astype(float).mean()):.4g}",
        seconds=fmt_seconds(seconds),
    )


def rows() -> list[dict[str, str]]:
    horizon = load("horizon_lyapunov_results.csv")
    slew = load("slew_constrained_horizon_results.csv")
    bandwidth = load("bandwidth_filter_audit_results.csv")
    krotov = load("krotov_baseline_results.csv")
    dcrab = load("dcrab_baseline_results.csv")
    dcrab_train8 = load("dcrab_train8_baseline_results.csv")
    grape = load("ensemble_grape_baseline_results.csv")
    process = load("process_horizon_results.csv")
    process_dcrab = load("process_dcrab_baseline_results.csv")
    process_dcrab_seeded = load("process_dcrab_seeded_horizon_results.csv")
    process_adj = load("process_adjoint_horizon_results.csv")
    open_noise = load("open_system_noise_results.csv")
    standalone_open = load("open_system_standalone_adjoint_results.csv")
    open_adj = load("open_system_adjoint_horizon_results.csv")
    leakage = load("transmon_leakage_results.csv")
    standalone_leakage = load("transmon_standalone_adjoint_results.csv")
    open_leakage = load("transmon_open_system_leakage_results.csv")
    standalone_open_leakage = load("transmon_open_leakage_adjoint_results.csv")
    integrated_open_leakage = load("open_leakage_integrated_sweep_results.csv")
    continuation_open_leakage = load("open_leakage_continuation_sweep_results.csv")
    high_fidelity_open_leakage = load("open_leakage_high_fidelity_sweep_results.csv")
    cap_open_leakage = load("open_leakage_cap_refinement_results.csv")
    worst_cap_open_leakage = load("open_leakage_worst_cap_refinement_results.csv")
    smooth_cap_open_leakage = load("open_leakage_smooth_cap_refinement_results.csv")
    full_pulse_open_leakage = load("open_leakage_full_pulse_refinement_results.csv")

    audit_rows: list[dict[str, str]] = []

    def add(
        regime: str,
        method: str,
        metric: str,
        summary: PairSummary,
        resources: str,
        role: str,
    ) -> None:
        audit_rows.append(
            {
                "regime": regime,
                "method": method,
                "metric": metric,
                "heldout_n": summary.n,
                "heldout_mean": summary.mean,
                "worst_or_leakage": summary.worst,
                "pulse_energy": summary.energy,
                "resource_profile": resources,
                "logged_design_seconds": summary.seconds,
                "role": role,
            }
        )

    add(
        "two-level transfer",
        "beam-horizon Lyapunov",
        "final state fidelity at delta=0.08",
        pair_summary(
            horizon,
            lambda df, task: df[
                (df["system"] == f"horizon_lyapunov_{task}")
                & (df["disorder_strength"] == 0.08)
            ],
            "final_fidelity",
        ),
        "100 seg; 16 train samples; q=6; B=6/8; finite grid",
        "proposed interpretable horizon controller",
    )
    add(
        "two-level transfer",
        "slew-constrained beam",
        "final state fidelity at delta=0.08",
        pair_summary(
            slew,
            lambda df, task: df[
                (df["task"] == task) & (df["slew_weight"].astype(float) == 0.005)
            ],
            "final_fidelity",
            seconds_columns=("design_seconds",),
        ),
        "60 seg; 16 train samples; q=4; B=6/8; slew weight=0.005",
        "physical-smoothness diagnostic",
    )
    add(
        "two-level transfer",
        "filtered slew beam",
        "final state fidelity at delta=0.08",
        pair_summary(
            bandwidth,
            lambda df, task: df[
                (df["task"] == task)
                & (df["base_slew_weight"].astype(float) == 0.005)
                & (df["filter"] == "boxcar3")
            ],
            "final_fidelity",
            seconds_columns=("design_seconds",),
        ),
        "60 seg; 16 train samples; q=4; B=6/8; slew weight=0.005; boxcar3 filter",
        "low-pass post-filter diagnostic",
    )
    add(
        "two-level transfer",
        "ensemble Krotov",
        "final state fidelity at delta=0.08",
        pair_summary(
            krotov,
            lambda df, task: df[(df["task"] == task) & (df["eval_strength"] == 0.08)],
            "state_transfer_fidelity",
            seconds_columns=("optimization_seconds",),
        ),
        "40 seg; 8 train seeds; 60 Krotov iter.; mean state-transfer functional",
        "Krotov-family terminal comparator",
    )
    add(
        "two-level transfer",
        "dCRAB terminal",
        "final state fidelity at delta=0.08",
        pair_summary(
            dcrab,
            lambda df, task: df[(df["task"] == task) & (df["eval_strength"] == 0.08)],
            "final_fidelity",
            seconds_columns=("optimization_seconds",),
        ),
        "40 seg; 4 train seeds; 3 Fourier modes; 3 basis refreshes",
        "derivative-free terminal ceiling",
    )
    add(
        "two-level transfer",
        "dCRAB train-8 terminal",
        "final state fidelity at delta=0.08",
        pair_summary(
            dcrab_train8,
            lambda df, task: df[(df["task"] == task) & (df["eval_strength"] == 0.08)],
            "final_fidelity",
            seconds_columns=("optimization_seconds",),
        ),
        "40 seg; 8 train seeds; 3 Fourier modes; 3 basis refreshes",
        "larger-training derivative-free terminal ceiling",
    )
    add(
        "two-level transfer",
        "ensemble GRAPE state",
        "final state fidelity at delta=0.08",
        pair_summary(
            grape,
            lambda df, task: df[
                (df["task"] == task)
                & (df["objective_kind"] == "state")
                & (df["disorder_strength"] == 0.08)
            ],
            "state_transfer_fidelity",
            seconds_columns=("optimization_seconds",),
        ),
        "60 seg; 16 train samples; exact Frechet gradients",
        "gradient terminal comparator",
    )

    add(
        "process fidelity",
        "finite process horizon",
        "average gate fidelity at delta=0.08",
        pair_summary(
            process,
            lambda df, task: df[(df["task"] == task) & (df["controller"] == "process_horizon")],
            "average_gate_fidelity",
        ),
        "100 seg; 8 train seeds; q=6; B=8; finite grid",
        "direct process-horizon diagnostic",
    )
    add(
        "process fidelity",
        "process dCRAB",
        "average gate fidelity at delta=0.08",
        pair_summary(
            process_dcrab,
            lambda df, task: df[(df["task"] == task) & (df["eval_strength"] == 0.08)],
            "average_gate_fidelity",
            seconds_columns=("optimization_seconds",),
        ),
        "40 seg; 8 train seeds; 3 Fourier modes; 3 basis refreshes",
        "derivative-free terminal process ceiling",
    )
    add(
        "process fidelity",
        "dCRAB-seeded process horizon",
        "average gate fidelity at delta=0.08",
        pair_summary(
            process_dcrab_seeded,
            lambda df, task: df[
                (df["task"] == task)
                & (df["controller"] == "dcrab_seeded_process_horizon")
            ],
            "average_gate_fidelity",
            seconds_columns=("reference_seconds", "horizon_seconds"),
        ),
        "40 seg; 8 train seeds; q=4; B=6; derivative-free process reference",
        "reference-assisted horizon diagnostic without a process-GRAPE reference",
    )
    add(
        "process fidelity",
        "adjoint process horizon",
        "average gate fidelity at delta=0.08",
        pair_summary(
            process_adj,
            lambda df, task: df[
                (df["task"] == task) & (df["controller"] == "adjoint_process_horizon")
            ],
            "average_gate_fidelity",
            seconds_columns=("reference_seconds", "horizon_seconds"),
        ),
        "60 seg; 16 train samples; q=4; reference-assisted adjoint horizon",
        "process objective inside horizon architecture",
    )
    add(
        "process fidelity",
        "process GRAPE",
        "average gate fidelity at delta=0.08",
        pair_summary(
            process_adj,
            lambda df, task: df[
                (df["task"] == task) & (df["controller"] == "process_grape_reference")
            ],
            "average_gate_fidelity",
            seconds_columns=("reference_seconds",),
        ),
        "60 seg; 16 train samples; exact Frechet gradients",
        "terminal process ceiling",
    )

    add(
        "open-system combined noise",
        "closed beam pulse",
        "final state fidelity at delta=0.08",
        pair_summary(
            open_noise,
            lambda df, task: df[
                (df["task"] == task)
                & (df["noise_case"] == "combined")
                & (df["disorder_strength"] == 0.08)
            ],
            "final_fidelity",
        ),
        "100 seg; closed-system beam pulse; evaluated under Lindblad noise",
        "stress test without dissipative training",
    )
    add(
        "open-system combined noise",
        "standalone Lindblad adjoint",
        "final state fidelity at delta=0.08",
        pair_summary(
            standalone_open,
            lambda df, task: df[
                (df["task"] == task)
                & (df["controller"] == "standalone_open_adjoint")
                & (df["eval_noise_case"] == "combined")
            ],
            "final_fidelity",
            seconds_columns=("reference_seconds", "horizon_seconds"),
        ),
        "60 seg; 3 train seeds; q=4; target-only adjoint horizon",
        "GRAPE-free dissipative adjoint diagnostic",
    )
    add(
        "open-system combined noise",
        "adjoint Lindblad horizon",
        "final state fidelity at delta=0.08",
        pair_summary(
            open_adj,
            lambda df, task: df[
                (df["task"] == task)
                & (df["controller"] == "adjoint_open_horizon")
                & (df["eval_noise_case"] == "combined")
            ],
            "final_fidelity",
            seconds_columns=("reference_seconds", "horizon_seconds"),
        ),
        "40 seg; 4 train seeds; q=4; Liouville Frechet gradients",
        "dissipative horizon diagnostic",
    )
    add(
        "open-system combined noise",
        "open-system GRAPE",
        "final state fidelity at delta=0.08",
        pair_summary(
            open_adj,
            lambda df, task: df[
                (df["task"] == task)
                & (df["controller"] == "open_grape_reference")
                & (df["eval_noise_case"] == "combined")
            ],
            "final_fidelity",
            seconds_columns=("reference_seconds",),
        ),
        "40 seg; 4 train seeds; Liouville Frechet gradients",
        "terminal dissipative ceiling",
    )

    add(
        "five-level leakage",
        "path horizon",
        "final state fidelity at delta=0.03",
        leakage_summary(leakage, "path_horizon"),
        "120 seg; 12 train samples; q=5; B=6; finite grid",
        "leakage-aware finite-candidate horizon",
    )
    add(
        "five-level leakage",
        "standalone adjoint horizon",
        "final state fidelity at delta=0.03",
        leakage_summary(standalone_leakage, "standalone_adjoint_horizon"),
        "120 seg; 12 train samples; q=5; path-seed adjoint polish",
        "GRAPE-free leakage-adjoint diagnostic",
    )
    add(
        "five-level leakage",
        "adjoint leakage horizon",
        "final state fidelity at delta=0.03",
        leakage_summary(leakage, "adjoint_horizon"),
        "80 seg; 12 train samples; q=5; trust-region adjoint polish",
        "gradient-assisted leakage horizon",
    )
    add(
        "five-level leakage",
        "leakage-penalized GRAPE",
        "final state fidelity at delta=0.03",
        leakage_summary(leakage, "leakage_penalized_grape"),
        "80 seg; 8 train samples; running leakage penalty",
        "terminal leakage-aware ceiling",
    )
    add(
        "five-level leakage + Lindblad",
        "path horizon",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(open_leakage, "path_horizon"),
        "120 seg; gamma_phi=0.001; gamma_relax=0.0005; evaluated under five-level Lindblad noise",
        "combined physical stress test",
    )
    add(
        "five-level leakage + Lindblad",
        "standalone open-leakage adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(standalone_open_leakage, "standalone_open_leakage_adjoint"),
        "120 seg; 4 train seeds; q=5; direct Lindblad leakage-adjoint horizon",
        "GRAPE-free direct combined-noise horizon diagnostic",
    )
    add(
        "five-level leakage + Lindblad",
        "target-biased open-leakage adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(standalone_open_leakage, "target_biased_open_leakage_adjoint"),
        "120 seg; 4 train seeds; q=5; terminal-target-biased direct Lindblad leakage-adjoint horizon",
        "GRAPE-free terminal-biased combined-noise horizon diagnostic",
    )
    add(
        "five-level leakage + Lindblad",
        "integrated open-leakage adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            integrated_open_leakage,
            "integrated_alpha1p0_lw1p5_trust004",
        ),
        "120 seg; 4 train seeds; q=5; terminal weight=1.0; leakage weight=1.5; trust radius=0.04",
        "GRAPE-free integrated combined-noise Pareto tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "two-stage target-biased open-leakage adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            standalone_open_leakage,
            "two_stage_target_biased_open_leakage_adjoint",
        ),
        "120 seg; 4 train seeds; q=5; target recovery then leakage repolish",
        "GRAPE-free fidelity-favoring combined-noise horizon tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "continuation open-leakage adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            continuation_open_leakage,
            "continuation_target08_leak12",
        ),
        "120 seg; 4 train seeds; q=5; low-leakage seed then two target/leakage continuation stages",
        "best GRAPE-free mean/leakage combined-noise horizon tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "fidelity-favoring continuation adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            continuation_open_leakage,
            "continuation_target08_leak08",
        ),
        "120 seg; 4 train seeds; q=5; lower leakage penalty in final continuation stage",
        "highest-fidelity GRAPE-free combined-noise horizon row, with higher leakage",
    )
    add(
        "five-level leakage + Lindblad",
        "target-push continuation adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            high_fidelity_open_leakage,
            "hf_leak05",
        ),
        "120 seg; 4 train seeds; q=5; low-leakage seed then leak-0.5 target-push continuation",
        "highest-fidelity GRAPE-free row that exceeds the reference-assisted mean, with higher leakage",
    )
    add(
        "five-level leakage + Lindblad",
        "leakage-cap target-push adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            cap_open_leakage,
            "cap050_w120",
        ),
        "120 seg; 4 train seeds; q=5; target-push seed then cap-0.05 leakage polish",
        "GRAPE-free cap-refined target-push tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "robust leakage-cap target-push adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            cap_open_leakage,
            "cap050_w120_mean02_worst05",
        ),
        "120 seg; 4 train seeds; q=5; cap-0.05 polish with worst weight 0.5 and mean-leakage term",
        "GRAPE-free balanced cap-refined target-push tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "worst-seed leakage-cap target-push adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            worst_cap_open_leakage,
            "worstcap050_w120_mean02_worst05",
        ),
        "120 seg; 4 train seeds; q=5; worst-weighted target-push seed then cap-0.05 polish",
        "GRAPE-free worst-seed-preserving cap-refined target-push tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "slew-aware leakage-cap target-push adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            smooth_cap_open_leakage,
            "smoothcap050_w120_slew10",
        ),
        "120 seg; 4 train seeds; q=5; cap-0.05 polish with worst weight 0.5, mean-leakage term, and slew weight 10",
        "GRAPE-free physical-regularity cap-refined target-push tradeoff",
    )
    add(
        "five-level leakage + Lindblad",
        "full-pulse capped target-push adjoint",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(
            full_pulse_open_leakage,
            "fullpulse050_w120_tail02_worst05_slew10",
        ),
        "120 seg; 4 train seeds; full-pulse Frechet adjoint; cap-0.05; tail target weight 0.2; slew weight 10",
        "GRAPE-free full-pulse diagnostic initialized by target-push horizon",
    )
    add(
        "five-level leakage + Lindblad",
        "adjoint leakage horizon",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(open_leakage, "adjoint_horizon"),
        "80 seg; gamma_phi=0.001; gamma_relax=0.0005; reference-assisted horizon",
        "combined physical horizon diagnostic",
    )
    add(
        "five-level leakage + Lindblad",
        "leakage-penalized GRAPE",
        "combined-noise final fidelity at delta=0.03",
        open_leakage_summary(open_leakage, "leakage_penalized_grape"),
        "80 seg; gamma_phi=0.001; gamma_relax=0.0005; terminal leakage-aware pulse",
        "combined physical terminal ceiling",
    )

    return audit_rows


def write_csv(audit_rows: list[dict[str, str]]) -> None:
    path = result_path("resource_audit_results.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(audit_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"wrote {len(audit_rows)} rows to {path}")


def write_summary(audit_rows: list[dict[str, str]]) -> None:
    path = result_path("resource_audit_summary.md")
    columns = [
        "regime",
        "method",
        "metric",
        "heldout_n",
        "heldout_mean",
        "worst_or_leakage",
        "pulse_energy",
        "resource_profile",
        "logged_design_seconds",
        "role",
    ]
    with path.open("w") as f:
        f.write("# Resource and Performance Audit\n\n")
        f.write(
            "This audit aggregates representative held-out performance and "
            "resource metadata from existing result files. Logged design seconds "
            "are reported only for scripts that wrote timing fields.\n\n"
        )
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in audit_rows:
            f.write("| " + " | ".join(row[col] for col in columns) + " |\n")
    print(f"wrote summary to {path}")


def main() -> None:
    audit_rows = rows()
    write_csv(audit_rows)
    write_summary(audit_rows)


if __name__ == "__main__":
    main()
