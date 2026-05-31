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
    dcrab = load("dcrab_baseline_results.csv")
    grape = load("ensemble_grape_baseline_results.csv")
    process = load("process_horizon_results.csv")
    process_adj = load("process_adjoint_horizon_results.csv")
    open_noise = load("open_system_noise_results.csv")
    standalone_open = load("open_system_standalone_adjoint_results.csv")
    open_adj = load("open_system_adjoint_horizon_results.csv")
    leakage = load("transmon_leakage_results.csv")
    standalone_leakage = load("transmon_standalone_adjoint_results.csv")
    open_leakage = load("transmon_open_system_leakage_results.csv")
    standalone_open_leakage = load("transmon_open_leakage_adjoint_results.csv")

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
    with path.open("w", newline="") as f:
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
