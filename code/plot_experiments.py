"""Generate experiment figures for the CAC manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import RESULTS_DIR


FIGURE_DIR = RESULTS_DIR / "figures"


def save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def grouped_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols)["final_fidelity"]
        .agg(["mean", "min", "std"])
        .reset_index()
    )


def plot_fidelity_vs_disorder() -> None:
    robust = pd.read_csv(RESULTS_DIR / "robustness_scan_results.csv")
    horizon = pd.read_csv(RESULTS_DIR / "horizon_lyapunov_results.csv").rename(
        columns={"disorder_strength": "eval_strength"}
    )
    polished_path = RESULTS_DIR / "polished_openloop_results.csv"
    polished = (
        pd.read_csv(polished_path)
        if polished_path.exists()
        else pd.DataFrame(columns=robust.columns)
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    colors = {
        "nominal": "#7a7a7a",
        "ensemble": "#1f77b4",
        "horizon": "#d62728",
        "polished": "#2ca02c",
    }

    for ax, task in zip(axes, ["Z", "H"]):
        nominal = grouped_stats(
            robust[(robust["task"] == task) & (robust["pulse_type"] == "nominal_open_loop")],
            ["eval_strength"],
        )
        ensemble = grouped_stats(
            robust[(robust["task"] == task) & (robust["pulse_type"] == "ensemble_open_loop")],
            ["eval_strength"],
        )
        htask = grouped_stats(
            horizon[horizon["system"] == f"horizon_lyapunov_{task}"],
            ["eval_strength"],
        )
        ax.plot(nominal["eval_strength"], nominal["mean"], "o-", color=colors["nominal"], label="Nominal")
        ax.plot(ensemble["eval_strength"], ensemble["mean"], "s-", color=colors["ensemble"], label="Ensemble open-loop")
        ax.plot(htask["eval_strength"], htask["mean"], "^-", color=colors["horizon"], label="Beam-horizon")
        if not polished.empty:
            ptask = grouped_stats(polished[polished["task"] == task], ["eval_strength"])
            ax.plot(ptask["eval_strength"], ptask["mean"], "D-", color=colors["polished"], label="Polished open-loop")
        ax.set_title(f"{task} transfer")
        ax.set_xlabel("Disorder strength")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.92, 1.0005)
    axes[0].set_ylabel("Held-out final fidelity")
    axes[1].legend(loc="lower left", fontsize=7, frameon=False)
    save(fig, "fidelity_vs_disorder")


def plot_heldout_distribution() -> None:
    horizon = pd.read_csv(RESULTS_DIR / "horizon_lyapunov_results.csv")
    polished = pd.read_csv(RESULTS_DIR / "polished_openloop_results.csv")
    robust = pd.read_csv(RESULTS_DIR / "robustness_scan_results.csv")

    data = []
    labels = []
    for task in ["Z", "H"]:
        data.append(
            robust[
                (robust["task"] == task)
                & (robust["pulse_type"] == "ensemble_open_loop")
                & (robust["eval_strength"] == 0.08)
            ]["final_fidelity"].to_numpy()
        )
        labels.append(f"{task}\nensemble")
        data.append(
            horizon[
                (horizon["system"] == f"horizon_lyapunov_{task}")
                & (horizon["disorder_strength"] == 0.08)
            ]["final_fidelity"].to_numpy()
        )
        labels.append(f"{task}\nhorizon")
        data.append(
            polished[
                (polished["task"] == task)
                & (polished["eval_strength"] == 0.08)
            ]["final_fidelity"].to_numpy()
        )
        labels.append(f"{task}\npolished")

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor("#86b6d8")
        body.set_edgecolor("#355c7d")
        body.set_alpha(0.75)
    for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
        parts[key].set_color("#333333")
        parts[key].set_linewidth(0.8)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final fidelity at $\\delta=0.08$")
    ax.set_ylim(0.986, 1.0005)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "heldout_distribution_delta008")


def plot_multilevel() -> None:
    multi = pd.read_csv(RESULTS_DIR / "multilevel_horizon_results.csv")
    stats = grouped_stats(multi, ["eval_strength"])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.errorbar(
        stats["eval_strength"],
        stats["mean"],
        yerr=stats["std"],
        marker="o",
        color="#8c564b",
        capsize=3,
    )
    ax.fill_between(stats["eval_strength"], stats["min"], stats["mean"], color="#c49c94", alpha=0.25)
    ax.set_xlabel("Disorder strength")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Three-level chain")
    ax.set_ylim(0.97, 1.0005)
    ax.grid(True, alpha=0.25)
    save(fig, "multilevel_horizon")


def main() -> None:
    plot_fidelity_vs_disorder()
    plot_heldout_distribution()
    plot_multilevel()
    print(f"wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
