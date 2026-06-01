"""Postprocess multi-step terminal fallback results as terminal-value evidence.

The multi-step terminal-fallback audit reports whether a bounded block lowers
the previous predicted terminal Lyapunov score.  This script asks a related
terminal-value question: after defining the best available L-step terminal
block score, does extending the terminal block from L to L+1 lower that best
score often enough to suggest a contraction-style terminal value?

It reads ``multistep_terminal_fallback_audit_results.csv`` and writes a compact
CSV/Markdown summary.  No trajectories are recomputed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import result_path


INPUT = result_path("multistep_terminal_fallback_audit_results.csv")
PAIRS = ((1, 2), (2, 3), (1, 3))


def load_block_scores(path: Path) -> dict[tuple[str, int], dict[int, float]]:
    scores: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (str(row["task"]), int(row["step"]))
            scores[key][int(row["block_steps"])] = float(row["best_block_phi"])
    return dict(scores)


def build_rows(scores: dict[tuple[str, int], dict[int, float]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for (task, step), block_scores in sorted(scores.items()):
        for shorter, longer in PAIRS:
            if shorter not in block_scores or longer not in block_scores:
                continue
            margin = block_scores[shorter] - block_scores[longer]
            rows.append(
                {
                    "task": task,
                    "step": step,
                    "comparison": f"{shorter}_to_{longer}",
                    "shorter_steps": shorter,
                    "longer_steps": longer,
                    "shorter_phi": block_scores[shorter],
                    "longer_phi": block_scores[longer],
                    "extension_margin": margin,
                    "extension_improves_value": int(margin > 0.0),
                }
            )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    keys = sorted({(str(row["task"]), str(row["comparison"])) for row in rows})
    for task, comparison in keys:
        group = [
            row
            for row in rows
            if str(row["task"]) == task and str(row["comparison"]) == comparison
        ]
        margins = np.array([float(row["extension_margin"]) for row in group])
        shorter_phi = np.array([float(row["shorter_phi"]) for row in group])
        longer_phi = np.array([float(row["longer_phi"]) for row in group])
        summary.append(
            {
                "task": task,
                "comparison": comparison,
                "audited_steps": str(len(group)),
                "positive_extension_fraction": f"{float(np.mean(margins > 0.0)):.6g}",
                "extension_margin_mean": f"{float(np.mean(margins)):.6g}",
                "extension_margin_median": f"{float(np.median(margins)):.6g}",
                "extension_margin_min": f"{float(np.min(margins)):.6g}",
                "extension_margin_max": f"{float(np.max(margins)):.6g}",
                "shorter_phi_mean": f"{float(np.mean(shorter_phi)):.6g}",
                "longer_phi_mean": f"{float(np.mean(longer_phi)):.6g}",
            }
        )
    return summary


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("terminal_value_extension_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("terminal_value_extension_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Terminal-Value Extension Audit Summary\n\n")
        f.write(
            "This postprocesses the multi-step terminal-fallback audit. A positive "
            "extension margin means that the best longer terminal block has a "
            "lower terminal Lyapunov score than the best shorter block on the "
            "same realized terminal state. It is a diagnostic for whether a "
            "short-block terminal-value surrogate is itself contractive.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    rows = build_rows(load_block_scores(INPUT))
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
