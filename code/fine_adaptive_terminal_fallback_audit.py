"""Fine-radius adaptive terminal fallback audit.

The coarse adaptive terminal-fallback audit tests descent radii starting at
0.25 and therefore mainly checks whether a relatively large one-segment
fallback can improve the terminal Lyapunov score.  This audit repeats the same
realized shifted-fallback trajectory with finer radii near zero.  It tests
whether local terminal descent exists but is too small to be captured by the
coarse amplitude grid.
"""

from __future__ import annotations

import csv

from adaptive_terminal_fallback_audit import (
    AdaptiveConfig,
    audit_task,
    summarize,
)
from paths import result_path
from shifted_fallback_margin_audit import MarginConfig


FINE_RADII = tuple(
    [0.0]
    + [i / 100.0 for i in range(1, 26)]
    + [0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
)


def relabel(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    relabeled = []
    for row in rows:
        new_row = dict(row)
        if new_row["method"] == "adaptive_gradient_line":
            new_row["method"] = "fine_adaptive_gradient_line"
        relabeled.append(new_row)
    return relabeled


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("fine_adaptive_terminal_fallback_audit_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("fine_adaptive_terminal_fallback_audit_summary.md").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Fine Adaptive Terminal-Fallback Audit Summary\n\n")
        f.write(
            "The realized shifted-fallback trajectory is kept fixed. The fine "
            "adaptive row uses dense line-search radii near zero before falling "
            "back to the larger radii used in the coarse adaptive audit. This "
            "tests whether local terminal descent exists but is too small for "
            "the coarse amplitude grid.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def main() -> None:
    margin_config = MarginConfig()
    adaptive_config = AdaptiveConfig(line_radii=FINE_RADII)
    rows = []
    for task in ("Z", "H"):
        print(f"auditing fine adaptive terminal fallback for {task}", flush=True)
        rows.extend(audit_task(task, margin_config, adaptive_config))
    rows = relabel(rows)
    write_outputs(rows)
    for row in summarize(rows):
        print(row)


if __name__ == "__main__":
    main()
