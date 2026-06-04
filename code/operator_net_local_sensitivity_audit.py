"""Local sensitivity audit for the operator-space finite-net results.

The deterministic finite-net certificate in ``finite_net_operator_audit.py``
uses the global terminal-infidelity Lipschitz constant ``2T``.  That bound is
rigorous but deliberately conservative.  This postprocessing audit reads the
operator-net CSV, builds local neighbor edges on the projected Pauli-coefficient
net, and reports finite-difference infidelity slopes.

The reported slopes are empirical diagnostics on the finite net.  They explain
why the certified ``2T h`` penalty can be loose, but they are not a replacement
for the deterministic continuous-ball certificate unless a separate derivative
or interpolation bound is proved.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import result_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def edge_records(
    task: str,
    method: str,
    rows: list[dict[str, str]],
    radius_multiplier: float,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    coords = np.array(
        [
            [float(row["coeff_x"]), float(row["coeff_y"]), float(row["coeff_z"])]
            for row in rows
        ],
        dtype=float,
    )
    infids = np.array([float(row["final_infidelity"]) for row in rows], dtype=float)
    net_indices = np.array([int(row["net_index"]) for row in rows], dtype=int)
    spacing = float(rows[0]["grid_spacing"])
    covering_radius = float(rows[0]["covering_radius"])
    time_horizon = float(rows[0]["time_horizon"])
    analytic_lipschitz = 2.0 * time_horizon
    axis_radius = radius_multiplier * spacing
    cell_radius = radius_multiplier * 2.0 * covering_radius
    distance_floor = 1e-12

    records: list[dict[str, float | int | str]] = []
    nearest_distances = np.full(len(rows), np.inf)
    nearest_slopes = np.zeros(len(rows))

    for i in range(len(rows) - 1):
        diff = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(diff, axis=1)
        valid = distances > distance_floor

        if np.any(valid):
            valid_indices = np.nonzero(valid)[0]
            valid_distances = distances[valid_indices]
            valid_js = i + 1 + valid_indices
            valid_slopes = np.abs(infids[valid_js] - infids[i]) / valid_distances
            nearest_offset = valid_indices[int(np.argmin(valid_distances))]
            nearest_j = i + 1 + nearest_offset
            nearest_distance = float(distances[nearest_offset])
            nearest_slope = float(
                abs(infids[nearest_j] - infids[i]) / nearest_distance
            )
            if nearest_distance < nearest_distances[i]:
                nearest_distances[i] = nearest_distance
                nearest_slopes[i] = nearest_slope
            better = valid_distances < nearest_distances[valid_js]
            nearest_distances[valid_js[better]] = valid_distances[better]
            nearest_slopes[valid_js[better]] = valid_slopes[better]

        mask = valid & (distances <= cell_radius)
        for offset in np.nonzero(mask)[0]:
            j = i + 1 + int(offset)
            distance = float(distances[offset])
            delta_infidelity = float(abs(infids[j] - infids[i]))
            slope = delta_infidelity / distance
            edge_class = "axis_radius" if distance <= axis_radius else "cell_radius_only"
            records.append(
                {
                    "task": task,
                    "method": method,
                    "edge_class": edge_class,
                    "net_i": int(net_indices[i]),
                    "net_j": int(net_indices[j]),
                    "distance": distance,
                    "delta_infidelity": delta_infidelity,
                    "slope": slope,
                    "grid_spacing": spacing,
                    "covering_radius": covering_radius,
                    "analytic_lipschitz_2T": analytic_lipschitz,
                }
            )

    edge_slopes = np.array([float(row["slope"]) for row in records], dtype=float)
    axis_slopes = np.array(
        [float(row["slope"]) for row in records if row["edge_class"] == "axis_radius"],
        dtype=float,
    )
    nearest_slopes = nearest_slopes[np.isfinite(nearest_distances)]
    nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
    nearest_stats = quantiles(nearest_slopes)
    axis_stats = quantiles(axis_slopes)
    cell_stats = quantiles(edge_slopes)
    worst_net_infidelity = float(np.max(infids))
    empirical_cell_penalty = cell_stats["max"] * covering_radius
    empirical_axis_penalty = axis_stats["max"] * covering_radius
    analytic_penalty = analytic_lipschitz * covering_radius
    summary = {
        "task": task,
        "method": method,
        "net_points": len(rows),
        "edge_radius_multiplier": radius_multiplier,
        "grid_spacing": spacing,
        "covering_radius_h": covering_radius,
        "axis_radius": axis_radius,
        "cell_radius": cell_radius,
        "neighbor_edges": len(records),
        "axis_edges": int(np.sum([row["edge_class"] == "axis_radius" for row in records])),
        "cell_only_edges": int(
            np.sum([row["edge_class"] == "cell_radius_only" for row in records])
        ),
        "nearest_distance_min": float(np.min(nearest_distances)),
        "nearest_distance_median": float(np.median(nearest_distances)),
        "nearest_slope_p95": nearest_stats["p95"],
        "nearest_slope_max": nearest_stats["max"],
        "axis_slope_p95": axis_stats["p95"],
        "axis_slope_p99": axis_stats["p99"],
        "axis_slope_max": axis_stats["max"],
        "cell_slope_p95": cell_stats["p95"],
        "cell_slope_p99": cell_stats["p99"],
        "cell_slope_max": cell_stats["max"],
        "analytic_lipschitz_2T": analytic_lipschitz,
        "cell_max_over_2T": cell_stats["max"] / analytic_lipschitz,
        "axis_max_over_2T": axis_stats["max"] / analytic_lipschitz,
        "worst_net_fidelity": 1.0 - worst_net_infidelity,
        "worst_net_infidelity": worst_net_infidelity,
        "analytic_h_penalty": analytic_penalty,
        "analytic_continuous_fidelity_lower_bound": 1.0
        - min(1.0, worst_net_infidelity + analytic_penalty),
        "empirical_axis_h_penalty_if_global": empirical_axis_penalty,
        "empirical_cell_h_penalty_if_global": empirical_cell_penalty,
        "empirical_cell_lower_bound_if_global": 1.0
        - min(1.0, worst_net_infidelity + empirical_cell_penalty),
    }
    return records, summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write("# Operator-Net Local Sensitivity Audit\n\n")
        f.write(
            "Postprocessing audit of neighboring finite differences on the "
            "projected Pauli-coefficient h-net used by the operator-space "
            "finite-net robustness audit. The slope columns are empirical "
            "local diagnostics on the finite net. They explain the "
            "conservatism of the analytic `2T h` penalty but do not replace "
            "the deterministic continuous-ball certificate.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            formatted = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    formatted.append(f"{value:.6g}")
                else:
                    formatted.append(str(value))
            f.write("| " + " | ".join(formatted) + " |\n")


def run(args: argparse.Namespace) -> None:
    input_path = result_path(args.input)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(input_path):
        groups[(row["task"], row["method"])].append(row)

    all_edges: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for task, method in sorted(groups):
        print(f"auditing local sensitivity for {task}/{method}", flush=True)
        records, summary = edge_records(
            task,
            method,
            groups[(task, method)],
            args.radius_multiplier,
        )
        all_edges.extend(records)
        summaries.append(summary)

    result_csv_path = result_path("operator_net_local_sensitivity_results.csv")
    summary_md_path = result_path("operator_net_local_sensitivity_summary.md")
    write_csv(result_csv_path, summaries)
    write_summary(summary_md_path, summaries)
    if args.write_edges:
        edge_path = result_path("operator_net_local_sensitivity_edges.csv")
        write_csv(edge_path, all_edges)
        print(f"wrote {len(all_edges)} edge rows to {edge_path}")
    print(f"wrote summary CSV to {result_csv_path}")
    print(f"wrote summary Markdown to {summary_md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="finite_net_operator_audit_results.csv")
    parser.add_argument(
        "--radius-multiplier",
        type=float,
        default=1.01,
        help="Multiplier applied to the grid spacing and 2h neighbor radii.",
    )
    parser.add_argument(
        "--write-edges",
        action="store_true",
        help="Also write every local neighbor edge to a large CSV artifact.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
