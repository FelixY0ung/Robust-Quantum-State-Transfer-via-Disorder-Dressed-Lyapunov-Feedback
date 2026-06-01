"""Large held-out audit for the two-level robust-transfer comparison.

The main tables use 50 held-out disorder seeds.  This audit reruns the same
beam-horizon and train-8 dCRAB design protocols, then evaluates the resulting
pulses on a larger disjoint held-out range.  The larger sample tightens the
Monte Carlo uncertainty for the central state-transfer comparison; it is still
an empirical audit rather than a distribution-free robustness certificate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

from paths import result_path
from strong_disorder_audit import (
    AuditConfig,
    evaluate_beam_horizon,
    evaluate_train8_dcrab,
    paired_deltas,
    summarize,
)


@dataclass(frozen=True)
class ExtendedHeldoutConfig:
    eval_strengths: tuple[float, ...] = (0.08, 0.12)
    eval_seed_start: int = 1000
    eval_seed_count: int = 200
    tasks: tuple[str, ...] = ("Z", "H")

    @property
    def eval_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.eval_seed_start, self.eval_seed_start + self.eval_seed_count)
        )

    def as_strong_config(self) -> AuditConfig:
        return AuditConfig(
            eval_strengths=self.eval_strengths,
            eval_seeds=self.eval_seeds,
            tasks=self.tasks,
        )


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("extended_heldout_audit_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    config: ExtendedHeldoutConfig,
    summary: list[dict[str, str]],
    deltas: list[dict[str, str]],
) -> None:
    with result_path("extended_heldout_audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Extended Held-Out Audit\n\n")
        f.write(
            "The beam-horizon controller is trained on delta 0.05/0.08 and "
            "the train-8 dCRAB comparator is trained at delta 0.08, using the "
            "same design protocols as the strong-disorder audit. The rows "
            f"below evaluate the regenerated pulses on {config.eval_seed_count} "
            f"disjoint held-out disorder seeds `{config.eval_seed_start}`.."
            f"`{config.eval_seed_start + config.eval_seed_count - 1}`. "
            "Rows at delta 0.12 are out-of-training-range tests; no controller "
            "is retuned for those rows. The empirical minima are minima over "
            "this finite held-out range, not guaranteed worst cases over the "
            "continuous disorder distribution.\n\n"
        )

        headers = list(summary[0].keys())
        f.write("## Held-Out Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

        f.write("\n## Paired Deltas\n\n")
        if deltas:
            delta_headers = list(deltas[0].keys())
            f.write("| " + " | ".join(delta_headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(delta_headers)) + " |\n")
            for row in deltas:
                f.write("| " + " | ".join(row[h] for h in delta_headers) + " |\n")
        else:
            f.write("No paired deltas were computed because fewer than two methods were run.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test audit.")
    parser.add_argument("--seed-start", type=int, default=ExtendedHeldoutConfig.eval_seed_start)
    parser.add_argument("--seed-count", type=int, default=ExtendedHeldoutConfig.eval_seed_count)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("beam_horizon", "dcrab_train8"),
        default=("beam_horizon", "dcrab_train8"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        config = ExtendedHeldoutConfig(
            eval_strengths=(0.08,),
            eval_seed_start=args.seed_start,
            eval_seed_count=min(args.seed_count, 8),
        )
    else:
        config = ExtendedHeldoutConfig(
            eval_seed_start=args.seed_start,
            eval_seed_count=args.seed_count,
        )

    audit_config = config.as_strong_config()
    rows: list[dict[str, float | int | str]] = []
    for task in audit_config.tasks:
        if "beam_horizon" in args.methods:
            print(f"designing and evaluating beam horizon for {task}", flush=True)
            rows.extend(evaluate_beam_horizon(task, audit_config, args.quick))
        if "dcrab_train8" in args.methods:
            print(f"optimizing and evaluating train-8 dCRAB for {task}", flush=True)
            rows.extend(evaluate_train8_dcrab(task, audit_config, args.quick))

    summary = summarize(rows)
    deltas = paired_deltas(rows)
    write_csv(rows)
    write_markdown(config, summary, deltas)
    print(f"wrote {len(rows)} rows to {result_path('extended_heldout_audit_results.csv')}")
    print(f"wrote summary to {result_path('extended_heldout_audit_summary.md')}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
