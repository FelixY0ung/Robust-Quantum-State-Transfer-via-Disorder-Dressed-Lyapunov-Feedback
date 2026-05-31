"""Open-system-trained beam-horizon diagnostic.

The existing open-system script evaluates closed-system beam-horizon pulses under
Lindblad perturbations.  This script trains a compact finite-candidate horizon
controller directly through the same Lindblad dynamics and compares it with the
closed-trained pulse on held-out disorder seeds.  It is a diagnostic for whether
simple open-system horizon training is already sufficient, not a claim of
optimal dissipative control.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import (
    candidate_controls,
    dagger,
    default_beam_width,
    fidelity,
    interaction_frame_operator,
    ket,
    problem,
    random_disorder,
    scenario_cost,
)
from open_system_noise import NoiseCase, evolve_open_system, rk4_open_step
from paths import figure_path, result_path


TRAIN_NOISE = NoiseCase("trained_combined", gamma_phi=0.005, gamma_relax=0.002)
EVAL_NOISE_CASES = (
    NoiseCase("static_only", 0.0, 0.0),
    NoiseCase("deph_0.005", 0.005, 0.0),
    NoiseCase("relax_0.002", 0.0, 0.002),
    NoiseCase("combined", 0.005, 0.002),
)


def step_open_precomputed(
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_i: tuple[np.ndarray, ...],
    disorders_i: tuple[np.ndarray, ...],
    dephasing_i: np.ndarray,
    relaxation_i: np.ndarray,
    gamma_phi: float,
    gamma_relax: float,
    dt: float,
    control: np.ndarray,
) -> tuple[np.ndarray, ...]:
    next_rhos = []
    for rho, strength, disorder_i in zip(rhos, strengths, disorders_i):
        hamiltonian = strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            hamiltonian = hamiltonian + coeff * hc_i
        next_rhos.append(
            rk4_open_step(
                rho,
                dt,
                hamiltonian,
                dephasing_i,
                relaxation_i,
                gamma_phi,
                gamma_relax,
            )
        )
    return tuple(next_rhos)


def select_open_control_beam(
    task: str,
    rhos: tuple[np.ndarray, ...],
    strengths: tuple[float, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dephasing_cache: tuple[np.ndarray, ...],
    relaxation_cache: tuple[np.ndarray, ...],
    start_index: int,
    dt: float,
    candidates: tuple[np.ndarray, ...],
    horizon_steps: int,
    beam_width: int,
    worst_weight: float,
    energy_weight: float,
    gamma_phi: float,
    gamma_relax: float,
) -> np.ndarray:
    p = problem(task)
    beams: list[tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], float]] = [
        ((), rhos, 0.0)
    ]
    for depth in range(horizon_steps):
        cache_index = min(start_index + depth, len(controls_cache) - 1)
        expanded = []
        for sequence, states, energy_sum in beams:
            for candidate in candidates:
                next_states = step_open_precomputed(
                    states,
                    strengths,
                    controls_cache[cache_index],
                    disorder_cache[cache_index],
                    dephasing_cache[cache_index],
                    relaxation_cache[cache_index],
                    gamma_phi,
                    gamma_relax,
                    dt,
                    candidate,
                )
                next_energy = energy_sum + float(np.dot(candidate, candidate))
                cost = scenario_cost(
                    p,
                    next_states,
                    next_energy / float(depth + 1),
                    worst_weight,
                    energy_weight,
                )
                expanded.append(
                    (cost, sequence + (candidate,), next_states, next_energy)
                )
        expanded.sort(key=lambda item: item[0])
        beams = [
            (sequence, states, energy_sum)
            for _, sequence, states, energy_sum in expanded[:beam_width]
        ]
    return beams[0][0][0]


def design_open_system_pulse(
    task: str,
    disorder_strength: float = 0.08,
    train_seeds: tuple[int, ...] = (0, 1, 2),
    noise: NoiseCase = TRAIN_NOISE,
    segments: int = 60,
    horizon_steps: int = 2,
    beam_width: int | None = None,
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0),
    worst_weight: float = 0.25,
    energy_weight: float = 0.0,
) -> np.ndarray:
    p = problem(task)
    if beam_width is None:
        beam_width = min(default_beam_width(task), 3)
    t_eval = np.linspace(0.0, p.t_final, segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    disorders = tuple(random_disorder(seed) for seed in train_seeds)
    strengths = tuple(disorder_strength for _ in disorders)
    rhos = tuple(p.initial.copy() for _ in disorders)
    candidates = candidate_controls(amplitudes)
    cache_times = tuple(
        float(t_eval[min(j, segments)]) for j in range(segments + horizon_steps)
    )
    controls_cache = tuple(
        tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        for t in cache_times
    )
    disorder_cache = tuple(
        tuple(interaction_frame_operator(p, disorder, t) for disorder in disorders)
        for t in cache_times
    )
    dephasing_operator = p.h0
    relaxation_operator = ket(1) @ dagger(ket(0))
    dephasing_cache = tuple(
        interaction_frame_operator(p, dephasing_operator, t) for t in cache_times
    )
    relaxation_cache = tuple(
        interaction_frame_operator(p, relaxation_operator, t) for t in cache_times
    )

    pulse = []
    for j in range(segments):
        u = select_open_control_beam(
            task,
            rhos,
            strengths,
            controls_cache,
            disorder_cache,
            dephasing_cache,
            relaxation_cache,
            j,
            dt,
            candidates,
            horizon_steps,
            beam_width,
            worst_weight,
            energy_weight,
            noise.gamma_phi,
            noise.gamma_relax,
        )
        pulse.append(u)
        rhos = step_open_precomputed(
            rhos,
            strengths,
            controls_cache[j],
            disorder_cache[j],
            dephasing_cache[j],
            relaxation_cache[j],
            noise.gamma_phi,
            noise.gamma_relax,
            dt,
            u,
        )
    return np.asarray(pulse)


def evaluate_training_modes(
    tasks: tuple[str, ...] = ("Z", "H"),
    disorder_strength: float = 0.08,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    closed_rows_path = result_path("open_system_noise_results.csv")
    eval_case_labels = {noise.label for noise in EVAL_NOISE_CASES}
    if closed_rows_path.exists():
        with closed_rows_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["task"] not in tasks or row["noise_case"] not in eval_case_labels:
                    continue
                rows.append(
                    {
                        "task": row["task"],
                        "pulse_type": "closed_trained",
                        "train_noise_case": "closed_static",
                        "eval_noise_case": row["noise_case"],
                        "disorder_strength": float(row["disorder_strength"]),
                        "gamma_phi": float(row["gamma_phi"]),
                        "gamma_relax": float(row["gamma_relax"]),
                        "seed": int(row["seed"]),
                        "final_fidelity": float(row["final_fidelity"]),
                        "final_infidelity": float(row["final_infidelity"]),
                        "final_purity": float(row["final_purity"]),
                        "pulse_energy": float(row["pulse_energy"]),
                    }
                )
    else:
        raise FileNotFoundError(
            "Run code/open_system_noise.py first so the closed-trained baseline "
            "can be reused without retraining."
        )

    for task in tasks:
        print(f"designing open-system-trained pulse for {task}", flush=True)
        open_pulse = design_open_system_pulse(
            task, disorder_strength=disorder_strength
        )
        energy = float(np.mean(np.sum(open_pulse * open_pulse, axis=1)))
        for noise in EVAL_NOISE_CASES:
            for seed in test_seeds:
                final_fidelity, purity = evolve_open_system(
                    task, open_pulse, disorder_strength, seed, noise
                )
                rows.append(
                    {
                        "task": task,
                        "pulse_type": "open_trained",
                        "train_noise_case": TRAIN_NOISE.label,
                        "eval_noise_case": noise.label,
                        "disorder_strength": disorder_strength,
                        "gamma_phi": noise.gamma_phi,
                        "gamma_relax": noise.gamma_relax,
                        "seed": seed,
                        "final_fidelity": final_fidelity,
                        "final_infidelity": 1.0 - final_fidelity,
                        "final_purity": purity,
                        "pulse_energy": energy,
                    }
                    )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["task"]),
                str(row["pulse_type"]),
                str(row["eval_noise_case"]),
            ),
            [],
        ).append(row)
    summary: list[dict[str, str]] = []
    for (task, pulse_type, eval_noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        first = items[0]
        summary.append(
            {
                "task": task,
                "pulse_type": pulse_type,
                "train_noise_case": str(first["train_noise_case"]),
                "eval_noise_case": eval_noise_case,
                "gamma_phi": f"{float(first['gamma_phi']):.4g}",
                "gamma_relax": f"{float(first['gamma_relax']):.4g}",
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_std": f"{np.std(fids):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{np.mean([float(row['pulse_energy']) for row in items]):.6g}",
            }
        )
    return summary


def plot_summary(summary: list[dict[str, str]]) -> None:
    cases = tuple(dict.fromkeys(row["eval_noise_case"] for row in summary))
    x = np.arange(len(cases))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, task in zip(axes, ("Z", "H")):
        for offset, pulse_type, color in (
            (-0.5, "closed_trained", "#7a7a7a"),
            (0.5, "open_trained", "#d62728"),
        ):
            rows = [
                row
                for row in summary
                if row["task"] == task and row["pulse_type"] == pulse_type
            ]
            means = np.array([float(row["final_fidelity_mean"]) for row in rows])
            mins = np.array([float(row["final_fidelity_min"]) for row in rows])
            ax.bar(
                x + offset * width,
                means,
                width=width,
                color=color,
                label=pulse_type.replace("_", " "),
            )
            ax.scatter(
                x + offset * width,
                mins,
                s=16,
                marker="x",
                color="black",
                linewidths=0.8,
            )
        ax.set_title(f"{task} transfer")
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out final fidelity")
    axes[0].set_ylim(0.92, 1.001)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path("open_system_training.pdf"))
    fig.savefig(figure_path("open_system_training.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("open_system_training_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("open_system_training_summary.md").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("# Open-System-Trained Horizon Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    plot_summary(summary)


def main() -> None:
    rows = evaluate_training_modes()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('open_system_training_results.csv')}")
    print(f"wrote aggregate table to {result_path('open_system_training_summary.md')}")
    print(f"wrote figure to {figure_path('open_system_training.pdf')}")


if __name__ == "__main__":
    main()
