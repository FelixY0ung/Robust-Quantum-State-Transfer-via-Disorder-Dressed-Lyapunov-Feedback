"""Open-system noise evaluation for beam-horizon Lyapunov pulses.

The main robustness experiments use closed, static-disorder Hamiltonian
dynamics in the interaction frame.  This script adds a separate Lindblad
evaluation layer with dephasing and amplitude-damping-like relaxation using the
same interaction-frame discretization.  The controller is still designed by the
closed-system beam-horizon procedure, then tested with extra dissipative terms
on held-out disorder seeds.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from horizon_lyapunov import (
    dagger,
    default_beam_width,
    design_pulse,
    fidelity,
    hermitize_trace_one,
    interaction_frame_operator,
    ket,
    problem,
    random_disorder,
)
from paths import figure_path, result_path


@dataclass(frozen=True)
class NoiseCase:
    label: str
    gamma_phi: float
    gamma_relax: float


NOISE_CASES = (
    NoiseCase("static_only", 0.0, 0.0),
    NoiseCase("deph_0.002", 0.002, 0.0),
    NoiseCase("deph_0.005", 0.005, 0.0),
    NoiseCase("deph_0.010", 0.010, 0.0),
    NoiseCase("relax_0.002", 0.0, 0.002),
    NoiseCase("combined", 0.005, 0.002),
)


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def dissipator(operator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    odag_o = dagger(operator) @ operator
    return operator @ rho @ dagger(operator) - 0.5 * (odag_o @ rho + rho @ odag_o)


def derivative(
    rho: np.ndarray,
    hamiltonian: np.ndarray,
    dephasing_operator: np.ndarray,
    relaxation_operator: np.ndarray,
    gamma_phi: float,
    gamma_relax: float,
) -> np.ndarray:
    drho = -1.0j * comm(hamiltonian, rho)
    if gamma_phi:
        drho = drho + gamma_phi * dissipator(dephasing_operator, rho)
    if gamma_relax:
        drho = drho + gamma_relax * dissipator(relaxation_operator, rho)
    return drho


def rk4_open_step(
    rho: np.ndarray,
    dt: float,
    hamiltonian: np.ndarray,
    dephasing_operator: np.ndarray,
    relaxation_operator: np.ndarray,
    gamma_phi: float,
    gamma_relax: float,
) -> np.ndarray:
    k1 = derivative(
        rho, hamiltonian, dephasing_operator, relaxation_operator, gamma_phi, gamma_relax
    )
    k2 = derivative(
        rho + 0.5 * dt * k1,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        gamma_phi,
        gamma_relax,
    )
    k3 = derivative(
        rho + 0.5 * dt * k2,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        gamma_phi,
        gamma_relax,
    )
    k4 = derivative(
        rho + dt * k3,
        hamiltonian,
        dephasing_operator,
        relaxation_operator,
        gamma_phi,
        gamma_relax,
    )
    return hermitize_trace_one(rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def evolve_open_system(
    task: str,
    pulse: np.ndarray,
    disorder_strength: float,
    disorder_seed: int,
    noise: NoiseCase,
) -> tuple[float, float]:
    p = problem(task)
    disorder = random_disorder(disorder_seed)
    t_eval = np.linspace(0.0, p.t_final, len(pulse) + 1)
    dt = float(t_eval[1] - t_eval[0])
    rho = p.initial.copy()
    dephasing_operator = p.h0
    relaxation_operator = ket(1) @ dagger(ket(0))

    for index, control in enumerate(pulse):
        t = float(t_eval[index])
        controls_i = tuple(interaction_frame_operator(p, hc, t) for hc in p.controls)
        disorder_i = interaction_frame_operator(p, disorder, t)
        dephasing_i = interaction_frame_operator(p, dephasing_operator, t)
        relaxation_i = interaction_frame_operator(p, relaxation_operator, t)
        h = disorder_strength * disorder_i
        for coeff, hc_i in zip(control, controls_i):
            h = h + coeff * hc_i
        rho = rk4_open_step(
            rho,
            dt,
            h,
            dephasing_i,
            relaxation_i,
            noise.gamma_phi,
            noise.gamma_relax,
        )

    final_fidelity = fidelity(rho, p.target)
    purity = float(np.real(np.trace(rho @ rho)))
    return final_fidelity, purity


def run_open_system_eval(
    tasks: tuple[str, ...] = ("Z", "H"),
    disorder_strength: float = 0.08,
    test_seeds: range = range(10, 60),
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for task in tasks:
        pulse = design_pulse(task, beam_width=default_beam_width(task))
        energy = float(np.mean(np.sum(pulse * pulse, axis=1)))
        for noise in NOISE_CASES:
            for seed in test_seeds:
                final_fidelity, purity = evolve_open_system(
                    task, pulse, disorder_strength, seed, noise
                )
                rows.append(
                    {
                        "task": task,
                        "noise_case": noise.label,
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
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["noise_case"])), []).append(row)

    summary: list[dict[str, str]] = []
    for (task, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        first = items[0]
        summary.append(
            {
                "task": task,
                "noise_case": noise_case,
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


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    with result_path("open_system_noise_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    with result_path("open_system_noise_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Open-System Noise Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_summary(summary)


def plot_summary(summary: list[dict[str, str]]) -> None:
    tasks = tuple(dict.fromkeys(row["task"] for row in summary))
    cases = tuple(dict.fromkeys(row["noise_case"] for row in summary))
    x = np.arange(len(cases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for offset, task in zip((-0.5, 0.5), tasks):
        rows = [row for row in summary if row["task"] == task]
        means = np.array([float(row["final_fidelity_mean"]) for row in rows])
        mins = np.array([float(row["final_fidelity_min"]) for row in rows])
        ax.bar(x + offset * width, means, width=width, label=f"{task} mean")
        ax.scatter(
            x + offset * width,
            mins,
            s=18,
            marker="x",
            color="black",
            linewidths=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=25, ha="right")
    ax.set_ylim(0.90, 1.001)
    ax.set_ylabel("Held-out final fidelity")
    ax.set_xlabel("Open-system noise case")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("open_system_noise.pdf"))
    fig.savefig(figure_path("open_system_noise.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    rows = run_open_system_eval()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows to {result_path('open_system_noise_results.csv')}")
    print(f"wrote aggregate table to {result_path('open_system_noise_summary.md')}")
    print(f"wrote figure to {figure_path('open_system_noise.pdf')}")


if __name__ == "__main__":
    main()
