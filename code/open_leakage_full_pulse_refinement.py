"""Full-pulse no-reference refinement for the open leakage benchmark.

The existing no-reference leakage-plus-Lindblad frontier is built from local
receding-horizon adjoint refinements initialized by a finite-candidate
Lyapunov path pulse.  This script tests a stronger journal-level question:
after that no-reference target-push pulse is obtained, can a single full-pulse
adjoint refinement close more of the leakage-GRAPE gap while retaining leakage
caps and pulse regularity?

The refinement is still not a terminal GRAPE reference: it is initialized from
the no-reference Lyapunov-horizon target-push pulse and uses the same training
disorder seeds, five-level Lindblad model, leakage cap, worst-seed term, and
optional slew penalty.  Its role is to diagnose whether the remaining gap is a
short-horizon conservatism issue or a broader objective/initialization issue.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from open_leakage_cap_refinement import (
    LOW_LEAKAGE_SEED,
    TARGET_PUSH_STAGES,
    adjoint_config_from_stage,
    evaluate_standard,
    load_reference_points,
    score_observable,
    stage_text,
)
from open_leakage_smooth_cap_refinement import pulse_regularization, slew_cost_and_gradient
from open_system_grape_baseline import commutator_super, liouvillian, vec
from paths import figure_path, result_path
from transmon_leakage_horizon import design_pulse, problem, random_disorder
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
    precompute_caches,
)


@dataclass(frozen=True)
class FullPulseConfig:
    segments: int = 120
    maxiter: int = 80
    train_strength: float = 0.03
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    eval_strength: float = 0.03
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    umax: float = 0.12
    trust_radius: float = 0.015
    terminal_target_weight: float = 1.0
    tail_target_weight: float = 0.2
    tail_fraction: float = 0.2
    leakage_cap: float = 0.05
    cap_weight: float = 120.0
    mean_leakage_weight: float = 0.2
    worst_weight: float = 0.5
    energy_weight: float = 1e-5
    trust_weight: float = 1e-3
    slew_weight: float = 10.0


@dataclass(frozen=True)
class FullPulseVariant:
    controller: str
    label: str
    trust_radius: float
    leakage_cap: float
    cap_weight: float
    mean_leakage_weight: float
    worst_weight: float
    tail_target_weight: float
    slew_weight: float
    maxiter: int = 80


FULL_PULSE_VARIANTS = (
    FullPulseVariant(
        "fullpulse050_w120_tail02_worst05_slew10",
        "full pulse, cap 0.050, weight 120, tail 0.2, worst 0.5, slew 10",
        trust_radius=0.015,
        leakage_cap=0.050,
        cap_weight=120.0,
        mean_leakage_weight=0.2,
        worst_weight=0.5,
        tail_target_weight=0.2,
        slew_weight=10.0,
        maxiter=80,
    ),
    FullPulseVariant(
        "fullpulse055_w80_tail02_worst05_slew10",
        "full pulse, cap 0.055, weight 80, tail 0.2, worst 0.5, slew 10",
        trust_radius=0.020,
        leakage_cap=0.055,
        cap_weight=80.0,
        mean_leakage_weight=0.2,
        worst_weight=0.5,
        tail_target_weight=0.2,
        slew_weight=10.0,
        maxiter=80,
    ),
)


def full_config_from_variant(
    base: OpenLeakageAdjointConfig,
    variant: FullPulseVariant,
) -> FullPulseConfig:
    return FullPulseConfig(
        segments=base.segments,
        maxiter=variant.maxiter,
        train_strength=base.train_strength,
        train_seeds=base.train_seeds,
        eval_strength=base.eval_strength,
        eval_seeds=base.eval_seeds,
        umax=base.umax,
        trust_radius=variant.trust_radius,
        leakage_cap=variant.leakage_cap,
        cap_weight=variant.cap_weight,
        mean_leakage_weight=variant.mean_leakage_weight,
        worst_weight=variant.worst_weight,
        energy_weight=base.energy_weight,
        trust_weight=base.trust_weight,
        tail_target_weight=variant.tail_target_weight,
        slew_weight=variant.slew_weight,
    )


def cache_config(config: FullPulseConfig) -> OpenLeakageAdjointConfig:
    return OpenLeakageAdjointConfig(
        segments=config.segments,
        horizon_steps=1,
        train_strength=config.train_strength,
        train_seeds=config.train_seeds,
        eval_strength=config.eval_strength,
        eval_seeds=config.eval_seeds,
        umax=config.umax,
        energy_weight=config.energy_weight,
        trust_weight=config.trust_weight,
    )


def full_pulse_objective_and_gradient(
    controls_flat: np.ndarray,
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dephasing_cache: tuple[np.ndarray, ...],
    relaxation_cache: tuple[np.ndarray, ...],
    reference_flat: np.ndarray,
    dt: float,
    config: FullPulseConfig,
) -> tuple[float, np.ndarray]:
    p = problem()
    n_controls = len(p.controls)
    controls = np.reshape(controls_flat, (config.segments, n_controls))
    target_row = vec(p.target.T)
    leak_observable = np.eye(p.h0.shape[0], dtype=complex) - p.computational_projector
    leak_row = vec(leak_observable.T)
    tail_start = max(1, int(np.floor((1.0 - config.tail_fraction) * config.segments)))
    tail_indices = tuple(range(tail_start, config.segments + 1))

    scenario_costs: list[float] = []
    scenario_grads: list[np.ndarray] = []

    for scenario_index in range(len(config.train_seeds)):
        states = [vec(p.initial)]
        propagators: list[np.ndarray] = []
        generators: list[np.ndarray] = []
        frechet_dirs: list[list[np.ndarray]] = []

        for index, coeffs in enumerate(controls):
            hamiltonian = config.train_strength * disorder_cache[index][scenario_index]
            for coeff, hc_i in zip(coeffs, controls_cache[index]):
                hamiltonian = hamiltonian + coeff * hc_i
            generator_dt = (
                liouvillian(
                    hamiltonian,
                    dephasing_cache[index],
                    relaxation_cache[index],
                    0.001,
                    0.0005,
                )
                * dt
            )
            propagator = expm(generator_dt)
            propagators.append(propagator)
            generators.append(generator_dt)
            frechet_dirs.append([commutator_super(hc_i) * dt for hc_i in controls_cache[index]])
            states.append(propagator @ states[-1])

        target_scores = np.array(
            [float(np.real(target_row @ states[index])) for index in tail_indices],
            dtype=float,
        )
        terminal_infidelity = 1.0 - float(np.real(target_row @ states[-1]))
        tail_infidelity = float(np.mean(1.0 - target_scores))
        leak_scores = np.array(
            [float(np.real(leak_row @ state)) for state in states[1:]],
            dtype=float,
        )
        leak_excess = np.maximum(0.0, leak_scores - config.leakage_cap)
        cap_cost = config.cap_weight * float(np.mean(np.square(leak_excess)))
        mean_leak_cost = config.mean_leakage_weight * float(np.mean(leak_scores))
        scenario_costs.append(
            config.terminal_target_weight * terminal_infidelity
            + config.tail_target_weight * tail_infidelity
            + cap_cost
            + mean_leak_cost
        )

        state_rows = [
            np.zeros_like(target_row) for _ in range(config.segments + 1)
        ]
        state_rows[-1] = state_rows[-1] - config.terminal_target_weight * target_row
        if config.tail_target_weight:
            tail_scale = config.tail_target_weight / float(len(tail_indices))
            for index in tail_indices:
                state_rows[index] = state_rows[index] - tail_scale * target_row
        leak_state_weights = (
            config.mean_leakage_weight / float(config.segments)
            + (2.0 * config.cap_weight / float(config.segments)) * leak_excess
        )
        for index, weight in enumerate(leak_state_weights, start=1):
            state_rows[index] = state_rows[index] + weight * leak_row

        costates: list[np.ndarray] = [
            np.zeros_like(target_row) for _ in range(config.segments + 1)
        ]
        costates[-1] = state_rows[-1]
        for index in reversed(range(1, config.segments)):
            costates[index] = state_rows[index] + propagators[index].T @ costates[index + 1]

        grad = np.zeros((config.segments, n_controls), dtype=float)
        for index in range(config.segments):
            for control_index in range(n_controls):
                d_propagator = expm_frechet(
                    generators[index],
                    frechet_dirs[index][control_index],
                    compute_expm=False,
                )
                value = costates[index + 1] @ (d_propagator @ states[index])
                grad[index, control_index] = float(np.real(value))
        scenario_grads.append(grad.reshape(-1))

    costs = np.array(scenario_costs, dtype=float)
    grads = np.vstack(scenario_grads)
    worst_index = int(np.argmax(costs))
    trust_delta = controls_flat - reference_flat
    slew_cost, slew_grad = slew_cost_and_gradient(
        controls,
        np.zeros(n_controls, dtype=float),
        config.slew_weight,
    )
    objective = float(
        np.mean(costs)
        + config.worst_weight * costs[worst_index]
        + config.energy_weight * np.mean(np.square(controls_flat))
        + config.trust_weight * np.mean(np.square(trust_delta))
        + slew_cost
    )
    gradient = (
        np.mean(grads, axis=0)
        + config.worst_weight * grads[worst_index]
        + config.energy_weight * (2.0 / controls_flat.size) * controls_flat
        + config.trust_weight * (2.0 / controls_flat.size) * trust_delta
        + slew_grad.reshape(-1)
    )
    return objective, gradient


def design_full_pulse_refinement(
    reference_pulse: np.ndarray,
    config: FullPulseConfig,
) -> tuple[np.ndarray, float, int, bool, float]:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    caches = precompute_caches(disorders, cache_config(config))
    initial = reference_pulse.reshape(-1).copy()
    lower = np.maximum(-config.umax, initial - config.trust_radius)
    upper = np.minimum(config.umax, initial + config.trust_radius)
    start = time.perf_counter()
    result = minimize(
        lambda x: full_pulse_objective_and_gradient(
            x,
            *caches,
            initial,
            dt,
            config,
        ),
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper)),
        options={"maxiter": config.maxiter, "ftol": 1e-10, "gtol": 1e-7},
    )
    return (
        np.reshape(result.x, reference_pulse.shape),
        float(result.fun),
        int(result.nit),
        bool(result.success),
        time.perf_counter() - start,
    )


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    label: str,
    reference: str,
    config: FullPulseConfig | None,
    pulse: np.ndarray,
) -> list[dict[str, float | int | str]]:
    regularity = pulse_regularization(pulse)
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "full_pulse_label": label,
                "reference_stages": reference,
                "leakage_cap": "" if config is None else config.leakage_cap,
                "cap_weight": "" if config is None else config.cap_weight,
                "mean_leakage_weight": ""
                if config is None
                else config.mean_leakage_weight,
                "worst_weight": "" if config is None else config.worst_weight,
                "tail_target_weight": "" if config is None else config.tail_target_weight,
                "trust_radius": "" if config is None else config.trust_radius,
                "slew_weight": "" if config is None else config.slew_weight,
                **regularity,
            }
        )
        annotated.append(item)
    return annotated


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), str(row["noise_case"])), []).append(row)
    summaries: list[dict[str, str]] = []
    for (controller, noise_case), items in sorted(groups.items()):
        fids = np.array([float(row["final_fidelity"]) for row in items])
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        final_leaks = np.array([float(row["final_leakage"]) for row in items])
        purities = np.array([float(row["final_purity"]) for row in items])
        first = items[0]
        ci95 = 1.96 * float(np.std(fids, ddof=1)) / np.sqrt(len(fids))
        summaries.append(
            {
                "controller": controller,
                "label": str(first["full_pulse_label"]),
                "noise_case": noise_case,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{float(first['pulse_energy']):.6g}",
                "pulse_slew_rms": f"{float(first['pulse_slew_rms']):.6g}",
                "pulse_slew_max": f"{float(first['pulse_slew_max']):.6g}",
                "pulse_abs_max": f"{float(first['pulse_abs_max']):.6g}",
                "segments": str(first["segments"]),
                "training_seconds": f"{float(first['training_seconds']):.4g}",
                "leakage_cap": str(first["leakage_cap"]),
                "cap_weight": str(first["cap_weight"]),
                "mean_leakage_weight": str(first["mean_leakage_weight"]),
                "worst_weight": str(first["worst_weight"]),
                "tail_target_weight": str(first["tail_target_weight"]),
                "trust_radius": str(first["trust_radius"]),
                "slew_weight": str(first["slew_weight"]),
                "reference_stages": str(first["reference_stages"]),
            }
        )
    return summaries


def combined_points(rows: list[dict[str, float | int | str]]) -> dict[str, tuple[float, float, float]]:
    points: dict[str, tuple[float, float, float]] = {}
    for controller in sorted({str(row["controller"]) for row in rows}):
        items = [
            row
            for row in rows
            if str(row["controller"]) == controller
            and str(row["noise_case"]) == "combined"
        ]
        if not items:
            continue
        fids = np.array([float(row["final_fidelity"]) for row in items])
        leaks = np.array([float(row["max_leakage"]) for row in items])
        points[controller] = (
            float(np.mean(fids)),
            float(np.min(fids)),
            float(np.mean(leaks)),
        )
    return points


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    points = load_reference_points()
    points.update(combined_points(rows))
    labels = {
        "fullpulse_target_push_reference": "Target push",
        "fullpulse050_w120_tail02_worst05_slew10": "Full pulse 0.050",
        "fullpulse055_w80_tail02_worst05_slew10": "Full pulse 0.055",
        "adjoint_horizon": "Ref.-adjoint",
        "leakage_penalized_grape": "Leakage-GRAPE",
        "cap050_w120": "Cap",
        "cap050_w120_mean02_worst05": "Robust cap",
        "smoothcap050_w120_slew10": "Smooth cap",
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for controller, (mean_fid, worst_fid, leak) in points.items():
        is_new = controller.startswith("fullpulse")
        if not is_new and controller not in {
            "adjoint_horizon",
            "leakage_penalized_grape",
            "cap050_w120",
            "cap050_w120_mean02_worst05",
            "smoothcap050_w120_slew10",
        }:
            continue
        color = "#9467bd" if is_new else "0.65"
        marker = "D" if is_new else "o"
        alpha = 0.95 if is_new else 0.60
        ax.scatter(leak, mean_fid, marker=marker, s=44, color=color, alpha=alpha)
        ax.scatter(leak, worst_fid, marker="x", s=28, color=color, alpha=alpha)
        ax.annotate(
            labels.get(controller, controller),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Full-pulse open-leakage refinement")
    ax.set_xlim(0.045, 0.095)
    ax.set_ylim(0.74, 0.96)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_full_pulse_refinement.pdf"))
    fig.savefig(figure_path("open_leakage_full_pulse_refinement.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_full_pulse_refinement_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_full_pulse_refinement_summary.md")
    headers = list(summary[0].keys())
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Full-Pulse Open-Leakage Refinement\n\n")
        f.write(
            "No-reference full-pulse adjoint refinements initialized from the "
            "target-push continuation pulse and trained through the five-level "
            "Lindblad leakage model with terminal/tail target pressure, leakage "
            "caps, worst-seed weighting, and slew regularization.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_full_pulse_refinement.pdf')}")


def config_from_args(args: argparse.Namespace) -> OpenLeakageAdjointConfig:
    if args.quick:
        return OpenLeakageAdjointConfig(
            segments=30,
            horizon_steps=3,
            horizon_maxiter=2,
            train_seeds=(0, 1),
            eval_seeds=tuple(range(10, 15)),
        )
    return OpenLeakageAdjointConfig(
        segments=args.segments,
        horizon_steps=args.horizon_steps,
        horizon_maxiter=args.horizon_maxiter,
    )


def run(args: argparse.Namespace) -> None:
    base_config = config_from_args(args)
    variants = FULL_PULSE_VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no full-pulse variants matched --only={args.only!r}")

    if args.quick:
        path_kwargs: dict[str, object] = {
            "train_strengths": (base_config.train_strength,),
            "train_seeds": base_config.train_seeds,
            "segments": base_config.segments,
            "horizon_steps": base_config.horizon_steps,
            "beam_width": 3,
            "amplitudes": (0.015, 0.035, 0.06),
            "leakage_weight": 0.5,
        }
    else:
        path_kwargs = {"segments": base_config.segments}

    print("designing finite-candidate path horizon seed", flush=True)
    start = time.perf_counter()
    path_pulse = design_pulse(**path_kwargs)
    path_seconds = time.perf_counter() - start
    reference_text = stage_text((LOW_LEAKAGE_SEED,) + TARGET_PUSH_STAGES)
    rows: list[dict[str, float | int | str]] = []

    print("running low-leakage seed", flush=True)
    seed_config = adjoint_config_from_stage(base_config, LOW_LEAKAGE_SEED)
    pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
        path_pulse,
        seed_config,
    )
    total_seconds = path_seconds + seconds
    total_iterations = iterations

    for stage in TARGET_PUSH_STAGES:
        print(
            "running target-push stage "
            f"target={stage.terminal_target_weight:g} leak={stage.leakage_weight:g}",
            flush=True,
        )
        stage_config = adjoint_config_from_stage(base_config, stage)
        pulse, objective, stage_iterations, stage_success, seconds = (
            design_open_leakage_adjoint_horizon(pulse, stage_config)
        )
        total_seconds += seconds
        total_iterations += stage_iterations
        success = success and stage_success

    target_push_pulse = pulse
    rows.extend(
        annotate_rows(
            evaluate_standard(
                target_push_pulse,
                "fullpulse_target_push_reference",
                total_seconds,
                objective,
                total_iterations,
                success,
                base_config,
            ),
            "target-push reference",
            reference_text,
            None,
            target_push_pulse,
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        config = full_config_from_variant(base_config, variant)
        refined_pulse, refined_objective, refined_iterations, refined_success, seconds = (
            design_full_pulse_refinement(target_push_pulse, config)
        )
        rows.extend(
            annotate_rows(
                evaluate_standard(
                    refined_pulse,
                    variant.controller,
                    total_seconds + seconds,
                    refined_objective,
                    total_iterations + refined_iterations,
                    refined_success,
                    base_config,
                ),
                variant.label,
                reference_text,
                config,
                refined_pulse,
            )
        )

    write_outputs(rows)


def gradient_check() -> None:
    config = FullPulseConfig(
        segments=8,
        maxiter=1,
        train_seeds=(0, 1),
        eval_seeds=tuple(range(10, 12)),
        leakage_cap=0.03,
        cap_weight=35.0,
        mean_leakage_weight=0.1,
        tail_target_weight=0.3,
        trust_radius=0.02,
        slew_weight=3.0,
    )
    p = problem()
    rng = np.random.default_rng(23)
    controls = rng.normal(scale=0.02, size=(config.segments, len(p.controls)))
    reference = controls + rng.normal(scale=0.002, size=controls.shape)
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    caches = precompute_caches(disorders, cache_config(config))
    dt = p.t_final / float(config.segments)
    value, grad = full_pulse_objective_and_gradient(
        controls.reshape(-1),
        *caches,
        reference.reshape(-1),
        dt,
        config,
    )
    direction = rng.normal(size=grad.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = full_pulse_objective_and_gradient(
        controls.reshape(-1) + eps * direction,
        *caches,
        reference.reshape(-1),
        dt,
        config,
    )[0]
    minus = full_pulse_objective_and_gradient(
        controls.reshape(-1) - eps * direction,
        *caches,
        reference.reshape(-1),
        dt,
        config,
    )[0]
    finite_difference = (plus - minus) / (2.0 * eps)
    analytic = float(np.dot(grad, direction))
    print(
        "value={:.8g} analytic={:.8g} finite_difference={:.8g} error={:.3g}".format(
            value,
            analytic,
            finite_difference,
            abs(analytic - finite_difference),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=6)
    parser.add_argument("--only", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gradient-check", action="store_true")
    args = parser.parse_args()
    if args.gradient_check:
        gradient_check()
    else:
        run(args)


if __name__ == "__main__":
    main()
