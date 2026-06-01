"""Leakage-cap refinement for the no-reference open-leakage frontier.

The high-fidelity continuation sweep shows that terminal-target pressure can
raise no-reference fidelity in the five-level leakage-plus-Lindblad benchmark,
but the best target-push rows pay a larger transient-leakage cost.  This audit
tests a narrower question: after a target-push pulse has been obtained without
a terminal GRAPE reference, can a short receding-horizon adjoint polish reduce
only the leakage excursions above a chosen cap while preserving terminal
fidelity?
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

from open_system_grape_baseline import commutator_super, liouvillian, vec
from paths import figure_path, result_path
from transmon_leakage_horizon import (
    design_pulse,
    path_projector,
    problem,
    random_disorder,
)
from transmon_open_leakage_adjoint_horizon import (
    OpenLeakageAdjointConfig,
    design_open_leakage_adjoint_horizon,
    load_result_rows,
    precompute_caches,
    step_liouville,
)
from transmon_open_system_leakage import evaluate_pulse


@dataclass(frozen=True)
class StageSpec:
    terminal_target_weight: float
    leakage_weight: float
    trust_radius: float
    horizon_maxiter: int
    worst_weight: float = 0.25


@dataclass(frozen=True)
class CapConfig:
    segments: int = 120
    horizon_steps: int = 5
    horizon_maxiter: int = 6
    train_strength: float = 0.03
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    eval_strength: float = 0.03
    eval_seeds: tuple[int, ...] = tuple(range(10, 60))
    umax: float = 0.12
    trust_radius: float = 0.012
    worst_weight: float = 0.25
    leakage_cap: float = 0.055
    cap_weight: float = 80.0
    mean_leakage_weight: float = 0.0
    energy_weight: float = 1e-5
    trust_weight: float = 1e-3
    terminal_target_weight: float = 1.0


@dataclass(frozen=True)
class CapVariant:
    controller: str
    label: str
    leakage_cap: float
    cap_weight: float
    trust_radius: float
    horizon_maxiter: int
    mean_leakage_weight: float = 0.0
    worst_weight: float = 0.25


LOW_LEAKAGE_SEED = StageSpec(
    terminal_target_weight=0.8,
    leakage_weight=1.5,
    trust_radius=0.04,
    horizon_maxiter=6,
)

TARGET_PUSH_STAGES = (
    StageSpec(1.0, 0.8, 0.02, 6),
    StageSpec(1.0, 0.5, 0.015, 6),
)

CAP_VARIANTS = (
    CapVariant(
        "cap055_w80",
        "cap 0.055, weight 80",
        leakage_cap=0.055,
        cap_weight=80.0,
        trust_radius=0.012,
        horizon_maxiter=6,
    ),
    CapVariant(
        "cap060_w60",
        "cap 0.060, weight 60",
        leakage_cap=0.060,
        cap_weight=60.0,
        trust_radius=0.012,
        horizon_maxiter=6,
    ),
    CapVariant(
        "cap050_w120",
        "cap 0.050, weight 120",
        leakage_cap=0.050,
        cap_weight=120.0,
        trust_radius=0.010,
        horizon_maxiter=6,
    ),
    CapVariant(
        "cap055_w80_mean02",
        "cap 0.055, weight 80, mean leak 0.2",
        leakage_cap=0.055,
        cap_weight=80.0,
        trust_radius=0.012,
        horizon_maxiter=6,
        mean_leakage_weight=0.2,
    ),
)


def adjoint_config_from_stage(
    base: OpenLeakageAdjointConfig,
    stage: StageSpec,
) -> OpenLeakageAdjointConfig:
    return OpenLeakageAdjointConfig(
        segments=base.segments,
        horizon_steps=base.horizon_steps,
        horizon_maxiter=stage.horizon_maxiter,
        train_strength=base.train_strength,
        train_seeds=base.train_seeds,
        eval_strength=base.eval_strength,
        eval_seeds=base.eval_seeds,
        umax=base.umax,
        trust_radius=stage.trust_radius,
        worst_weight=stage.worst_weight,
        leakage_weight=stage.leakage_weight,
        energy_weight=base.energy_weight,
        trust_weight=base.trust_weight,
        terminal_target_weight=stage.terminal_target_weight,
    )


def cap_config_from_variant(
    base: OpenLeakageAdjointConfig,
    variant: CapVariant,
) -> CapConfig:
    return CapConfig(
        segments=base.segments,
        horizon_steps=base.horizon_steps,
        horizon_maxiter=variant.horizon_maxiter,
        train_strength=base.train_strength,
        train_seeds=base.train_seeds,
        eval_strength=base.eval_strength,
        eval_seeds=base.eval_seeds,
        umax=base.umax,
        trust_radius=variant.trust_radius,
        worst_weight=variant.worst_weight,
        leakage_cap=variant.leakage_cap,
        cap_weight=variant.cap_weight,
        mean_leakage_weight=variant.mean_leakage_weight,
        energy_weight=base.energy_weight,
        trust_weight=base.trust_weight,
        terminal_target_weight=1.0,
    )


def score_observable(progress: float, config: CapConfig) -> np.ndarray:
    p = problem()
    path = path_projector(progress, p.h0.shape[0])
    weight = min(1.0, max(0.0, config.terminal_target_weight))
    return (1.0 - weight) * path + weight * p.target


def cap_objective_and_gradient(
    controls_flat: np.ndarray,
    current_states: tuple[np.ndarray, ...],
    controls_cache: tuple[tuple[np.ndarray, ...], ...],
    disorder_cache: tuple[tuple[np.ndarray, ...], ...],
    dephasing_cache: tuple[np.ndarray, ...],
    relaxation_cache: tuple[np.ndarray, ...],
    reference_flat: np.ndarray,
    start_index: int,
    dt: float,
    config: CapConfig,
) -> tuple[float, np.ndarray]:
    p = problem()
    n_controls = len(p.controls)
    horizon_steps = controls_flat.size // n_controls
    controls = np.reshape(controls_flat, (horizon_steps, n_controls))
    target = score_observable(
        min(config.segments, start_index + horizon_steps) / float(config.segments),
        config,
    )
    target_row = vec(target.T)
    leak_observable = np.eye(p.h0.shape[0], dtype=complex) - p.computational_projector
    leak_row = vec(leak_observable.T)
    scenario_costs: list[float] = []
    scenario_grads: list[np.ndarray] = []

    for scenario_index, state0 in enumerate(current_states):
        states = [state0]
        propagators: list[np.ndarray] = []
        generators: list[np.ndarray] = []
        frechet_dirs: list[list[np.ndarray]] = []

        for depth, coeffs in enumerate(controls):
            cache_index = min(start_index + depth, len(controls_cache) - 1)
            hamiltonian = config.train_strength * disorder_cache[cache_index][scenario_index]
            for coeff, hc_i in zip(coeffs, controls_cache[cache_index]):
                hamiltonian = hamiltonian + coeff * hc_i
            generator_dt = (
                liouvillian(
                    hamiltonian,
                    dephasing_cache[cache_index],
                    relaxation_cache[cache_index],
                    0.001,
                    0.0005,
                )
                * dt
            )
            propagator = expm(generator_dt)
            generators.append(generator_dt)
            propagators.append(propagator)
            frechet_dirs.append(
                [commutator_super(hc_i) * dt for hc_i in controls_cache[cache_index]]
            )
            states.append(propagator @ states[-1])

        target_score = float(np.real(target_row @ states[-1]))
        leak_scores = np.array(
            [float(np.real(leak_row @ state)) for state in states[1:]],
            dtype=float,
        )
        leak_excess = np.maximum(0.0, leak_scores - config.leakage_cap)
        cap_cost = config.cap_weight * float(np.mean(np.square(leak_excess)))
        mean_leak_cost = config.mean_leakage_weight * float(np.mean(leak_scores))
        scenario_cost = 1.0 - target_score + cap_cost + mean_leak_cost
        scenario_costs.append(scenario_cost)

        leak_state_weights = (
            config.mean_leakage_weight / float(horizon_steps)
            + (2.0 * config.cap_weight / float(horizon_steps)) * leak_excess
        )
        state_rows = [weight * leak_row for weight in leak_state_weights]
        costates: list[np.ndarray] = [
            np.zeros_like(target_row) for _ in range(horizon_steps + 1)
        ]
        costates[-1] = -target_row + state_rows[-1]
        for depth in reversed(range(1, horizon_steps)):
            costates[depth] = state_rows[depth - 1] + propagators[depth].T @ costates[depth + 1]

        grad = np.zeros((horizon_steps, n_controls), dtype=float)
        for depth in range(horizon_steps):
            for control_index in range(n_controls):
                d_propagator = expm_frechet(
                    generators[depth],
                    frechet_dirs[depth][control_index],
                    compute_expm=False,
                )
                value = costates[depth + 1] @ (d_propagator @ states[depth])
                grad[depth, control_index] = float(np.real(value))
        scenario_grads.append(grad.reshape(-1))

    costs = np.array(scenario_costs, dtype=float)
    grads = np.vstack(scenario_grads)
    worst_index = int(np.argmax(costs))
    trust_delta = controls_flat - reference_flat
    objective = float(
        np.mean(costs)
        + config.worst_weight * costs[worst_index]
        + config.energy_weight * np.mean(np.square(controls_flat))
        + config.trust_weight * np.mean(np.square(trust_delta))
    )
    gradient = (
        np.mean(grads, axis=0)
        + config.worst_weight * grads[worst_index]
        + config.energy_weight * (2.0 / controls_flat.size) * controls_flat
        + config.trust_weight * (2.0 / controls_flat.size) * trust_delta
    )
    return objective, gradient


def design_cap_refinement(
    reference_pulse: np.ndarray,
    config: CapConfig,
) -> tuple[np.ndarray, float, int, bool, float]:
    p = problem()
    t_eval = np.linspace(0.0, p.t_final, config.segments + 1)
    dt = float(t_eval[1] - t_eval[0])
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    states = tuple(vec(p.initial) for _ in disorders)
    caches = precompute_caches(disorders, config)  # type: ignore[arg-type]
    controls_cache, disorder_cache, dephasing_cache, relaxation_cache = caches
    pulse: list[np.ndarray] = []
    objectives: list[float] = []
    iterations: list[int] = []
    successes: list[bool] = []
    start = time.perf_counter()

    for index in range(config.segments):
        horizon = min(config.horizon_steps, config.segments - index)
        initial = reference_pulse[index : index + horizon].reshape(-1).copy()
        lower = np.maximum(-config.umax, initial - config.trust_radius)
        upper = np.minimum(config.umax, initial + config.trust_radius)
        result = minimize(
            lambda x: cap_objective_and_gradient(
                x,
                states,
                controls_cache,
                disorder_cache,
                dephasing_cache,
                relaxation_cache,
                initial,
                index,
                dt,
                config,
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(lower, upper)),
            options={"maxiter": config.horizon_maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        control = np.reshape(result.x, (horizon, len(p.controls)))[0]
        pulse.append(control)
        objectives.append(float(result.fun))
        iterations.append(int(result.nit))
        successes.append(bool(result.success))
        states = step_liouville(
            states,
            controls_cache[index],
            disorder_cache[index],
            dephasing_cache[index],
            relaxation_cache[index],
            dt,
            control,
            config,  # type: ignore[arg-type]
        )

    return (
        np.vstack(pulse),
        float(np.mean(objectives)),
        int(sum(iterations)),
        all(successes),
        time.perf_counter() - start,
    )


def stage_text(stages: tuple[StageSpec, ...]) -> str:
    return "; ".join(
        f"a={stage.terminal_target_weight:g},lw={stage.leakage_weight:g},"
        f"r={stage.trust_radius:g},w={stage.worst_weight:g}"
        for stage in stages
    )


def annotate_rows(
    rows: list[dict[str, float | int | str]],
    label: str,
    reference: str,
    cap_config: CapConfig | None,
) -> list[dict[str, float | int | str]]:
    annotated: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "cap_label": label,
                "reference_stages": reference,
                "leakage_cap": "" if cap_config is None else cap_config.leakage_cap,
                "cap_weight": "" if cap_config is None else cap_config.cap_weight,
                "mean_leakage_weight": ""
                if cap_config is None
                else cap_config.mean_leakage_weight,
                "trust_radius": "" if cap_config is None else cap_config.trust_radius,
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
                "label": str(first["cap_label"]),
                "noise_case": noise_case,
                "n": str(len(items)),
                "final_fidelity_mean": f"{np.mean(fids):.6g}",
                "final_fidelity_min": f"{np.min(fids):.6g}",
                "final_fidelity_ci95": f"{ci95:.6g}",
                "final_leakage_mean": f"{np.mean(final_leaks):.6g}",
                "max_leakage_mean": f"{np.mean(max_leaks):.6g}",
                "final_purity_mean": f"{np.mean(purities):.6g}",
                "pulse_energy_mean": f"{float(first['pulse_energy']):.6g}",
                "segments": str(first["segments"]),
                "training_seconds": f"{float(first['training_seconds']):.4g}",
                "leakage_cap": str(first["leakage_cap"]),
                "cap_weight": str(first["cap_weight"]),
                "mean_leakage_weight": str(first["mean_leakage_weight"]),
                "trust_radius": str(first["trust_radius"]),
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
        max_leaks = np.array([float(row["max_leakage"]) for row in items])
        points[controller] = (
            float(np.mean(fids)),
            float(np.min(fids)),
            float(np.mean(max_leaks)),
        )
    return points


def load_reference_points() -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, float | int | str]] = []
    rows.extend(load_result_rows("open_leakage_high_fidelity_sweep_results.csv"))
    rows.extend(
        row
        for row in load_result_rows("transmon_open_system_leakage_results.csv")
        if str(row.get("controller")) in {"adjoint_horizon", "leakage_penalized_grape"}
    )
    return combined_points(rows)


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    reference_points = load_reference_points()
    new_points = combined_points(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for controller, (mean_fid, worst_fid, leak) in reference_points.items():
        if controller.startswith("hf_") and controller not in {"hf_leak05", "hf_leak06"}:
            continue
        ax.scatter(leak, mean_fid, marker="o", s=24, color="0.68", alpha=0.55)
        ax.scatter(leak, worst_fid, marker="x", s=18, color="0.68", alpha=0.55)
    for controller, (mean_fid, worst_fid, leak) in new_points.items():
        if controller in {"cap_path_seed", "cap_low_leak_seed"}:
            continue
        color = "#1f77b4" if controller == "cap_target_push_reference" else "#2ca02c"
        marker = "s" if controller == "cap_target_push_reference" else "D"
        ax.scatter(leak, mean_fid, marker=marker, s=42, color=color)
        ax.annotate(
            controller.replace("cap_", ""),
            (leak, mean_fid),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
        ax.scatter(leak, worst_fid, marker="x", s=28, color=color)
    ax.set_xlabel("Mean maximum leakage")
    ax.set_ylabel("Held-out final fidelity")
    ax.set_title("Leakage-cap no-reference refinement")
    ax.set_xlim(0.0, 0.09)
    ax.set_ylim(0.50, 0.97)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path("open_leakage_cap_refinement.pdf"))
    fig.savefig(figure_path("open_leakage_cap_refinement.png"), dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    result_file = result_path("open_leakage_cap_refinement_results.csv")
    with result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_file = result_path("open_leakage_cap_refinement_summary.md")
    headers = list(summary[0].keys())
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Leakage-Cap Open-Leakage Refinement\n\n")
        f.write(
            "No-reference cap refinements initialized from the target-push "
            "continuation pulse and trained through the five-level Lindblad "
            "leakage model.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")

    plot_results(rows)
    print(f"wrote {len(rows)} rows to {result_file}")
    print(f"wrote summary to {summary_file}")
    print(f"wrote figure to {figure_path('open_leakage_cap_refinement.pdf')}")


def evaluate_standard(
    pulse: np.ndarray,
    controller: str,
    training_seconds: float,
    objective: float | None,
    iterations: int | None,
    success: bool | None,
    base_config: OpenLeakageAdjointConfig,
) -> list[dict[str, float | int | str]]:
    return evaluate_pulse(
        pulse,
        controller,
        training_seconds,
        objective,
        iterations,
        success,
        disorder_strength=base_config.eval_strength,
        test_seeds=range(min(base_config.eval_seeds), max(base_config.eval_seeds) + 1),
    )


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
    variants = CAP_VARIANTS
    if args.only:
        wanted = set(args.only.split(","))
        variants = tuple(variant for variant in variants if variant.controller in wanted)
        if not variants:
            raise ValueError(f"no cap variants matched --only={args.only!r}")

    path_kwargs: dict[str, object]
    if args.quick:
        path_kwargs = {
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
    rows: list[dict[str, float | int | str]] = []
    reference_text = stage_text((LOW_LEAKAGE_SEED,) + TARGET_PUSH_STAGES)
    rows.extend(
        annotate_rows(
            evaluate_standard(
                path_pulse,
                "cap_path_seed",
                path_seconds,
                None,
                None,
                None,
                base_config,
            ),
            "path seed",
            "",
            None,
        )
    )

    print("running low-leakage seed", flush=True)
    seed_config = adjoint_config_from_stage(base_config, LOW_LEAKAGE_SEED)
    pulse, objective, iterations, success, seconds = design_open_leakage_adjoint_horizon(
        path_pulse,
        seed_config,
    )
    total_seconds = path_seconds + seconds
    total_iterations = iterations
    rows.extend(
        annotate_rows(
            evaluate_standard(
                pulse,
                "cap_low_leak_seed",
                total_seconds,
                objective,
                total_iterations,
                success,
                base_config,
            ),
            "low-leakage seed",
            stage_text((LOW_LEAKAGE_SEED,)),
            None,
        )
    )

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
                "cap_target_push_reference",
                total_seconds,
                objective,
                total_iterations,
                success,
                base_config,
            ),
            "target-push reference",
            reference_text,
            None,
        )
    )

    for variant in variants:
        print(f"running {variant.controller}", flush=True)
        cap_config = cap_config_from_variant(base_config, variant)
        pulse, objective, iterations, success, seconds = design_cap_refinement(
            target_push_pulse,
            cap_config,
        )
        rows.extend(
            annotate_rows(
                evaluate_standard(
                    pulse,
                    variant.controller,
                    total_seconds + seconds,
                    objective,
                    total_iterations + iterations,
                    success,
                    base_config,
                ),
                variant.label,
                reference_text,
                cap_config,
            )
        )

    write_outputs(rows)


def gradient_check() -> None:
    config = CapConfig(
        segments=12,
        horizon_steps=3,
        horizon_maxiter=1,
        train_seeds=(0, 1),
        eval_seeds=tuple(range(10, 12)),
        leakage_cap=0.03,
        cap_weight=35.0,
        mean_leakage_weight=0.1,
    )
    p = problem()
    rng = np.random.default_rng(11)
    controls = rng.normal(scale=0.02, size=(config.horizon_steps, len(p.controls)))
    reference = controls + rng.normal(scale=0.002, size=controls.shape)
    disorders = tuple(random_disorder(seed, p.h0.shape[0]) for seed in config.train_seeds)
    states = tuple(vec(p.initial) for _ in disorders)
    caches = precompute_caches(disorders, config)  # type: ignore[arg-type]
    dt = p.t_final / float(config.segments)
    value, grad = cap_objective_and_gradient(
        controls.reshape(-1),
        states,
        *caches,
        reference.reshape(-1),
        2,
        dt,
        config,
    )
    direction = rng.normal(size=grad.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    plus = cap_objective_and_gradient(
        controls.reshape(-1) + eps * direction,
        states,
        *caches,
        reference.reshape(-1),
        2,
        dt,
        config,
    )[0]
    minus = cap_objective_and_gradient(
        controls.reshape(-1) - eps * direction,
        states,
        *caches,
        reference.reshape(-1),
        2,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--only", help="comma-separated cap controller names")
    parser.add_argument("--segments", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--horizon-maxiter", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_check:
        gradient_check()
        return
    run(args)


if __name__ == "__main__":
    main()
