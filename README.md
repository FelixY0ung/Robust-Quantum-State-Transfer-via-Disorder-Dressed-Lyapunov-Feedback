# Robust Quantum State Transfer via Disorder-Dressed Lyapunov Feedback

Reproducible code and generated numerical results for the disorder-dressed Lyapunov feedback study are open source in this repository.

## Repository Layout

- `code/`: simulation, control, robustness-scan, and plotting scripts.
- `results/`: CSV outputs, Markdown summaries, and generated paper figures.
- `docs/`: reproducibility and physical-consistency audit notes.
- `requirements.txt`: Python dependencies used by the core scripts.
- `requirements-krotov.txt`: optional dependencies for the Krotov-package comparator.

## Environment

The scripts are plain Python and were run with Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The optional Krotov-package comparator uses the external `krotov`/QuTiP 4 stack, which is best isolated in a Python 3.9/3.10 environment:

```bash
python3 -m venv .venv-krotov
source .venv-krotov/bin/activate
pip install -r requirements-krotov.txt
```

## Reproducing The Reported Results

Run commands from the repository root. The scripts use fixed random seeds and write outputs into `results/`.

```bash
python3 code/stationarity_analysis.py
python3 code/reproducible_simulation.py --full
python3 code/tracking_simulation.py
python3 code/robustness_scan.py
python3 code/ensemble_openloop.py
python3 code/ensemble_lyapunov.py
python3 code/crab_baseline.py
python3 code/dcrab_baseline.py
python3 code/dcrab_baseline.py --training-seed-count 8 --output-prefix dcrab_train8_baseline --baseline-label dcrab_train8
python3 code/dcrab_baseline.py --basis-count 4 --refreshes 5 --maxiter 24 --popsize 5 --output-prefix dcrab_large_budget --baseline-label dcrab_large
python3 code/horizon_lyapunov.py
python3 code/strong_disorder_audit.py
python3 code/slew_constrained_horizon.py
python3 code/bandwidth_filter_audit.py
python3 code/shifted_fallback_horizon.py
python3 code/shifted_fallback_margin_audit.py
python3 code/state_adjoint_horizon.py
python3 code/horizon_ablation.py
python3 code/open_system_noise.py
python3 code/open_system_horizon_training.py
python3 code/open_system_grape_baseline.py
python3 code/open_system_adjoint_horizon.py
python3 code/open_system_standalone_adjoint_horizon.py
python3 code/gate_fidelity_probe.py
python3 code/gate_process_baseline.py
python3 code/ensemble_grape_baseline.py
python3 code/krotov_baseline.py
python3 code/process_horizon.py
python3 code/process_standalone_adjoint_horizon.py
python3 code/process_seeded_horizon.py
python3 code/process_adjoint_horizon.py
python3 code/statistical_audit.py
python3 code/resource_audit.py
python3 code/multilevel_horizon.py
python3 code/transmon_leakage_horizon.py
python3 code/transmon_standalone_adjoint_horizon.py
python3 code/transmon_open_system_leakage.py
python3 code/transmon_open_leakage_adjoint_horizon.py
python3 code/transmon_open_leakage_adjoint_horizon.py --plot-only
python3 code/open_leakage_integrated_sweep.py --extended
python3 code/open_leakage_pareto_refinement.py
python3 code/open_leakage_continuation_sweep.py
python3 code/open_leakage_training_ensemble_audit.py
python3 code/open_leakage_long_horizon_sweep.py
python3 code/open_leakage_pareto_audit.py
python3 code/polished_openloop.py
python3 code/plot_experiments.py
python3 code/reproducibility_manifest.py
```

The leakage and open-system GRAPE benchmarks also have quick derivative checks:

```bash
python3 code/transmon_leakage_horizon.py --gradient-check
python3 code/open_system_grape_baseline.py --gradient-check
python3 code/open_system_adjoint_horizon.py --gradient-check
python3 code/state_adjoint_horizon.py --gradient-check
python3 code/process_standalone_adjoint_horizon.py --gradient-check
python3 code/process_adjoint_horizon.py --gradient-check
python3 code/transmon_open_leakage_adjoint_horizon.py --gradient-check
```

The longer scripts are the open-loop optimization, Krotov-package comparator, CRAB and dCRAB reduced-basis baselines, strong-disorder extrapolation audit, terminal process-fidelity baseline, ensemble-GRAPE baseline, process-horizon diagnostics, slew-constrained horizon audit, bandwidth filter audit, five-level leakage benchmark, five-level leakage-plus-Lindblad stress test, direct, target-biased, two-stage, integrated, Pareto-refined, continuation, seeded-continuation, training-ensemble, and long-horizon open-leakage adjoint diagnostics, open-leakage Pareto audit, open-system horizon-training diagnostic, open-system GRAPE baseline, adjoint open-system horizon diagnostic, shifted-fallback horizon comparison and margin audit, standalone state-adjoint horizon diagnostic, standalone process-adjoint horizon diagnostic, and horizon-search experiments. The checked outputs currently included in `results/` use held-out disorder seeds `10..59` for the key two-level, strong-disorder extrapolation, three-level, five-level leakage, five-level leakage-plus-Lindblad, direct/target-biased/two-stage/integrated/Pareto-refined/continuation/seeded-continuation/long-horizon open-leakage adjoint, open-leakage Pareto, Krotov-package, CRAB/dCRAB, slew-constrained, bandwidth-filter, horizon-ablation, shifted-fallback, standalone state-adjoint, standalone process-adjoint, open-system stress-test, open-system training diagnostic, open-system GRAPE, adjoint open-system horizon, gate-fidelity diagnostic, terminal process-baseline, ensemble-GRAPE, process-horizon, process-seeded horizon, process-adjoint horizon, and statistical-audit results; the training-ensemble open-leakage audit uses a fresh disjoint held-out range `60..109`.

The five-level weakly anharmonic leakage benchmark compares path-horizon control, gradient-seeded horizon control, adjoint-polished horizon control, terminal GRAPE, and leakage-penalized GRAPE. At `delta = 0.03`, gradient-seeded horizon control reaches mean final fidelity `0.916971` with mean maximum leakage `0.037966`; adjoint-polished horizon control reaches mean final fidelity `0.948305` with mean maximum leakage `0.053031`; leakage-penalized GRAPE reaches mean final fidelity `0.969447` while reducing mean maximum leakage to `0.053939`.

The direct leakage-adjoint horizon diagnostic uses the finite path-horizon pulse as the local initializer and does not use a GRAPE reference. At `delta = 0.03`, the conservative path-score variant raises mean final fidelity from the path seed's `0.837761` to `0.856543` in the static five-level model while reducing mean maximum leakage from `0.046844` to `0.017790`. With the default terminal target weight `--terminal-target-weight 0.8`, the target-biased direct variant reaches static mean fidelity `0.932303` with mean maximum leakage `0.042106`. These are GRAPE-free local horizon diagnostics, not terminal-optimization replacements.

The five-level leakage-plus-Lindblad stress test evaluates the path horizon, conservative direct open-leakage adjoint, target-biased direct open-leakage adjoint, integrated single-stage direct sweep, two-stage direct target/leakage repolish, continuation direct sweep, reference-assisted adjoint horizon, and leakage-penalized GRAPE pulse under weak dephasing and relaxation with `gamma_phi = 0.001` and `gamma_relax = 0.0005`. At `delta = 0.03` under combined noise, the path seed reaches mean final fidelity `0.826420`, the conservative direct adjoint reaches `0.840595` with mean maximum leakage `0.017652`, the integrated direct sweep point reaches `0.897659` with mean maximum leakage `0.040141`, and the target-biased direct adjoint reaches `0.910936` with mean maximum leakage `0.042021`. Increasing the worst-seed weight in the integrated sweep raises the best single-stage worst held-out fidelity to `0.774726`, but that robust-weighted point remains dominated in the mean-fidelity/mean-leakage plane. The two-stage direct variant reaches mean final fidelity `0.919179` and worst held-out fidelity `0.784895`, but raises mean maximum leakage to `0.065347`. The leakage-controlled continuation variant reaches mean final fidelity `0.920133`, worst held-out fidelity `0.766810`, and mean maximum leakage `0.053478`, dominating the two-stage row in the mean-fidelity/mean-leakage plane. A fidelity-favoring continuation variant reaches mean final fidelity `0.929533` and worst held-out fidelity `0.812158`, but raises mean maximum leakage to `0.071371`; two third-stage leakage-repolish variants lower the combined-noise mean to `0.915414` or `0.910847`, so they do not improve the frontier. The reference-assisted adjoint horizon reaches `0.932452`, while leakage-penalized GRAPE reaches `0.952952`. This is a stress test and direct-horizon diagnostic; the highest-fidelity rows still show the remaining terminal-optimization ceiling.

The open-leakage Pareto audit reuses the same combined-noise rows and reports the mean final fidelity versus mean maximum leakage tradeoff. The direct adjoint is the lowest-leakage Pareto point, the Pareto-refinement sweep fills the low-to-mid leakage no-reference front, the integrated direct sweep adds a higher-fidelity no-reference Pareto point, the target-biased direct row raises the GRAPE-free front, and the continuation sweep separates leakage-controlled, balanced, and fidelity-favoring no-reference points. The leakage-controlled continuation remains on the global Pareto front; the balanced and fidelity-favoring continuation rows improve GRAPE-free fidelity but are dominated in mean fidelity/leakage by the reference-assisted horizon and leakage-GRAPE, and the audited third-stage leakage repolish variants do not improve this frontier. The generated files are `results/open_leakage_integrated_sweep_results.csv`, `results/open_leakage_integrated_sweep_summary.md`, `results/figures/open_leakage_integrated_sweep.*`, `results/open_leakage_pareto_refinement_results.csv`, `results/open_leakage_pareto_refinement_summary.md`, `results/figures/open_leakage_pareto_refinement.*`, `results/open_leakage_continuation_sweep_results.csv`, `results/open_leakage_continuation_sweep_summary.md`, `results/figures/open_leakage_continuation_sweep.*`, `results/open_leakage_pareto_audit_results.csv`, `results/open_leakage_pareto_audit_summary.md`, and `results/figures/open_leakage_pareto_audit.*`.

The seeded-continuation audit tests whether starting from a stronger no-reference target/leakage seed closes the open-leakage gap. The target-1.0/leakage-1.5 seed reaches combined-noise mean fidelity `0.897659` with mean maximum leakage `0.040141`; after continuation, the best seeded row reaches `0.927850` mean fidelity, `0.805900` worst held-out fidelity, and `0.070628` mean maximum leakage. This does not improve the existing low-leak-seeded high-fidelity row (`0.929533`, `0.812158`, `0.071371`) or the cleaner leakage-controlled row (`0.920133`, `0.766810`, `0.053478`), so it is recorded as a negative frontier audit. The generated files are `results/open_leakage_seed_continuation_sweep_results.csv`, `results/open_leakage_seed_continuation_sweep_summary.md`, and `results/figures/open_leakage_seed_continuation_sweep.*`.

The training-ensemble audit tests whether the hardest open-leakage gap is mainly a small-sample training artifact. It reruns the same low-leakage seed plus fidelity-favoring continuation recipe with the original four training seeds, eight training seeds, and a hard-sample-enriched set `(0,1,2,3,13,37)`, then evaluates all rows on fresh seeds `60..109`. Under combined noise, the four-seed row reaches mean/worst/max-leak `0.930098/0.818198/0.072162`; the eight-seed row is essentially unchanged in mean but slightly improves the worst seed to `0.929951/0.823891/0.073161`; the hard-sample-enriched row is worse on the fresh held-out range at `0.926738/0.795235/0.073488`. The generated files are `results/open_leakage_training_ensemble_audit_results.csv`, `results/open_leakage_training_ensemble_audit_summary.md`, and `results/figures/open_leakage_training_ensemble_audit.*`.

The long-horizon audit tests whether the hardest open-leakage gap is mainly caused by the five-step local lookahead in the continuation frontier. It reruns no-reference direct Lindblad leakage-aware continuation with `q = 8` and `q = 10` horizons and wider trust regions. Under combined noise, the best tested long-horizon row is `long_q8_leak08` with mean/worst/max-leak `0.920249/0.740572/0.079176`, below the existing fidelity-favoring continuation row `0.929533/0.812158/0.071371`; the `q = 10` row drops to `0.911872/0.674742/0.097828`. The generated files are `results/open_leakage_long_horizon_sweep_results.csv`, `results/open_leakage_long_horizon_sweep_summary.md`, and `results/figures/open_leakage_long_horizon_sweep.*`.

The reduced-basis CRAB baseline is an independent derivative-free terminal-control comparator. With three randomized Fourier modes per control and differential-evolution training at `delta = 0.08`, it reaches held-out mean fidelity `0.999240` for Z and `0.997811` for H at `delta = 0.08`.

The optional Krotov-package ensemble comparator uses the same two-level Hamiltonian resources, a drift-rotated terminal target equivalent to the interaction-frame state-transfer objective, 40 control segments, 60 Krotov iterations, and eight training disorder seeds at `delta = 0.08`. On the standard 50 held-out seeds, it reaches mean fidelity `0.958963` for Z and `0.987255` for H at `delta = 0.08`, with worst held-out fidelities `0.945603/0.978499`. This is a direct prior-family comparator, not the strongest terminal ceiling.

The dCRAB-style reduced-basis baseline refreshes the randomized Fourier basis over three sequential terminal-optimization rounds and accepts only corrections that improve the robust training score. At `delta = 0.08`, it reaches held-out mean fidelity `0.999635` for Z and `0.999578` for H, with worst held-out fidelities above `0.9983`.

The train-8 dCRAB comparator keeps the same three Fourier modes and three basis refreshes but doubles the training disorder seeds. At `delta = 0.08`, it reaches held-out mean fidelity `0.999713` for Z and `0.999789` for H, with worst held-out fidelities `0.998775/0.999393`. A separate four-mode, five-refresh diagnostic is also included; it generalizes worse on the held-out split (`0.998664/0.992961` for Z/H at `delta = 0.08`) and is kept as an over-parameterization diagnostic rather than the main terminal ceiling.

The strong-disorder extrapolation audit reruns the existing beam-horizon protocol trained on `delta = 0.05/0.08` and the train-8 dCRAB protocol trained at `delta = 0.08`, then evaluates the resulting pulses at `delta = 0.08`, `0.10`, and `0.12` without retuning. At `delta = 0.12`, beam-horizon control retains held-out mean fidelity `0.995700/0.996804` for Z/H, while train-8 dCRAB reaches `0.998811/0.999259`. Paired on the same held-out seeds, the train-8 dCRAB advantage at `delta = 0.12` is `0.00311051 ± 0.00091051` for Z and `0.00245446 ± 0.00081157` for H.

The process-GRAPE-seeded horizon diagnostic uses a 60-segment process-GRAPE reference and a four-step local receding horizon. At `delta = 0.08`, it preserves held-out average gate fidelity `0.994732` for Z and `0.999573` for H. This is a gradient-reference diagnostic, not an independent replacement for process GRAPE.

The adjoint-polished process-horizon diagnostic puts exact Frechet gradients inside the same four-step receding-horizon process score while staying in a trust region around the 60-segment process-GRAPE reference. At `delta = 0.08`, it preserves held-out average gate fidelity `0.994641` for Z and `0.999717` for H. This is a reference-assisted horizon diagnostic, not a standalone process-GRAPE replacement.

The shifted-fallback horizon comparison implements the candidate-inclusion part of the practical-decrease certificate: at each step it also scores the shifted tail of the previous horizon sequence with every admissible appended fallback. At `delta = 0.08`, it reaches held-out mean fidelity `0.995809` for Z and `0.997657` for H, while lowering mean pulse energy to `4.625/5.285`. This is a certification-oriented comparison, not the best-fidelity beam-horizon row.

The shifted-fallback margin audit recomputes the training trajectory and checks the terminal-progress part of the practical-decrease certificate. Candidate inclusion is enforced at every post-initial step, while a positive shifted-fallback margin is available on `16.2%` of audited Z steps and `14.1%` of audited H steps. This records the certificate as a sufficient-condition diagnostic, not as an unconditional convergence proof.

The standalone state-adjoint horizon diagnostic uses the finite-candidate horizon only as a local initializer, then optimizes the same short state-transfer Lyapunov score by exact Frechet derivatives without a terminal GRAPE reference. At `delta = 0.08`, it reaches held-out mean fidelity `0.997445` for Z and `0.996096` for H; the corresponding 60-segment finite-candidate seed reaches `0.997407/0.989428`. This is evidence for adjoint-assisted horizon scoring, not a global optimality claim.

The standalone process-adjoint horizon diagnostic applies the same idea to average gate fidelity without a terminal process-GRAPE reference. At `delta = 0.08`, it reaches held-out average gate fidelity `0.924093` for Z and `0.927613` for H, close to the finite process seed's `0.927537/0.928319`. This negative result is included to document that exact short-horizon process gradients alone do not solve the global process-search difficulty.

The open-system horizon-training diagnostic compares the existing closed-system-trained horizon pulses with compact Lindblad-trained finite-candidate horizon pulses. Under combined dephasing and relaxation at `delta = 0.08`, the compact open-trained pulses reach mean fidelity `0.948556` for Z and `0.942853` for H, below the closed-trained horizon's `0.957140` and `0.956625`, but with lower pulse energy. This diagnostic is included as a limitation rather than an improvement claim.

The open-system GRAPE baseline is a terminal open-loop comparator trained through the combined Lindblad model, not a Lyapunov feedback law. Under the same combined-noise held-out test, it reaches mean fidelity `0.977461` for Z and `0.978929` for H, showing that the remaining open-system gap is mainly an optimizer/method gap for this two-level state-transfer task.

The adjoint open-system horizon diagnostic carries most of the open-system GRAPE performance into a reference-assisted receding-horizon form with exact Frechet derivatives. Under the combined-noise held-out test at `delta = 0.08`, it reaches mean fidelity `0.974684` for Z and `0.977779` for H. It remains reference-assisted and should not be read as an independent open-system Lyapunov proof.

The standalone open-system adjoint horizon diagnostic uses the compact finite-candidate Lindblad horizon pulse only as a trust-region initializer, sets the reference-tracking weight to zero, and optimizes the direct target score through the Lindblad model. Under combined noise at `delta = 0.08`, it reaches mean fidelity `0.948934` for Z and `0.943610` for H, only slightly above the finite-candidate seed. This records a GRAPE-free dissipative adjoint diagnostic and a remaining optimizer gap.

The slew-constrained horizon audit adds a quadratic step-to-step control penalty to the compact two-level beam search. With `nu = 0.005` at `delta = 0.08`, it reaches held-out mean fidelity `0.995054` for Z and `0.995260` for H while reducing pulse energy to `0.8083/0.4542` and high-frequency fraction to `0.0409/0.0632`. This is a smoothness diagnostic rather than a calibrated hardware bandwidth model.

The bandwidth filter audit post-filters the compact beam pulses with deterministic low-pass FIR filters. The unpenalized pulse is filter-sensitive: `boxcar3` lowers Z/H mean fidelity to `0.960064/0.958034`, with paired losses of `-0.0304 ± 0.0036` and `-0.0385 ± 0.0046`. The `nu = 0.005` pulse is more tolerant: under the same `boxcar3` filter it retains `0.994356/0.995036` mean fidelity with energy `0.7052/0.3982`, and paired losses shrink to `-0.00070 ± 0.00075` and `-0.00022 ± 0.00115`. Stronger filters are included to show the remaining boundary.

The resource audit aggregates representative held-out performance, pulse energy, segment counts, training sample counts, horizon parameters, and logged design seconds where available. It is included to make the journal comparisons transparent; it is not a universal runtime benchmark because the earliest finite-candidate runs did not write wall-clock timing fields.

The reproducibility manifest records SHA-256 hashes, file sizes, CSV row counts, and CSV columns for the checked code and result artifacts without rerunning simulations. Regenerate it with `python3 code/reproducibility_manifest.py`; the outputs are `results/reproducibility_manifest.json` and `results/reproducibility_manifest.md`.

## Main Outputs

- `results/horizon_lyapunov_summary.md`
- `results/slew_constrained_horizon_summary.md`
- `results/bandwidth_filter_audit_summary.md`
- `results/crab_baseline_summary.md`
- `results/dcrab_baseline_summary.md`
- `results/dcrab_train8_baseline_summary.md`
- `results/dcrab_large_budget_summary.md`
- `results/strong_disorder_audit_summary.md`
- `results/shifted_fallback_horizon_summary.md`
- `results/shifted_fallback_margin_audit_summary.md`
- `results/state_adjoint_horizon_summary.md`
- `results/horizon_ablation_summary.md`
- `results/open_system_noise_summary.md`
- `results/open_system_training_summary.md`
- `results/open_system_grape_summary.md`
- `results/open_system_adjoint_horizon_summary.md`
- `results/open_system_standalone_adjoint_summary.md`
- `results/gate_fidelity_probe_summary.md`
- `results/gate_process_baseline_summary.md`
- `results/ensemble_grape_baseline_summary.md`
- `results/krotov_baseline_summary.md`
- `results/process_horizon_summary.md`
- `results/process_standalone_adjoint_summary.md`
- `results/process_seeded_horizon_summary.md`
- `results/process_adjoint_horizon_summary.md`
- `results/statistical_audit_summary.md`
- `results/resource_audit_summary.md`
- `results/reproducibility_manifest.md`
- `results/reproducibility_manifest.json`
- `results/multilevel_horizon_summary.md`
- `results/transmon_leakage_summary.md`
- `results/transmon_standalone_adjoint_summary.md`
- `results/transmon_open_system_leakage_summary.md`
- `results/transmon_open_leakage_adjoint_summary.md`
- `results/open_leakage_integrated_sweep_summary.md`
- `results/open_leakage_pareto_refinement_summary.md`
- `results/open_leakage_continuation_sweep_summary.md`
- `results/open_leakage_training_ensemble_audit_summary.md`
- `results/open_leakage_long_horizon_sweep_summary.md`
- `results/open_leakage_pareto_audit_summary.md`
- `results/robustness_scan_summary.md`
- `results/polished_openloop_summary.md`
- `results/figures/`

See `docs/code_reproducibility_audit.md` for the train/test split, physical-consistency checks, and modeling scope.
