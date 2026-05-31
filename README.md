# Robust Quantum State Transfer via Disorder-Dressed Lyapunov Feedback

Reproducible code and generated numerical results for the disorder-dressed Lyapunov feedback study are open source in this repository.

## Repository Layout

- `code/`: simulation, control, robustness-scan, and plotting scripts.
- `results/`: CSV outputs, Markdown summaries, and generated paper figures.
- `docs/`: reproducibility and physical-consistency audit notes.
- `requirements.txt`: Python dependencies used by the scripts.

## Environment

The scripts are plain Python and were run with Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
python3 code/horizon_lyapunov.py
python3 code/shifted_fallback_horizon.py
python3 code/shifted_fallback_margin_audit.py
python3 code/state_adjoint_horizon.py
python3 code/horizon_ablation.py
python3 code/open_system_noise.py
python3 code/open_system_horizon_training.py
python3 code/open_system_grape_baseline.py
python3 code/open_system_adjoint_horizon.py
python3 code/gate_fidelity_probe.py
python3 code/gate_process_baseline.py
python3 code/ensemble_grape_baseline.py
python3 code/process_horizon.py
python3 code/process_standalone_adjoint_horizon.py
python3 code/process_seeded_horizon.py
python3 code/process_adjoint_horizon.py
python3 code/statistical_audit.py
python3 code/multilevel_horizon.py
python3 code/transmon_leakage_horizon.py
python3 code/polished_openloop.py
python3 code/plot_experiments.py
```

The leakage and open-system GRAPE benchmarks also have quick derivative checks:

```bash
python3 code/transmon_leakage_horizon.py --gradient-check
python3 code/open_system_grape_baseline.py --gradient-check
python3 code/open_system_adjoint_horizon.py --gradient-check
python3 code/state_adjoint_horizon.py --gradient-check
python3 code/process_standalone_adjoint_horizon.py --gradient-check
python3 code/process_adjoint_horizon.py --gradient-check
```

The longer scripts are the open-loop optimization, CRAB and dCRAB reduced-basis baselines, terminal process-fidelity baseline, ensemble-GRAPE baseline, process-horizon diagnostics, five-level leakage benchmark, open-system horizon-training diagnostic, open-system GRAPE baseline, adjoint open-system horizon diagnostic, shifted-fallback horizon comparison and margin audit, standalone state-adjoint horizon diagnostic, standalone process-adjoint horizon diagnostic, and horizon-search experiments. The checked outputs currently included in `results/` use held-out disorder seeds `10..59` for the key two-level, three-level, five-level leakage, CRAB/dCRAB, horizon-ablation, shifted-fallback, standalone state-adjoint, standalone process-adjoint, open-system stress-test, open-system training diagnostic, open-system GRAPE, adjoint open-system horizon, gate-fidelity diagnostic, terminal process-baseline, ensemble-GRAPE, process-horizon, process-seeded horizon, process-adjoint horizon, and statistical-audit results.

The five-level weakly anharmonic leakage benchmark compares path-horizon control, gradient-seeded horizon control, adjoint-polished horizon control, terminal GRAPE, and leakage-penalized GRAPE. At `delta = 0.03`, gradient-seeded horizon control reaches mean final fidelity `0.916971` with mean maximum leakage `0.037966`; adjoint-polished horizon control reaches mean final fidelity `0.948305` with mean maximum leakage `0.053031`; leakage-penalized GRAPE reaches mean final fidelity `0.969447` while reducing mean maximum leakage to `0.053939`.

The reduced-basis CRAB baseline is an independent derivative-free terminal-control comparator. With three randomized Fourier modes per control and differential-evolution training at `delta = 0.08`, it reaches held-out mean fidelity `0.999240` for Z and `0.997811` for H at `delta = 0.08`.

The dCRAB-style reduced-basis baseline refreshes the randomized Fourier basis over three sequential terminal-optimization rounds and accepts only corrections that improve the robust training score. At `delta = 0.08`, it reaches held-out mean fidelity `0.999635` for Z and `0.999578` for H, with worst held-out fidelities above `0.9983`.

The process-GRAPE-seeded horizon diagnostic uses a 60-segment process-GRAPE reference and a four-step local receding horizon. At `delta = 0.08`, it preserves held-out average gate fidelity `0.994732` for Z and `0.999573` for H. This is a gradient-reference diagnostic, not an independent replacement for process GRAPE.

The adjoint-polished process-horizon diagnostic puts exact Frechet gradients inside the same four-step receding-horizon process score while staying in a trust region around the 60-segment process-GRAPE reference. At `delta = 0.08`, it preserves held-out average gate fidelity `0.994641` for Z and `0.999717` for H. This is a reference-assisted horizon diagnostic, not a standalone process-GRAPE replacement.

The shifted-fallback horizon comparison implements the candidate-inclusion part of the practical-decrease certificate: at each step it also scores the shifted tail of the previous horizon sequence with every admissible appended fallback. At `delta = 0.08`, it reaches held-out mean fidelity `0.995809` for Z and `0.997657` for H, while lowering mean pulse energy to `4.625/5.285`. This is a certification-oriented comparison, not the best-fidelity beam-horizon row.

The shifted-fallback margin audit recomputes the training trajectory and checks the terminal-progress part of the practical-decrease certificate. Candidate inclusion is enforced at every post-initial step, while a positive shifted-fallback margin is available on `16.2%` of audited Z steps and `14.1%` of audited H steps. This records the certificate as a sufficient-condition diagnostic, not as an unconditional convergence proof.

The standalone state-adjoint horizon diagnostic uses the finite-candidate horizon only as a local initializer, then optimizes the same short state-transfer Lyapunov score by exact Frechet derivatives without a terminal GRAPE reference. At `delta = 0.08`, it reaches held-out mean fidelity `0.997445` for Z and `0.996096` for H; the corresponding 60-segment finite-candidate seed reaches `0.997407/0.989428`. This is evidence for adjoint-assisted horizon scoring, not a global optimality claim.

The standalone process-adjoint horizon diagnostic applies the same idea to average gate fidelity without a terminal process-GRAPE reference. At `delta = 0.08`, it reaches held-out average gate fidelity `0.924093` for Z and `0.927613` for H, close to the finite process seed's `0.927537/0.928319`. This negative result is included to document that exact short-horizon process gradients alone do not solve the global process-search difficulty.

The open-system horizon-training diagnostic compares the existing closed-system-trained horizon pulses with compact Lindblad-trained finite-candidate horizon pulses. Under combined dephasing and relaxation at `delta = 0.08`, the compact open-trained pulses reach mean fidelity `0.948556` for Z and `0.942853` for H, below the closed-trained horizon's `0.957140` and `0.956625`, but with lower pulse energy. This diagnostic is included as a limitation rather than an improvement claim.

The open-system GRAPE baseline is a terminal open-loop comparator trained through the combined Lindblad model, not a Lyapunov feedback law. Under the same combined-noise held-out test, it reaches mean fidelity `0.977461` for Z and `0.978929` for H, showing that the remaining open-system gap is mainly an optimizer/method gap for this two-level state-transfer task.

The adjoint open-system horizon diagnostic carries most of the open-system GRAPE performance into a reference-assisted receding-horizon form with exact Frechet derivatives. Under the combined-noise held-out test at `delta = 0.08`, it reaches mean fidelity `0.974684` for Z and `0.977779` for H. It remains reference-assisted and should not be read as an independent open-system Lyapunov proof.

## Main Outputs

- `results/horizon_lyapunov_summary.md`
- `results/crab_baseline_summary.md`
- `results/dcrab_baseline_summary.md`
- `results/shifted_fallback_horizon_summary.md`
- `results/shifted_fallback_margin_audit_summary.md`
- `results/state_adjoint_horizon_summary.md`
- `results/horizon_ablation_summary.md`
- `results/open_system_noise_summary.md`
- `results/open_system_training_summary.md`
- `results/open_system_grape_summary.md`
- `results/open_system_adjoint_horizon_summary.md`
- `results/gate_fidelity_probe_summary.md`
- `results/gate_process_baseline_summary.md`
- `results/ensemble_grape_baseline_summary.md`
- `results/process_horizon_summary.md`
- `results/process_standalone_adjoint_summary.md`
- `results/process_seeded_horizon_summary.md`
- `results/process_adjoint_horizon_summary.md`
- `results/statistical_audit_summary.md`
- `results/multilevel_horizon_summary.md`
- `results/transmon_leakage_summary.md`
- `results/robustness_scan_summary.md`
- `results/polished_openloop_summary.md`
- `results/figures/`

See `docs/code_reproducibility_audit.md` for the train/test split, physical-consistency checks, and modeling scope.
