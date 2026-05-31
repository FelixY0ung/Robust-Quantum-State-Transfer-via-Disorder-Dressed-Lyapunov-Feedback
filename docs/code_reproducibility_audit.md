# Code Reproducibility and Physical-Consistency Audit

Date: 2026-05-28

Updated: 2026-05-31

## Scope

Audited the simulation code used by `manuscript/cac2026_paper.tex`, especially:

- `code/horizon_lyapunov.py`
- `code/multilevel_horizon.py`
- `code/robustness_scan.py`
- `code/crab_baseline.py`
- `code/polished_openloop.py`
- `code/plot_experiments.py`
- `code/open_system_horizon_training.py`
- `code/open_system_grape_baseline.py`
- `code/transmon_leakage_horizon.py`
- `code/ensemble_lyapunov.py`
- `code/process_seeded_horizon.py`
- `code/process_adjoint_horizon.py`
- `code/tracking_simulation.py`
- `code/stationarity_analysis.py`

## Checks Performed

1. Syntax/reproducibility checks
   - Ran `python3 -m py_compile` on all core scripts.
   - Re-ran `code/horizon_lyapunov.py` and `code/multilevel_horizon.py` after changing their default held-out seeds to `range(10, 60)`.
   - Confirmed output sizes:
     - `results/horizon_lyapunov_results.csv`: 400 rows, strengths `0, 0.02, 0.05, 0.08`, seeds `10..59`.
     - `results/multilevel_horizon_results.csv`: 200 rows, strengths `0, 0.03, 0.05, 0.08`, seeds `10..59`.
     - `results/robustness_scan_results.csv`: 800 rows, 2 tasks, 2 pulse types,
       strengths `0, 0.02, 0.05, 0.08`, seeds `10..59`.
     - `results/crab_baseline_results.csv`: 200 rows, 2 tasks, strengths
       `0.05, 0.08`, seeds `10..59`.
     - `results/polished_openloop_results.csv`: 200 rows, 2 tasks, strengths
       `0.05, 0.08`, seeds `10..59`.
     - `results/ensemble_results.csv`: 100 rows, Z/H tasks, seeds `10..59`.
    - `results/ensemble_lyapunov_results.csv`: 100 rows, Z/H tasks, seeds
      `10..59`.
     - `results/open_system_training_results.csv`: 800 rows, 2 tasks, 2
       training modes, 4 noise cases, seeds `10..59`.
     - `results/open_system_grape_results.csv`: 400 rows, 2 tasks, 4
       evaluation noise cases, seeds `10..59`.
     - `results/process_seeded_horizon_results.csv`: 200 rows, 2 tasks, 2
       controllers, seeds `10..59`.
     - `results/process_adjoint_horizon_results.csv`: 200 rows, 2 tasks, 2
       controllers, seeds `10..59`.
    - `results/transmon_leakage_results.csv`: 1000 rows, 5 controllers,
       strengths `0, 0.01, 0.02, 0.03`, seeds `10..59`.
   - Ran `python3 code/transmon_leakage_horizon.py --gradient-check`; directional
     derivative errors were about `1e-9` for both the terminal and leakage-
     penalized GRAPE objectives and about `1e-10` for the short-horizon adjoint
     objective.
   - Ran `python3 code/open_system_grape_baseline.py --gradient-check`; the
     directional derivative error was about `1.63e-11`.
   - Ran `python3 code/process_adjoint_horizon.py --gradient-check`; the
     directional derivative error was about `5.08e-11`.

2. Train/test separation
   - Two-level horizon training uses seeds `0..7`, test uses `10..59`.
   - Three-level horizon training uses seeds `0..3`, test uses `10..59`.
   - Open-loop robustness scan uses training seeds `0..3` and test seeds `10..59`.
   - Polished terminal open-loop uses training seeds `0..7` at `delta = 0.08`
     and test seeds `10..59`.
   - CRAB reduced-basis training uses disorder seeds `0..3` at `delta = 0.08`,
     a fixed randomized Fourier basis seed `31`, and test seeds `10..59`.
   - Standalone open-loop and one-step ensemble Lyapunov diagnostics now use
     test seeds `10..59`.
   - No overlap was found in the key reported training/test split.
   - Five-level leakage path-horizon training uses seeds `0..5`; terminal
     GRAPE and leakage-penalized GRAPE use training seeds `0..3`; the
     gradient-seeded horizon uses a leakage-penalized reference pulse and
     horizon training seeds `0..3`; the adjoint-polished horizon optimizes each
     short horizon in a trust region around the same leakage-penalized
     reference; all five use test seeds `10..59`.
   - The compact open-system-trained horizon diagnostic uses Lindblad training
     seeds `0..2` with 60 segments, two-step lookahead, and beam width three;
     it reuses the closed-trained evaluation rows from `open_system_noise` and
     evaluates the open-trained pulse on test seeds `10..59`.
   - The open-system GRAPE baseline trains through the combined Lindblad model
     with disorder seeds `0..3`, restart seeds `17` and `29`, and evaluates on
     test seeds `10..59`. It is a terminal open-loop comparator, not Lyapunov
     feedback.
   - The process-seeded horizon diagnostic trains/reuses a process-GRAPE
     reference with disorder seeds `0..7` and strengths `0.05, 0.08`, then runs
     a local four-step horizon around that reference and evaluates on test seeds
     `10..59`.
   - The process-adjoint horizon diagnostic uses the same process-GRAPE
     reference configuration, then continuously optimizes each four-step
     process horizon by exact Frechet derivatives inside a trust region and
     evaluates on test seeds `10..59`.

3. Physical state checks
   - Checked representative two-level and three-level evolutions for trace preservation and Hermiticity.
   - Two-level representative minimum density-matrix eigenvalue at `delta = 0.08`: `2.14e-4`.
   - Three-level representative minimum density-matrix eigenvalue at `delta = 0.08`: `-2.64e-16`, consistent with floating-point roundoff.
   - Final fidelities are in the physical range `[0, 1]` within numerical tolerance.
   - The five-level leakage benchmark explicitly reports leakage outside the
     computational subspace `|0>, |1>` as both final leakage and maximum leakage.

4. Manuscript consistency
   - `results/horizon_lyapunov_summary.md`, `results/multilevel_horizon_summary.md`, `results/extended_heldout_summary.md`, and `manuscript/cac2026_paper.tex` now use the same 50-seed horizon statistics.
   - `results/robustness_scan_summary.md` and the open-loop tables in `manuscript/cac2026_paper.tex` now use the regenerated 50-seed open-loop statistics.
   - `results/polished_openloop_summary.md` is cited as a terminal-optimization
     ceiling and not as a Lyapunov feedback result.
   - `results/figures/*.pdf` and `results/figures/*.png` are generated from
     current CSV outputs by `code/plot_experiments.py`.
   - Old 10-seed horizon and open-loop statistics were replaced in the core summary files.
   - `results/transmon_leakage_summary.md` and `results/figures/transmon_leakage.*`
     are generated by `code/transmon_leakage_horizon.py` and are cited as a
     physically richer stress test, not as a solved hardware benchmark.
   - `results/crab_baseline_summary.md` and `results/figures/crab_baseline.*`
     are generated by `code/crab_baseline.py` and are cited as an independent
     reduced-basis terminal-control comparator.
   - `results/process_seeded_horizon_summary.md` and
     `results/figures/process_seeded_horizon.*` are generated by
     `code/process_seeded_horizon.py` and are cited as a gradient-reference
     process-horizon diagnostic.
   - `results/process_adjoint_horizon_summary.md` and
     `results/figures/process_adjoint_horizon.*` are generated by
     `code/process_adjoint_horizon.py` and are cited as an adjoint-polished
     reference-assisted process-horizon diagnostic.
   - `results/open_system_training_summary.md` and
     `results/figures/open_system_training.*` are generated by
     `code/open_system_horizon_training.py` and are cited as a diagnostic
     limitation, not as a performance improvement.
   - `results/open_system_grape_summary.md` and
     `results/figures/open_system_grape.*` are generated by
     `code/open_system_grape_baseline.py` and are cited as a terminal
     open-system comparator, not as a Lyapunov feedback result.

## Findings

### No hard physical impossibility found

The main two-level and three-level horizon simulations use Hermitian Hamiltonians, unitary or RK4 approximations to closed-system dynamics, trace-one density matrices, fixed random seeds, and disjoint train/test disorder samples. The reported state-transfer fidelities are therefore numerically reproducible within the stated simplified model.

### Important modeling limitations

These limitations are scientifically acceptable only because the manuscript now states conservative claims:

- The static disorder model is a normalized random Hamiltonian perturbation, not a calibrated device-specific noise model.
- The disorder-dressed master-equation form in the theory section is a design/diagnostic model, while the key horizon simulations evaluate static Hamiltonian disorder realizations in the interaction frame.
- The beam-horizon controller is an offline receding-horizon-style candidate search over known training disorder samples; it is not a hardware-real-time feedback controller with measurement backaction.
- The main Lyapunov result remains a state-transfer result; the process-fidelity
  rows are diagnostics and reference-assisted horizon extensions, not a
  standalone full-gate Lyapunov controller.
- The open-loop baseline is compact and reproducible, but not a full GRAPE/Krotov reimplementation.
- The CRAB baseline is a reduced-basis terminal optimizer with a compact
  differential-evolution budget. It is useful as an independent modern
  comparator, but it is not a Lyapunov feedback controller and its optimizer
  reports max-iteration termination rather than formal convergence.
- The process-seeded horizon diagnostic preserves high process fidelity only
  because it uses a process-GRAPE reference trajectory. It is evidence that
  gradient references can be embedded into a receding-horizon architecture, not
  a standalone process-level Lyapunov controller.
- The adjoint-polished process-horizon diagnostic uses exact Frechet gradients
  inside the short-horizon process score, but it still relies on the
  process-GRAPE reference and trust region. It should be presented as stronger
  evidence for gradient-assisted horizon scoring, not as a replacement for
  terminal process GRAPE.
- The compact open-system-trained horizon diagnostic lowers pulse energy but
  does not improve final fidelity over the closed-trained horizon under the
  combined dephasing/relaxation test. It should be read as evidence that
  dissipative horizon training needs richer gradient-assisted or process-level
  optimization, not as a solved open-system controller.
- The open-system GRAPE baseline improves combined-noise state-transfer
  fidelity, but it is a terminal open-loop optimizer. It should be used as a
  reachability/comparator result and not described as an open-system Lyapunov
  feedback controller.
- The five-level leakage benchmark shows a remaining method gap: terminal GRAPE
  outperforms the finite-candidate path-horizon controller on final fidelity.
  Gradient-seeded horizon control improves the finite-candidate horizon while
  keeping transient leakage low, and adjoint-polished horizon control closes
  more of the final-fidelity gap by continuously optimizing short horizons.
  Leakage-penalized GRAPE nearly preserves terminal-GRAPE fidelity while
  reducing transient leakage, indicating that the next horizon controller should
  go beyond reference seeding or local polishing and incorporate explicit
  leakage shaping and gradient information inside the horizon optimization.

## Changes Made During Audit

- Updated `code/horizon_lyapunov.py` default `test_seeds` from `range(10, 20)` to `range(10, 60)`.
- Updated `code/multilevel_horizon.py` default `test_seeds` from `range(10, 20)` to `range(10, 60)`.
- Updated `code/robustness_scan.py` to evaluate test seeds `10..59`.
- Re-ran the robustness scan so its CSV and summary outputs match the 50-seed statistics used in the manuscript.
- Updated `code/ensemble_openloop.py` and `code/ensemble_lyapunov.py` to
  evaluate test seeds `10..59`, then regenerated `results/ensemble_summary.md`
  and `results/ensemble_lyapunov_summary.md`.
- Added `code/polished_openloop.py`, regenerated near-unit-fidelity terminal
  baseline results, and added `code/plot_experiments.py` for reproducible
  experiment figures.
- Added `code/crab_baseline.py`, generated reduced-basis CRAB results and
  figures, and recorded the result as an independent derivative-free terminal
  robust-control comparator.
- Added `code/process_seeded_horizon.py`, generated process-GRAPE-seeded
  horizon results and figures, and recorded the result as a bridge toward
  adjoint-assisted process-level horizon control.
- Added `code/process_adjoint_horizon.py`, generated adjoint-polished
  process-horizon results and figures, and recorded the result as a
  reference-assisted process-horizon diagnostic using exact Frechet gradients.
- Added `code/transmon_leakage_horizon.py`, regenerated five-level leakage CSV,
  summary, and figure outputs, and extended the benchmark with gradient-seeded
  horizon, adjoint-polished horizon, and leakage-penalized GRAPE comparators.
- Added `code/open_system_horizon_training.py`, generated a compact
  open-system-trained horizon diagnostic, and recorded the resulting
  lower-energy but lower-fidelity Lindblad-trained pulses as a modeling
  limitation.
- Added `code/open_system_grape_baseline.py`, generated terminal open-system
  GRAPE results and figures, and recorded the result as an optimizer ceiling
  for the dissipative two-level state-transfer task.

## Judgment

The code is rigorous and reproducible for a compact CAC-style numerical control paper, provided the manuscript keeps its current conservative positioning. It should not be presented as a hardware-validated robust quantum gate controller or as a full optimal-control benchmark against GRAPE/Krotov. Under the current state-transfer and diagnostic framing, no code-level issue was found that invalidates the main conclusions.
