# Code Reproducibility and Physical-Consistency Audit

Date: 2026-05-28

Updated: 2026-05-31

## Scope

Audited the simulation code used by the journal manuscript, especially:

- `code/horizon_lyapunov.py`
- `code/shifted_fallback_horizon.py`
- `code/state_adjoint_horizon.py`
- `code/multilevel_horizon.py`
- `code/robustness_scan.py`
- `code/crab_baseline.py`
- `code/polished_openloop.py`
- `code/plot_experiments.py`
- `code/open_system_horizon_training.py`
- `code/open_system_grape_baseline.py`
- `code/open_system_adjoint_horizon.py`
- `code/slew_constrained_horizon.py`
- `code/bandwidth_filter_audit.py`
- `code/transmon_leakage_horizon.py`
- `code/transmon_open_system_leakage.py`
- `code/transmon_open_leakage_adjoint_horizon.py`
- `code/ensemble_lyapunov.py`
- `code/process_seeded_horizon.py`
- `code/process_standalone_adjoint_horizon.py`
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
     - `results/process_standalone_adjoint_results.csv`: 200 rows, 2 tasks,
       2 controllers, seeds `10..59`.
     - `results/shifted_fallback_horizon_results.csv`: 400 rows, 2 tasks,
       strengths `0, 0.02, 0.05, 0.08`, seeds `10..59`.
     - `results/state_adjoint_horizon_results.csv`: 400 rows, 2 tasks, 2
       controllers, strengths `0.05, 0.08`, seeds `10..59`.
     - `results/open_system_adjoint_horizon_results.csv`: 800 rows, 2 tasks,
       2 controllers, 4 evaluation noise cases, seeds `10..59`.
     - `results/slew_constrained_horizon_results.csv`: 300 rows, 2 tasks,
       3 slew weights, strength `0.08`, seeds `10..59`.
     - `results/bandwidth_filter_audit_results.csv`: 800 rows, 2 tasks,
       2 base slew weights, 4 filters, strength `0.08`, seeds `10..59`.
     - `results/transmon_leakage_results.csv`: 1000 rows, 5 controllers,
       strengths `0, 0.01, 0.02, 0.03`, seeds `10..59`.
     - `results/transmon_open_system_leakage_results.csv`: 600 rows, 3
       controllers, 4 noise cases, strength `0.03`, seeds `10..59`.
     - `results/transmon_open_leakage_adjoint_results.csv`: 800 rows, 4
       controllers, 4 noise cases, strength `0.03`, seeds `10..59`.
   - Ran `python3 code/transmon_leakage_horizon.py --gradient-check`; directional
     derivative errors were about `1e-9` for both the terminal and leakage-
     penalized GRAPE objectives and about `1e-10` for the short-horizon adjoint
     objective.
   - Ran `python3 code/open_system_grape_baseline.py --gradient-check`; the
     directional derivative error was about `1.63e-11`.
   - Ran `python3 code/open_system_adjoint_horizon.py --gradient-check`; the
     directional derivative error was about `8.22e-11`.
   - Ran `python3 code/state_adjoint_horizon.py --gradient-check`; the
     directional derivative error was about `1.41e-11`.
   - Ran `python3 code/process_standalone_adjoint_horizon.py --gradient-check`;
     the directional derivative error was about `1.12e-10`.
   - Ran `python3 code/process_adjoint_horizon.py --gradient-check`; the
     directional derivative error was about `5.08e-11`.
   - Ran `python3 code/transmon_open_leakage_adjoint_horizon.py --gradient-check`;
     the directional derivative error was about `1.46e-10`.

2. Train/test separation
   - Two-level horizon training uses seeds `0..7`, test uses `10..59`.
   - The shifted-fallback horizon comparison uses the same two-level horizon
     training seeds `0..7`, the same training strengths `0.05, 0.08`, and test
     seeds `10..59`; it augments the candidate set by scoring the shifted tail
     of the previous selected horizon sequence with every admissible fallback.
   - The state-adjoint horizon diagnostic uses the same two-level training
     strengths and training seeds `0..7`, uses the finite-candidate horizon
     only as a local initializer, then optimizes each short horizon by exact
     Frechet derivatives; test seeds are `10..59`.
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
   - The five-level leakage-plus-Lindblad stress test reuses the path horizon,
     reference-assisted adjoint horizon, and leakage-penalized GRAPE pulse
     families, then evaluates them under four Lindblad noise cases on test
     seeds `10..59`. It is an evaluation stress test, not jointly trained
     open-system leakage optimization.
   - The compact open-system-trained horizon diagnostic uses Lindblad training
     seeds `0..2` with 60 segments, two-step lookahead, and beam width three;
     it reuses the closed-trained evaluation rows from `open_system_noise` and
     evaluates the open-trained pulse on test seeds `10..59`.
   - The open-system GRAPE baseline trains through the combined Lindblad model
     with disorder seeds `0..3`, restart seeds `17` and `29`, and evaluates on
     test seeds `10..59`. It is a terminal open-loop comparator, not Lyapunov
     feedback.
   - The adjoint open-system horizon diagnostic uses the same open-system
     GRAPE reference configuration, then optimizes each four-step receding
     horizon through the Lindblad model by exact Frechet derivatives in a trust
     region and evaluates on test seeds `10..59`.
   - The process-seeded horizon diagnostic trains/reuses a process-GRAPE
     reference with disorder seeds `0..7` and strengths `0.05, 0.08`, then runs
     a local four-step horizon around that reference and evaluates on test seeds
     `10..59`.
   - The standalone process-adjoint horizon diagnostic uses the same process
     training strengths and seeds, uses the finite-candidate process horizon
     only as a local initializer, and evaluates on test seeds `10..59`.
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
   - `results/horizon_lyapunov_summary.md`, `results/multilevel_horizon_summary.md`, and `results/extended_heldout_summary.md` now use the same 50-seed horizon statistics reported in the journal manuscript.
   - `results/robustness_scan_summary.md` and the manuscript open-loop tables now use the regenerated 50-seed open-loop statistics.
   - `results/polished_openloop_summary.md` is cited as a terminal-optimization
     ceiling and not as a Lyapunov feedback result.
   - `results/figures/*.pdf` and `results/figures/*.png` are generated from
     current CSV outputs by `code/plot_experiments.py`.
   - Old 10-seed horizon and open-loop statistics were replaced in the core summary files.
   - `results/transmon_leakage_summary.md` and `results/figures/transmon_leakage.*`
     are generated by `code/transmon_leakage_horizon.py` and are cited as a
     physically richer stress test, not as a solved hardware benchmark.
   - `results/transmon_open_system_leakage_summary.md` and
     `results/figures/transmon_open_system_leakage.*` are generated by
     `code/transmon_open_system_leakage.py` and are cited as a combined
     leakage-plus-Lindblad stress test, not as a jointly trained open-system
     leakage optimizer.
   - `results/transmon_open_leakage_adjoint_summary.md` is generated by
     `code/transmon_open_leakage_adjoint_horizon.py` and is cited as a
     GRAPE-free direct Lindblad leakage-adjoint horizon diagnostic. It includes
     the conservative direct row, a terminal-target-biased direct row, and a
     two-stage target/leakage repolish row. The conservative row lowers leakage
     strongly but gives only a modest fidelity gain over the path seed; the
     two-stage row raises combined-noise fidelity and worst-seed performance at
     the cost of higher maximum leakage. The same script supports `--plot-only`
     and writes `results/figures/transmon_open_leakage_combined.*`, combining
     these direct adjoint rows with the path, reference-assisted adjoint, and
     leakage-GRAPE stress-test rows.
   - `results/open_leakage_integrated_sweep_summary.md` and
     `results/figures/open_leakage_integrated_sweep.*` are generated by
     `code/open_leakage_integrated_sweep.py --extended`. They test single-stage
     no-reference open-leakage objectives with stronger terminal target pressure
     and leakage penalties, plus robust variants with larger worst-seed weights.
     The best retained mean/leakage point is an intermediate Pareto tradeoff,
     and the best robust-weighted point improves worst held-out fidelity without
     escaping mean/leakage domination. These rows strengthen the boundary audit
     rather than closing the gap to the reference-assisted or terminal rows.
   - `results/open_leakage_pareto_audit_summary.md` and
     `results/figures/open_leakage_pareto_audit.*` are generated by
     `code/open_leakage_pareto_audit.py`. They reuse the same combined-noise
     held-out rows and the integrated sweep point to classify the
     mean-fidelity/mean-maximum-leakage tradeoff. The audit shows that direct,
     integrated, and target-biased no-reference horizons provide useful Pareto
     tradeoff points, while the two-stage direct row improves the no-reference
     worst seed but is dominated in the two-dimensional Pareto audit by the
     reference-assisted horizon and leakage-GRAPE.
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
   - `results/process_standalone_adjoint_summary.md` is generated by
     `code/process_standalone_adjoint_horizon.py` and is cited as a
     GRAPE-free process-horizon limitation result.
   - `results/shifted_fallback_horizon_summary.md` is generated by
     `code/shifted_fallback_horizon.py` and is cited as a
     certification-oriented comparison that enforces the candidate-inclusion
     part of the shifted-fallback theorem, not as the best-fidelity horizon row.
   - `results/state_adjoint_horizon_summary.md` is generated by
     `code/state_adjoint_horizon.py` and is cited as a GRAPE-free
     adjoint-assisted state-transfer horizon diagnostic.
   - `results/open_system_training_summary.md` and
     `results/figures/open_system_training.*` are generated by
     `code/open_system_horizon_training.py` and are cited as a diagnostic
     limitation, not as a performance improvement.
   - `results/open_system_grape_summary.md` and
     `results/figures/open_system_grape.*` are generated by
     `code/open_system_grape_baseline.py` and are cited as a terminal
     open-system comparator, not as a Lyapunov feedback result.
   - `results/open_system_adjoint_horizon_summary.md` and
     `results/figures/open_system_adjoint_horizon.*` are generated by
     `code/open_system_adjoint_horizon.py` and are cited as a
     reference-assisted dissipative receding-horizon diagnostic.

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
- The adjoint open-system horizon diagnostic preserves most of the open-system
  GRAPE combined-noise fidelity in a receding-horizon form, but it is still
  reference-assisted and depends on a GRAPE trajectory and trust region.
- The shifted-fallback horizon comparison enforces the candidate-inclusion
  hypothesis of the practical-decrease proposition, but the terminal-progress
  margin remains a sufficient condition rather than an automatically verified
  guarantee at every receding-horizon step.
- The state-adjoint horizon diagnostic improves the 60-segment finite-candidate
  seed, especially for H transfer, and uses no terminal GRAPE reference.
  However, it remains a local nonconvex short-horizon optimizer initialized by
  the finite-candidate controller; it should not be described as a global
  optimal-control solver.
- The standalone process-adjoint horizon diagnostic does not materially improve
  held-out average gate fidelity over the finite process seed. It should be
  cited as evidence that exact short-horizon process gradients alone do not
  remove the global process-search difficulty.
- The five-level leakage benchmark shows a remaining method gap: terminal GRAPE
  outperforms the finite-candidate path-horizon controller on final fidelity.
  Gradient-seeded horizon control improves the finite-candidate horizon while
  keeping transient leakage low, and adjoint-polished horizon control closes
  more of the final-fidelity gap by continuously optimizing short horizons.
  Leakage-penalized GRAPE nearly preserves terminal-GRAPE fidelity while
  reducing transient leakage, indicating that the next horizon controller should
  go beyond reference seeding or local polishing and incorporate explicit
  leakage shaping and gradient information inside the horizon optimization.
- The combined five-level leakage-plus-Lindblad stress test preserves the same
  conclusion under weak dephasing and relaxation: the reference-assisted adjoint
  horizon substantially improves over the path horizon, but leakage-penalized
  GRAPE remains the terminal ceiling. This supports the manuscript's claim
  boundary that integrated leakage-aware, open-system horizon optimization is
  still future work.

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
- Added `code/open_system_adjoint_horizon.py`, generated reference-assisted
  adjoint Lindblad horizon results and figures, and recorded the result as a
  dissipative receding-horizon diagnostic using exact Frechet derivatives.
- Added `code/shifted_fallback_horizon.py`, generated shifted-fallback horizon
  comparison rows, and recorded the result as a certification-performance
  tradeoff for the beam-horizon controller.
- Added `code/state_adjoint_horizon.py`, generated standalone adjoint
  state-transfer horizon rows, and recorded the result as a GRAPE-free
  adjoint-assisted horizon diagnostic.
- Added `code/process_standalone_adjoint_horizon.py`, generated standalone
  process-adjoint horizon rows, and recorded the result as a GRAPE-free
  process-level limitation diagnostic.
- Added `code/transmon_open_system_leakage.py`, generated combined five-level
  leakage-plus-Lindblad rows and figures, and recorded the result as a physical
  stress test of the leakage-aware horizon family.
- Added `code/transmon_open_leakage_adjoint_horizon.py`, generated standalone
  direct five-level Lindblad leakage-adjoint rows, and recorded the result as a
  GRAPE-free direct combined-noise horizon diagnostic.
- Extended `code/transmon_open_leakage_adjoint_horizon.py` with a two-stage
  target-recovery/leakage-repolish variant and regenerated the direct
  open-leakage adjoint results, combined figure, statistical audit, and resource
  audit.
- Added `code/open_leakage_integrated_sweep.py`, generated the extended
  single-stage no-reference open-leakage sweep, and updated the statistical,
  resource, and Pareto audits with the intermediate integrated Pareto point.
- Added `code/open_leakage_pareto_audit.py`, generated the open-leakage Pareto
  CSV, summary, and figure, and recorded the fidelity/leakage tradeoff used in
  the journal manuscript's hardest benchmark discussion.
- Added `code/slew_constrained_horizon.py`, generated a compact two-level
  smoothness audit with quadratic first-difference penalties, and recorded the
  result as a physical-realism diagnostic rather than a hardware bandwidth
  model.
- Added `code/bandwidth_filter_audit.py`, generated post-design low-pass filter
  robustness rows, and recorded the result as bandwidth-aware evidence rather
  than a calibrated hardware transfer-function study.
- Extended `code/statistical_audit.py` with paired smoothness/filter
  comparisons, including the boxcar3 fidelity loss before and after slew
  shaping.

## Judgment

The code is rigorous and reproducible for a journal-facing numerical control paper, provided the manuscript keeps its current conservative positioning. It should not be presented as a hardware-validated robust quantum gate controller or as a full optimal-control benchmark against GRAPE/Krotov. Under the current state-transfer and diagnostic framing, no code-level issue was found that invalidates the main conclusions.
