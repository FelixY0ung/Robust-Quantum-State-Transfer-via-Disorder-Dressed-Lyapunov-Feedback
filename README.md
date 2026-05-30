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
python3 code/horizon_lyapunov.py
python3 code/horizon_ablation.py
python3 code/open_system_noise.py
python3 code/gate_fidelity_probe.py
python3 code/gate_process_baseline.py
python3 code/ensemble_grape_baseline.py
python3 code/multilevel_horizon.py
python3 code/polished_openloop.py
python3 code/plot_experiments.py
```

The longer scripts are the open-loop optimization, terminal process-fidelity baseline, ensemble-GRAPE baseline, and horizon-search experiments. The checked outputs currently included in `results/` use held-out disorder seeds `10..59` for the key two-level, three-level, horizon-ablation, open-system stress-test, gate-fidelity diagnostic, terminal process-baseline, and ensemble-GRAPE results.

## Main Outputs

- `results/horizon_lyapunov_summary.md`
- `results/horizon_ablation_summary.md`
- `results/open_system_noise_summary.md`
- `results/gate_fidelity_probe_summary.md`
- `results/gate_process_baseline_summary.md`
- `results/ensemble_grape_baseline_summary.md`
- `results/multilevel_horizon_summary.md`
- `results/robustness_scan_summary.md`
- `results/polished_openloop_summary.md`
- `results/figures/`

See `docs/code_reproducibility_audit.md` for the train/test split, physical-consistency checks, and modeling scope.
