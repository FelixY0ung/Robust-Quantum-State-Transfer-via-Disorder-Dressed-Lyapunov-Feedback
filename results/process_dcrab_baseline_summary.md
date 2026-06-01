# Process dCRAB Baseline Summary

| task | eval_strength | n | state_fidelity_mean | state_fidelity_min | avg_gate_fidelity_mean | avg_gate_fidelity_min | avg_gate_fidelity_std | avg_gate_fidelity_ci95 | pulse_energy_mean | basis_count | segments | refreshes | n_training_seeds | frequency_seed_path | optimizer_success_all | optimization_seconds | training_objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0.08 | 50 | 0.997744 | 0.992236 | 0.997256 | 0.990112 | 0.00262362 | 0.000734615 | 3.25377 | 3 | 40 | 3 | 8 | 31,43,59 | False | 87.309 | 0.00255756 |
| Z | 0.08 | 50 | 0.999312 | 0.998191 | 0.996985 | 0.988854 | 0.00298382 | 0.000835469 | 5.06693 | 3 | 40 | 3 | 8 | 31,43,59 | False | 65.311 | 0.00363592 |

## Refresh Logs

| task | refresh_index | frequency_seed | previous_objective | candidate_objective | objective | accepted | optimizer_iterations | optimizer_success | optimization_seconds | correction_bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z | 0 | 31 | 0.729324 | 0.00363592 | 0.00363592 | True | 18 | False | 31.128 | 4 |
| Z | 1 | 43 | 0.00363592 | 0.00650802 | 0.00363592 | False | 18 | False | 17.8579 | 2 |
| Z | 2 | 59 | 0.00363592 | 0.00652108 | 0.00363592 | False | 18 | False | 16.325 | 2 |
| H | 0 | 31 | 0.782246 | 0.00255756 | 0.00255756 | True | 18 | False | 40.9614 | 4 |
| H | 1 | 43 | 0.00255756 | 0.00818752 | 0.00255756 | False | 18 | False | 20.8214 | 2 |
| H | 2 | 59 | 0.00255756 | 0.00950185 | 0.00255756 | False | 18 | False | 25.5267 | 2 |
