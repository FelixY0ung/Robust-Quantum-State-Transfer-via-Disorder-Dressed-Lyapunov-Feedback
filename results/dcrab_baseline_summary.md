# dCRAB-Style Baseline Summary

| task | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | basis_count | segments | refreshes | frequency_seed_path | optimizer_success_all | optimization_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0.05 | 50 | 0.999753 | 0.99908 | 0.000268558 | 5.14016 | 3 | 40 | 3 | 31,43,59 | False | 48.133 |
| H | 0.08 | 50 | 0.999578 | 0.998374 | 0.000412751 | 5.14016 | 3 | 40 | 3 | 31,43,59 | False | 48.133 |
| Z | 0.05 | 50 | 0.999869 | 0.999526 | 0.00011474 | 8.65196 | 3 | 40 | 3 | 31,43,59 | False | 38.754 |
| Z | 0.08 | 50 | 0.999635 | 0.998336 | 0.000386238 | 8.65196 | 3 | 40 | 3 | 31,43,59 | False | 38.754 |

## Refresh Logs

| task | refresh_index | frequency_seed | previous_objective | candidate_objective | objective | accepted | optimizer_iterations | optimizer_success | optimization_seconds | correction_bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z | 0 | 31 | 1.11263 | 0.00175861 | 0.00175861 | True | 18 | False | 17.2639 | 4 |
| Z | 1 | 43 | 0.00175861 | 0.0020065 | 0.00175861 | False | 18 | False | 6.03291 | 2 |
| Z | 2 | 59 | 0.00175861 | 0.00175851 | 0.00175851 | True | 18 | False | 15.4571 | 2 |
| H | 0 | 31 | 0.659139 | 0.00103234 | 0.00103234 | True | 18 | False | 20.3674 | 4 |
| H | 1 | 43 | 0.00103234 | 0.00184525 | 0.00103234 | False | 18 | False | 13.9675 | 2 |
| H | 2 | 59 | 0.00103234 | 0.00162533 | 0.00103234 | False | 18 | False | 13.798 | 2 |
