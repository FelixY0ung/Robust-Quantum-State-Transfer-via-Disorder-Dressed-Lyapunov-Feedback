# dCRAB-Style Baseline Summary

| task | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | basis_count | segments | refreshes | frequency_seed_path | optimizer_success_all | optimization_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0.05 | 50 | 0.999932 | 0.999798 | 5.99668e-05 | 4.97162 | 3 | 40 | 3 | 31,43,59 | False | 71.008 |
| H | 0.08 | 50 | 0.999789 | 0.999393 | 0.000183508 | 4.97162 | 3 | 40 | 3 | 31,43,59 | False | 71.008 |
| Z | 0.05 | 50 | 0.999934 | 0.999766 | 7.02945e-05 | 8.60437 | 3 | 40 | 3 | 31,43,59 | False | 51.806 |
| Z | 0.08 | 50 | 0.999713 | 0.998775 | 0.000277975 | 8.60437 | 3 | 40 | 3 | 31,43,59 | False | 51.806 |

## Refresh Logs

| task | refresh_index | frequency_seed | previous_objective | candidate_objective | objective | accepted | optimizer_iterations | optimizer_success | optimization_seconds | correction_bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z | 0 | 31 | 1.09285 | 0.00190827 | 0.00190827 | True | 18 | False | 19.3278 | 4 |
| Z | 1 | 43 | 0.00190827 | 0.00221402 | 0.00190827 | False | 18 | False | 13.7985 | 2 |
| Z | 2 | 59 | 0.00190827 | 0.00216116 | 0.00190827 | False | 18 | False | 18.6795 | 2 |
| H | 0 | 31 | 0.650111 | 0.00400975 | 0.00400975 | True | 18 | False | 16.3239 | 4 |
| H | 1 | 43 | 0.00400975 | 0.00110523 | 0.00110523 | True | 18 | False | 37.1449 | 2 |
| H | 2 | 59 | 0.00110523 | 0.00767405 | 0.00110523 | False | 18 | False | 17.5388 | 2 |
