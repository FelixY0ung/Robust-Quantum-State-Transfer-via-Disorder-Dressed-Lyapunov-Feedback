# dCRAB-Style Baseline Summary

| task | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | basis_count | segments | refreshes | frequency_seed_path | optimizer_success_all | optimization_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0.05 | 50 | 0.995687 | 0.985677 | 0.00458912 | 7.49843 | 4 | 40 | 5 | 31,43,59,71,83 | False | 99.434 |
| H | 0.08 | 50 | 0.992961 | 0.973139 | 0.00839971 | 7.49843 | 4 | 40 | 5 | 31,43,59,71,83 | False | 99.434 |
| Z | 0.05 | 50 | 0.999136 | 0.997239 | 0.000835807 | 4.59972 | 4 | 40 | 5 | 31,43,59,71,83 | False | 104.860 |
| Z | 0.08 | 50 | 0.998664 | 0.994864 | 0.00149459 | 4.59972 | 4 | 40 | 5 | 31,43,59,71,83 | False | 104.860 |

## Refresh Logs

| task | refresh_index | frequency_seed | previous_objective | candidate_objective | objective | accepted | optimizer_iterations | optimizer_success | optimization_seconds | correction_bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z | 0 | 31 | 1.11263 | 0.00109777 | 0.00109777 | True | 24 | False | 25.7234 | 4 |
| Z | 1 | 43 | 0.00109777 | 0.00150561 | 0.00109777 | False | 24 | False | 23.6566 | 2 |
| Z | 2 | 59 | 0.00109777 | 0.00181073 | 0.00109777 | False | 24 | False | 18.1937 | 2 |
| Z | 3 | 71 | 0.00109777 | 0.00107223 | 0.00107223 | True | 24 | False | 20.0079 | 2 |
| Z | 4 | 83 | 0.00107223 | 0.0013019 | 0.00107223 | False | 24 | False | 17.278 | 2 |
| H | 0 | 31 | 0.659139 | 0.00213524 | 0.00213524 | True | 24 | False | 23.2336 | 4 |
| H | 1 | 43 | 0.00213524 | 0.00213074 | 0.00213074 | True | 24 | False | 22.5805 | 2 |
| H | 2 | 59 | 0.00213074 | 0.00209883 | 0.00209883 | True | 24 | False | 21.5749 | 2 |
| H | 3 | 71 | 0.00209883 | 0.00233711 | 0.00209883 | False | 24 | False | 16.5789 | 2 |
| H | 4 | 83 | 0.00209883 | 0.00210637 | 0.00209883 | False | 24 | False | 15.4658 | 2 |
