# Extended Held-Out Horizon Summary

Training pulses are unchanged from `code/horizon_lyapunov.py` and `code/multilevel_horizon.py`. This check expands only the held-out evaluation from 10 seeds (`10..19`) to 50 seeds (`10..59`).

## Two-Level Beam-Horizon Lyapunov

| system | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | tail_infidelity_mean | pulse_energy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| horizon_lyapunov_H | 0.05 | 50 | 0.998805 | 0.997088 | 0.000772541 | 0.000943604 | 5.6075 |
| horizon_lyapunov_H | 0.08 | 50 | 0.998156 | 0.994749 | 0.00144612 | 0.00136403 | 5.6075 |
| horizon_lyapunov_Z | 0.05 | 50 | 0.998523 | 0.996815 | 0.000638528 | 0.00161278 | 4.9225 |
| horizon_lyapunov_Z | 0.08 | 50 | 0.997582 | 0.993966 | 0.00132018 | 0.00265704 | 4.9225 |

## Three-Level Chain Beam-Horizon Lyapunov

| system | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | tail_infidelity_mean | pulse_energy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three_level_chain | 0.05 | 50 | 0.996530 | 0.986516 | 0.00269025 | 0.00338026 | 3.47812 |
| three_level_chain | 0.08 | 50 | 0.993592 | 0.973613 | 0.00483547 | 0.00610371 | 3.47812 |
