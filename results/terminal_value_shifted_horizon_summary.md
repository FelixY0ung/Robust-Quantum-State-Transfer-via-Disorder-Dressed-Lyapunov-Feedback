# Terminal-Value Shifted-Horizon Summary

This controller uses the finite-set fallback-value terminal score inside the online shifted-fallback horizon ranking. Beam pruning is unchanged, shifted candidates are forced into the final scored set, and the scheduled Bellman margin is audited along the resulting training trajectory.

## Held-Out Performance

| task | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | paired_delta_plain | paired_delta_shifted | pulse_energy_mean | shifted_selected_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0 | 50 | 0.999402 | 0.999402 | 1.11022e-16 | 0.000155022 | 0.000213923 | 5.2425 | 0.141414 |
| H | 0.02 | 50 | 0.999318 | 0.998992 | 0.000186898 | 0.000153702 | 0.000256072 | 5.2425 | 0.141414 |
| H | 0.05 | 50 | 0.99899 | 0.997946 | 0.000516849 | 0.000184381 | 0.000436335 | 5.2425 | 0.141414 |
| H | 0.08 | 50 | 0.998396 | 0.99637 | 0.000953723 | 0.000240268 | 0.000739279 | 5.2425 | 0.141414 |
| Z | 0 | 50 | 0.999071 | 0.999071 | 0 | -5.49558e-05 | 0.000433687 | 4.705 | 0.131313 |
| Z | 0.02 | 50 | 0.998954 | 0.998298 | 0.000312203 | -8.03408e-05 | 0.000565819 | 4.705 | 0.131313 |
| Z | 0.05 | 50 | 0.998314 | 0.995882 | 0.00104307 | -0.000209125 | 0.000867885 | 4.705 | 0.131313 |
| Z | 0.08 | 50 | 0.997141 | 0.991973 | 0.00217514 | -0.000440962 | 0.00133186 | 4.705 | 0.131313 |

## Scheduled Terminal-Value Margins

| task | audited_steps | terminal_outside_steps | positive_margin_fraction | positive_terminal_outside_fraction | certified_epsilon_min_outside | margin_mean | margin_min | terminal_phi_mean | terminal_phi_min | max_bellman_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 99 | 93 | 0.939394 | 1 | 3.25313e-05 | 0.00155078 | 0 | 0.00252784 | 0.000448775 | 8.67e-19 |
| Z | 99 | 99 | 1 | 1 | 0.000116004 | 0.00332589 | 0.000116004 | 0.00432589 | 0.001116 | 8.67e-19 |
