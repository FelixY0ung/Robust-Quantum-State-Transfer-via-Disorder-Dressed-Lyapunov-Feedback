# Adaptive Operator-Net Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon on a coarse Pauli coefficient net, identifies hard points on the 2145-point evaluation net, and then redesigns on the coarse net augmented by those hard points. It tests whether adaptive net enrichment can improve the direct-net controller; it is not a fixed-depth all-time theorem.

## Held-Out Performance

| task | design_label | eval_strength | n | score_mode | worst_weight | train_net_points | hard_points_used | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | 0 | 50 | mean_worst | 0.5 | 335 | 0 | 0.999008 | 0.999008 | 0 | 5.4275 | 0.161616 | 24.71 |
| H | coarse | 0.02 | 50 | mean_worst | 0.5 | 335 | 0 | 0.998808 | 0.997888 | 0.000482961 | 5.4275 | 0.161616 | 24.71 |
| H | coarse | 0.05 | 50 | mean_worst | 0.5 | 335 | 0 | 0.998068 | 0.995031 | 0.00128709 | 5.4275 | 0.161616 | 24.71 |
| H | coarse | 0.08 | 50 | mean_worst | 0.5 | 335 | 0 | 0.99679 | 0.990701 | 0.00232648 | 5.4275 | 0.161616 | 24.71 |
| H | hard_augmented | 0 | 50 | mean_worst | 0.5 | 548 | 213 | 0.999481 | 0.999481 | 0 | 5.6175 | 0.141414 | 38.8 |
| H | hard_augmented | 0.02 | 50 | mean_worst | 0.5 | 548 | 213 | 0.999309 | 0.998606 | 0.00029981 | 5.6175 | 0.141414 | 38.8 |
| H | hard_augmented | 0.05 | 50 | mean_worst | 0.5 | 548 | 213 | 0.998496 | 0.99578 | 0.000925453 | 5.6175 | 0.141414 | 38.8 |
| H | hard_augmented | 0.08 | 50 | mean_worst | 0.5 | 548 | 213 | 0.997026 | 0.991017 | 0.00196096 | 5.6175 | 0.141414 | 38.8 |
| Z | coarse | 0 | 50 | mean_worst | 0.5 | 335 | 0 | 0.998882 | 0.998882 | 0 | 3.66 | 0.191919 | 21.62 |
| Z | coarse | 0.02 | 50 | mean_worst | 0.5 | 335 | 0 | 0.998538 | 0.997368 | 0.00062002 | 3.66 | 0.191919 | 21.62 |
| Z | coarse | 0.05 | 50 | mean_worst | 0.5 | 335 | 0 | 0.997052 | 0.99285 | 0.00201853 | 3.66 | 0.191919 | 21.62 |
| Z | coarse | 0.08 | 50 | mean_worst | 0.5 | 335 | 0 | 0.994414 | 0.985742 | 0.00411654 | 3.66 | 0.191919 | 21.62 |
| Z | hard_augmented | 0 | 50 | mean_worst | 0.5 | 549 | 214 | 0.999025 | 0.999025 | 1.11022e-16 | 4.775 | 0.0909091 | 32.35 |
| Z | hard_augmented | 0.02 | 50 | mean_worst | 0.5 | 549 | 214 | 0.998778 | 0.998243 | 0.000228467 | 4.775 | 0.0909091 | 32.35 |
| Z | hard_augmented | 0.05 | 50 | mean_worst | 0.5 | 549 | 214 | 0.99756 | 0.99489 | 0.00114362 | 4.775 | 0.0909091 | 32.35 |
| Z | hard_augmented | 0.08 | 50 | mean_worst | 0.5 | 549 | 214 | 0.995349 | 0.988981 | 0.00281629 | 4.775 | 0.0909091 | 32.35 |

## Operator-Net Performance

| task | design_label | net_label | score_mode | worst_weight | net_points | train_net_points | hard_points_used | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | eval_net | mean_worst | 0.5 | 2145 | 335 | 0 | 13 | 0.011547 | 0.990701 | 0.997541 | 0.00929886 | 5.4275 | 0.161616 | 24.71 |
| H | coarse | train_net | mean_worst | 0.5 | 335 | 335 | 0 | 7 | 0.023094 | 0.9908 | 0.997477 | 0.00920027 | 5.4275 | 0.161616 | 24.71 |
| H | hard_augmented | eval_net | mean_worst | 0.5 | 2145 | 548 | 213 | 13 | 0.011547 | 0.990715 | 0.997445 | 0.00928543 | 5.6175 | 0.141414 | 38.8 |
| H | hard_augmented | train_net | mean_worst | 0.5 | 548 | 548 | 213 | 7 | 0.023094 | 0.990734 | 0.997624 | 0.00926636 | 5.6175 | 0.141414 | 38.8 |
| Z | coarse | eval_net | mean_worst | 0.5 | 2145 | 335 | 0 | 13 | 0.011547 | 0.985564 | 0.995531 | 0.0144364 | 3.66 | 0.191919 | 21.62 |
| Z | coarse | train_net | mean_worst | 0.5 | 335 | 335 | 0 | 7 | 0.023094 | 0.985676 | 0.995385 | 0.0143242 | 3.66 | 0.191919 | 21.62 |
| Z | hard_augmented | eval_net | mean_worst | 0.5 | 2145 | 549 | 214 | 13 | 0.011547 | 0.988764 | 0.996207 | 0.0112361 | 4.775 | 0.0909091 | 32.35 |
| Z | hard_augmented | train_net | mean_worst | 0.5 | 549 | 549 | 214 | 7 | 0.023094 | 0.988764 | 0.994866 | 0.0112361 | 4.775 | 0.0909091 | 32.35 |

## Scheduled Margin Audit

| task | design_label | score_mode | worst_weight | train_net_points | eval_net_points | hard_points_used | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | mean_worst | 0.5 | 335 | 2145 | 0 | 99 | 99 | 99 | 1 | 1 | 0.000883191 | 0.000859704 | 0.00344118 | 0.000859704 |
| H | hard_augmented | mean_worst | 0.5 | 548 | 2145 | 213 | 99 | 99 | 99 | 1 | 1 | 0.000544691 | 0.00059592 | 0.00320816 | 0.00059592 |
| Z | coarse | mean_worst | 0.5 | 335 | 2145 | 0 | 99 | 99 | 99 | 1 | 1 | 0.00142009 | 0.00140678 | 0.00746689 | 0.00140678 |
| Z | hard_augmented | mean_worst | 0.5 | 549 | 2145 | 214 | 99 | 99 | 99 | 1 | 1 | 0.00196449 | 0.00203242 | 0.00602222 | 0.00203242 |
