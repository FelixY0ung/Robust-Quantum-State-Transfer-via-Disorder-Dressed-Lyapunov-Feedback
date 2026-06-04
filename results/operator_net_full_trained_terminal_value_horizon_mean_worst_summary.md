# Operator-Net Trained Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon directly on an explicit Pauli coefficient net using two-level Bloch-vector propagation, then evaluates the resulting pulse on the standard held-out seeds and on a denser operator net. It tests the direct-net training route; it is not a new fixed-depth theorem.

## Held-Out Performance

| task | eval_strength | n | score_mode | train_net_points | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0 | 50 | mean_worst | 2145 | 0.999156 | 0.999156 | 1.11022e-16 | 5.7525 | 0.131313 | 170.6 |
| H | 0.02 | 50 | mean_worst | 2145 | 0.999089 | 0.998628 | 0.000243412 | 5.7525 | 0.131313 | 170.6 |
| H | 0.05 | 50 | mean_worst | 2145 | 0.998723 | 0.997499 | 0.00064877 | 5.7525 | 0.131313 | 170.6 |
| H | 0.08 | 50 | mean_worst | 2145 | 0.998027 | 0.995822 | 0.00116869 | 5.7525 | 0.131313 | 170.6 |
| Z | 0 | 50 | mean_worst | 2145 | 0.999136 | 0.999136 | 1.11022e-16 | 3.71 | 0.151515 | 146.5 |
| Z | 0.02 | 50 | mean_worst | 2145 | 0.998837 | 0.997895 | 0.000368047 | 3.71 | 0.151515 | 146.5 |
| Z | 0.05 | 50 | mean_worst | 2145 | 0.997403 | 0.993341 | 0.00152129 | 3.71 | 0.151515 | 146.5 |
| Z | 0.08 | 50 | mean_worst | 2145 | 0.994788 | 0.985677 | 0.00354865 | 3.71 | 0.151515 | 146.5 |

## Operator-Net Performance

| task | net_label | score_mode | net_points | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | eval_net | mean_worst | 2145 | 13 | 0.011547 | 0.995533 | 0.998115 | 0.00446697 | 5.7525 | 0.131313 | 170.6 |
| H | train_net | mean_worst | 2145 | 13 | 0.011547 | 0.995533 | 0.998115 | 0.00446697 | 5.7525 | 0.131313 | 170.6 |
| Z | eval_net | mean_worst | 2145 | 13 | 0.011547 | 0.985665 | 0.995403 | 0.0143353 | 3.71 | 0.151515 | 146.5 |
| Z | train_net | mean_worst | 2145 | 13 | 0.011547 | 0.985665 | 0.995403 | 0.0143353 | 3.71 | 0.151515 | 146.5 |

## Scheduled Margin Audit

| task | score_mode | train_net_points | eval_net_points | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | mean_worst | 2145 | 2145 | 99 | 91 | 91 | 1 | 1 | 3.61824e-05 | 3.61824e-05 | 0.00115043 | 0 |
| Z | mean_worst | 2145 | 2145 | 99 | 99 | 99 | 1 | 1 | 0.000831988 | 0.000831988 | 0.00517851 | 0.000831988 |
