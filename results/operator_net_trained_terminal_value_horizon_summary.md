# Operator-Net Trained Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon directly on an explicit Pauli coefficient net using two-level Bloch-vector propagation, then evaluates the resulting pulse on the standard held-out seeds and on a denser operator net. It tests the direct-net training route; it is not a new fixed-depth theorem.

## Held-Out Performance

| task | eval_strength | n | score_mode | train_net_points | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0 | 50 | max_net | 335 | 0.999328 | 0.999328 | 1.11022e-16 | 4.585 | 0.131313 | 24.42 |
| H | 0.02 | 50 | max_net | 335 | 0.999035 | 0.998039 | 0.000488862 | 4.585 | 0.131313 | 24.42 |
| H | 0.05 | 50 | max_net | 335 | 0.997785 | 0.994132 | 0.00155914 | 4.585 | 0.131313 | 24.42 |
| H | 0.08 | 50 | max_net | 335 | 0.99557 | 0.987982 | 0.00325108 | 4.585 | 0.131313 | 24.42 |
| Z | 0 | 50 | max_net | 335 | 0.999198 | 0.999198 | 0 | 4.4275 | 0.111111 | 22.16 |
| Z | 0.02 | 50 | max_net | 335 | 0.999038 | 0.998729 | 0.000152896 | 4.4275 | 0.111111 | 22.16 |
| Z | 0.05 | 50 | max_net | 335 | 0.998221 | 0.99698 | 0.000610529 | 4.4275 | 0.111111 | 22.16 |
| Z | 0.08 | 50 | max_net | 335 | 0.996733 | 0.993934 | 0.00140668 | 4.4275 | 0.111111 | 22.16 |

## Operator-Net Performance

| task | net_label | score_mode | net_points | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | eval_net | max_net | 2145 | 13 | 0.011547 | 0.987202 | 0.996591 | 0.0127983 | 4.585 | 0.131313 | 24.42 |
| H | train_net | max_net | 335 | 7 | 0.023094 | 0.987266 | 0.996472 | 0.0127344 | 4.585 | 0.131313 | 24.42 |
| Z | eval_net | max_net | 2145 | 13 | 0.011547 | 0.993921 | 0.997097 | 0.00607879 | 4.4275 | 0.111111 | 22.16 |
| Z | train_net | max_net | 335 | 7 | 0.023094 | 0.993986 | 0.997006 | 0.00601367 | 4.4275 | 0.111111 | 22.16 |

## Scheduled Margin Audit

| task | score_mode | train_net_points | eval_net_points | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | max_net | 335 | 2145 | 99 | 99 | 99 | 1 | 1 | 0.000716775 | 0.000734612 | 0.00584715 | 0.000734612 |
| Z | max_net | 335 | 2145 | 99 | 99 | 99 | 1 | 1 | 3.12675e-05 | 3.12675e-05 | 0.00777389 | 3.12675e-05 |
