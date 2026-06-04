# Operator-Net Trained Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon directly on an explicit Pauli coefficient net using two-level Bloch-vector propagation, then evaluates the resulting pulse on the standard held-out seeds and on a denser operator net. It tests the direct-net training route; it is not a new fixed-depth theorem.

## Held-Out Performance

| task | eval_strength | n | score_mode | train_net_points | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0 | 50 | mean_worst | 335 | 0.999108 | 0.999108 | 1.11022e-16 | 5.72 | 0.20202 | 24.63 |
| H | 0.02 | 50 | mean_worst | 335 | 0.998978 | 0.998122 | 0.000418841 | 5.72 | 0.20202 | 24.63 |
| H | 0.05 | 50 | mean_worst | 335 | 0.998352 | 0.995671 | 0.00118865 | 5.72 | 0.20202 | 24.63 |
| H | 0.08 | 50 | mean_worst | 335 | 0.997205 | 0.991988 | 0.00221151 | 5.72 | 0.20202 | 24.63 |
| Z | 0 | 50 | mean_worst | 335 | 0.998844 | 0.998844 | 2.22045e-16 | 3.63 | 0.242424 | 21.79 |
| Z | 0.02 | 50 | mean_worst | 335 | 0.9985 | 0.997556 | 0.000469595 | 3.63 | 0.242424 | 21.79 |
| Z | 0.05 | 50 | mean_worst | 335 | 0.996903 | 0.99306 | 0.001729 | 3.63 | 0.242424 | 21.79 |
| Z | 0.08 | 50 | mean_worst | 335 | 0.99399 | 0.985394 | 0.00387005 | 3.63 | 0.242424 | 21.79 |

## Operator-Net Performance

| task | net_label | score_mode | net_points | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | eval_net | mean_worst | 2145 | 13 | 0.011547 | 0.991961 | 0.997424 | 0.00803875 | 5.72 | 0.20202 | 24.63 |
| H | train_net | mean_worst | 335 | 7 | 0.023094 | 0.991967 | 0.997349 | 0.00803281 | 5.72 | 0.20202 | 24.63 |
| Z | eval_net | mean_worst | 2145 | 13 | 0.011547 | 0.984793 | 0.995142 | 0.0152065 | 3.63 | 0.242424 | 21.79 |
| Z | train_net | mean_worst | 335 | 7 | 0.023094 | 0.985208 | 0.99498 | 0.0147921 | 3.63 | 0.242424 | 21.79 |

## Scheduled Margin Audit

| task | score_mode | train_net_points | eval_net_points | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | mean_worst | 335 | 2145 | 99 | 88 | 88 | 1 | 1 | 9.67849e-05 | 8.10763e-05 | 0.00165531 | 0 |
| Z | mean_worst | 335 | 2145 | 99 | 99 | 99 | 1 | 1 | 0.000791865 | 0.000770138 | 0.0053348 | 0.000770138 |
