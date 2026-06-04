# Operator-Net Trained Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon directly on an explicit Pauli coefficient net using two-level Bloch-vector propagation, then evaluates the resulting pulse on the standard held-out seeds and on a denser operator net. It tests the direct-net training route; it is not a new fixed-depth theorem.

## Held-Out Performance

| task | eval_strength | n | score_mode | train_net_points | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0 | 50 | max_net | 2145 | 0.999591 | 0.999591 | 0 | 5.0475 | 0.171717 | 170.9 |
| H | 0.02 | 50 | max_net | 2145 | 0.999529 | 0.999359 | 9.19828e-05 | 5.0475 | 0.171717 | 170.9 |
| H | 0.05 | 50 | max_net | 2145 | 0.999234 | 0.99853 | 0.000323666 | 5.0475 | 0.171717 | 170.9 |
| H | 0.08 | 50 | max_net | 2145 | 0.998702 | 0.996999 | 0.000703896 | 5.0475 | 0.171717 | 170.9 |
| Z | 0 | 50 | max_net | 2145 | 0.998703 | 0.998703 | 1.11022e-16 | 4.285 | 0.0808081 | 146 |
| Z | 0.02 | 50 | max_net | 2145 | 0.998362 | 0.997306 | 0.000612048 | 4.285 | 0.0808081 | 146 |
| Z | 0.05 | 50 | max_net | 2145 | 0.996932 | 0.993459 | 0.00181826 | 4.285 | 0.0808081 | 146 |
| Z | 0.08 | 50 | max_net | 2145 | 0.994344 | 0.987445 | 0.00351743 | 4.285 | 0.0808081 | 146 |

## Operator-Net Performance

| task | net_label | score_mode | net_points | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | eval_net | max_net | 2145 | 13 | 0.011547 | 0.996974 | 0.998839 | 0.00302563 | 5.0475 | 0.171717 | 170.9 |
| H | train_net | max_net | 2145 | 13 | 0.011547 | 0.996974 | 0.998839 | 0.00302563 | 5.0475 | 0.171717 | 170.9 |
| Z | eval_net | max_net | 2145 | 13 | 0.011547 | 0.986928 | 0.995574 | 0.0130716 | 4.285 | 0.0808081 | 146 |
| Z | train_net | max_net | 2145 | 13 | 0.011547 | 0.986928 | 0.995574 | 0.0130716 | 4.285 | 0.0808081 | 146 |

## Scheduled Margin Audit

| task | score_mode | train_net_points | eval_net_points | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | max_net | 2145 | 2145 | 99 | 99 | 99 | 1 | 1 | 0.000556819 | 0.000556819 | 0.00306019 | 0.000556819 |
| Z | max_net | 2145 | 2145 | 99 | 99 | 99 | 1 | 1 | 4.88998e-05 | 4.88998e-05 | 0.00768585 | 4.88998e-05 |
