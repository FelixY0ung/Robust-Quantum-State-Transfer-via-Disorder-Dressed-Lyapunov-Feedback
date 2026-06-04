# Adaptive Operator-Net Terminal-Value Horizon Summary

This diagnostic trains the terminal-value shifted-fallback horizon on a coarse Pauli coefficient net, identifies hard points on the 2145-point evaluation net, and then redesigns on the coarse net augmented by those hard points. It tests whether adaptive net enrichment can improve the direct-net controller; it is not a fixed-depth all-time theorem.

## Held-Out Performance

| task | design_label | eval_strength | n | score_mode | worst_weight | train_net_points | hard_points_used | final_fidelity_mean | final_fidelity_min | final_fidelity_std | pulse_energy_mean | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | 0 | 50 | max_net | 0.25 | 335 | 0 | 0.999328 | 0.999328 | 1.11022e-16 | 4.585 | 0.131313 | 24.49 |
| H | coarse | 0.02 | 50 | max_net | 0.25 | 335 | 0 | 0.999035 | 0.998039 | 0.000488862 | 4.585 | 0.131313 | 24.49 |
| H | coarse | 0.05 | 50 | max_net | 0.25 | 335 | 0 | 0.997785 | 0.994132 | 0.00155914 | 4.585 | 0.131313 | 24.49 |
| H | coarse | 0.08 | 50 | max_net | 0.25 | 335 | 0 | 0.99557 | 0.987982 | 0.00325108 | 4.585 | 0.131313 | 24.49 |
| H | coarse | 0 | 50 | mean_worst | 0.25 | 335 | 0 | 0.999108 | 0.999108 | 1.11022e-16 | 5.72 | 0.20202 | 24.55 |
| H | coarse | 0.02 | 50 | mean_worst | 0.25 | 335 | 0 | 0.998978 | 0.998122 | 0.000418841 | 5.72 | 0.20202 | 24.55 |
| H | coarse | 0.05 | 50 | mean_worst | 0.25 | 335 | 0 | 0.998352 | 0.995671 | 0.00118865 | 5.72 | 0.20202 | 24.55 |
| H | coarse | 0.08 | 50 | mean_worst | 0.25 | 335 | 0 | 0.997205 | 0.991988 | 0.00221151 | 5.72 | 0.20202 | 24.55 |
| H | hard_augmented | 0 | 50 | max_net | 0.25 | 549 | 214 | 0.999177 | 0.999177 | 0 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | 0.02 | 50 | max_net | 0.25 | 549 | 214 | 0.999025 | 0.997705 | 0.000572282 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | 0.05 | 50 | max_net | 0.25 | 549 | 214 | 0.997866 | 0.993452 | 0.00156968 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | 0.08 | 50 | max_net | 0.25 | 549 | 214 | 0.995632 | 0.986655 | 0.00324492 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | 0 | 50 | mean_worst | 0.25 | 547 | 212 | 0.999354 | 0.999354 | 1.11022e-16 | 5.3675 | 0.0707071 | 38.35 |
| H | hard_augmented | 0.02 | 50 | mean_worst | 0.25 | 547 | 212 | 0.999237 | 0.998749 | 0.000176143 | 5.3675 | 0.0707071 | 38.35 |
| H | hard_augmented | 0.05 | 50 | mean_worst | 0.25 | 547 | 212 | 0.998497 | 0.996622 | 0.000762484 | 5.3675 | 0.0707071 | 38.35 |
| H | hard_augmented | 0.08 | 50 | mean_worst | 0.25 | 547 | 212 | 0.997082 | 0.99306 | 0.00188539 | 5.3675 | 0.0707071 | 38.35 |
| Z | coarse | 0 | 50 | max_net | 0.25 | 335 | 0 | 0.999198 | 0.999198 | 0 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | 0.02 | 50 | max_net | 0.25 | 335 | 0 | 0.999038 | 0.998729 | 0.000152896 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | 0.05 | 50 | max_net | 0.25 | 335 | 0 | 0.998221 | 0.99698 | 0.000610529 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | 0.08 | 50 | max_net | 0.25 | 335 | 0 | 0.996733 | 0.993934 | 0.00140668 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | 0 | 50 | mean_worst | 0.25 | 335 | 0 | 0.998844 | 0.998844 | 2.22045e-16 | 3.63 | 0.242424 | 20.78 |
| Z | coarse | 0.02 | 50 | mean_worst | 0.25 | 335 | 0 | 0.9985 | 0.997556 | 0.000469595 | 3.63 | 0.242424 | 20.78 |
| Z | coarse | 0.05 | 50 | mean_worst | 0.25 | 335 | 0 | 0.996903 | 0.99306 | 0.001729 | 3.63 | 0.242424 | 20.78 |
| Z | coarse | 0.08 | 50 | mean_worst | 0.25 | 335 | 0 | 0.99399 | 0.985394 | 0.00387005 | 3.63 | 0.242424 | 20.78 |
| Z | hard_augmented | 0 | 50 | max_net | 0.25 | 545 | 210 | 0.999198 | 0.999198 | 0 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | 0.02 | 50 | max_net | 0.25 | 545 | 210 | 0.999038 | 0.998729 | 0.000152896 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | 0.05 | 50 | max_net | 0.25 | 545 | 210 | 0.998221 | 0.99698 | 0.000610529 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | 0.08 | 50 | max_net | 0.25 | 545 | 210 | 0.996733 | 0.993934 | 0.00140668 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | 0 | 50 | mean_worst | 0.25 | 547 | 212 | 0.99924 | 0.99924 | 0 | 4.1775 | 0.212121 | 32.34 |
| Z | hard_augmented | 0.02 | 50 | mean_worst | 0.25 | 547 | 212 | 0.999152 | 0.999038 | 5.18563e-05 | 4.1775 | 0.212121 | 32.34 |
| Z | hard_augmented | 0.05 | 50 | mean_worst | 0.25 | 547 | 212 | 0.998673 | 0.998088 | 0.000282808 | 4.1775 | 0.212121 | 32.34 |
| Z | hard_augmented | 0.08 | 50 | mean_worst | 0.25 | 547 | 212 | 0.997806 | 0.996333 | 0.000713238 | 4.1775 | 0.212121 | 32.34 |

## Operator-Net Performance

| task | design_label | net_label | score_mode | worst_weight | net_points | train_net_points | hard_points_used | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | pulse_energy | shifted_selected_fraction | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | eval_net | max_net | 0.25 | 2145 | 335 | 0 | 13 | 0.011547 | 0.987202 | 0.996591 | 0.0127983 | 4.585 | 0.131313 | 24.49 |
| H | coarse | train_net | max_net | 0.25 | 335 | 335 | 0 | 7 | 0.023094 | 0.987266 | 0.996472 | 0.0127344 | 4.585 | 0.131313 | 24.49 |
| H | coarse | eval_net | mean_worst | 0.25 | 2145 | 335 | 0 | 13 | 0.011547 | 0.991961 | 0.997424 | 0.00803875 | 5.72 | 0.20202 | 24.55 |
| H | coarse | train_net | mean_worst | 0.25 | 335 | 335 | 0 | 7 | 0.023094 | 0.991967 | 0.997349 | 0.00803281 | 5.72 | 0.20202 | 24.55 |
| H | hard_augmented | eval_net | max_net | 0.25 | 2145 | 549 | 214 | 13 | 0.011547 | 0.983425 | 0.99592 | 0.016575 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | train_net | max_net | 0.25 | 549 | 549 | 214 | 7 | 0.023094 | 0.983556 | 0.996591 | 0.0164435 | 4.865 | 0.0808081 | 39.13 |
| H | hard_augmented | eval_net | mean_worst | 0.25 | 2145 | 547 | 212 | 13 | 0.011547 | 0.991796 | 0.997439 | 0.00820414 | 5.3675 | 0.0707071 | 38.35 |
| H | hard_augmented | train_net | mean_worst | 0.25 | 547 | 547 | 212 | 7 | 0.023094 | 0.991906 | 0.997987 | 0.00809378 | 5.3675 | 0.0707071 | 38.35 |
| Z | coarse | eval_net | max_net | 0.25 | 2145 | 335 | 0 | 13 | 0.011547 | 0.993921 | 0.997097 | 0.00607879 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | train_net | max_net | 0.25 | 335 | 335 | 0 | 7 | 0.023094 | 0.993986 | 0.997006 | 0.00601367 | 4.4275 | 0.111111 | 22.09 |
| Z | coarse | eval_net | mean_worst | 0.25 | 2145 | 335 | 0 | 13 | 0.011547 | 0.984793 | 0.995142 | 0.0152065 | 3.63 | 0.242424 | 20.78 |
| Z | coarse | train_net | mean_worst | 0.25 | 335 | 335 | 0 | 7 | 0.023094 | 0.985208 | 0.99498 | 0.0147921 | 3.63 | 0.242424 | 20.78 |
| Z | hard_augmented | eval_net | max_net | 0.25 | 2145 | 545 | 210 | 13 | 0.011547 | 0.993921 | 0.997097 | 0.00607879 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | train_net | max_net | 0.25 | 545 | 545 | 210 | 7 | 0.023094 | 0.993921 | 0.996158 | 0.00607879 | 4.4275 | 0.111111 | 32.75 |
| Z | hard_augmented | eval_net | mean_worst | 0.25 | 2145 | 547 | 212 | 13 | 0.011547 | 0.996302 | 0.997958 | 0.00369827 | 4.1775 | 0.212121 | 32.34 |
| Z | hard_augmented | train_net | mean_worst | 0.25 | 547 | 547 | 212 | 7 | 0.023094 | 0.996345 | 0.997967 | 0.00365522 | 4.1775 | 0.212121 | 32.34 |

## Scheduled Margin Audit

| task | design_label | score_mode | worst_weight | train_net_points | eval_net_points | hard_points_used | audited_steps | train_outside_steps | eval_outside_steps | train_positive_outside_fraction | eval_positive_outside_fraction | train_min_outside_margin | eval_min_outside_margin | eval_margin_mean | eval_margin_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | coarse | max_net | 0.25 | 335 | 2145 | 0 | 99 | 99 | 99 | 1 | 1 | 0.000716775 | 0.000734612 | 0.00584715 | 0.000734612 |
| H | coarse | mean_worst | 0.25 | 335 | 2145 | 0 | 99 | 88 | 88 | 1 | 1 | 9.67849e-05 | 8.10763e-05 | 0.00165531 | 0 |
| H | hard_augmented | max_net | 0.25 | 549 | 2145 | 214 | 99 | 99 | 99 | 1 | 1 | 0.000716775 | 0.000734612 | 0.00594386 | 0.000734612 |
| H | hard_augmented | mean_worst | 0.25 | 547 | 2145 | 212 | 99 | 99 | 99 | 1 | 1 | 0.000357039 | 0.000454107 | 0.00247861 | 0.000454107 |
| Z | coarse | max_net | 0.25 | 335 | 2145 | 0 | 99 | 99 | 99 | 1 | 1 | 3.12675e-05 | 3.12675e-05 | 0.00777389 | 3.12675e-05 |
| Z | coarse | mean_worst | 0.25 | 335 | 2145 | 0 | 99 | 99 | 99 | 1 | 1 | 0.000791865 | 0.000770138 | 0.0053348 | 0.000770138 |
| Z | hard_augmented | max_net | 0.25 | 545 | 2145 | 210 | 99 | 99 | 99 | 1 | 1 | 3.12675e-05 | 3.12675e-05 | 0.00777235 | 3.12675e-05 |
| Z | hard_augmented | mean_worst | 0.25 | 547 | 2145 | 212 | 99 | 99 | 99 | 1 | 1 | 0.0012363 | 0.00131953 | 0.00717198 | 0.00131953 |
