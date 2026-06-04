# Operator-Net Training Strategy Comparison

Postprocessed comparison of direct coarse-net, adaptive hard-point, and full 2145-point Pauli-ball terminal-value horizon designs. Best rows are selected by worst fidelity on the deterministic 2145-point certificate net. This is a design-selection diagnostic, not an independent distribution-free generalization claim.

## Best Worst-Net Rows

| task | method | train_net_points | hard_points_used | worst_net_fidelity | mean_net_fidelity | heldout_delta_0_08_mean | heldout_delta_0_08_min | eval_min_outside_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | full-2145-max_net | 2145 | 0 | 0.996974 | 0.998839 | 0.998702 | 0.996999 | 0.000556819 |
| Z | adaptive_w0.25-hard_augmented-mean_worst | 547 | 212 | 0.996302 | 0.997958 | 0.997806 | 0.996333 | 0.00131953 |

## All Strategy Rows

| task | method | worst_net_fidelity | mean_net_fidelity | heldout_delta_0_08_mean | heldout_delta_0_08_min | eval_positive_outside_fraction | eval_min_outside_margin | pulse_energy | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | full-2145-max_net | 0.996974 | 0.998839 | 0.998702 | 0.996999 | 1 | 0.000556819 | 5.0475 | 170.909 |
| H | full-2145-mean_worst | 0.995533 | 0.998115 | 0.998027 | 0.995822 | 1 | 3.61824e-05 | 5.7525 | 170.564 |
| H | adaptive_w0.25-coarse-mean_worst | 0.991961 | 0.997424 | 0.997205 | 0.991988 | 1 | 8.10763e-05 | 5.72 | 24.5479 |
| H | adaptive_w0.25-hard_augmented-mean_worst | 0.991796 | 0.997439 | 0.997082 | 0.99306 | 1 | 0.000454107 | 5.3675 | 38.3535 |
| H | adaptive_w0.5-hard_augmented-mean_worst | 0.990715 | 0.997445 | 0.997026 | 0.991017 | 1 | 0.00059592 | 5.6175 | 38.795 |
| H | adaptive_w0.5-coarse-mean_worst | 0.990701 | 0.997541 | 0.99679 | 0.990701 | 1 | 0.000859704 | 5.4275 | 24.711 |
| H | adaptive_w0.25-coarse-max_net | 0.987202 | 0.996591 | 0.99557 | 0.987982 | 1 | 0.000734612 | 4.585 | 24.4863 |
| H | adaptive_w0.25-hard_augmented-max_net | 0.983425 | 0.99592 | 0.995632 | 0.986655 | 1 | 0.000734612 | 4.865 | 39.1288 |
| Z | adaptive_w0.25-hard_augmented-mean_worst | 0.996302 | 0.997958 | 0.997806 | 0.996333 | 1 | 0.00131953 | 4.1775 | 32.3351 |
| Z | adaptive_w0.25-coarse-max_net | 0.993921 | 0.997097 | 0.996733 | 0.993934 | 1 | 3.12675e-05 | 4.4275 | 22.0942 |
| Z | adaptive_w0.25-hard_augmented-max_net | 0.993921 | 0.997097 | 0.996733 | 0.993934 | 1 | 3.12675e-05 | 4.4275 | 32.7529 |
| Z | adaptive_w0.5-hard_augmented-mean_worst | 0.988764 | 0.996207 | 0.995349 | 0.988981 | 1 | 0.00203242 | 4.775 | 32.347 |
| Z | full-2145-max_net | 0.986928 | 0.995574 | 0.994344 | 0.987445 | 1 | 4.88998e-05 | 4.285 | 146.049 |
| Z | full-2145-mean_worst | 0.985665 | 0.995403 | 0.994788 | 0.985677 | 1 | 0.000831988 | 3.71 | 146.512 |
| Z | adaptive_w0.5-coarse-mean_worst | 0.985564 | 0.995531 | 0.994414 | 0.985742 | 1 | 0.00140678 | 3.66 | 21.616 |
| Z | adaptive_w0.25-coarse-mean_worst | 0.984793 | 0.995142 | 0.99399 | 0.985394 | 1 | 0.000770138 | 3.63 | 20.777 |
