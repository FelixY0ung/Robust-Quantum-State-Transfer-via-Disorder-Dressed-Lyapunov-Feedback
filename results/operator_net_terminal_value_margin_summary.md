# Operator-Net Terminal-Value Margin Audit

This audit propagates the explicit Pauli-coefficient h-net in parallel with the terminal-value shifted-horizon controller and reports the scheduled Bellman margin on the shifted-tail terminal states. The margin is the energy-free terminal-value identity `[Phi-tau]_+`; the training-ensemble terminal-value audit verifies the same identity by explicit successor scoring. The `max_net` score mode uses worst net infidelity as Phi, while `mean_worst` uses the controller-style mean-plus-worst score.

| task | score_mode | net_points | ball_radius | points_per_axis | covering_radius_h | audited_steps | terminal_outside_steps | positive_terminal_outside_fraction | certified_epsilon_min_outside | margin_mean | margin_min | terminal_score_mean | terminal_score_min | final_worst_net_fidelity | final_mean_net_fidelity | shifted_selected_fraction | audit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | max_net | 2145 | 0.08 | 13 | 0.011547 | 99 | 95 | 1 | 0.000292932 | 0.00407419 | 0 | 0.00506337 | 0.000572185 | 0.996117 | 0.998708 | 0.141414 | 363.6 |
| H | mean_worst | 2145 | 0.08 | 13 | 0.011547 | 99 | 93 | 1 | 8.70005e-05 | 0.00223497 | 0 | 0.00321749 | 0.000527401 | 0.996117 | 0.998708 | 0.141414 | 363.6 |
| Z | max_net | 2145 | 0.08 | 13 | 0.011547 | 99 | 99 | 1 | 0.000780183 | 0.00683261 | 0.000780183 | 0.00783261 | 0.00178018 | 0.991103 | 0.997275 | 0.131313 | 324.9 |
| Z | mean_worst | 2145 | 0.08 | 13 | 0.011547 | 99 | 99 | 1 | 0.000472507 | 0.00393287 | 0.000472507 | 0.00493287 | 0.00147251 | 0.991103 | 0.997275 | 0.131313 | 324.9 |
