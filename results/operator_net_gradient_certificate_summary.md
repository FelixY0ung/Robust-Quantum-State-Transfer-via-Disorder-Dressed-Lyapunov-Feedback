# Operator-Net Gradient Certificate Audit

First-order terminal-infidelity sensitivities are propagated on the same Pauli-ball h-net used by the operator-space finite-net audit. The certified sensitivity bound is `max_net ||grad ell|| + 4T^2 h`; the older `2T` bound is shown for comparison.

| task | method | net_points | ball_radius | points_per_axis | covering_radius_h | worst_net_fidelity | mean_net_fidelity | worst_net_infidelity | gradient_norm_mean | gradient_norm_p95 | gradient_norm_p99 | gradient_norm_max | hessian_bound_4T2 | gradient_net_lipschitz_bound | gradient_net_h_penalty | gradient_net_continuous_infidelity_bound | gradient_net_continuous_fidelity_lower_bound | analytic_lipschitz_2T | analytic_h_penalty | analytic_continuous_fidelity_lower_bound | pulse_energy | design_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | terminal_value_shifted | 2145 | 0.08 | 13 | 0.011547 | 0.996117 | 0.998708 | 0.0038828 | 0.0280475 | 0.0578236 | 0.0634231 | 0.0672411 | 256 | 3.02327 | 0.0349098 | 0.0387926 | 0.961207 | 16 | 0.184752 | 0.811365 | 5.2425 | 309 |
| Z | terminal_value_shifted | 2145 | 0.08 | 13 | 0.011547 | 0.991103 | 0.997275 | 0.00889709 | 0.0745115 | 0.14346 | 0.159509 | 0.172701 | 256 | 3.12873 | 0.0361275 | 0.0450246 | 0.954975 | 16 | 0.184752 | 0.806351 | 4.705 | 271 |
