# Fixed-Depth Terminal-Value Margin Audit

This audit tests the stronger, unsupported fixed-depth condition `G_{j,L}(z) > G_{j+1,L}(F_k z)` on shifted-tail terminal states. The scheduled certificate used in the manuscript instead checks `G_{j,L}(z) > G_{j+1,L-1}(F_k z)`, whose Bellman margin is `[Phi(z)-tau]_+`. Negative fixed-depth rows therefore do not contradict the scheduled certificate; they show why the current score/controller should not be claimed as a fixed-depth all-time value theorem.

| task | controller | value_depth | audit_stride | audited_steps | terminal_outside_steps | fixed_positive_fraction | fixed_positive_outside_fraction | fixed_min_outside | fixed_mean | fixed_min | scheduled_positive_outside_fraction | scheduled_min_outside | tail_phi_mean | tail_phi_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | terminal_value_shifted | 1 | 10 | 9 | 8 | 0 | 0 | -0.00121832 | -0.00053128 | -0.00121832 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 10 | 9 | 8 | 0 | 0 | -0.00195169 | -0.000889392 | -0.00195169 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| Z | terminal_value_shifted | 1 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00153259 | -0.000403227 | -0.00153259 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.00245111 | -0.000686482 | -0.00245111 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
