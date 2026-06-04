# Fixed-Depth Terminal-Value Margin Audit

This audit tests the stronger, unsupported fixed-depth condition `G_{j,L}(z) > G_{j+1,L}(F_k z)` on shifted-tail terminal states. The scheduled certificate used in the manuscript instead checks `G_{j,L}(z) > G_{j+1,L-1}(F_k z)`, whose Bellman margin is `[Phi(z)-tau]_+`. Negative fixed-depth rows therefore do not contradict the scheduled certificate; they show why the current score/controller should not be claimed as a fixed-depth all-time value theorem. Residual-threshold sweeps test whether a looser terminal region can make this fixed-depth condition nonvacuously positive. Rows with `stage_discount < 1` test discounted future fallback values inside the same fixed-depth comparison; they are not scheduled depth-decrement certificates.

| task | controller | value_depth | residual_threshold | stage_discount | terminal_weight | control_stage_weight | audit_stride | audited_steps | terminal_outside_steps | fixed_positive_fraction | fixed_positive_outside_fraction | fixed_min_outside | fixed_mean | fixed_min | scheduled_positive_outside_fraction | scheduled_min_outside | tail_phi_mean | tail_phi_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | terminal_value_shifted | 1 | 0.001 | 1 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00121832 | -0.00053128 | -0.00121832 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.0015 | 1 | 1 | 0 | 10 | 9 | 7 | 0 | 0 | -0.00121832 | -0.00052807 | -0.00121832 | 1 | 0.000717241 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.002 | 1 | 1 | 0 | 10 | 9 | 7 | 0 | 0 | -0.00121832 | -0.000519002 | -0.00121832 | 1 | 0.000217241 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.0025 | 1 | 1 | 0 | 10 | 9 | 5 | 0.111111 | 0 | -0.00121832 | -0.000483284 | -0.00121832 | 1 | 0.000130514 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.003 | 1 | 1 | 0 | 10 | 9 | 3 | 0.111111 | 0 | -0.00121832 | -0.000407324 | -0.00121832 | 1 | 0.000129985 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.004 | 1 | 1 | 0 | 10 | 9 | 1 | 0.111111 | 0 | -0.00121832 | -0.000285052 | -0.00121832 | 1 | 0.000969528 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.005 | 1 | 1 | 0 | 10 | 9 | 0 | 0.222222 | nan | nan | -0.000252049 | -0.00118785 | nan | nan | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.0075 | 1 | 1 | 0 | 10 | 9 | 0 | 0.222222 | nan | nan | -0.000162122 | -0.00065187 | nan | nan | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 1 | 0.01 | 1 | 1 | 0 | 10 | 9 | 0 | 0.222222 | nan | nan | -0.000161048 | -0.000642203 | nan | nan | 0.00265087 | 0.000817896 |
| Z | terminal_value_shifted | 1 | 0.001 | 1 | 1 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00153259 | -0.000403227 | -0.00153259 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.0015 | 1 | 1 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00153259 | -0.000403227 | -0.00153259 | 1 | 0.000494837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.002 | 1 | 1 | 0 | 10 | 9 | 8 | 0.222222 | 0.125 | -0.00153259 | -0.000426779 | -0.00153259 | 1 | 0.000420047 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.0025 | 1 | 1 | 0 | 10 | 9 | 7 | 0.222222 | 0.142857 | -0.00153259 | -0.000414325 | -0.00153259 | 1 | 0.000564422 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.003 | 1 | 1 | 0 | 10 | 9 | 7 | 0.222222 | 0.142857 | -0.00153259 | -0.000384614 | -0.00153259 | 1 | 6.44225e-05 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.004 | 1 | 1 | 0 | 10 | 9 | 6 | 0.222222 | 0.166667 | -0.00153259 | -0.000333885 | -0.00153259 | 1 | 0.000397373 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.005 | 1 | 1 | 0 | 10 | 9 | 3 | 0.222222 | 0.333333 | -0.00153259 | -0.00024527 | -0.00153259 | 1 | 0.000752835 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.0075 | 1 | 1 | 0 | 10 | 9 | 0 | 0.444444 | nan | nan | -0.000150448 | -0.000819656 | nan | nan | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 1 | 0.01 | 1 | 1 | 0 | 10 | 9 | 0 | 0.444444 | nan | nan | -0.000132302 | -0.000656343 | nan | nan | 0.00438166 | 0.00199484 |
