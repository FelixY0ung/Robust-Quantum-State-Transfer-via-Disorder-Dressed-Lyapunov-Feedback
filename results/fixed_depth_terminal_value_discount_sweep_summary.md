# Fixed-Depth Terminal-Value Margin Audit

This audit tests the stronger, unsupported fixed-depth condition `G_{j,L}(z) > G_{j+1,L}(F_k z)` on shifted-tail terminal states. The scheduled certificate used in the manuscript instead checks `G_{j,L}(z) > G_{j+1,L-1}(F_k z)`, whose Bellman margin is `[Phi(z)-tau]_+`. Negative fixed-depth rows therefore do not contradict the scheduled certificate; they show why the current score/controller should not be claimed as a fixed-depth all-time value theorem. Rows with `stage_discount < 1` test discounted future fallback values inside the same fixed-depth comparison; they are not scheduled depth-decrement certificates.

| task | controller | value_depth | stage_discount | terminal_weight | control_stage_weight | audit_stride | audited_steps | terminal_outside_steps | fixed_positive_fraction | fixed_positive_outside_fraction | fixed_min_outside | fixed_mean | fixed_min | scheduled_positive_outside_fraction | scheduled_min_outside | tail_phi_mean | tail_phi_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | terminal_value_shifted | 2 | 0 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.000566451 | -0.000218398 | -0.000566451 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.000566451 | -0.000218398 | -0.000566451 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.25 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.000729418 | -0.000293689 | -0.000729418 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.25 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.000775254 | -0.000318214 | -0.000775254 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.5 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.000892386 | -0.00036898 | -0.000892386 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.5 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00107573 | -0.000465388 | -0.00107573 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.75 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00105535 | -0.000443321 | -0.00105535 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.75 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00146787 | -0.000657913 | -0.00146787 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.9 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00115313 | -0.000487723 | -0.00115313 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 0.9 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00174716 | -0.000794685 | -0.00174716 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 1 | 0 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00121832 | -0.000517324 | -0.00121832 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| H | terminal_value_shifted | 2 | 1 | 1 | 0 | 10 | 9 | 8 | 0 | 0 | -0.00195169 | -0.000889392 | -0.00195169 | 1 | 0.000348059 | 0.00265087 | 0.000817896 |
| Z | terminal_value_shifted | 2 | 0 | 0 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.000712933 | -0.000127706 | -0.000712933 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0 | 1 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.000712933 | -0.000127706 | -0.000712933 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.25 | 0 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.000917847 | -0.000196586 | -0.000917847 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.25 | 1 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.000975255 | -0.000218406 | -0.000975255 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.5 | 0 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00112276 | -0.000265466 | -0.00112276 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.5 | 1 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00135239 | -0.000352709 | -0.00135239 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.75 | 0 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00132767 | -0.000334347 | -0.00132767 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.75 | 1 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.00184434 | -0.000515387 | -0.00184434 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.9 | 0 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00145062 | -0.000375675 | -0.00145062 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 0.9 | 1 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.00219463 | -0.000617035 | -0.00219463 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 1 | 0 | 0 | 10 | 9 | 9 | 0.222222 | 0.222222 | -0.00153259 | -0.000403227 | -0.00153259 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
| Z | terminal_value_shifted | 2 | 1 | 1 | 0 | 10 | 9 | 9 | 0.333333 | 0.333333 | -0.00245111 | -0.000686482 | -0.00245111 | 1 | 0.000994837 | 0.00438166 | 0.00199484 |
