# Selected Strategy Fixed-Depth Audit

This audit reruns the selected finite-library Pauli-ball strategies and checks the stricter fixed-depth comparison on the 2145-point evaluation net. It is a theorem-upgrade test: negative fixed-depth rows show that the selected direct/full-net controllers still do not support a fixed-depth all-time value theorem.

| task | method | value_depth | audit_stride | audited_steps | eval_outside_steps | fixed_positive_outside_fraction | fixed_min_outside | fixed_mean | fixed_min | scheduled_positive_outside_fraction | scheduled_min_outside | eval_phi_mean | eval_phi_max | audit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | selected_full2145_max_net | 1 | 10 | 9 | 9 | 0 | -0.00147837 | -0.000886431 | -0.00147837 | 1 | 0.000743598 | 0.00425307 | 0.00622591 | 177.2 |
| Z | selected_hard_augmented_mean_worst | 1 | 10 | 9 | 9 | 0.444444 | -0.00118628 | -0.000234207 | -0.00118628 | 1 | 0.00374136 | 0.00885912 | 0.0127753 | 45.87 |
