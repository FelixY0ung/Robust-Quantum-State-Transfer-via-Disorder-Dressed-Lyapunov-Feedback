# Terminal-Score Candidate Audit Summary

This audit tests the terminal-score replacement proposition on the realized shifted-fallback trajectory. The candidate score is `G_beta(z)=Phi(z)-beta*max(0, Phi(z)-min_k Phi(F_k(z)))`. `beta=0` recovers the raw terminal Lyapunov score, while `beta=1` is the one-step terminal-value score. A positive margin means that some appended fallback decreases this replacement score at the previous predicted terminal ensemble state. The audit uses every 4th post-initial control step to keep the nested successor scoring tractable. The result is a certificate-design diagnostic, not a replacement controller run.

| task | beta | audited_steps | positive_margin_fraction | terminal_outside_steps | positive_terminal_outside_fraction | g_margin_mean | g_margin_median | g_margin_min | g_margin_max | g_terminal_mean | availability_mean | nonnegative_fraction | max_tail_phi_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 0.00 | 24 | 0.125 | 22 | 0.0909091 | -0.000230543 | -0.000251758 | -0.000518561 | 0.000155784 | 0.00278753 | 1.36253e-05 | 1 | 0 |
| H | 0.25 | 24 | 0.125 | 22 | 0.0909091 | -0.000231772 | -0.000251758 | -0.000518561 | 0.000158274 | 0.00278412 | 1.36253e-05 | 1 | 0 |
| H | 0.50 | 24 | 0.125 | 22 | 0.0909091 | -0.000233 | -0.000251758 | -0.000518561 | 0.000160765 | 0.00278072 | 1.36253e-05 | 1 | 0 |
| H | 0.75 | 24 | 0.125 | 22 | 0.0909091 | -0.0002342 | -0.000251758 | -0.000518561 | 0.000163952 | 0.00277731 | 1.36253e-05 | 1 | 0 |
| H | 1.00 | 24 | 0.166667 | 22 | 0.181818 | -0.000164067 | -0.000201548 | -0.000505824 | 0.000513157 | 0.00277391 | 1.36253e-05 | 1 | 0 |
| Z | 0.00 | 24 | 0.166667 | 24 | 0.166667 | -0.000260178 | -0.000266035 | -0.000880171 | 0.000154055 | 0.0044126 | 1.63512e-05 | 1 | 0 |
| Z | 0.25 | 24 | 0.166667 | 24 | 0.166667 | -0.000263561 | -0.000266035 | -0.000880171 | 0.000131037 | 0.00440852 | 1.63512e-05 | 1 | 0 |
| Z | 0.50 | 24 | 0.166667 | 24 | 0.166667 | -0.000266945 | -0.000266035 | -0.000880171 | 0.000108019 | 0.00440443 | 1.63512e-05 | 1 | 0 |
| Z | 0.75 | 24 | 0.166667 | 24 | 0.166667 | -0.000263818 | -0.000266035 | -0.000880171 | 0.000141996 | 0.00440034 | 1.63512e-05 | 1 | 0 |
| Z | 1.00 | 24 | 0.208333 | 24 | 0.208333 | -0.000205659 | -0.000266035 | -0.000880171 | 0.000498328 | 0.00439625 | 1.63512e-05 | 1 | 0 |
