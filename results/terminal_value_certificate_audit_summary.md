# Terminal-Value Certificate Audit Summary

This audit tests a certificate-active terminal score for the shifted-fallback theorem. The score is a finite-set fallback value `G_{j,L}` with running cost `[Phi-tau]_+` and terminal cost `Phi`. For exact propagation, the Bellman identity makes the best one-segment shifted-fallback margin equal to the positive running cost whenever the terminal state lies outside `Phi <= tau`. This is an executable terminal-score certificate audit, not yet a new closed-loop controller result.

| task | value_depth | audited_steps | terminal_outside_steps | positive_margin_fraction | positive_terminal_outside_fraction | certified_epsilon_min_outside | margin_mean | margin_median | margin_min | margin_max | terminal_phi_mean | terminal_phi_min | max_bellman_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 1 | 99 | 93 | 0.939394 | 1 | 2.89492e-05 | 0.00177638 | 0.00185394 | 0 | 0.00450473 | 0.00275264 | 0.000448775 | 8.67e-19 |
| H | 2 | 99 | 93 | 0.939394 | 1 | 2.89492e-05 | 0.00177638 | 0.00185394 | 0 | 0.00450473 | 0.00275264 | 0.000448775 | 8.67e-19 |
| Z | 1 | 99 | 99 | 1 | 1 | 0.000274843 | 0.00344742 | 0.00368353 | 0.000274843 | 0.00721561 | 0.00444742 | 0.00127484 | 8.67e-19 |
| Z | 2 | 99 | 99 | 1 | 1 | 0.000274843 | 0.00344742 | 0.00368353 | 0.000274843 | 0.00721561 | 0.00444742 | 0.00127484 | 8.67e-19 |
