# Shifted-Fallback Margin Audit Summary

Margins are computed on the training ensemble along the implemented shifted-fallback trajectory. Positive margin means that at least one appended fallback satisfies the terminal-fallback inequality for the previous predicted terminal ensemble state. The terminal-outside columns use the previous selected terminal Lyapunov score, matching the terminal-fallback proposition.

| task | audited_steps | positive_margin_fraction | current_outside_residual_steps | positive_current_outside_fraction | terminal_outside_residual_steps | positive_terminal_outside_fraction | terminal_phi_mean | terminal_phi_median | margin_mean | margin_median | margin_min | margin_p10 | margin_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 99 | 0.141414 | 93 | 0.129032 | 93 | 0.139785 | 0.00275264 | 0.00285394 | -0.000229381 | -0.000252991 | -0.00062318 | -0.000415038 | 0.000155784 |
| Z | 99 | 0.161616 | 99 | 0.161616 | 99 | 0.161616 | 0.00444742 | 0.00468353 | -0.000278002 | -0.000282364 | -0.000880171 | -0.000578302 | 0.000339939 |
