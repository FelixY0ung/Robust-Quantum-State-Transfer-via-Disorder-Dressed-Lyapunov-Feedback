# Multi-Step Terminal-Fallback Audit Summary

The realized shifted-fallback trajectory is kept fixed. Each row evaluates whether a short bounded terminal fallback block can reduce the previous predicted terminal Lyapunov score. These blocks are terminal-region diagnostics only; they do not replace the controller or constitute the one-step shifted-fallback theorem.

| task | fallback_spec | block_steps | audited_steps | positive_margin_fraction | terminal_outside_steps | positive_terminal_outside_fraction | margin_mean | margin_median | margin_min | margin_max | raw_margin_mean | best_block_phi_mean | expanded_per_step_mean | max_tail_phi_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | one_step_exhaustive | 1 | 49 | 0.142857 | 46 | 0.130435 | -0.000223657 | -0.00025256 | -0.00062318 | 0.000155784 | -0.000223657 | 0.00297145 | 41 | 0 |
| H | three_step_beam8 | 3 | 49 | 0.102041 | 46 | 0.108696 | -0.000604873 | -0.000669568 | -0.00203961 | 0.00101125 | -0.000604873 | 0.00335266 | 697 | 0 |
| H | two_step_beam8 | 2 | 49 | 0.122449 | 46 | 0.108696 | -0.000375686 | -0.000412557 | -0.00137922 | 0.000771148 | -0.000375686 | 0.00312348 | 369 | 0 |
| Z | one_step_exhaustive | 1 | 49 | 0.163265 | 49 | 0.163265 | -0.000273594 | -0.000272472 | -0.000880171 | 0.000339939 | -0.000273594 | 0.00471912 | 41 | 0 |
| Z | three_step_beam8 | 3 | 49 | 0.22449 | 49 | 0.22449 | -0.000756736 | -0.000820363 | -0.00270647 | 0.000720171 | -0.000756736 | 0.00520226 | 697 | 0 |
| Z | two_step_beam8 | 2 | 49 | 0.204082 | 49 | 0.204082 | -0.000481262 | -0.00048063 | -0.00181599 | 0.000625299 | -0.000481262 | 0.00492679 | 369 | 0 |
