# Terminal-Value Extension Audit Summary

This postprocesses the multi-step terminal-fallback audit. A positive extension margin means that the best longer terminal block has a lower terminal Lyapunov score than the best shorter block on the same realized terminal state. It is a diagnostic for whether a short-block terminal-value surrogate is itself contractive.

| task | comparison | audited_steps | positive_extension_fraction | extension_margin_mean | extension_margin_median | extension_margin_min | extension_margin_max | shorter_phi_mean | longer_phi_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 1_to_2 | 49 | 0.204082 | -0.000152028 | -0.000179957 | -0.00075604 | 0.000688881 | 0.00297145 | 0.00312348 |
| H | 1_to_3 | 49 | 0.122449 | -0.000381216 | -0.000423727 | -0.00141642 | 0.000928984 | 0.00297145 | 0.00335266 |
| H | 2_to_3 | 49 | 0.102041 | -0.000229188 | -0.000246825 | -0.000660385 | 0.000240103 | 0.00312348 | 0.00335266 |
| Z | 1_to_2 | 49 | 0.244898 | -0.000207669 | -0.000238439 | -0.000935823 | 0.000376879 | 0.00471912 | 0.00492679 |
| Z | 1_to_3 | 49 | 0.22449 | -0.000483142 | -0.000489452 | -0.0018263 | 0.000658788 | 0.00471912 | 0.00520226 |
| Z | 2_to_3 | 49 | 0.163265 | -0.000275473 | -0.000268476 | -0.000922974 | 0.000281908 | 0.00492679 | 0.00520226 |
