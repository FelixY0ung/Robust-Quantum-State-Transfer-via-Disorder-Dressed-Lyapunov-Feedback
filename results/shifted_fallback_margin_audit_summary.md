# Shifted-Fallback Margin Audit Summary

Margins are computed on the training ensemble along the implemented shifted-fallback trajectory. Positive margin means that at least one appended fallback satisfies the proposition's terminal-progress inequality at that step.

| task | audited_steps | positive_margin_fraction | outside_residual_steps | positive_outside_residual_fraction | margin_mean | margin_median | margin_min | margin_p10 | margin_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 99 | 0.141414 | 93 | 0.129032 | -0.000229381 | -0.000252991 | -0.00062318 | -0.000415038 | 0.000155784 |
| Z | 99 | 0.161616 | 99 | 0.161616 | -0.000278002 | -0.000282364 | -0.000880171 | -0.000578302 | 0.000339939 |
