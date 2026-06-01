# Adaptive Terminal-Fallback Audit Summary

The realized shifted-fallback trajectory is kept fixed. The adaptive row estimates the local one-segment terminal-score gradient at the predicted terminal ensemble state and line-searches along the descent direction inside the same amplitude cap. This is a certificate-design audit, not a replacement held-out controller result.

| task | method | audited_steps | positive_margin_fraction | terminal_outside_steps | positive_terminal_outside_fraction | margin_mean | margin_median | margin_min | margin_max | max_tail_phi_abs_error | line_radius_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | adaptive_gradient_line | 99 | 0.131313 | 93 | 0.129032 | -0.000240201 | -0.000262821 | -0.00062318 | 0.000155784 | 0 | 0 |
| H | default_alphabet | 99 | 0.141414 | 93 | 0.139785 | -0.000229381 | -0.000252991 | -0.00062318 | 0.000155784 | 0 | nan |
| Z | adaptive_gradient_line | 99 | 0.161616 | 99 | 0.161616 | -0.0002877 | -0.000282364 | -0.000880171 | 0.00028536 | 0 | 0 |
| Z | default_alphabet | 99 | 0.161616 | 99 | 0.161616 | -0.000278002 | -0.000282364 | -0.000880171 | 0.000339939 | 0 | nan |
