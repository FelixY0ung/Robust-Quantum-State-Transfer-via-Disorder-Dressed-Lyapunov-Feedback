# Fine Adaptive Terminal-Fallback Audit Summary

The realized shifted-fallback trajectory is kept fixed. The fine adaptive row uses dense line-search radii near zero before falling back to the larger radii used in the coarse adaptive audit. This tests whether local terminal descent exists but is too small for the coarse amplitude grid.

| task | method | audited_steps | positive_margin_fraction | terminal_outside_steps | positive_terminal_outside_fraction | margin_mean | margin_median | margin_min | margin_max | max_tail_phi_abs_error | line_radius_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | default_alphabet | 99 | 0.141414 | 93 | 0.139785 | -0.000229381 | -0.000252991 | -0.00062318 | 0.000155784 | 0 | nan |
| H | fine_adaptive_gradient_line | 99 | 0.151515 | 93 | 0.150538 | -0.000214821 | -0.000240072 | -0.00062318 | 0.0002062 | 0 | 0.0253535 |
| Z | default_alphabet | 99 | 0.161616 | 99 | 0.161616 | -0.000278002 | -0.000282364 | -0.000880171 | 0.000339939 | 0 | nan |
| Z | fine_adaptive_gradient_line | 99 | 0.171717 | 99 | 0.171717 | -0.00026455 | -0.00026964 | -0.000857025 | 0.000294478 | 0 | 0.0223232 |
