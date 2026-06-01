# Terminal-Fallback Alphabet Sweep Summary

The realized shifted-fallback trajectory is kept fixed while the terminal fallback alphabet is varied within the same amplitude cap. This is a certificate-design audit, not a replacement held-out controller result.

| task | fallback_spec | candidate_count | audited_steps | positive_margin_fraction | terminal_outside_steps | positive_terminal_outside_fraction | margin_mean | margin_median | margin_min | margin_p10 | margin_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | default | 41 | 99 | 0.141414 | 93 | 0.139785 | -0.000229381 | -0.000252991 | -0.00062318 | -0.000415038 | 0.000155784 |
| H | dense_inrange | 97 | 99 | 0.141414 | 93 | 0.139785 | -0.000223881 | -0.000250957 | -0.00062318 | -0.000415038 | 0.000207118 |
| H | midpoint_inrange | 65 | 99 | 0.141414 | 93 | 0.139785 | -0.000229356 | -0.000252991 | -0.00062318 | -0.000415038 | 0.000155784 |
| Z | default | 41 | 99 | 0.161616 | 99 | 0.161616 | -0.000278002 | -0.000282364 | -0.000880171 | -0.000578302 | 0.000339939 |
| Z | dense_inrange | 97 | 99 | 0.161616 | 99 | 0.161616 | -0.000270653 | -0.000280519 | -0.000880171 | -0.000578302 | 0.000339939 |
| Z | midpoint_inrange | 65 | 99 | 0.161616 | 99 | 0.161616 | -0.000278002 | -0.000282364 | -0.000880171 | -0.000578302 | 0.000339939 |
