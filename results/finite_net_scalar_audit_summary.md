# Scalar Finite-Net Audit

Dense scalar disorder-strength grid for fixed held-out disorder directions. This is a deterministic one-dimensional grid audit, not a cover of all disorder directions.

## Task Summary

| task | directions | grid_points | fidelity_min_all | fidelity_mean_at_max_strength | fidelity_min_at_max_strength | max_empirical_infidelity_slope | pulse_energy | design_seconds | resource_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 5 | 25 | 0.990849 | 0.99568 | 0.990849 | 0.129836 | 5.6075 | 129.8 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam |
| Z | 5 | 25 | 0.991114 | 0.99535 | 0.991114 | 0.107357 | 4.9225 | 97.83 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam |

## Direction Summary

| task | seed | strength_min | strength_max | grid_points | grid_radius | fidelity_min | fidelity_at_max_strength | max_adjacent_infidelity_slope | mean_adjacent_infidelity_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | 10 | 0 | 0.12 | 25 | 0.0025 | 0.999176 | 0.999176 | 0.0089364 | 0.00436573 |
| H | 13 | 0 | 0.12 | 25 | 0.0025 | 0.991093 | 0.991093 | 0.129836 | 0.0679456 |
| H | 20 | 0 | 0.12 | 25 | 0.0025 | 0.997626 | 0.997626 | 0.0200018 | 0.0135085 |
| H | 37 | 0 | 0.12 | 25 | 0.0025 | 0.999247 | 0.999658 | 0.00913199 | 0.00342884 |
| H | 50 | 0 | 0.12 | 25 | 0.0025 | 0.990849 | 0.990849 | 0.122 | 0.0699794 |
| Z | 10 | 0 | 0.12 | 25 | 0.0025 | 0.996529 | 0.996529 | 0.0391358 | 0.0216447 |
| Z | 13 | 0 | 0.12 | 25 | 0.0025 | 0.991114 | 0.991114 | 0.107357 | 0.0667706 |
| Z | 20 | 0 | 0.12 | 25 | 0.0025 | 0.997171 | 0.997171 | 0.0339962 | 0.0174381 |
| Z | 37 | 0 | 0.12 | 25 | 0.0025 | 0.996757 | 0.996757 | 0.0374706 | 0.0197434 |
| Z | 50 | 0 | 0.12 | 25 | 0.0025 | 0.995181 | 0.995181 | 0.0482715 | 0.0328783 |
