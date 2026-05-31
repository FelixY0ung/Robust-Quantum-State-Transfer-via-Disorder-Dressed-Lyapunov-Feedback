# Process-Adjoint Horizon Summary

| task | controller | n | state_fidelity_mean | state_fidelity_min | state_fidelity_ci95 | avg_gate_fidelity_mean | avg_gate_fidelity_min | avg_gate_fidelity_ci95 | pulse_energy_mean | segments | horizon_steps | reference_iterations | reference_success | horizon_iterations | horizon_success | reference_seconds | horizon_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | adjoint_process_horizon | 50 | 0.999744 | 0.998956 | 6.47198e-05 | 0.999717 | 0.999258 | 5.15302e-05 | 3.02681 | 60 | 4 | 120 | False | 479 | False | 125.942 | 15.262 |
| H | process_grape_reference | 50 | 0.999769 | 0.999179 | 5.15911e-05 | 0.999742 | 0.999325 | 4.84915e-05 | 3.06649 | 60 | 0 | 120 | False | 0 | True | 125.942 | 0.000 |
| Z | adjoint_process_horizon | 50 | 0.9987 | 0.997116 | 0.000199497 | 0.994641 | 0.98493 | 0.00111059 | 1.40135 | 60 | 4 | 13 | True | 474 | False | 39.993 | 13.531 |
| Z | process_grape_reference | 50 | 0.998935 | 0.997151 | 0.000177503 | 0.994732 | 0.985022 | 0.00111402 | 1.40241 | 60 | 0 | 13 | True | 0 | True | 39.993 | 0.000 |
