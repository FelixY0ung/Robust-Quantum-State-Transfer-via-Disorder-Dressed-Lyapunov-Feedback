# Open-System Adjoint Horizon Summary

| task | controller | eval_noise_case | n | final_fidelity_mean | final_fidelity_min | final_fidelity_ci95 | final_purity_mean | pulse_energy_mean | segments | horizon_steps | reference_iterations | reference_success | horizon_iterations | horizon_success | reference_seconds | horizon_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | adjoint_open_horizon | combined | 50 | 0.977779 | 0.973761 | 0.000638621 | 0.96313 | 0.248281 | 40 | 4 | 100 | False | 318 | False | 34.960 | 2.429 |
| H | adjoint_open_horizon | deph_0.005 | 50 | 0.993018 | 0.988852 | 0.000657315 | 0.992915 | 0.248281 | 40 | 4 | 100 | False | 318 | False | 34.960 | 2.429 |
| H | adjoint_open_horizon | relax_0.002 | 50 | 0.979813 | 0.975616 | 0.000618229 | 0.967818 | 0.248281 | 40 | 4 | 100 | False | 318 | False | 34.960 | 2.429 |
| H | adjoint_open_horizon | static_only | 50 | 0.99511 | 0.990844 | 0.000634683 | 0.997854 | 0.248281 | 40 | 4 | 100 | False | 318 | False | 34.960 | 2.429 |
| H | open_grape_reference | combined | 50 | 0.978929 | 0.975951 | 0.000430871 | 0.962201 | 0.326679 | 40 | 0 | 100 | False | 0 | True | 34.960 | 0.000 |
| H | open_grape_reference | deph_0.005 | 50 | 0.993936 | 0.99079 | 0.000452118 | 0.991496 | 0.326679 | 40 | 0 | 100 | False | 0 | True | 34.960 | 0.000 |
| H | open_grape_reference | relax_0.002 | 50 | 0.982128 | 0.979552 | 0.00038182 | 0.968792 | 0.326679 | 40 | 0 | 100 | False | 0 | True | 34.960 | 0.000 |
| H | open_grape_reference | static_only | 50 | 0.997222 | 0.994589 | 0.000399787 | 0.998458 | 0.326679 | 40 | 0 | 100 | False | 0 | True | 34.960 | 0.000 |
| Z | adjoint_open_horizon | combined | 50 | 0.974684 | 0.969816 | 0.000623996 | 0.956776 | 0.600226 | 40 | 4 | 100 | False | 317 | False | 31.537 | 2.312 |
| Z | adjoint_open_horizon | deph_0.005 | 50 | 0.974487 | 0.96956 | 0.000630485 | 0.956456 | 0.600226 | 40 | 4 | 100 | False | 317 | False | 31.537 | 2.312 |
| Z | adjoint_open_horizon | relax_0.002 | 50 | 0.976831 | 0.972175 | 0.00060707 | 0.961461 | 0.600226 | 40 | 4 | 100 | False | 317 | False | 31.537 | 2.312 |
| Z | adjoint_open_horizon | static_only | 50 | 0.976654 | 0.971967 | 0.000612922 | 0.961187 | 0.600226 | 40 | 4 | 100 | False | 317 | False | 31.537 | 2.312 |
| Z | open_grape_reference | combined | 50 | 0.977461 | 0.973611 | 0.000545563 | 0.960609 | 0.63968 | 40 | 0 | 100 | False | 0 | True | 31.537 | 0.000 |
| Z | open_grape_reference | deph_0.005 | 50 | 0.977301 | 0.973407 | 0.000550954 | 0.960347 | 0.63968 | 40 | 0 | 100 | False | 0 | True | 31.537 | 0.000 |
| Z | open_grape_reference | relax_0.002 | 50 | 0.980339 | 0.97665 | 0.000524134 | 0.966574 | 0.63968 | 40 | 0 | 100 | False | 0 | True | 31.537 | 0.000 |
| Z | open_grape_reference | static_only | 50 | 0.980206 | 0.976473 | 0.000529003 | 0.966368 | 0.63968 | 40 | 0 | 100 | False | 0 | True | 31.537 | 0.000 |
