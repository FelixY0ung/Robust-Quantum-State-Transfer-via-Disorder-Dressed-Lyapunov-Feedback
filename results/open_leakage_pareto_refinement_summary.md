# Open-Leakage Pareto Refinement

No-reference target/leakage variants trained through the five-level Lindblad leakage model.

| controller | label | noise_case | n | final_fidelity_mean | final_fidelity_min | final_fidelity_ci95 | final_leakage_mean | max_leakage_mean | final_purity_mean | pulse_energy_mean | segments | training_seconds | terminal_target_weight | leakage_weight | worst_weight | trust_radius | horizon_maxiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| open_leakage_path_seed | path seed | combined | 50 | 0.82642 | 0.521083 | 0.0251353 | 0.0423641 | 0.0466763 | 0.964078 | 0.00138437 | 120 | 66.1 |  |  |  |  |  |
| open_leakage_path_seed | path seed | deph_0.001 | 50 | 0.832238 | 0.521722 | 0.025606 | 0.0426705 | 0.0470868 | 0.981711 | 0.00138437 | 120 | 66.1 |  |  |  |  |  |
| open_leakage_path_seed | path seed | relax_0.0005 | 50 | 0.831909 | 0.520429 | 0.0256722 | 0.0418744 | 0.0464294 | 0.981852 | 0.00138437 | 120 | 66.1 |  |  |  |  |  |
| open_leakage_path_seed | path seed | static_only | 50 | 0.837761 | 0.521005 | 0.0261519 | 0.0421673 | 0.0468438 | 1 | 0.00138437 | 120 | 66.1 |  |  |  |  |  |
| pareto_alpha0p8_lw0p8_trust004 | alpha 0.8, leak 0.8, trust 0.04 | combined | 50 | 0.881557 | 0.720702 | 0.0216829 | 0.00200101 | 0.0365263 | 0.950976 | 0.00189409 | 120 | 80.29 | 0.8 | 0.8 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw0p8_trust004 | alpha 0.8, leak 0.8, trust 0.04 | deph_0.001 | 50 | 0.898471 | 0.736578 | 0.0218648 | 0.00206207 | 0.0367355 | 0.983345 | 0.00189409 | 120 | 80.29 | 0.8 | 0.8 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw0p8_trust004 | alpha 0.8, leak 0.8, trust 0.04 | relax_0.0005 | 50 | 0.886846 | 0.724857 | 0.0221044 | 0.00136509 | 0.0362941 | 0.966923 | 0.00189409 | 120 | 80.29 | 0.8 | 0.8 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw0p8_trust004 | alpha 0.8, leak 0.8, trust 0.04 | static_only | 50 | 0.903872 | 0.74084 | 0.0222904 | 0.00139885 | 0.0365013 | 1 | 0.00189409 | 120 | 80.29 | 0.8 | 0.8 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p0_trust004 | alpha 0.8, leak 1.0, trust 0.04 | combined | 50 | 0.87249 | 0.701739 | 0.0232446 | 0.0029901 | 0.0309483 | 0.951472 | 0.00182539 | 120 | 79.63 | 0.8 | 1.0 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p0_trust004 | alpha 0.8, leak 1.0, trust 0.04 | deph_0.001 | 50 | 0.889109 | 0.717048 | 0.0234163 | 0.00307321 | 0.031127 | 0.982675 | 0.00182539 | 120 | 79.63 | 0.8 | 1.0 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p0_trust004 | alpha 0.8, leak 1.0, trust 0.04 | relax_0.0005 | 50 | 0.877838 | 0.705975 | 0.0237022 | 0.00243069 | 0.0307556 | 0.968102 | 0.00182539 | 120 | 79.63 | 0.8 | 1.0 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p0_trust004 | alpha 0.8, leak 1.0, trust 0.04 | static_only | 50 | 0.894566 | 0.721386 | 0.023878 | 0.00248984 | 0.0309325 | 1 | 0.00182539 | 120 | 79.63 | 0.8 | 1.0 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p2_trust004 | alpha 0.8, leak 1.2, trust 0.04 | combined | 50 | 0.866819 | 0.687288 | 0.0241454 | 0.00333162 | 0.0263027 | 0.951963 | 0.00175441 | 120 | 79.04 | 0.8 | 1.2 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p2_trust004 | alpha 0.8, leak 1.2, trust 0.04 | deph_0.001 | 50 | 0.883196 | 0.702196 | 0.0243187 | 0.00342043 | 0.0264481 | 0.982071 | 0.00175441 | 120 | 79.04 | 0.8 | 1.2 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p2_trust004 | alpha 0.8, leak 1.2, trust 0.04 | relax_0.0005 | 50 | 0.872276 | 0.691626 | 0.0246169 | 0.00283344 | 0.0261278 | 0.969208 | 0.00175441 | 120 | 79.04 | 0.8 | 1.2 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p2_trust004 | alpha 0.8, leak 1.2, trust 0.04 | static_only | 50 | 0.88876 | 0.706634 | 0.0247947 | 0.00290095 | 0.0262714 | 1 | 0.00175441 | 120 | 79.04 | 0.8 | 1.2 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p5_trust004 | alpha 0.8, leak 1.5, trust 0.04 | combined | 50 | 0.857829 | 0.661094 | 0.0252906 | 0.00379889 | 0.022663 | 0.952657 | 0.00165559 | 120 | 79.19 | 0.8 | 1.5 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p5_trust004 | alpha 0.8, leak 1.5, trust 0.04 | deph_0.001 | 50 | 0.873944 | 0.675383 | 0.0254735 | 0.00389611 | 0.0227748 | 0.981213 | 0.00165559 | 120 | 79.19 | 0.8 | 1.5 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p5_trust004 | alpha 0.8, leak 1.5, trust 0.04 | relax_0.0005 | 50 | 0.863411 | 0.665437 | 0.02577 | 0.00336986 | 0.0225378 | 0.970771 | 0.00165559 | 120 | 79.19 | 0.8 | 1.5 | 0.25 | 0.04 | 6 |
| pareto_alpha0p8_lw1p5_trust004 | alpha 0.8, leak 1.5, trust 0.04 | static_only | 50 | 0.87963 | 0.679823 | 0.0259577 | 0.00344871 | 0.0226473 | 1 | 0.00165559 | 120 | 79.19 | 0.8 | 1.5 | 0.25 | 0.04 | 6 |
