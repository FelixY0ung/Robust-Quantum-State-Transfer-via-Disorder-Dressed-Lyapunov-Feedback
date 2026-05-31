# Integrated Open-Leakage Horizon Sweep

Single-stage no-reference horizons trained through the five-level Lindblad leakage model.

| controller | label | noise_case | n | final_fidelity_mean | final_fidelity_min | final_fidelity_ci95 | final_leakage_mean | max_leakage_mean | final_purity_mean | pulse_energy_mean | segments | training_seconds | terminal_target_weight | leakage_weight | trust_radius | horizon_maxiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| integrated_alpha0p9_lw1p0 | alpha 0.9, leak 1.0 | combined | 50 | 0.907817 | 0.681171 | 0.015557 | 0.00713042 | 0.0453514 | 0.954306 | 0.001754 | 120 | 74.96 | 0.9 | 1.0 | 0.025 | 4 |
| integrated_alpha0p9_lw1p0 | alpha 0.9, leak 1.0 | deph_0.001 | 50 | 0.923209 | 0.690482 | 0.0159608 | 0.00725401 | 0.04559 | 0.984207 | 0.001754 | 120 | 74.96 | 0.9 | 1.0 | 0.025 | 4 |
| integrated_alpha0p9_lw1p0 | alpha 0.9, leak 1.0 | relax_0.0005 | 50 | 0.91334 | 0.683171 | 0.0157776 | 0.00643701 | 0.0452154 | 0.969469 | 0.001754 | 120 | 74.96 | 0.9 | 1.0 | 0.025 | 4 |
| integrated_alpha0p9_lw1p0 | alpha 0.9, leak 1.0 | static_only | 50 | 0.928835 | 0.692489 | 0.0161866 | 0.00653154 | 0.0454528 | 1 | 0.001754 | 120 | 74.96 | 0.9 | 1.0 | 0.025 | 4 |
| integrated_alpha1p0_lw1p2 | alpha 1.0, leak 1.2 | combined | 50 | 0.903259 | 0.658347 | 0.0171034 | 0.00688921 | 0.0459055 | 0.954288 | 0.00175667 | 120 | 75.22 | 1.0 | 1.2 | 0.025 | 4 |
| integrated_alpha1p0_lw1p2 | alpha 1.0, leak 1.2 | deph_0.001 | 50 | 0.91864 | 0.667419 | 0.0175586 | 0.00701229 | 0.0461689 | 0.984133 | 0.00175667 | 120 | 75.22 | 1.0 | 1.2 | 0.025 | 4 |
| integrated_alpha1p0_lw1p2 | alpha 1.0, leak 1.2 | relax_0.0005 | 50 | 0.90856 | 0.659863 | 0.0173375 | 0.00619041 | 0.0457704 | 0.969522 | 0.00175667 | 120 | 75.22 | 1.0 | 1.2 | 0.025 | 4 |
| integrated_alpha1p0_lw1p2 | alpha 1.0, leak 1.2 | static_only | 50 | 0.924041 | 0.668935 | 0.0177991 | 0.00628389 | 0.0460324 | 1 | 0.00175667 | 120 | 75.22 | 1.0 | 1.2 | 0.025 | 4 |
| integrated_alpha1p0_lw1p2_trust004 | alpha 1.0, leak 1.2, trust 0.04 | combined | 50 | 0.90457 | 0.756149 | 0.0157763 | 0.00308591 | 0.0490586 | 0.950881 | 0.00196522 | 120 | 78.99 | 1.0 | 1.2 | 0.04 | 6 |
| integrated_alpha1p0_lw1p2_trust004 | alpha 1.0, leak 1.2, trust 0.04 | deph_0.001 | 50 | 0.921904 | 0.772721 | 0.01589 | 0.00316421 | 0.0493131 | 0.984693 | 0.00196522 | 120 | 78.99 | 1.0 | 1.2 | 0.04 | 6 |
| integrated_alpha1p0_lw1p2_trust004 | alpha 1.0, leak 1.2, trust 0.04 | relax_0.0005 | 50 | 0.909957 | 0.760424 | 0.0160879 | 0.00238627 | 0.0487677 | 0.965501 | 0.00196522 | 120 | 78.99 | 1.0 | 1.2 | 0.04 | 6 |
| integrated_alpha1p0_lw1p2_trust004 | alpha 1.0, leak 1.2, trust 0.04 | static_only | 50 | 0.927408 | 0.777107 | 0.0162028 | 0.00243411 | 0.0490193 | 1 | 0.00196522 | 120 | 78.99 | 1.0 | 1.2 | 0.04 | 6 |
| integrated_alpha1p0_lw1p5 | alpha 1.0, leak 1.5 | combined | 50 | 0.906301 | 0.669933 | 0.0158863 | 0.00798902 | 0.0421367 | 0.954652 | 0.00174927 | 120 | 75.07 | 1.0 | 1.5 | 0.025 | 4 |
| integrated_alpha1p0_lw1p5 | alpha 1.0, leak 1.5 | deph_0.001 | 50 | 0.921341 | 0.678592 | 0.0162933 | 0.00813477 | 0.0423558 | 0.983756 | 0.00174927 | 120 | 75.07 | 1.0 | 1.5 | 0.025 | 4 |
| integrated_alpha1p0_lw1p5 | alpha 1.0, leak 1.5 | relax_0.0005 | 50 | 0.912027 | 0.671899 | 0.0161138 | 0.00734744 | 0.0419993 | 0.970277 | 0.00174927 | 120 | 75.07 | 1.0 | 1.5 | 0.025 | 4 |
| integrated_alpha1p0_lw1p5 | alpha 1.0, leak 1.5 | static_only | 50 | 0.927168 | 0.680559 | 0.0165259 | 0.00746631 | 0.0422171 | 1 | 0.00174927 | 120 | 75.07 | 1.0 | 1.5 | 0.025 | 4 |
| integrated_alpha1p0_lw1p5_trust004 | alpha 1.0, leak 1.5, trust 0.04 | combined | 50 | 0.897659 | 0.726939 | 0.0172714 | 0.00336531 | 0.0401413 | 0.951328 | 0.00190076 | 120 | 79.21 | 1.0 | 1.5 | 0.04 | 6 |
| integrated_alpha1p0_lw1p5_trust004 | alpha 1.0, leak 1.5, trust 0.04 | deph_0.001 | 50 | 0.914738 | 0.742804 | 0.0173731 | 0.00345315 | 0.0403497 | 0.983707 | 0.00190076 | 120 | 79.21 | 1.0 | 1.5 | 0.04 | 6 |
| integrated_alpha1p0_lw1p5_trust004 | alpha 1.0, leak 1.5, trust 0.04 | relax_0.0005 | 50 | 0.903224 | 0.731228 | 0.0176079 | 0.00276208 | 0.0398871 | 0.96695 | 0.00190076 | 120 | 79.21 | 1.0 | 1.5 | 0.04 | 6 |
| integrated_alpha1p0_lw1p5_trust004 | alpha 1.0, leak 1.5, trust 0.04 | static_only | 50 | 0.920415 | 0.747197 | 0.0177113 | 0.00282374 | 0.040093 | 1 | 0.00190076 | 120 | 79.21 | 1.0 | 1.5 | 0.04 | 6 |
| open_leakage_path_seed | path seed | combined | 50 | 0.82642 | 0.521083 | 0.0251353 | 0.0423641 | 0.0466763 | 0.964078 | 0.00138437 | 120 | 66.64 |  |  |  |  |
| open_leakage_path_seed | path seed | deph_0.001 | 50 | 0.832238 | 0.521722 | 0.025606 | 0.0426705 | 0.0470868 | 0.981711 | 0.00138437 | 120 | 66.64 |  |  |  |  |
| open_leakage_path_seed | path seed | relax_0.0005 | 50 | 0.831909 | 0.520429 | 0.0256722 | 0.0418744 | 0.0464294 | 0.981852 | 0.00138437 | 120 | 66.64 |  |  |  |  |
| open_leakage_path_seed | path seed | static_only | 50 | 0.837761 | 0.521005 | 0.0261519 | 0.0421673 | 0.0468438 | 1 | 0.00138437 | 120 | 66.64 |  |  |  |  |
