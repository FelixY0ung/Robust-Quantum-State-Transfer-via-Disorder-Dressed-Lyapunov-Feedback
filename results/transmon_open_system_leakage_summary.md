# Transmon Open-System Leakage Summary

| system | controller | noise_case | eval_strength | gamma_phi | gamma_relax | n | final_fidelity_mean | final_fidelity_min | final_fidelity_ci95 | final_leakage_mean | max_leakage_mean | final_purity_mean | pulse_energy_mean | segments | training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| five_level_weak_anharmonic | adjoint_horizon | combined | 0.03 | 0.001 | 0.0005 | 50 | 0.932452 | 0.78764 | 0.0102647 | 0.015825 | 0.0538238 | 0.967478 | 0.00202043 | 80 | 19.887 |
| five_level_weak_anharmonic | adjoint_horizon | deph_0.001 | 0.03 | 0.001 | 0 | 50 | 0.941623 | 0.795501 | 0.0103741 | 0.0158995 | 0.0540005 | 0.985062 | 0.00202043 | 80 | 19.887 |
| five_level_weak_anharmonic | adjoint_horizon | relax_0.0005 | 0.03 | 0 | 0.0005 | 50 | 0.939023 | 0.792995 | 0.0103635 | 0.0142044 | 0.0528186 | 0.981958 | 0.00202043 | 80 | 19.887 |
| five_level_weak_anharmonic | adjoint_horizon | static_only | 0.03 | 0 | 0 | 50 | 0.948305 | 0.800964 | 0.0104747 | 0.0142235 | 0.0530326 | 0.999998 | 0.00202043 | 80 | 19.887 |
| five_level_weak_anharmonic | leakage_penalized_grape | combined | 0.03 | 0.001 | 0.0005 | 50 | 0.952952 | 0.830203 | 0.0090783 | 0.00236709 | 0.0548929 | 0.965698 | 0.00266075 | 80 | 16.108 |
| five_level_weak_anharmonic | leakage_penalized_grape | deph_0.001 | 0.03 | 0.001 | 0 | 50 | 0.961127 | 0.836212 | 0.00918748 | 0.0023946 | 0.0549536 | 0.982472 | 0.00266075 | 80 | 16.108 |
| five_level_weak_anharmonic | leakage_penalized_grape | relax_0.0005 | 0.03 | 0 | 0.0005 | 50 | 0.961143 | 0.83604 | 0.00921276 | 0.00108562 | 0.053873 | 0.982682 | 0.00266075 | 80 | 16.108 |
| five_level_weak_anharmonic | leakage_penalized_grape | static_only | 0.03 | 0 | 0 | 50 | 0.969446 | 0.842123 | 0.00932397 | 0.00108263 | 0.0539417 | 0.999997 | 0.00266075 | 80 | 16.108 |
| five_level_weak_anharmonic | path_horizon | combined | 0.03 | 0.001 | 0.0005 | 50 | 0.82642 | 0.521083 | 0.0251353 | 0.0423641 | 0.0466763 | 0.964078 | 0.00138437 | 120 | 67.835 |
| five_level_weak_anharmonic | path_horizon | deph_0.001 | 0.03 | 0.001 | 0 | 50 | 0.832238 | 0.521722 | 0.025606 | 0.0426705 | 0.0470868 | 0.981711 | 0.00138437 | 120 | 67.835 |
| five_level_weak_anharmonic | path_horizon | relax_0.0005 | 0.03 | 0 | 0.0005 | 50 | 0.831909 | 0.520429 | 0.0256722 | 0.0418744 | 0.0464294 | 0.981852 | 0.00138437 | 120 | 67.835 |
| five_level_weak_anharmonic | path_horizon | static_only | 0.03 | 0 | 0 | 50 | 0.837761 | 0.521005 | 0.0261519 | 0.0421673 | 0.0468438 | 1 | 0.00138437 | 120 | 67.835 |
