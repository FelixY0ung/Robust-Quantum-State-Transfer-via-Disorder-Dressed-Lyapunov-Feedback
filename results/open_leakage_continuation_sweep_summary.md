# Open-Leakage Continuation Sweep

No-reference continuation horizons trained through the five-level Lindblad leakage model.

| controller | label | noise_case | n | final_fidelity_mean | final_fidelity_min | final_fidelity_ci95 | final_leakage_mean | max_leakage_mean | final_purity_mean | pulse_energy_mean | segments | training_seconds | stage_description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuation_path_seed | path seed | combined | 50 | 0.82642 | 0.521083 | 0.0251353 | 0.0423641 | 0.0466763 | 0.964078 | 0.00138437 | 120 | 66.49 |  |
| continuation_path_seed | path seed | deph_0.001 | 50 | 0.832238 | 0.521722 | 0.025606 | 0.0426705 | 0.0470868 | 0.981711 | 0.00138437 | 120 | 66.49 |  |
| continuation_path_seed | path seed | relax_0.0005 | 50 | 0.831909 | 0.520429 | 0.0256722 | 0.0418744 | 0.0464294 | 0.981852 | 0.00138437 | 120 | 66.49 |  |
| continuation_path_seed | path seed | static_only | 50 | 0.837761 | 0.521005 | 0.0261519 | 0.0421673 | 0.0468438 | 1 | 0.00138437 | 120 | 66.49 |  |
| continuation_robust_leak12 | low-leak seed, robust target 0.8 then leak 1.2 | combined | 50 | 0.918177 | 0.762203 | 0.0120081 | 0.00435639 | 0.0536495 | 0.948554 | 0.00303915 | 120 | 106.9 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.025,w=0.5; a=1,lw=1.2,r=0.015,w=0.5 |
| continuation_robust_leak12 | low-leak seed, robust target 0.8 then leak 1.2 | deph_0.001 | 50 | 0.936651 | 0.77475 | 0.0122383 | 0.00445898 | 0.053808 | 0.984836 | 0.00303915 | 120 | 106.9 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.025,w=0.5; a=1,lw=1.2,r=0.015,w=0.5 |
| continuation_robust_leak12 | low-leak seed, robust target 0.8 then leak 1.2 | relax_0.0005 | 50 | 0.923791 | 0.766089 | 0.0122194 | 0.00355287 | 0.0533739 | 0.962991 | 0.00303915 | 120 | 106.9 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.025,w=0.5; a=1,lw=1.2,r=0.015,w=0.5 |
| continuation_robust_leak12 | low-leak seed, robust target 0.8 then leak 1.2 | static_only | 50 | 0.942393 | 0.778685 | 0.0124463 | 0.00361721 | 0.0535308 | 0.999997 | 0.00303915 | 120 | 106.9 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.025,w=0.5; a=1,lw=1.2,r=0.015,w=0.5 |
| continuation_stageA_alpha0p8_lw1p5 | low-leakage seed | combined | 50 | 0.857829 | 0.661094 | 0.0252906 | 0.00379889 | 0.022663 | 0.952657 | 0.00165559 | 120 | 79.45 | a=0.8,lw=1.5,r=0.04,w=0.25 |
| continuation_stageA_alpha0p8_lw1p5 | low-leakage seed | deph_0.001 | 50 | 0.873944 | 0.675383 | 0.0254735 | 0.00389611 | 0.0227748 | 0.981213 | 0.00165559 | 120 | 79.45 | a=0.8,lw=1.5,r=0.04,w=0.25 |
| continuation_stageA_alpha0p8_lw1p5 | low-leakage seed | relax_0.0005 | 50 | 0.863411 | 0.665437 | 0.02577 | 0.00336986 | 0.0225378 | 0.970771 | 0.00165559 | 120 | 79.45 | a=0.8,lw=1.5,r=0.04,w=0.25 |
| continuation_stageA_alpha0p8_lw1p5 | low-leakage seed | static_only | 50 | 0.87963 | 0.679823 | 0.0259577 | 0.00344871 | 0.0226473 | 1 | 0.00165559 | 120 | 79.45 | a=0.8,lw=1.5,r=0.04,w=0.25 |
| continuation_target05_leak15 | low-leak seed, target 0.5 then leak 1.5 | combined | 50 | 0.909376 | 0.689982 | 0.0145869 | 0.00411964 | 0.0552942 | 0.948818 | 0.00277453 | 120 | 106.3 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.5,r=0.025,w=0.25; a=1,lw=1.5,r=0.015,w=0.25 |
| continuation_target05_leak15 | low-leak seed, target 0.5 then leak 1.5 | deph_0.001 | 50 | 0.927816 | 0.701885 | 0.0148891 | 0.00421351 | 0.0554851 | 0.983931 | 0.00277453 | 120 | 106.3 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.5,r=0.025,w=0.25; a=1,lw=1.5,r=0.015,w=0.25 |
| continuation_target05_leak15 | low-leak seed, target 0.5 then leak 1.5 | relax_0.0005 | 50 | 0.914949 | 0.692244 | 0.0148023 | 0.00340642 | 0.0552617 | 0.964193 | 0.00277453 | 120 | 106.3 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.5,r=0.025,w=0.25; a=1,lw=1.5,r=0.015,w=0.25 |
| continuation_target05_leak15 | low-leak seed, target 0.5 then leak 1.5 | static_only | 50 | 0.933508 | 0.70417 | 0.0151051 | 0.00346548 | 0.0554519 | 0.999998 | 0.00277453 | 120 | 106.3 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.5,r=0.025,w=0.25; a=1,lw=1.5,r=0.015,w=0.25 |
| continuation_target08_leak12 | low-leak seed, target 0.8 then leak 1.2 | combined | 50 | 0.920133 | 0.76681 | 0.0119527 | 0.00434561 | 0.0534782 | 0.94888 | 0.00296001 | 120 | 103.7 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.02,w=0.25; a=1,lw=1.2,r=0.015,w=0.25 |
| continuation_target08_leak12 | low-leak seed, target 0.8 then leak 1.2 | deph_0.001 | 50 | 0.938723 | 0.779464 | 0.0121639 | 0.00445628 | 0.0536249 | 0.984739 | 0.00296001 | 120 | 103.7 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.02,w=0.25; a=1,lw=1.2,r=0.015,w=0.25 |
| continuation_target08_leak12 | low-leak seed, target 0.8 then leak 1.2 | relax_0.0005 | 50 | 0.925741 | 0.770833 | 0.0121469 | 0.00357297 | 0.0531285 | 0.963429 | 0.00296001 | 120 | 103.7 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.02,w=0.25; a=1,lw=1.2,r=0.015,w=0.25 |
| continuation_target08_leak12 | low-leak seed, target 0.8 then leak 1.2 | static_only | 50 | 0.94446 | 0.783537 | 0.0123542 | 0.00364723 | 0.0532731 | 0.999998 | 0.00296001 | 120 | 103.7 | a=0.8,lw=1.5,r=0.04,w=0.25; a=1,lw=0.8,r=0.02,w=0.25; a=1,lw=1.2,r=0.015,w=0.25 |
