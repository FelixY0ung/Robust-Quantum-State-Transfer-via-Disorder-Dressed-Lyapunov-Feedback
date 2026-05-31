# Statistical Audit Summary

Confidence intervals use 1.96 standard errors over held-out seeds. Paired deltas are computed by matching the same held-out seed indices.

| kind | task | comparison | metric | n | mean | ci95_halfwidth | paired_delta | paired_delta_ci95 | paired_effect_dz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | Z | beam horizon transfer | final_fidelity | 50 | 0.997581558 | 0.00037 |  |  |  |
| summary | H | beam horizon transfer | final_fidelity | 50 | 0.998155764 | 0.000405 |  |  |  |
| summary | Z | dCRAB transfer ceiling | final_fidelity | 50 | 0.999635386 | 0.000108 |  |  |  |
| summary | H | dCRAB transfer ceiling | final_fidelity | 50 | 0.999577856 | 0.000116 |  |  |  |
| summary | Z | polished transfer ceiling | final_fidelity | 50 | 0.99996295 | 1.44e-05 |  |  |  |
| summary | H | polished transfer ceiling | final_fidelity | 50 | 0.99998439 | 5.92e-06 |  |  |  |
| summary | Z | process horizon gate | average_gate_fidelity | 50 | 0.926604095 | 0.0169 |  |  |  |
| summary | H | process horizon gate | average_gate_fidelity | 50 | 0.925944246 | 0.0173 |  |  |  |
| summary | Z | seeded process horizon gate | average_gate_fidelity | 50 | 0.994732095 | 0.00113 |  |  |  |
| summary | H | seeded process horizon gate | average_gate_fidelity | 50 | 0.999572854 | 4.53e-05 |  |  |  |
| summary | Z | adjoint process horizon gate | average_gate_fidelity | 50 | 0.9946405 | 0.00112 |  |  |  |
| summary | H | adjoint process horizon gate | average_gate_fidelity | 50 | 0.999716609 | 5.21e-05 |  |  |  |
| summary | Z | standalone process-adjoint gate | average_gate_fidelity | 50 | 0.924093227 | 0.0174 |  |  |  |
| summary | H | standalone process-adjoint gate | average_gate_fidelity | 50 | 0.927613406 | 0.0168 |  |  |  |
| summary | Z | GRAPE process gate | average_gate_fidelity | 50 | 0.99473236 | 0.00113 |  |  |  |
| summary | H | GRAPE process gate | average_gate_fidelity | 50 | 0.999742127 | 4.9e-05 |  |  |  |
| summary | Z | GRAPE state gate | average_gate_fidelity | 50 | 0.570795647 | 0.0444 |  |  |  |
| summary | H | GRAPE state gate | average_gate_fidelity | 50 | 0.545252071 | 0.0423 |  |  |  |
| summary | Z | transfer horizon gate | average_gate_fidelity | 50 | 0.377711277 | 0.00426 |  |  |  |
| summary | H | transfer horizon gate | average_gate_fidelity | 50 | 0.906708784 | 0.00568 |  |  |  |
| summary | Z | open noise combined | final_fidelity | 50 | 0.957140406 | 0.000329 |  |  |  |
| summary | H | open noise combined | final_fidelity | 50 | 0.956625219 | 0.00034 |  |  |  |
| summary | Z | standalone Lindblad seed combined | final_fidelity | 50 | 0.948556211 | 0.000325 |  |  |  |
| summary | H | standalone Lindblad seed combined | final_fidelity | 50 | 0.94285266 | 0.00378 |  |  |  |
| summary | Z | standalone Lindblad adjoint combined | final_fidelity | 50 | 0.948933959 | 0.00049 |  |  |  |
| summary | H | standalone Lindblad adjoint combined | final_fidelity | 50 | 0.943610037 | 0.00361 |  |  |  |
| summary | Z | adjoint Lindblad horizon combined | final_fidelity | 50 | 0.974683969 | 0.00063 |  |  |  |
| summary | H | adjoint Lindblad horizon combined | final_fidelity | 50 | 0.977779184 | 0.000645 |  |  |  |
| summary | Z | open-system GRAPE combined | final_fidelity | 50 | 0.97746055 | 0.000551 |  |  |  |
| summary | H | open-system GRAPE combined | final_fidelity | 50 | 0.978929484 | 0.000435 |  |  |  |
| summary | L | leakage path seed fidelity | final_fidelity | 50 | 0.837761152 | 0.0264 |  |  |  |
| summary | L | standalone leakage adjoint fidelity | final_fidelity | 50 | 0.846605872 | 0.0311 |  |  |  |
| summary | L | leakage-GRAPE fidelity | final_fidelity | 50 | 0.969446711 | 0.00942 |  |  |  |
| summary | L | leakage path seed max leakage | max_leakage | 50 | 0.0468435426 | 0.00266 |  |  |  |
| summary | L | standalone leakage adjoint max leakage | max_leakage | 50 | 0.0146423576 | 0.000871 |  |  |  |
| summary | L | leakage-GRAPE max leakage | max_leakage | 50 | 0.0539387677 | 0.00274 |  |  |  |
| summary | OL | open leakage path combined fidelity | final_fidelity | 50 | 0.826420064 | 0.0254 |  |  |  |
| summary | OL | standalone open leakage adjoint combined fidelity | final_fidelity | 50 | 0.840595369 | 0.029 |  |  |  |
| summary | OL | open leakage adjoint combined fidelity | final_fidelity | 50 | 0.932451986 | 0.0104 |  |  |  |
| summary | OL | open leakage-GRAPE combined fidelity | final_fidelity | 50 | 0.95295242 | 0.00917 |  |  |  |
| summary | OL | open leakage path combined max leakage | max_leakage | 50 | 0.0466762953 | 0.0026 |  |  |  |
| summary | OL | standalone open leakage adjoint combined max leakage | max_leakage | 50 | 0.0176521996 | 0.000984 |  |  |  |
| summary | OL | open leakage adjoint combined max leakage | max_leakage | 50 | 0.053823841 | 0.00274 |  |  |  |
| summary | OL | open leakage-GRAPE combined max leakage | max_leakage | 50 | 0.0548929297 | 0.00272 |  |  |  |
| summary | Z | compact beam no-slew transfer | final_fidelity | 50 | 0.990507913 | 0.000508 |  |  |  |
| summary | H | compact beam no-slew transfer | final_fidelity | 50 | 0.996504934 | 0.000437 |  |  |  |
| summary | Z | compact slew-constrained transfer | final_fidelity | 50 | 0.995053745 | 0.000382 |  |  |  |
| summary | H | compact slew-constrained transfer | final_fidelity | 50 | 0.995259938 | 0.000726 |  |  |  |
| summary | Z | filtered no-slew boxcar3 transfer | final_fidelity | 50 | 0.960063585 | 0.00337 |  |  |  |
| summary | H | filtered no-slew boxcar3 transfer | final_fidelity | 50 | 0.958034091 | 0.00482 |  |  |  |
| summary | Z | filtered slew boxcar3 transfer | final_fidelity | 50 | 0.994356367 | 0.000778 |  |  |  |
| summary | H | filtered slew boxcar3 transfer | final_fidelity | 50 | 0.995035697 | 0.00113 |  |  |  |
| paired | Z | beam horizon minus one-step horizon | final_fidelity | 50 |  |  | 0.0402918502 | 0.00928 | 1.2 |
| paired | H | beam horizon minus one-step horizon | final_fidelity | 50 |  |  | 0.0421727374 | 0.00977 | 1.2 |
| paired | Z | dCRAB ceiling minus beam horizon | final_fidelity | 50 |  |  | 0.00205382791 | 0.000431 | 1.32 |
| paired | H | dCRAB ceiling minus beam horizon | final_fidelity | 50 |  |  | 0.00142209189 | 0.000466 | 0.846 |
| paired | Z | polished ceiling minus dCRAB ceiling | final_fidelity | 50 |  |  | 0.000327564561 | 0.000102 | 0.891 |
| paired | H | polished ceiling minus dCRAB ceiling | final_fidelity | 50 |  |  | 0.000406533891 | 0.000116 | 0.968 |
| paired | Z | polished ceiling minus beam horizon | final_fidelity | 50 |  |  | 0.00238139247 | 0.000375 | 1.76 |
| paired | H | polished ceiling minus beam horizon | final_fidelity | 50 |  |  | 0.00182862578 | 0.000407 | 1.24 |
| paired | Z | adjoint Lindblad horizon minus closed horizon | final_fidelity | 50 |  |  | 0.0175435631 | 0.000732 | 6.64 |
| paired | Z | standalone Lindblad adjoint minus finite Lindblad seed | final_fidelity | 50 |  |  | 0.000377747949 | 0.000281 | 0.372 |
| paired | H | standalone Lindblad adjoint minus finite Lindblad seed | final_fidelity | 50 |  |  | 0.00075737681 | 0.000661 | 0.318 |
| paired | Z | closed horizon minus standalone Lindblad adjoint | final_fidelity | 50 |  |  | 0.00820644731 | 0.000618 | 3.68 |
| paired | H | closed horizon minus standalone Lindblad adjoint | final_fidelity | 50 |  |  | 0.0130151814 | 0.00375 | 0.962 |
| paired | Z | adjoint Lindblad horizon minus standalone Lindblad adjoint | final_fidelity | 50 |  |  | 0.0257500104 | 0.00083 | 8.6 |
| paired | H | adjoint Lindblad horizon minus closed horizon | final_fidelity | 50 |  |  | 0.0211539652 | 0.000816 | 7.19 |
| paired | H | adjoint Lindblad horizon minus standalone Lindblad adjoint | final_fidelity | 50 |  |  | 0.0341691466 | 0.00321 | 2.95 |
| paired | Z | open-system GRAPE minus closed horizon | final_fidelity | 50 |  |  | 0.0203201437 | 0.000674 | 8.35 |
| paired | H | open-system GRAPE minus closed horizon | final_fidelity | 50 |  |  | 0.0223042654 | 0.000664 | 9.31 |
| paired | Z | open-system GRAPE minus adjoint Lindblad horizon | final_fidelity | 50 |  |  | 0.00277658065 | 0.000122 | 6.33 |
| paired | H | open-system GRAPE minus adjoint Lindblad horizon | final_fidelity | 50 |  |  | 0.00115030017 | 0.000251 | 1.27 |
| paired | Z | seeded process horizon minus process horizon | average_gate_fidelity | 50 |  |  | 0.0681279998 | 0.0163 | 1.16 |
| paired | H | seeded process horizon minus process horizon | average_gate_fidelity | 50 |  |  | 0.0736286086 | 0.0173 | 1.18 |
| paired | Z | adjoint process horizon minus process horizon | average_gate_fidelity | 50 |  |  | 0.0680364053 | 0.0163 | 1.16 |
| paired | H | adjoint process horizon minus process horizon | average_gate_fidelity | 50 |  |  | 0.0737723637 | 0.0173 | 1.18 |
| paired | Z | GRAPE process gate minus adjoint process horizon | average_gate_fidelity | 50 |  |  | 9.18595941e-05 | 9.7e-05 | 0.262 |
| paired | H | GRAPE process gate minus adjoint process horizon | average_gate_fidelity | 50 |  |  | 2.55174015e-05 | 2.27e-05 | 0.311 |
| paired | Z | standalone process adjoint minus finite process seed | average_gate_fidelity | 50 |  |  | -0.00344355219 | 0.00409 | -0.233 |
| paired | H | standalone process adjoint minus finite process seed | average_gate_fidelity | 50 |  |  | -0.000705797142 | 0.000969 | -0.202 |
| paired | Z | GRAPE process gate minus process horizon | average_gate_fidelity | 50 |  |  | 0.0681282649 | 0.0163 | 1.16 |
| paired | H | GRAPE process gate minus process horizon | average_gate_fidelity | 50 |  |  | 0.0737978811 | 0.0173 | 1.18 |
| paired | Z | process horizon gate minus transfer horizon gate | average_gate_fidelity | 50 |  |  | 0.548892818 | 0.018 | 8.44 |
| paired | H | process horizon gate minus transfer horizon gate | average_gate_fidelity | 50 |  |  | 0.0192354616 | 0.0188 | 0.284 |
| paired | L | standalone leakage adjoint minus path seed | final_fidelity | 50 |  |  | 0.00884471999 | 0.0322 | 0.0762 |
| paired | L | leakage-GRAPE minus standalone leakage adjoint | final_fidelity | 50 |  |  | 0.12284084 | 0.0278 | 1.22 |
| paired | L | path seed max leakage minus standalone leakage adjoint | max_leakage | 50 |  |  | 0.032201185 | 0.00213 | 4.19 |
| paired | L | leakage-GRAPE max leakage minus standalone leakage adjoint | max_leakage | 50 |  |  | 0.0392964101 | 0.00259 | 4.21 |
| paired | OL | open leakage adjoint minus path horizon | final_fidelity | 50 |  |  | 0.106031923 | 0.02 | 1.47 |
| paired | OL | standalone open leakage adjoint minus path seed | final_fidelity | 50 |  |  | 0.014175305 | 0.0278 | 0.141 |
| paired | OL | reference-assisted open leakage adjoint minus standalone open leakage adjoint | final_fidelity | 50 |  |  | 0.0918566178 | 0.0251 | 1.02 |
| paired | OL | open leakage-GRAPE minus adjoint horizon | final_fidelity | 50 |  |  | 0.0205004332 | 0.00712 | 0.798 |
| paired | OL | open leakage-GRAPE minus standalone open leakage adjoint | final_fidelity | 50 |  |  | 0.112357051 | 0.0261 | 1.19 |
| paired | OL | open leakage path max leakage minus adjoint horizon | max_leakage | 50 |  |  | -0.00714754568 | 0.0038 | -0.521 |
| paired | OL | open leakage path seed max leakage minus standalone open leakage adjoint | max_leakage | 50 |  |  | 0.0290240958 | 0.00199 | 4.04 |
| paired | OL | reference-assisted open leakage adjoint max leakage minus standalone open leakage adjoint | max_leakage | 50 |  |  | 0.0361716414 | 0.00259 | 3.87 |
| paired | OL | open leakage-GRAPE max leakage minus adjoint horizon | max_leakage | 50 |  |  | 0.00106908866 | 0.00267 | 0.111 |
| paired | Z | compact slew-constrained minus no-slew beam | final_fidelity | 50 |  |  | 0.00454583191 | 0.000622 | 2.02 |
| paired | H | compact slew-constrained minus no-slew beam | final_fidelity | 50 |  |  | -0.00124499609 | 0.000674 | -0.512 |
| paired | Z | boxcar3 no-slew filtered minus no-slew beam | final_fidelity | 50 |  |  | -0.0304443285 | 0.00364 | -2.32 |
| paired | H | boxcar3 no-slew filtered minus no-slew beam | final_fidelity | 50 |  |  | -0.038470843 | 0.0046 | -2.32 |
| paired | Z | boxcar3 filtered slew minus slew beam | final_fidelity | 50 |  |  | -0.00069737842 | 0.000752 | -0.257 |
| paired | H | boxcar3 filtered slew minus slew beam | final_fidelity | 50 |  |  | -0.000224240975 | 0.00115 | -0.0542 |
| paired | Z | boxcar3 filtered slew minus filtered no-slew | final_fidelity | 50 |  |  | 0.034292782 | 0.00286 | 3.33 |
| paired | H | boxcar3 filtered slew minus filtered no-slew | final_fidelity | 50 |  |  | 0.037001606 | 0.00542 | 1.89 |
| paired | Z | gaussian7 filtered slew minus slew beam | final_fidelity | 50 |  |  | -0.0201968259 | 0.00223 | -2.51 |
| paired | H | gaussian7 filtered slew minus slew beam | final_fidelity | 50 |  |  | -0.00669288856 | 0.0026 | -0.714 |
