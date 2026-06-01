# Open Leakage Pareto Audit

Combined five-level leakage-plus-Lindblad evaluation at $\delta=0.03$, $\gamma_\phi=0.001$, and $\gamma_1=0.0005$.

| label | family | mean fidelity | worst fidelity | mean max leakage | Pareto status |
| --- | --- | --- | --- | --- | --- |
| Path horizon | GRAPE-free horizon | 0.826420 | 0.521083 | 0.046676 | dominated by Direct adjoint; Target-biased direct; Integrated direct; Pareto alpha0.8 leak1.5; Pareto alpha0.8 leak1.2; Pareto alpha0.8 leak1.0; Pareto alpha0.8 leak0.8 |
| Direct adjoint | GRAPE-free horizon | 0.840595 | 0.551524 | 0.017652 | Pareto |
| Target-biased direct | GRAPE-free horizon | 0.910936 | 0.706954 | 0.042021 | Pareto |
| Integrated direct | GRAPE-free horizon | 0.897659 | 0.726939 | 0.040141 | Pareto |
| Pareto alpha0.8 leak1.5 | GRAPE-free horizon | 0.857829 | 0.661094 | 0.022663 | Pareto |
| Pareto alpha0.8 leak1.2 | GRAPE-free horizon | 0.866819 | 0.687288 | 0.026303 | Pareto |
| Pareto alpha0.8 leak1.0 | GRAPE-free horizon | 0.872490 | 0.701739 | 0.030948 | Pareto |
| Pareto alpha0.8 leak0.8 | GRAPE-free horizon | 0.881557 | 0.720702 | 0.036526 | Pareto |
| Two-stage direct | GRAPE-free horizon | 0.919179 | 0.784895 | 0.065347 | dominated by Continuation controlled; Continuation balanced; Reference-assisted horizon; Leakage-GRAPE |
| Continuation controlled | GRAPE-free horizon | 0.920133 | 0.766810 | 0.053478 | Pareto |
| Continuation balanced | GRAPE-free horizon | 0.924778 | 0.795870 | 0.060700 | dominated by Reference-assisted horizon; Leakage-GRAPE |
| Continuation high-fidelity | GRAPE-free horizon | 0.929533 | 0.812158 | 0.071371 | dominated by Leakage-cap target-push; Reference-assisted horizon; Leakage-GRAPE |
| HF continuation leak0.6 | GRAPE-free horizon | 0.933503 | 0.810714 | 0.076874 | dominated by Leakage-cap target-push; Leakage-GRAPE |
| HF continuation leak0.5 | GRAPE-free horizon | 0.935047 | 0.800444 | 0.079428 | dominated by Leakage-cap target-push; Leakage-GRAPE |
| Leakage-cap target-push | GRAPE-free horizon | 0.938530 | 0.766622 | 0.069783 | dominated by Leakage-GRAPE |
| HF continuation leak0.4 | GRAPE-free horizon | 0.935466 | 0.784925 | 0.084094 | dominated by Leakage-cap target-push; Leakage-GRAPE |
| Reference-assisted horizon | reference-assisted horizon | 0.932452 | 0.787640 | 0.053824 | Pareto |
| Leakage-GRAPE | terminal optimizer | 0.952952 | 0.830203 | 0.054893 | Pareto |
