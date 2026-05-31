# Open Leakage Pareto Audit

Combined five-level leakage-plus-Lindblad evaluation at $\delta=0.03$, $\gamma_\phi=0.001$, and $\gamma_1=0.0005$.

| label | family | mean fidelity | worst fidelity | mean max leakage | Pareto status |
| --- | --- | --- | --- | --- | --- |
| Path horizon | GRAPE-free horizon | 0.826420 | 0.521083 | 0.046676 | dominated by Direct adjoint; Target-biased direct; Integrated direct |
| Direct adjoint | GRAPE-free horizon | 0.840595 | 0.551524 | 0.017652 | Pareto |
| Target-biased direct | GRAPE-free horizon | 0.910936 | 0.706954 | 0.042021 | Pareto |
| Integrated direct | GRAPE-free horizon | 0.897659 | 0.726939 | 0.040141 | Pareto |
| Two-stage direct | GRAPE-free horizon | 0.919179 | 0.784895 | 0.065347 | dominated by Reference-assisted horizon; Leakage-GRAPE |
| Reference-assisted horizon | reference-assisted horizon | 0.932452 | 0.787640 | 0.053824 | Pareto |
| Leakage-GRAPE | terminal optimizer | 0.952952 | 0.830203 | 0.054893 | Pareto |
