# Scheduled Depth-Budget Audit

This postprocesses the terminal-value margin CSVs into the depth budget required by the coherent scheduled finite-entry clause. It is not a fixed-depth or all-time certificate: the online depth-one run verifies one-step scheduled Bellman margins, while finite entry requires a preallocated schedule with `L_{j+1}=L_j-1`.

| source | task | depth | outside | first entry | eps min | first value | needed L0 | audited covers L0 | entry <= L0 | max gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_terminal_value_shifted | H | 1 | 93/99 | 36 | 3.25313e-05 | 0.00204861 | 63 | 1 | 1 | 8.67e-19 |
| online_terminal_value_shifted | Z | 1 | 99/99 | -- | 0.000116004 | 0.003313 | 29 | 1 | 0 | 8.67e-19 |
| offline_shifted_fallback | H | 1 | 93/99 | 36 | 2.89492e-05 | 0.00205141 | 71 | 1 | 1 | 8.67e-19 |
| offline_shifted_fallback | H | 2 | 93/99 | 36 | 2.89492e-05 | 0.00310095 | 108 | 0 | 1 | 8.67e-19 |
| offline_shifted_fallback | Z | 1 | 99/99 | -- | 0.000274843 | 0.003313 | 13 | 1 | 0 | 8.67e-19 |
| offline_shifted_fallback | Z | 2 | 99/99 | -- | 0.000274843 | 0.00484461 | 18 | 1 | 0 | 8.67e-19 |

The `needed L0` column is `floor(first_current_value / eps_min) + 1`. For H, the recorded terminal score can enter the residual set before this conservative bound; for Z, the depth-one recorded run remains outside, so the row should be read only as the coherent-schedule budget required by the theorem.
