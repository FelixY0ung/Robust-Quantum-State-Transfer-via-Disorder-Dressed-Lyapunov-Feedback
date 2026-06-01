# Extended Held-Out Audit

The beam-horizon controller is trained on delta 0.05/0.08 and the train-8 dCRAB comparator is trained at delta 0.08, using the same design protocols as the strong-disorder audit. The rows below evaluate the regenerated pulses on 200 disjoint held-out disorder seeds `1000`..`1199`. Rows at delta 0.12 are out-of-training-range tests; no controller is retuned for those rows. The empirical minima are minima over this finite held-out range, not guaranteed worst cases over the continuous disorder distribution.

## Held-Out Summary

| task | method | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | final_fidelity_ci95 | pulse_energy_mean | design_seconds | resource_profile | training_strengths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | beam_horizon | 0.08 | 200 | 0.998151 | 0.994788 | 0.00134319 | 0.000186157 | 5.6075 | 131.045 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| H | beam_horizon | 0.12 | 200 | 0.996729 | 0.990313 | 0.00246283 | 0.000341331 | 5.6075 | 131.045 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| H | dcrab_train8 | 0.08 | 200 | 0.999777 | 0.999392 | 0.000188467 | 2.61202e-05 | 4.97162 | 69.608 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| H | dcrab_train8 | 0.12 | 200 | 0.999252 | 0.997646 | 0.000628008 | 8.70374e-05 | 4.97162 | 69.608 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| Z | beam_horizon | 0.08 | 200 | 0.997537 | 0.993907 | 0.00130833 | 0.000181325 | 4.9225 | 99.037 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| Z | beam_horizon | 0.12 | 200 | 0.995588 | 0.988162 | 0.00270006 | 0.000374209 | 4.9225 | 99.037 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| Z | dcrab_train8 | 0.08 | 200 | 0.999716 | 0.998649 | 0.000294418 | 4.08042e-05 | 8.60437 | 50.752 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| Z | dcrab_train8 | 0.12 | 200 | 0.998772 | 0.994295 | 0.00132116 | 0.000183103 | 8.60437 | 50.752 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |

## Paired Deltas

| task | comparison | eval_strength | n | delta_mean | delta_min | delta_max | delta_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H | dcrab_train8_minus_beam_horizon | 0.08 | 200 | 0.0016263179 | -0.00027639596 | 0.005158406 | 0.00020151776 |
| H | dcrab_train8_minus_beam_horizon | 0.12 | 200 | 0.0025231943 | -0.0014008726 | 0.0094321246 | 0.00037986504 |
| Z | dcrab_train8_minus_beam_horizon | 0.08 | 200 | 0.0021795679 | -4.0155679e-05 | 0.0059675132 | 0.00019450627 |
| Z | dcrab_train8_minus_beam_horizon | 0.12 | 200 | 0.0031844294 | -0.0038112064 | 0.011703485 | 0.00046726407 |
