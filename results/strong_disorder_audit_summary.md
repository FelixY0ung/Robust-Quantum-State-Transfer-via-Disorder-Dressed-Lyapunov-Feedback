# Strong-Disorder Extrapolation Audit

The beam-horizon controller is trained on delta 0.05/0.08 and the dCRAB comparator is trained at delta 0.08. Rows at delta 0.10 and 0.12 are held-out extrapolation tests, not retuned designs.

## Held-Out Summary

| task | method | eval_strength | n | final_fidelity_mean | final_fidelity_min | final_fidelity_std | final_fidelity_ci95 | pulse_energy_mean | design_seconds | resource_profile | training_strengths |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | beam_horizon | 0.08 | 50 | 0.998156 | 0.994749 | 0.00144612 | 0.000400845 | 5.6075 | 132.670 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| H | beam_horizon | 0.10 | 50 | 0.997553 | 0.992677 | 0.00201964 | 0.000559816 | 5.6075 | 132.670 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| H | beam_horizon | 0.12 | 50 | 0.996804 | 0.990176 | 0.00271157 | 0.000751608 | 5.6075 | 132.670 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| H | dcrab_train8 | 0.08 | 50 | 0.999789 | 0.999393 | 0.000183508 | 5.08658e-05 | 4.97162 | 70.287 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| H | dcrab_train8 | 0.10 | 50 | 0.999589 | 0.998674 | 0.000358004 | 9.92336e-05 | 4.97162 | 70.287 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| H | dcrab_train8 | 0.12 | 50 | 0.999259 | 0.997429 | 0.000654274 | 0.000181356 | 4.97162 | 70.287 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| Z | beam_horizon | 0.08 | 50 | 0.997582 | 0.993966 | 0.00132018 | 0.000365936 | 4.9225 | 100.214 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| Z | beam_horizon | 0.10 | 50 | 0.996728 | 0.991482 | 0.00191688 | 0.000531333 | 4.9225 | 100.214 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| Z | beam_horizon | 0.12 | 50 | 0.9957 | 0.988545 | 0.0026274 | 0.000728277 | 4.9225 | 100.214 | 100 seg.; 8 train seeds; train delta 0.05/0.08; q=6; finite beam | 0.05,0.08 |
| Z | dcrab_train8 | 0.08 | 50 | 0.999713 | 0.998775 | 0.000277975 | 7.70506e-05 | 8.60437 | 51.094 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| Z | dcrab_train8 | 0.10 | 50 | 0.999378 | 0.997357 | 0.000607729 | 0.000168454 | 8.60437 | 51.094 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |
| Z | dcrab_train8 | 0.12 | 50 | 0.998811 | 0.995072 | 0.0011728 | 0.000325085 | 8.60437 | 51.094 | 40 seg.; 8 train seeds; train delta 0.08; 3 Fourier modes; 3 refreshes | 0.08 |

## Paired Deltas

| task | comparison | eval_strength | n | delta_mean | delta_min | delta_max | delta_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H | dcrab_train8_minus_beam_horizon | 0.08 | 50 | 0.0016329188 | -0.00019826776 | 0.0051385775 | 0.00042544778 |
| H | dcrab_train8_minus_beam_horizon | 0.10 | 50 | 0.0020361686 | -0.00058561826 | 0.0070406434 | 0.00059834489 |
| H | dcrab_train8_minus_beam_horizon | 0.12 | 50 | 0.0024544597 | -0.0012175666 | 0.0093383998 | 0.00081156544 |
| Z | dcrab_train8_minus_beam_horizon | 0.08 | 50 | 0.0021314502 | -1.5130494e-05 | 0.0058657267 | 0.00039135677 |
| Z | dcrab_train8_minus_beam_horizon | 0.10 | 50 | 0.002650787 | -0.001116156 | 0.0083373043 | 0.00061181868 |
| Z | dcrab_train8_minus_beam_horizon | 0.12 | 50 | 0.0031105065 | -0.0029684008 | 0.011282176 | 0.00091050622 |
