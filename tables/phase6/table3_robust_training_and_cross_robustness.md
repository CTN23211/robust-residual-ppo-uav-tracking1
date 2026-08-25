# Table 3. Robust training and cross-robustness

## Table 3A. Targeted velocity-fault robust training

| alpha | Region | Metric | Nominal Residual PPO | Robust Residual PPO | Change | Change unit |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | Nominal | Success (%) | 100.000000 | 100.000000 | 0.00 | percentage points |
| 0.00 | Nominal | Steady RMSE (m) | 0.017268 | 0.008760 | -49.27 | % |
| 0.00 | Nominal | Capture time (s) | 3.884000 | 4.136000 | 6.49 | % |
| 0.00 | Nominal | Forward overshoot (m) | 0.005320 | 0.001285 | -75.84 | % |
| 0.00 | Nominal | Command smoothness | 0.000036 | 0.000018 | -50.67 | % |
| 0.75 | Training boundary | Success (%) | 100.000000 | 100.000000 | 0.00 | percentage points |
| 0.75 | Training boundary | Steady RMSE (m) | 0.008852 | 0.003350 | -62.16 | % |
| 0.75 | Training boundary | Capture time (s) | 3.691500 | 3.606000 | -2.32 | % |
| 0.75 | Training boundary | Forward overshoot (m) | 0.026793 | 0.006086 | -77.28 | % |
| 0.75 | Training boundary | Command smoothness | 0.000133 | 0.000133 | -0.20 | % |
| 1.00 | Held-out severity | Success (%) | 58.000000 | 95.000000 | 37.00 | percentage points |
| 1.00 | Held-out severity | Steady RMSE (m) | 0.065019 | 0.037941 | -41.65 | % |
| 1.00 | Held-out severity | Capture time (s) | 7.753000 | 6.092000 | -21.42 | % |
| 1.00 | Held-out severity | Forward overshoot (m) | 0.187184 | 0.122927 | -34.33 | % |
| 1.00 | Held-out severity | Command smoothness | 0.000316 | 0.000324 | 2.58 | % |

## Table 3B. Cross-perturbation validation

| Condition | Nominal success | Robust success | Success delta (pp) | Primary error change (%) | Smoothness change (%) | Rescued seeds | Regressed seeds | Robust lower-error seeds /100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal | 100.0 | 100.0 | 0.0 | -49.3 | -50.7 | 0 | 0 | 100 |
| Position noise 0.10 m | 100.0 | 100.0 | 0.0 | -15.3 | 118.6 | 0 | 0 | 83 |
| Velocity noise 0.10 m/s | 99.0 | 97.0 | -2.0 | -27.5 | 21.6 | 0 | 2 | 99 |
| Delay 400 ms | 100.0 | 100.0 | 0.0 | -24.2 | -13.9 | 0 | 0 | 83 |
| Vision loss 1.0 s | 100.0 | 100.0 | 0.0 | -51.3 | 78.7 | 0 | 0 | 100 |
| Speed step | 100.0 | 100.0 | 0.0 | -46.0 | 28.4 | 0 | 0 | 100 |
| Lateral sine | 100.0 | 100.0 | 0.0 | -12.3 | 5.3 | 0 | 0 | 100 |
| Constant turn | 100.0 | 100.0 | 0.0 | 8.1 | -36.5 | 0 | 0 | 9 |
