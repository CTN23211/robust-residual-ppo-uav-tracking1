| Condition | Severity | Primary error metric | PD+FF success (%) | PD+FF error (m) | Direct PPO success (%) | Direct PPO error (m) | Residual PPO success (%) | Residual PPO error (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Velocity fault | alpha=1.0 | steady RMSE | 2.00 | 0.1379 | — | — | 58.00 | 0.0650 |
| Position noise | sigma_p=0.10 m | steady RMSE | 100.00 | 0.0578 | 86.00 | 0.0289 | 100.00 | 0.0269 |
| Velocity noise | sigma_v=0.10 m/s | steady RMSE | 100.00 | 0.0660 | 79.00 | 0.0346 | 99.00 | 0.0391 |
| Observation delay | 400 ms | steady RMSE | 100.00 | 0.0317 | 0.00 | 0.1221 | 100.00 | 0.0089 |
| Vision loss | 1.0 s | steady RMSE | 100.00 | 0.0467 | 100.00 | 0.0198 | 100.00 | 0.0113 |
| Speed step | +0.20 m/s at t=6 s | post-maneuver RMSE | 100.00 | 0.0489 | 100.00 | 0.0109 | 100.00 | 0.0183 |
| Lateral sine | A=0.40 m, T=8 s | post-maneuver RMSE | 100.00 | 0.0592 | 100.00 | 0.1031 | 100.00 | 0.0431 |
| Constant turn | omega=0.25 rad/s | post-maneuver RMSE | 100.00 | 0.0513 | 53.00 | 0.1509 | 100.00 | 0.0244 |
