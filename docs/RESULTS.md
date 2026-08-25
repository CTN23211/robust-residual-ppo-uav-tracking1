# Results Summary

## Nominal architecture trade-off

| Method | Success | Steady RMSE | Capture time | Command smoothness |
|---|---:|---:|---:|---:|
| PD+FF | 100% | 0.0558 m | 5.569 s | 0.000010 |
| Direct PPO | 100% | 0.0063 m | 3.179 s | 0.002325 |
| Residual PPO | 100% | 0.0173 m | 3.884 s | 0.000036 |

Direct PPO is strongest nominally in speed/accuracy but is much more control-active. Residual PPO occupies an intermediate accuracy–capture–smoothness compromise around a structured controller.

## Severe zero-shot endpoints

- Velocity-estimation fault, `alpha=1.0`: nominal Residual PPO retains 58% success while PD+FF falls to 2%.
- Position noise, `sigma_p=0.10 m`: Residual PPO retains 100% success; Direct PPO 86%.
- Velocity noise, `sigma_v=0.10 m/s`: Residual PPO 99%; Direct PPO 79%.
- Observation delay, 400 ms: Residual PPO 100%; Direct PPO 0%.
- Constant-turn target: Residual PPO 100%; Direct PPO 53%.

## Targeted robust training

Robust Residual PPO is trained from scratch with `alpha ~ U(0, 0.75)` and tested at held-out `alpha=1.0`.

At the held-out severity:

- Success: **58% → 95%** (+37 percentage points)
- Steady RMSE: **0.0650 m → 0.0379 m** (−41.6%)
- Forward overshoot: **0.187 m → 0.123 m** (−34.3%)

## Cross-robustness

Targeted training improves the primary tracking error in 7 of 8 cross-validation conditions. The main reported trade-offs are:

- velocity noise 0.10 m/s: success 99% → 97%
- constant-turn target: primary error +8.1%
- severe position noise: command-smoothness cost +118.6%

The appropriate conclusion is **broad positive transfer with identifiable trade-offs**, not uniform improvement under every perturbation.


## Continuous 3D robustness landscape

Figure 7 visualizes the paired steady-state error advantage of Robust Residual PPO over Nominal Residual PPO across two measured dimensions: velocity-fault severity `alpha` and randomized UGV target speed. The plotted height is `steady_rmse_nominal - steady_rmse_robust`, so positive values favor the robust policy.

The source is `results/robust_residual_alpha_sweep_raw.csv`. For each of the five tested alpha levels, the 100 paired seeds are grouped into eight equal-width target-speed bins. The resulting 40 empirical cell means are stored in `results/phase6/fig7_robustness_landscape_cells.csv`; the rendered surface uses linear interpolation only to connect those measured cells visually. It is not evidence of additional unmeasured alpha/speed combinations.
