# Phase 6 — Consolidated Results and Evidence Chain

## 1. Core study question

The project compares a classical PD+feedforward controller, a Direct PPO controller, and a Residual PPO controller, then tests whether targeted domain randomization of the feedforward velocity-estimation fault improves severe-fault robustness without broad cross-perturbation collapse.

## 2. Nominal architecture trade-off

Under nominal randomized tracking, Direct PPO is fastest and most accurate (steady RMSE 0.0063 m; capture 3.179 s), PD+FF is smoothest but slower (steady RMSE 0.0558 m; capture 5.569 s), and Residual PPO occupies the intermediate trade-off (steady RMSE 0.0173 m; capture 3.884 s; command smoothness 0.000036).

## 3. Zero-shot robustness pattern

At the controller-path velocity fault alpha=1.0, PD+FF success falls to 2% with 0.409 m forward overshoot, while nominal Residual PPO retains 58% success with 0.187 m overshoot.
Under severe observation noise, Residual PPO retains 100% success at sigma_p=0.10 m and 99% at sigma_v=0.10 m/s, compared with Direct PPO at 86% and 79% respectively.
At 400 ms observation delay, Direct PPO success falls to 0% whereas Residual PPO remains at 100%. At 1.0 s temporary vision loss, all controllers still succeed, but Direct PPO forward overshoot is 0.126 m versus 0.009 m for Residual PPO.
For the unseen constant-turn target, Direct PPO success is 53% with post-maneuver RMSE 0.1509 m, while Residual PPO remains at 100% success with 0.0244 m RMSE.

## 4. Targeted robust training closes the main failure mode

Robust Residual PPO is trained from scratch with alpha sampled once per episode from U(0, 0.75). Alpha=1.0 is therefore a held-out severity (not a wholly unseen fault type).
At held-out alpha=1.0, success increases from 58% to 95% (+37 percentage points), steady RMSE decreases by 41.6%, and forward overshoot decreases by 34.3%.

## 5. Cross-robustness validation

Targeted robust training improves the condition-specific primary tracking error in 7/8 cross-validation conditions. The only success-rate regression occurs for Velocity noise 0.10 m/s, where success changes by -2 percentage points.
The main accuracy trade-off is constant-turn tracking (+8.1% primary error), while the clearest control-activity trade-off appears under severe position noise (+118.6% smoothness cost).

## 6. Claims supported by the data

- Direct PPO provides the strongest nominal speed/accuracy but is the most sensitive to observation delay and high observation noise.
- Residual PPO offers a stronger robustness–smoothness compromise by limiting learned control authority around a structured PD+feedforward baseline.
- Targeted alpha-domain randomization substantially improves held-out severe-fault performance and mostly transfers positively to unrelated perturbations.
- Robust training is not uniformly free: severe observation-noise control activity and a small high-velocity-noise success regression remain measurable trade-offs.

## 7. Interpretation constraints

- Phase 5.1 corrupts only the feedforward velocity estimate; PPO observations remain true. Do not call it a full sensor-observation corruption test.
- Alpha=1.0 is a held-out severity of the same fault family, not a completely unseen fault type.
- Temporary vision loss uses last-valid-target hold while the target remains constant velocity; recovery-time comparisons are confounded by the moving restoration time during the pursuit transient.
- Lower steady RMSE at some delayed-controller settings should not be generalized as evidence that delay is beneficial.
- Full-episode RMSE can hide transient overshoot; use success, condition-specific primary error, stable ratio, and forward overshoot together.

## 8. Recommended main-paper figure/table order

1. Figure 1 — Method overview and robust-training design
2. Figure 2 — Nominal performance trade-off
3. Figure 3 — Zero-shot robustness map
4. Figure 4 — Velocity-estimation fault sweep
5. Figure 5 — Targeted robust training and held-out alpha=1.0
6. Figure 6 — Cross-robustness trade-offs
7. Table 1 — Nominal controller comparison
8. Table 2 — Severe-endpoint zero-shot robustness
9. Table 3 — Robust training and cross-robustness
