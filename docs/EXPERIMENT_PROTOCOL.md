# Experiment Protocol

## Simulation

- Time step: 0.05 s
- Episode duration: 20 s (400 steps)
- UAV command: planar velocity command
- UAV max speed: 1.0 m/s
- UAV max acceleration: 0.8 m/s²
- UAV velocity time constant: 0.30 s

## Randomized nominal tracking

At reset:

- UAV x position: Uniform(-2.0, -0.5) m
- UAV y position: Uniform(-1.0, 1.0) m
- UGV longitudinal speed: Uniform(0.20, 0.50) m/s

## Observation and action

The common policy observation has 10 dimensions:

`[relative position (2), relative velocity (2), UAV velocity (2), UGV velocity (2), previous action (2)]`

Direct PPO outputs the full normalized velocity command. Residual PPO outputs a bounded correction added to the structured PD+feedforward command.

## Residual controller

`u = v_UGV + Kp e_p + Kd e_v + Delta u_RL`

with:

- Kp = 0.45
- Kd = 0.10
- residual authority = ±0.15 m/s per axis

## Robust-training fault

`v_hat_UGV = v_UGV + alpha (v_UAV - v_UGV)`

Robust Residual PPO samples one `alpha` per episode from `U(0, 0.75)`. `alpha=1.0` is excluded from training and used only as a held-out severity of the same fault family.

## Evaluation

Core robustness evaluations use 100 episodes per method / severity with test seeds starting at 1000 where specified by the evaluation scripts.

Metrics include success rate, full RMSE, condition-specific tracking error, capture / recovery time, stable ratio, forward overshoot, and physical-command smoothness.

See `docs/PHASE6_ANALYSIS.md` for the frozen reporting rules and interpretation boundaries.
