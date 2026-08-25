# Technical Contributions

This document separates the work implemented in this project from third-party reinforcement-learning infrastructure.

## Work implemented in this repository

1. **Tracking simulator and Gymnasium environments**
   - Built a 2D UAV–UGV tracking simulator with first-order UAV velocity response, velocity saturation, and acceleration limits.
   - Defined a common 10-dimensional observation space and randomized initial conditions / target speeds.

2. **Controller architectures**
   - Implemented a classical PD + target-velocity feedforward baseline.
   - Implemented Direct PPO velocity control.
   - Implemented Residual PPO around the structured PD+feedforward baseline, with residual authority limited to ±0.15 m/s per axis.

3. **Failure-mode formulation**
   - Formalized velocity-estimation contamination as

     `v_hat_UGV = v_UGV + alpha (v_UAV - v_UGV)`

   - Kept the fault localized to the explicit feedforward path so that its effect could be isolated from generic observation corruption.

4. **Robust training design**
   - Trained Robust Residual PPO from scratch with one fault severity sampled per episode from `alpha ~ U(0, 0.75)`.
   - Intentionally excluded `alpha = 1.0` from training and used it as a held-out severe fault severity.

5. **Robustness evaluation suite**
   - Implemented sweeps for velocity-estimation faults, position/velocity observation noise, observation delay, temporary vision loss, and unseen target-motion patterns.
   - Used fixed evaluation seeds for paired comparisons and retained raw episode-level CSVs.

6. **Evidence consolidation and reproducibility layer**
   - Built the Phase 6 analysis pipeline that validates expected row counts and nominal consistency, constructs a unified result table, generates the six core figures, produces publication-style tables, and writes a compact evidence-chain report.

## Third-party infrastructure

- PPO implementation: Stable-Baselines3
- Environment API: Gymnasium
- Numerical / data tools: NumPy, pandas
- Plotting: Matplotlib
- Deep-learning backend: PyTorch

The project contribution is therefore in the **control formulation, simulator/environment design, residual architecture, fault model, robust-training protocol, evaluation design, and analysis pipeline**, rather than in re-implementing PPO itself.
