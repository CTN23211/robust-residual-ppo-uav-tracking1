# Phase 6 — Consolidated Results Package

This folder adds a reproducible Phase 6 analysis layer on top of the existing Phase 5 results. It does **not** retrain any controller and does **not** rerun Phase 5 environments. It only reads the already-generated CSV files under `results/`, validates their consistency, and produces the final paper/report-level figures and tables.

## One-command execution

From the project root:

```powershell
conda activate uav_rl
cd robust-residual-ppo-uav-tracking
python -m analysis.run_phase6
```

Expected output folders:

```text
results/phase6/
plots/phase6/
tables/phase6/
```

## New code

```text
analysis/
├── __init__.py
├── phase6_common.py
├── build_master_results.py
├── make_core_tables.py
├── make_core_figures.py
├── make_phase6_report.py
└── run_phase6.py
```

### `build_master_results.py`

Reads all validated Phase 5 summary/raw CSVs, checks expected row counts, checks nominal consistency across zero-perturbation evaluators, checks the robust-training held-out contract, and creates:

```text
results/phase6/phase6_master_results.csv
results/phase6/phase6_severe_endpoints_long.csv
results/phase6/phase6_validation_report.txt
```

### `make_core_figures.py`

Creates six main figures as both 300-DPI PNG and vector SVG:

```text
fig1_method_overview
fig2_nominal_tradeoff
fig3_zero_shot_robustness_map
fig4_velocity_fault_sweep
fig5_targeted_robust_training
fig6_cross_robustness_tradeoffs
```

Figures 2, 4, and 5 use deterministic bootstrap 95% confidence intervals where raw episode data are available.

### `make_core_tables.py`

Creates CSV + Markdown + LaTeX versions of the three logical core tables:

```text
Table 1: nominal controller comparison
Table 2: severe-endpoint zero-shot robustness
Table 3A: targeted robust training
Table 3B: cross-robustness validation
```

Table 3A and 3B are two sections of the same logical Table 3.

### `make_phase6_report.py`

Creates:

```text
results/phase6/PHASE6_KEY_FINDINGS.md
```

This is the compact evidence-chain narrative for a project report or future paper draft.

## Final evidence chain

The recommended story is **not** a chronological Phase 1→5 diary. Use this order:

1. Engineering motivation: velocity-estimation contamination can produce excessive feedforward compensation and forward overshoot.
2. Architecture comparison: PD+FF vs Direct PPO vs Residual PPO.
3. Nominal trade-off: Direct PPO is fastest/most accurate; PD+FF is smoothest; Residual PPO is the middle robustness–smoothness compromise.
4. Zero-shot stress map: noise, delay, temporary perception loss, OOD target motion, and the velocity-estimation fault.
5. Failure-mode analysis: severe velocity-estimation corruption is the dominant weakness of nominal Residual PPO.
6. Targeted robust training: train from scratch with `alpha ~ U(0, 0.75)` and hold out `alpha=1.0`.
7. Headline result: held-out success increases from 58% to 95%, with lower steady RMSE and forward overshoot.
8. Cross-robustness: targeted training mostly transfers positively, but high observation noise exposes measurable control-activity/reliability trade-offs.

## Reporting rules / scientific caveats

- Call `alpha=1.0` a **held-out fault severity** or **severity extrapolation**, not a completely unseen fault type.
- The Phase 5.1 fault corrupts only the explicit feedforward estimate. Direct PPO has no equivalent feedforward path, so it is intentionally `N/A` in that row.
- For noise/delay/vision tests, the Phase 6 “primary error” is steady-state RMSE.
- For target-motion tests, the Phase 6 “primary error” is post-maneuver RMSE.
- Vision-loss testing is a last-valid-target-hold experiment under a constant-velocity target, with dropout starting at `t=3 s`; do not generalize it to arbitrary target maneuvers during dropout.
- Do not interpret reduced RMSE under some delay levels as evidence that delay is generally beneficial.
- Full RMSE alone is insufficient; always interpret it together with success, stable ratio, forward overshoot, and physical command smoothness.

## Phase 6 freeze rule

After these artifacts are generated and visually checked, freeze Phase 5/6 numerical results. Do not tune disturbance severities or retrain the robust model after inspecting the held-out results unless a new, explicitly pre-registered experiment is started.
