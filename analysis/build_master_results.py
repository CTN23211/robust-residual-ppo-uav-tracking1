from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.phase6_common import (
    PHASE6_RESULTS_DIR,
    REQUIRED_RAW_FILES,
    REQUIRED_SUMMARY_FILES,
    ensure_phase6_dirs,
    load_csv,
    require_columns,
    require_files,
    safe_float,
)


def _append(rows: list[dict], **kwargs) -> None:
    template = {
        "phase": "",
        "family": "",
        "condition": "",
        "severity": np.nan,
        "severity_unit": "",
        "method": "",
        "success_rate": np.nan,
        "full_rmse_mean": np.nan,
        "primary_error_name": "",
        "primary_error_mean": np.nan,
        "stable_ratio_mean": np.nan,
        "capture_time_mean": np.nan,
        "settling_or_recovery_time_mean": np.nan,
        "forward_overshoot_mean": np.nan,
        "command_smoothness_mean": np.nan,
        "max_command_change_mean": np.nan,
        "held_out": False,
        "source_file": "",
        "notes": "",
    }
    template.update(kwargs)
    rows.append(template)


def build_master() -> pd.DataFrame:
    ensure_phase6_dirs()
    require_files(REQUIRED_SUMMARY_FILES + REQUIRED_RAW_FILES)
    rows: list[dict] = []

    # --------------------------------------------------------
    # Phase 5.1: feedforward velocity-estimation fault
    # --------------------------------------------------------
    df = load_csv("velocity_compensation_fault_summary.csv")
    require_columns(df, ["alpha", "method", "success_rate", "rmse_mean", "steady_rmse_mean",
                         "capture_time_mean", "stable_ratio_mean", "forward_overshoot_mean",
                         "command_smoothness_mean"], "velocity_compensation_fault_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5.1",
            family="velocity_fault",
            condition="velocity_coupling_alpha",
            severity=float(r.alpha),
            severity_unit="alpha",
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="steady_rmse",
            primary_error_mean=float(r.steady_rmse_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            capture_time_mean=safe_float(r.capture_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            source_file="velocity_compensation_fault_summary.csv",
            notes="Controller-path feedforward estimate fault; PPO observations remain true. Direct PPO not exposed to this mechanism.",
        )

    # --------------------------------------------------------
    # Phase 5.2: observation noise
    # --------------------------------------------------------
    df = load_csv("observation_noise_summary.csv")
    require_columns(df, ["noise_type", "noise_std", "method", "success_rate", "rmse_mean",
                         "steady_rmse_mean", "stable_ratio_mean", "capture_time_mean",
                         "settling_time_mean", "forward_overshoot_mean", "command_smoothness_mean",
                         "max_command_change_mean"], "observation_noise_summary.csv")
    for _, r in df.iterrows():
        unit = "m" if r.noise_type == "position" else "m/s"
        _append(
            rows,
            phase="5.2",
            family=f"{r.noise_type}_noise",
            condition=f"{r.noise_type}_observation_noise",
            severity=float(r.noise_std),
            severity_unit=unit,
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="steady_rmse",
            primary_error_mean=float(r.steady_rmse_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            capture_time_mean=safe_float(r.capture_time_mean),
            settling_or_recovery_time_mean=safe_float(r.settling_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            max_command_change_mean=float(r.max_command_change_mean),
            source_file="observation_noise_summary.csv",
        )

    # --------------------------------------------------------
    # Phase 5.3: observation delay
    # --------------------------------------------------------
    df = load_csv("observation_delay_summary.csv")
    require_columns(df, ["delay_ms", "method", "success_rate", "rmse_mean", "steady_rmse_mean",
                         "stable_ratio_mean", "capture_time_mean", "settling_time_mean",
                         "forward_overshoot_mean", "command_smoothness_mean", "max_command_change_mean"],
                    "observation_delay_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5.3",
            family="observation_delay",
            condition="observation_delay",
            severity=float(r.delay_ms),
            severity_unit="ms",
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="steady_rmse",
            primary_error_mean=float(r.steady_rmse_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            capture_time_mean=safe_float(r.capture_time_mean),
            settling_or_recovery_time_mean=safe_float(r.settling_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            max_command_change_mean=float(r.max_command_change_mean),
            source_file="observation_delay_summary.csv",
            notes="Legacy settling-time values near episode end can be misleading for failed Direct PPO cases; use success/stable ratio as primary evidence.",
        )

    # --------------------------------------------------------
    # Phase 5.4: temporary vision loss
    # --------------------------------------------------------
    df = load_csv("temporary_vision_loss_summary.csv")
    require_columns(df, ["loss_duration", "method", "success_rate", "rmse_mean", "steady_rmse_mean",
                         "stable_ratio_mean", "capture_time_mean", "recovery_time_mean",
                         "forward_overshoot_mean", "command_smoothness_mean", "max_command_change_mean"],
                    "temporary_vision_loss_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5.4",
            family="vision_loss",
            condition="temporary_vision_loss",
            severity=float(r.loss_duration),
            severity_unit="s",
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="steady_rmse",
            primary_error_mean=float(r.steady_rmse_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            capture_time_mean=safe_float(r.capture_time_mean),
            settling_or_recovery_time_mean=safe_float(r.recovery_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            max_command_change_mean=float(r.max_command_change_mean),
            source_file="temporary_vision_loss_summary.csv",
            notes="Last-valid target hold with constant-velocity target; dropout begins at t=3 s during pursuit transient.",
        )

    # --------------------------------------------------------
    # Phase 5.5: unseen target motion
    # --------------------------------------------------------
    df = load_csv("unseen_trajectory_summary.csv")
    require_columns(df, ["scenario", "method", "success_rate", "rmse_mean", "post_maneuver_rmse_mean",
                         "post_stable_ratio_mean", "reacquisition_time_mean", "forward_overshoot_mean",
                         "command_smoothness_mean", "max_command_change_mean"], "unseen_trajectory_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5.5",
            family="target_motion",
            condition=str(r.scenario),
            severity=np.nan,
            severity_unit="scenario",
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="post_maneuver_rmse",
            primary_error_mean=float(r.post_maneuver_rmse_mean),
            stable_ratio_mean=float(r.post_stable_ratio_mean),
            settling_or_recovery_time_mean=safe_float(r.reacquisition_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            max_command_change_mean=float(r.max_command_change_mean),
            source_file="unseen_trajectory_summary.csv",
            notes="For persistent maneuvers, post-maneuver RMSE/P95/stable ratio are more informative than reacquisition time.",
        )

    # --------------------------------------------------------
    # Phase 5B-1: targeted robust training alpha sweep
    # --------------------------------------------------------
    df = load_csv("robust_residual_alpha_sweep_summary.csv")
    require_columns(df, ["method", "alpha", "held_out", "success_rate", "rmse_mean", "steady_rmse_mean",
                         "capture_time_mean", "stable_ratio_mean", "forward_overshoot_mean",
                         "command_smoothness_mean"], "robust_residual_alpha_sweep_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5B-1",
            family="robust_training_alpha_sweep",
            condition="velocity_coupling_alpha",
            severity=float(r.alpha),
            severity_unit="alpha",
            method=r.method,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.rmse_mean),
            primary_error_name="steady_rmse",
            primary_error_mean=float(r.steady_rmse_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            capture_time_mean=safe_float(r.capture_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            held_out=bool(r.held_out),
            source_file="robust_residual_alpha_sweep_summary.csv",
            notes="Robust policy trained from scratch with alpha ~ U(0, 0.75); alpha=1.0 is a held-out severity, not an unseen fault type.",
        )

    # --------------------------------------------------------
    # Phase 5B-2: cross-robustness
    # --------------------------------------------------------
    df = load_csv("cross_robustness_summary.csv")
    require_columns(df, ["controller", "condition", "success_rate", "full_rmse_mean", "primary_error_mean",
                         "stable_ratio_mean", "forward_overshoot_mean", "command_smoothness_mean",
                         "secondary_time_mean"], "cross_robustness_summary.csv")
    for _, r in df.iterrows():
        _append(
            rows,
            phase="5B-2",
            family="cross_robustness",
            condition=str(r.condition),
            severity=np.nan,
            severity_unit="endpoint",
            method=r.controller,
            success_rate=float(r.success_rate),
            full_rmse_mean=float(r.full_rmse_mean),
            primary_error_name="condition_specific_primary_error",
            primary_error_mean=float(r.primary_error_mean),
            stable_ratio_mean=float(r.stable_ratio_mean),
            settling_or_recovery_time_mean=safe_float(r.secondary_time_mean),
            forward_overshoot_mean=float(r.forward_overshoot_mean),
            command_smoothness_mean=float(r.command_smoothness_mean),
            source_file="cross_robustness_summary.csv",
            notes="Primary error = steady RMSE for noise/delay/vision; post-maneuver RMSE for target-motion scenarios.",
        )

    master = pd.DataFrame(rows)
    return master


def validate_inputs(master: pd.DataFrame) -> list[str]:
    """Scientific sanity checks. Fail loudly for hard inconsistencies; report softer caveats."""
    messages: list[str] = []

    expected_summary_rows = {
        "velocity_compensation_fault_summary.csv": 10,
        "observation_noise_summary.csv": 30,
        "observation_delay_summary.csv": 15,
        "temporary_vision_loss_summary.csv": 15,
        "unseen_trajectory_summary.csv": 12,
        "robust_residual_alpha_sweep_summary.csv": 10,
        "cross_robustness_summary.csv": 16,
        "cross_robustness_paired_comparison.csv": 8,
    }
    for name, expected in expected_summary_rows.items():
        actual = len(load_csv(name))
        status = "PASS" if actual == expected else "FAIL"
        messages.append(f"[{status}] {name}: {actual} rows (expected {expected})")
        if actual != expected:
            raise ValueError(f"Unexpected row count in {name}: {actual} != {expected}")

    expected_raw_rows = {
        "velocity_compensation_fault_raw.csv": 1000,
        "observation_noise_raw.csv": 3000,
        "observation_delay_raw.csv": 1500,
        "temporary_vision_loss_raw.csv": 1500,
        "unseen_trajectory_raw.csv": 1200,
        "robust_residual_alpha_sweep_raw.csv": 1000,
        "cross_robustness_raw.csv": 1600,
    }
    for name, expected in expected_raw_rows.items():
        actual = len(load_csv(name))
        status = "PASS" if actual == expected else "FAIL"
        messages.append(f"[{status}] {name}: {actual} rows (expected {expected})")
        if actual != expected:
            raise ValueError(f"Unexpected row count in {name}: {actual} != {expected}")

    # Nominal results should match across zero-perturbation evaluators.
    noise = load_csv("observation_noise_summary.csv")
    delay = load_csv("observation_delay_summary.csv")
    vision = load_csv("temporary_vision_loss_summary.csv")
    for method in ["PD+FF", "Direct PPO", "Residual PPO"]:
        n = noise[(noise.noise_type == "position") & (noise.noise_std == 0) & (noise.method == method)].iloc[0]
        d = delay[(delay.delay_ms == 0) & (delay.method == method)].iloc[0]
        v = vision[(vision.loss_duration == 0) & (vision.method == method)].iloc[0]
        values = np.array([n.rmse_mean, d.rmse_mean, v.rmse_mean], dtype=float)
        spread = float(values.max() - values.min())
        status = "PASS" if spread < 1e-6 else "FAIL"
        messages.append(f"[{status}] Nominal RMSE consistency for {method}: spread={spread:.3e}")
        if spread >= 1e-6:
            raise ValueError(f"Nominal RMSE mismatch for {method}: {values}")

    # Robust training contract.
    robust = load_csv("robust_residual_alpha_sweep_summary.csv")
    held = robust[robust.held_out == True]  # noqa: E712
    held_alphas = sorted(set(float(x) for x in held.alpha))
    if held_alphas != [1.0]:
        raise ValueError(f"Expected only alpha=1.0 held out, got {held_alphas}")
    messages.append("[PASS] Robust-training held-out contract: alpha=1.0 only.")

    messages.append(f"[PASS] Canonical master table built with {len(master)} rows.")
    messages.append("[NOTE] Direct PPO is intentionally N/A for controller-path velocity-fault tests.")
    messages.append("[NOTE] Vision-loss results are last-valid-target hold with a constant-velocity target and a t=3 s dropout anchor.")
    messages.append("[NOTE] Do not claim delay is intrinsically beneficial when steady RMSE decreases; this is configuration-specific closed-loop behavior.")
    return messages


def build_endpoint_matrix(master: pd.DataFrame) -> pd.DataFrame:
    """Long-form severe-endpoint data used by the paper heatmap/table."""
    rows: list[dict] = []

    def add(label, subset):
        for _, r in subset.iterrows():
            rows.append({
                "condition": label,
                "method": r.method,
                "success_rate": r.success_rate,
                "primary_error_name": r.primary_error_name,
                "primary_error_mean": r.primary_error_mean,
                "forward_overshoot_mean": r.forward_overshoot_mean,
                "command_smoothness_mean": r.command_smoothness_mean,
            })

    add("Velocity fault alpha=1.0", master[(master.phase == "5.1") & (master.severity == 1.0)])
    add("Position noise sigma=0.10 m", master[(master.phase == "5.2") & (master.family == "position_noise") & (master.severity == 0.10)])
    add("Velocity noise sigma=0.10 m/s", master[(master.phase == "5.2") & (master.family == "velocity_noise") & (master.severity == 0.10)])
    add("Observation delay 400 ms", master[(master.phase == "5.3") & (master.severity == 400.0)])
    add("Vision loss 1.0 s", master[(master.phase == "5.4") & (master.severity == 1.0)])
    for scenario, label in [
        ("speed_step", "Speed step"),
        ("lateral_sine", "Lateral sine"),
        ("constant_turn", "Constant turn"),
    ]:
        add(label, master[(master.phase == "5.5") & (master.condition == scenario)])

    return pd.DataFrame(rows)


def main() -> None:
    master = build_master()
    messages = validate_inputs(master)

    master_path = PHASE6_RESULTS_DIR / "phase6_master_results.csv"
    master.to_csv(master_path, index=False)

    endpoints = build_endpoint_matrix(master)
    endpoints_path = PHASE6_RESULTS_DIR / "phase6_severe_endpoints_long.csv"
    endpoints.to_csv(endpoints_path, index=False)

    report_path = PHASE6_RESULTS_DIR / "phase6_validation_report.txt"
    report_path.write_text("\n".join(messages) + "\n", encoding="utf-8")

    print("Phase 6 master data built.")
    print(f"  {master_path}")
    print(f"  {endpoints_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
