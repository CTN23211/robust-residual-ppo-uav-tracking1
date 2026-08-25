from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.phase6_common import (
    TABLES_DIR,
    ensure_phase6_dirs,
    load_csv,
    markdown_table,
)


def save_table(df: pd.DataFrame, stem: str, formats: dict[str, str] | None = None) -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    tex_path = TABLES_DIR / f"{stem}.tex"

    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df, formats) + "\n", encoding="utf-8")

    # LaTeX keeps raw numeric precision moderate; user can style booktabs later.
    tex_path.write_text(
        df.to_latex(index=False, na_rep="--", float_format=lambda x: f"{x:.4g}"),
        encoding="utf-8",
    )


def table1_nominal() -> pd.DataFrame:
    df = load_csv("observation_delay_summary.csv")
    d = df[df.delay_ms == 0].copy()
    d["Success (%)"] = 100 * d.success_rate
    d["Stable ratio (%)"] = 100 * d.stable_ratio_mean
    out = d[[
        "method", "Success (%)", "rmse_mean", "steady_rmse_mean", "capture_time_mean",
        "Stable ratio (%)", "forward_overshoot_mean", "command_smoothness_mean",
    ]].rename(columns={
        "method": "Method",
        "rmse_mean": "Full RMSE (m)",
        "steady_rmse_mean": "Steady RMSE (m)",
        "capture_time_mean": "Capture time (s)",
        "forward_overshoot_mean": "Forward overshoot (m)",
        "command_smoothness_mean": "Command smoothness",
    })
    return out


def table2_zero_shot() -> pd.DataFrame:
    fault = load_csv("velocity_compensation_fault_summary.csv")
    noise = load_csv("observation_noise_summary.csv")
    delay = load_csv("observation_delay_summary.csv")
    vision = load_csv("temporary_vision_loss_summary.csv")
    traj = load_csv("unseen_trajectory_summary.csv")

    condition_specs = [
        ("Velocity fault", "alpha=1.0", "steady RMSE", fault[fault.alpha == 1.0], "steady_rmse_mean"),
        ("Position noise", "sigma_p=0.10 m", "steady RMSE", noise[(noise.noise_type == "position") & (noise.noise_std == 0.10)], "steady_rmse_mean"),
        ("Velocity noise", "sigma_v=0.10 m/s", "steady RMSE", noise[(noise.noise_type == "velocity") & (noise.noise_std == 0.10)], "steady_rmse_mean"),
        ("Observation delay", "400 ms", "steady RMSE", delay[delay.delay_ms == 400], "steady_rmse_mean"),
        ("Vision loss", "1.0 s", "steady RMSE", vision[vision.loss_duration == 1.0], "steady_rmse_mean"),
        ("Speed step", "+0.20 m/s at t=6 s", "post-maneuver RMSE", traj[traj.scenario == "speed_step"], "post_maneuver_rmse_mean"),
        ("Lateral sine", "A=0.40 m, T=8 s", "post-maneuver RMSE", traj[traj.scenario == "lateral_sine"], "post_maneuver_rmse_mean"),
        ("Constant turn", "omega=0.25 rad/s", "post-maneuver RMSE", traj[traj.scenario == "constant_turn"], "post_maneuver_rmse_mean"),
    ]

    rows = []
    for condition, severity, metric_name, sdf, metric_col in condition_specs:
        row = {"Condition": condition, "Severity": severity, "Primary error metric": metric_name}
        for method in ["PD+FF", "Direct PPO", "Residual PPO"]:
            m = sdf[sdf.method == method]
            if len(m) == 0:
                row[f"{method} success (%)"] = np.nan
                row[f"{method} error (m)"] = np.nan
            else:
                r = m.iloc[0]
                row[f"{method} success (%)"] = 100 * float(r.success_rate)
                row[f"{method} error (m)"] = float(r[metric_col])
        rows.append(row)
    return pd.DataFrame(rows)


def table3a_robust_training() -> pd.DataFrame:
    df = load_csv("robust_residual_alpha_sweep_summary.csv")
    rows = []
    for alpha, region in [(0.0, "Nominal"), (0.75, "Training boundary"), (1.0, "Held-out severity")]:
        sub = df[df.alpha == alpha]
        n = sub[sub.method == "Nominal Residual PPO"].iloc[0]
        r = sub[sub.method == "Robust Residual PPO"].iloc[0]
        metrics = [
            ("Success (%)", 100 * n.success_rate, 100 * r.success_rate, "pp"),
            ("Steady RMSE (m)", n.steady_rmse_mean, r.steady_rmse_mean, "%"),
            ("Capture time (s)", n.capture_time_mean, r.capture_time_mean, "%"),
            ("Forward overshoot (m)", n.forward_overshoot_mean, r.forward_overshoot_mean, "%"),
            ("Command smoothness", n.command_smoothness_mean, r.command_smoothness_mean, "%"),
        ]
        for metric, old, new, change_type in metrics:
            if change_type == "pp":
                change = new - old
                change_label = "percentage points"
            else:
                change = 100 * (new - old) / old if abs(old) > 1e-12 else np.nan
                change_label = "%"
            rows.append({
                "alpha": alpha,
                "Region": region,
                "Metric": metric,
                "Nominal Residual PPO": old,
                "Robust Residual PPO": new,
                "Change": change,
                "Change unit": change_label,
            })
    return pd.DataFrame(rows)


def table3b_cross() -> pd.DataFrame:
    df = load_csv("cross_robustness_paired_comparison.csv").copy()
    return df[[
        "condition",
        "nominal_success_rate",
        "robust_success_rate",
        "success_delta_percentage_points",
        "primary_error_change_percent",
        "command_smoothness_change_percent",
        "nominal_fail_robust_success",
        "nominal_success_robust_fail",
        "robust_lower_primary_error_count",
    ]].rename(columns={
        "condition": "Condition",
        "nominal_success_rate": "Nominal success",
        "robust_success_rate": "Robust success",
        "success_delta_percentage_points": "Success delta (pp)",
        "primary_error_change_percent": "Primary error change (%)",
        "command_smoothness_change_percent": "Smoothness change (%)",
        "nominal_fail_robust_success": "Rescued seeds",
        "nominal_success_robust_fail": "Regressed seeds",
        "robust_lower_primary_error_count": "Robust lower-error seeds /100",
    })


def main() -> None:
    ensure_phase6_dirs()

    t1 = table1_nominal()
    save_table(t1, "table1_nominal_controller_comparison", {
        "Success (%)": ".1f",
        "Full RMSE (m)": ".4f",
        "Steady RMSE (m)": ".4f",
        "Capture time (s)": ".3f",
        "Stable ratio (%)": ".2f",
        "Forward overshoot (m)": ".4f",
        "Command smoothness": ".6f",
    })

    t2 = table2_zero_shot()
    formats2 = {c: ".2f" for c in t2.columns if "success" in c}
    formats2.update({c: ".4f" for c in t2.columns if "error (m)" in c})
    save_table(t2, "table2_zero_shot_severe_robustness", formats2)

    t3a = table3a_robust_training()
    save_table(t3a, "table3a_targeted_robust_training", {
        "alpha": ".2f",
        "Nominal Residual PPO": ".6f",
        "Robust Residual PPO": ".6f",
        "Change": ".2f",
    })

    t3b = table3b_cross()
    t3b_display = t3b.copy()
    t3b_display["Nominal success"] *= 100
    t3b_display["Robust success"] *= 100
    save_table(t3b_display, "table3b_cross_robustness", {
        "Nominal success": ".1f",
        "Robust success": ".1f",
        "Success delta (pp)": ".1f",
        "Primary error change (%)": ".1f",
        "Smoothness change (%)": ".1f",
    })

    combined = (
        "# Table 3. Robust training and cross-robustness\n\n"
        "## Table 3A. Targeted velocity-fault robust training\n\n"
        + markdown_table(t3a, {
            "alpha": ".2f",
            "Nominal Residual PPO": ".6f",
            "Robust Residual PPO": ".6f",
            "Change": ".2f",
        })
        + "\n\n## Table 3B. Cross-perturbation validation\n\n"
        + markdown_table(t3b_display, {
            "Nominal success": ".1f",
            "Robust success": ".1f",
            "Success delta (pp)": ".1f",
            "Primary error change (%)": ".1f",
            "Smoothness change (%)": ".1f",
        })
        + "\n"
    )
    (TABLES_DIR / "table3_robust_training_and_cross_robustness.md").write_text(combined, encoding="utf-8")

    print(f"Phase 6 core tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
