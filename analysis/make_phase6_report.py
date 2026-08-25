from __future__ import annotations

import pandas as pd

from analysis.phase6_common import PHASE6_RESULTS_DIR, load_csv, percent_change


def pct_reduction(new, old):
    return 100.0 * (old - new) / old


def main() -> None:
    nominal = load_csv("observation_delay_summary.csv")
    nominal = nominal[nominal.delay_ms == 0]
    fault = load_csv("velocity_compensation_fault_summary.csv")
    robust = load_csv("robust_residual_alpha_sweep_summary.csv")
    cross = load_csv("cross_robustness_paired_comparison.csv")
    noise = load_csv("observation_noise_summary.csv")
    delay = load_csv("observation_delay_summary.csv")
    vision = load_csv("temporary_vision_loss_summary.csv")
    traj = load_csv("unseen_trajectory_summary.csv")

    pd0 = nominal[nominal.method == "PD+FF"].iloc[0]
    direct0 = nominal[nominal.method == "Direct PPO"].iloc[0]
    residual0 = nominal[nominal.method == "Residual PPO"].iloc[0]

    f_pd = fault[(fault.alpha == 1.0) & (fault.method == "PD+FF")].iloc[0]
    f_res = fault[(fault.alpha == 1.0) & (fault.method == "Residual PPO")].iloc[0]

    n1 = robust[(robust.alpha == 1.0) & (robust.method == "Nominal Residual PPO")].iloc[0]
    r1 = robust[(robust.alpha == 1.0) & (robust.method == "Robust Residual PPO")].iloc[0]

    pos_direct = noise[(noise.noise_type == "position") & (noise.noise_std == 0.10) & (noise.method == "Direct PPO")].iloc[0]
    pos_res = noise[(noise.noise_type == "position") & (noise.noise_std == 0.10) & (noise.method == "Residual PPO")].iloc[0]
    vel_direct = noise[(noise.noise_type == "velocity") & (noise.noise_std == 0.10) & (noise.method == "Direct PPO")].iloc[0]
    vel_res = noise[(noise.noise_type == "velocity") & (noise.noise_std == 0.10) & (noise.method == "Residual PPO")].iloc[0]
    d400 = delay[(delay.delay_ms == 400) & (delay.method == "Direct PPO")].iloc[0]
    r400 = delay[(delay.delay_ms == 400) & (delay.method == "Residual PPO")].iloc[0]
    v1d = vision[(vision.loss_duration == 1.0) & (vision.method == "Direct PPO")].iloc[0]
    v1r = vision[(vision.loss_duration == 1.0) & (vision.method == "Residual PPO")].iloc[0]
    ct_d = traj[(traj.scenario == "constant_turn") & (traj.method == "Direct PPO")].iloc[0]
    ct_r = traj[(traj.scenario == "constant_turn") & (traj.method == "Residual PPO")].iloc[0]

    lines = []
    lines.append("# Phase 6 — Consolidated Results and Evidence Chain")
    lines.append("")
    lines.append("## 1. Core study question")
    lines.append("")
    lines.append(
        "The project compares a classical PD+feedforward controller, a Direct PPO controller, and a Residual PPO controller, "
        "then tests whether targeted domain randomization of the feedforward velocity-estimation fault improves severe-fault robustness without broad cross-perturbation collapse."
    )
    lines.append("")
    lines.append("## 2. Nominal architecture trade-off")
    lines.append("")
    lines.append(
        f"Under nominal randomized tracking, Direct PPO is fastest and most accurate (steady RMSE {direct0.steady_rmse_mean:.4f} m; capture {direct0.capture_time_mean:.3f} s), "
        f"PD+FF is smoothest but slower (steady RMSE {pd0.steady_rmse_mean:.4f} m; capture {pd0.capture_time_mean:.3f} s), and Residual PPO occupies the intermediate trade-off "
        f"(steady RMSE {residual0.steady_rmse_mean:.4f} m; capture {residual0.capture_time_mean:.3f} s; command smoothness {residual0.command_smoothness_mean:.6f})."
    )
    lines.append("")
    lines.append("## 3. Zero-shot robustness pattern")
    lines.append("")
    lines.append(
        f"At the controller-path velocity fault alpha=1.0, PD+FF success falls to {100*f_pd.success_rate:.0f}% with {f_pd.forward_overshoot_mean:.3f} m forward overshoot, while nominal Residual PPO retains {100*f_res.success_rate:.0f}% success with {f_res.forward_overshoot_mean:.3f} m overshoot."
    )
    lines.append(
        f"Under severe observation noise, Residual PPO retains {100*pos_res.success_rate:.0f}% success at sigma_p=0.10 m and {100*vel_res.success_rate:.0f}% at sigma_v=0.10 m/s, compared with Direct PPO at {100*pos_direct.success_rate:.0f}% and {100*vel_direct.success_rate:.0f}% respectively."
    )
    lines.append(
        f"At 400 ms observation delay, Direct PPO success falls to {100*d400.success_rate:.0f}% whereas Residual PPO remains at {100*r400.success_rate:.0f}%. At 1.0 s temporary vision loss, all controllers still succeed, but Direct PPO forward overshoot is {v1d.forward_overshoot_mean:.3f} m versus {v1r.forward_overshoot_mean:.3f} m for Residual PPO."
    )
    lines.append(
        f"For the unseen constant-turn target, Direct PPO success is {100*ct_d.success_rate:.0f}% with post-maneuver RMSE {ct_d.post_maneuver_rmse_mean:.4f} m, while Residual PPO remains at {100*ct_r.success_rate:.0f}% success with {ct_r.post_maneuver_rmse_mean:.4f} m RMSE."
    )
    lines.append("")
    lines.append("## 4. Targeted robust training closes the main failure mode")
    lines.append("")
    lines.append(
        "Robust Residual PPO is trained from scratch with alpha sampled once per episode from U(0, 0.75). Alpha=1.0 is therefore a held-out severity (not a wholly unseen fault type)."
    )
    lines.append(
        f"At held-out alpha=1.0, success increases from {100*n1.success_rate:.0f}% to {100*r1.success_rate:.0f}% (+{100*(r1.success_rate-n1.success_rate):.0f} percentage points), "
        f"steady RMSE decreases by {pct_reduction(r1.steady_rmse_mean, n1.steady_rmse_mean):.1f}%, and forward overshoot decreases by {pct_reduction(r1.forward_overshoot_mean, n1.forward_overshoot_mean):.1f}%."
    )
    lines.append("")
    lines.append("## 5. Cross-robustness validation")
    lines.append("")
    improved = int((cross.primary_error_change_percent < 0).sum())
    total = len(cross)
    worst_success = cross.loc[cross.success_delta_percentage_points.idxmin()]
    lines.append(
        f"Targeted robust training improves the condition-specific primary tracking error in {improved}/{total} cross-validation conditions. "
        f"The only success-rate regression occurs for {worst_success.condition}, where success changes by {worst_success.success_delta_percentage_points:.0f} percentage points."
    )
    ct = cross[cross.condition == "Constant turn"].iloc[0]
    pos = cross[cross.condition == "Position noise 0.10 m"].iloc[0]
    lines.append(
        f"The main accuracy trade-off is constant-turn tracking (+{ct.primary_error_change_percent:.1f}% primary error), while the clearest control-activity trade-off appears under severe position noise (+{pos.command_smoothness_change_percent:.1f}% smoothness cost)."
    )
    lines.append("")
    lines.append("## 6. Claims supported by the data")
    lines.append("")
    lines.append("- Direct PPO provides the strongest nominal speed/accuracy but is the most sensitive to observation delay and high observation noise.")
    lines.append("- Residual PPO offers a stronger robustness–smoothness compromise by limiting learned control authority around a structured PD+feedforward baseline.")
    lines.append("- Targeted alpha-domain randomization substantially improves held-out severe-fault performance and mostly transfers positively to unrelated perturbations.")
    lines.append("- Robust training is not uniformly free: severe observation-noise control activity and a small high-velocity-noise success regression remain measurable trade-offs.")
    lines.append("")
    lines.append("## 7. Interpretation constraints")
    lines.append("")
    lines.append("- Phase 5.1 corrupts only the feedforward velocity estimate; PPO observations remain true. Do not call it a full sensor-observation corruption test.")
    lines.append("- Alpha=1.0 is a held-out severity of the same fault family, not a completely unseen fault type.")
    lines.append("- Temporary vision loss uses last-valid-target hold while the target remains constant velocity; recovery-time comparisons are confounded by the moving restoration time during the pursuit transient.")
    lines.append("- Lower steady RMSE at some delayed-controller settings should not be generalized as evidence that delay is beneficial.")
    lines.append("- Full-episode RMSE can hide transient overshoot; use success, condition-specific primary error, stable ratio, and forward overshoot together.")
    lines.append("")
    lines.append("## 8. Recommended main-paper figure/table order")
    lines.append("")
    lines.append("1. Figure 1 — Method overview and robust-training design")
    lines.append("2. Figure 2 — Nominal performance trade-off")
    lines.append("3. Figure 3 — Zero-shot robustness map")
    lines.append("4. Figure 4 — Velocity-estimation fault sweep")
    lines.append("5. Figure 5 — Targeted robust training and held-out alpha=1.0")
    lines.append("6. Figure 6 — Cross-robustness trade-offs")
    lines.append("7. Table 1 — Nominal controller comparison")
    lines.append("8. Table 2 — Severe-endpoint zero-shot robustness")
    lines.append("9. Table 3 — Robust training and cross-robustness")

    path = PHASE6_RESULTS_DIR / "PHASE6_KEY_FINDINGS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Phase 6 key findings saved to: {path}")


if __name__ == "__main__":
    main()
