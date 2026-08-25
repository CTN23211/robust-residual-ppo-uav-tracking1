from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

from analysis.phase6_common import (
    METHOD_COLORS,
    METHOD_ORDER,
    PHASE6_RESULTS_DIR,
    PLOTS_DIR,
    RESIDUAL_ORDER,
    bootstrap_mean_ci,
    ensure_phase6_dirs,
    load_csv,
    save_png_svg,
)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def _box(ax, xy, width, height, text, fc="#F7F7F7", ec="#333333", lw=1.2, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, text=None):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#444444")
    ax.add_patch(arr)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=9)


def fig1_method_overview():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Tracking architectures and targeted robust training", ha="center", va="top", fontsize=14, weight="bold")

    _box(ax, (0.04, 0.72), 0.18, 0.13, "State / observation\n$e_p, e_v, v_{UAV}, v_{UGV}$", fc="#EAF2F8")
    _box(ax, (0.32, 0.72), 0.23, 0.13, "PD + feedforward\n$u_{base}=\\hat v_{UGV}+K_p e_p+K_d e_v$", fc="#E8F5E9")
    _box(ax, (0.32, 0.43), 0.23, 0.13, "Residual PPO\n$\\Delta u=0.15\\,\\pi(s)$", fc="#F3E5F5")
    _box(ax, (0.66, 0.58), 0.20, 0.16, "Residual control\n$u=clip(u_{base}+\\Delta u)$", fc="#FFF8E1")
    _box(ax, (0.66, 0.28), 0.20, 0.13, "Direct PPO\n$u=\\pi(s)$", fc="#FFF3E0")
    _box(ax, (0.90, 0.48), 0.08, 0.15, "UAV\nplant", fc="#ECEFF1")

    _arrow(ax, (0.22, 0.785), (0.32, 0.785))
    _arrow(ax, (0.22, 0.755), (0.32, 0.50), "state")
    _arrow(ax, (0.55, 0.785), (0.66, 0.675), "$u_{base}$")
    _arrow(ax, (0.55, 0.50), (0.66, 0.625), "$\\Delta u$")
    _arrow(ax, (0.22, 0.735), (0.66, 0.345), "state")
    _arrow(ax, (0.86, 0.66), (0.90, 0.575))
    _arrow(ax, (0.86, 0.345), (0.90, 0.535))

    # Fault and training block
    ax.add_patch(Rectangle((0.05, 0.08), 0.50, 0.22, fill=False, linestyle="--", linewidth=1.3, edgecolor="#555555"))
    ax.text(0.06, 0.275, "Velocity-estimation fault model", fontsize=10, weight="bold", va="top")
    ax.text(0.08, 0.205, "$\\hat v_{UGV}=v_{UGV}+\\alpha(v_{UAV}-v_{UGV})$", fontsize=12)
    ax.text(0.08, 0.135, "Robust training:  $\\alpha \\sim U(0,0.75)$", fontsize=10)
    ax.text(0.08, 0.095, "Held-out severity:  $\\alpha=1.0$", fontsize=10, weight="bold")
    _arrow(ax, (0.47, 0.30), (0.43, 0.72), "corrupt FF estimate")

    ax.text(0.64, 0.11,
            "Phase 6 evidence chain:\nnominal trade-off  →  zero-shot stress tests  →  failure mode\n→ targeted robust training  →  cross-robustness validation",
            fontsize=10, va="bottom", ha="left")

    save_png_svg(fig, "fig1_method_overview")
    plt.close(fig)


def _nominal_raw():
    df = load_csv("observation_delay_raw.csv")
    return df[df.delay_steps == 0].copy()


def fig2_nominal_tradeoff():
    raw = _nominal_raw()
    metrics = [
        ("steady_rmse", "Steady RMSE (m)", False),
        ("capture_time", "Capture time (s)", False),
        ("command_smoothness", "Physical command smoothness", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    for ax, (metric, ylabel, logscale) in zip(axes, metrics):
        means, lows, highs = [], [], []
        for i, method in enumerate(METHOD_ORDER):
            vals = raw.loc[raw.method == method, metric].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_mean_ci(vals, seed=20260820 + i)
            means.append(mean); lows.append(lo); highs.append(hi)
        x = np.arange(len(METHOD_ORDER))
        yerr = np.array([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
        bars = ax.bar(x, means, yerr=yerr, capsize=3,
                      color=[METHOD_COLORS[m] for m in METHOD_ORDER], alpha=0.9)
        ax.set_xticks(x, ["PD+FF", "Direct\nPPO", "Residual\nPPO"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if logscale:
            ax.set_yscale("log")
        for b, v in zip(bars, means):
            if logscale:
                ax.text(b.get_x()+b.get_width()/2, v*1.25, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(b.get_x()+b.get_width()/2, v + 0.03*max(means), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_title("(a) Tracking precision")
    axes[1].set_title("(b) Initial capture")
    axes[2].set_title("(c) Control activity")
    fig.suptitle("Nominal performance reveals a speed–accuracy–smoothness trade-off", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, "fig2_nominal_tradeoff")
    plt.close(fig)


def _severe_matrices():
    fault = load_csv("velocity_compensation_fault_summary.csv")
    noise = load_csv("observation_noise_summary.csv")
    delay = load_csv("observation_delay_summary.csv")
    vision = load_csv("temporary_vision_loss_summary.csv")
    traj = load_csv("unseen_trajectory_summary.csv")

    conditions = [
        "Velocity fault\n$\\alpha=1.0$",
        "Position noise\n$\\sigma_p=0.10$ m",
        "Velocity noise\n$\\sigma_v=0.10$ m/s",
        "Delay\n400 ms",
        "Vision loss\n1.0 s",
        "Speed\nstep",
        "Lateral\nsine",
        "Constant\nturn",
    ]
    success = np.full((len(conditions), 3), np.nan)
    ratio = np.full_like(success, np.nan)

    # 1) velocity fault
    for j, method in enumerate(METHOD_ORDER):
        cur = fault[(fault.alpha == 1.0) & (fault.method == method)]
        base = fault[(fault.alpha == 0.0) & (fault.method == method)]
        if len(cur):
            success[0, j] = float(cur.iloc[0].success_rate)
            ratio[0, j] = float(cur.iloc[0].steady_rmse_mean / base.iloc[0].steady_rmse_mean)

    # 2-3) noise
    for row_idx, ntype in [(1, "position"), (2, "velocity")]:
        for j, method in enumerate(METHOD_ORDER):
            cur = noise[(noise.noise_type == ntype) & (noise.noise_std == 0.10) & (noise.method == method)].iloc[0]
            base = noise[(noise.noise_type == ntype) & (noise.noise_std == 0.0) & (noise.method == method)].iloc[0]
            success[row_idx, j] = cur.success_rate
            ratio[row_idx, j] = cur.steady_rmse_mean / base.steady_rmse_mean

    # 4) delay
    for j, method in enumerate(METHOD_ORDER):
        cur = delay[(delay.delay_ms == 400) & (delay.method == method)].iloc[0]
        base = delay[(delay.delay_ms == 0) & (delay.method == method)].iloc[0]
        success[3, j] = cur.success_rate
        ratio[3, j] = cur.steady_rmse_mean / base.steady_rmse_mean

    # 5) vision loss
    for j, method in enumerate(METHOD_ORDER):
        cur = vision[(vision.loss_duration == 1.0) & (vision.method == method)].iloc[0]
        base = vision[(vision.loss_duration == 0.0) & (vision.method == method)].iloc[0]
        success[4, j] = cur.success_rate
        ratio[4, j] = cur.steady_rmse_mean / base.steady_rmse_mean

    # 6-8) trajectory
    for row_idx, scenario in [(5, "speed_step"), (6, "lateral_sine"), (7, "constant_turn")]:
        for j, method in enumerate(METHOD_ORDER):
            cur = traj[(traj.scenario == scenario) & (traj.method == method)].iloc[0]
            base = traj[(traj.scenario == "nominal") & (traj.method == method)].iloc[0]
            success[row_idx, j] = cur.success_rate
            ratio[row_idx, j] = cur.post_maneuver_rmse_mean / base.post_maneuver_rmse_mean

    return conditions, success, ratio


def fig3_zero_shot_map():
    conditions, success, ratio = _severe_matrices()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.0), gridspec_kw={"width_ratios": [1, 1.1]})

    # Success heatmap
    ax = axes[0]
    im = ax.imshow(success * 100, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(3), ["PD+FF", "Direct PPO", "Residual PPO"])
    ax.set_yticks(range(len(conditions)), conditions)
    ax.set_title("(a) Success rate (%)")
    for i in range(success.shape[0]):
        for j in range(success.shape[1]):
            if np.isnan(success[i, j]):
                txt = "N/A"
            else:
                txt = f"{100*success[i,j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Relative error heatmap uses log10 to handle 0.1x ... 26x.
    ax = axes[1]
    log_ratio = np.log10(ratio)
    finite = log_ratio[np.isfinite(log_ratio)]
    vmax = max(abs(finite.min()), abs(finite.max()))
    im2 = ax.imshow(log_ratio, vmin=-vmax, vmax=vmax, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(3), ["PD+FF", "Direct PPO", "Residual PPO"])
    ax.set_yticks(range(len(conditions)), [""] * len(conditions))
    ax.set_title("(b) Tracking-error ratio vs matched nominal")
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            txt = "N/A" if np.isnan(ratio[i, j]) else f"{ratio[i,j]:.1f}×"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(error / matched nominal error)")

    fig.suptitle("Zero-shot robustness map at pre-defined severe endpoints", fontsize=13, weight="bold")
    fig.text(0.5, 0.015,
             "Primary error: steady RMSE for fault/noise/delay/vision; post-maneuver RMSE for target-motion scenarios. "
             "Direct PPO is N/A for the controller-path feedforward fault.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_png_svg(fig, "fig3_zero_shot_robustness_map")
    plt.close(fig)


def _group_ci(raw, group_cols, metric, seed_offset=0):
    records = []
    for idx, (keys, g) in enumerate(raw.groupby(group_cols, sort=True)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean, lo, hi = bootstrap_mean_ci(g[metric].to_numpy(), seed=20260820 + seed_offset + idx)
        rec = dict(zip(group_cols, keys))
        rec.update(mean=mean, lower=lo, upper=hi)
        records.append(rec)
    return pd.DataFrame(records)


def fig4_velocity_fault_sweep():
    raw = load_csv("velocity_compensation_fault_raw.csv")
    methods = ["PD+FF", "Residual PPO"]
    metrics = [
        ("success", "Success rate (%)", 100.0),
        ("forward_overshoot", "Forward overshoot (m)", 1.0),
        ("steady_rmse", "Steady RMSE (m)", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        ci = _group_ci(raw, ["method", "alpha"], metric, seed_offset=10)
        for method in methods:
            d = ci[ci.method == method].sort_values("alpha")
            x = d.alpha.to_numpy()
            y = scale * d["mean"].to_numpy()
            lo = scale * d.lower.to_numpy(); hi = scale * d.upper.to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, label=method, color=METHOD_COLORS[method])
            ax.fill_between(x, lo, hi, alpha=0.15, color=METHOD_COLORS[method])
        ax.set_xlabel("Velocity-coupling severity $\\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("(a) Task success")
    axes[1].set_title("(b) Forward overshoot")
    axes[2].set_title("(c) Steady tracking")
    axes[0].legend(loc="best")
    fig.suptitle("Zero-shot response to feedforward velocity-estimation corruption", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, "fig4_velocity_fault_sweep")
    plt.close(fig)


def fig5_targeted_robust_training():
    raw = load_csv("robust_residual_alpha_sweep_raw.csv")
    metrics = [
        ("success", "Success rate (%)", 100.0),
        ("forward_overshoot", "Forward overshoot (m)", 1.0),
        ("steady_rmse", "Steady RMSE (m)", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        ci = _group_ci(raw, ["method", "alpha"], metric, seed_offset=100)
        for method in RESIDUAL_ORDER:
            d = ci[ci.method == method].sort_values("alpha")
            x = d.alpha.to_numpy(); y = scale * d["mean"].to_numpy()
            lo = scale * d.lower.to_numpy(); hi = scale * d.upper.to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, label=method, color=METHOD_COLORS[method])
            ax.fill_between(x, lo, hi, alpha=0.15, color=METHOD_COLORS[method])
        ax.axvline(0.75, color="#555555", linestyle="--", linewidth=1.2)
        ax.axvspan(0.75, 1.02, color="#999999", alpha=0.08)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Velocity-coupling severity $\\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].set_title("(a) Task success")
    axes[1].set_title("(b) Forward overshoot")
    axes[2].set_title("(c) Steady tracking")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].text(0.77, axes[1].get_ylim()[1]*0.92, "held-out\nseverity", fontsize=8, va="top")

    # Headline annotation at alpha=1.
    s = load_csv("robust_residual_alpha_sweep_summary.csv")
    n = s[(s.method == "Nominal Residual PPO") & (s.alpha == 1.0)].iloc[0]
    r = s[(s.method == "Robust Residual PPO") & (s.alpha == 1.0)].iloc[0]
    axes[0].annotate("58% → 95%", xy=(1.0, 95), xytext=(0.63, 72),
                     arrowprops=dict(arrowstyle="->", lw=1), fontsize=9, weight="bold")
    axes[1].text(0.98, max(n.forward_overshoot_mean, r.forward_overshoot_mean)*0.60,
                 f"−{100*(n.forward_overshoot_mean-r.forward_overshoot_mean)/n.forward_overshoot_mean:.1f}%",
                 ha="right", fontsize=9, weight="bold")
    axes[2].text(0.98, max(n.steady_rmse_mean, r.steady_rmse_mean)*0.60,
                 f"−{100*(n.steady_rmse_mean-r.steady_rmse_mean)/n.steady_rmse_mean:.1f}%",
                 ha="right", fontsize=9, weight="bold")

    fig.suptitle("Targeted fault randomization generalizes to held-out severity $\\alpha=1.0$", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_png_svg(fig, "fig5_targeted_robust_training")
    plt.close(fig)


def fig6_cross_robustness_tradeoffs():
    df = load_csv("cross_robustness_paired_comparison.csv").copy()
    # Keep order from the validated evaluator.
    labels = [
        "Nominal", "Position noise", "Velocity noise", "Delay 400 ms",
        "Vision loss", "Speed step", "Lateral sine", "Constant turn",
    ]
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)

    error_change = df.primary_error_change_percent.to_numpy(dtype=float)
    smooth_change = df.command_smoothness_change_percent.to_numpy(dtype=float)

    axes[0].barh(y, error_change, color=["#59A14F" if v <= 0 else "#E15759" for v in error_change])
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Primary tracking error change (%)")
    axes[0].set_title("(a) Robust vs nominal tracking error")
    axes[0].grid(axis="x", alpha=0.25)
    for i, (v, pp) in enumerate(zip(error_change, df.success_delta_percentage_points)):
        xtext = v + (2 if v >= 0 else -2)
        axes[0].text(xtext, i, f"{v:+.1f}%\nΔS={pp:+.0f} pp",
                     ha="left" if v >= 0 else "right", va="center", fontsize=8)

    axes[1].barh(y, smooth_change, color=["#59A14F" if v <= 0 else "#E15759" for v in smooth_change])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel("Command smoothness cost change (%)")
    axes[1].set_title("(b) Control-activity trade-off")
    axes[1].grid(axis="x", alpha=0.25)
    for i, v in enumerate(smooth_change):
        xtext = v + (4 if v >= 0 else -4)
        axes[1].text(xtext, i, f"{v:+.1f}%", ha="left" if v >= 0 else "right", va="center", fontsize=8)

    fig.suptitle("Cross-robustness validation: broad transfer with identifiable trade-offs", fontsize=13, weight="bold")
    fig.text(0.5, 0.015, "Negative change = improvement for both plotted metrics. ΔS = success-rate change in percentage points.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_png_svg(fig, "fig6_cross_robustness_tradeoffs")
    plt.close(fig)



def fig7_3d_robustness_landscape():
    """Create a continuous 3D landscape from paired alpha-sweep evaluations.

    The surface is not an additional experiment. It visualizes the empirical
    paired advantage of Robust Residual PPO over Nominal Residual PPO as a
    function of velocity-fault severity and randomized UGV target speed.

    Raw paired episodes are aggregated into 8 equal-width target-speed bins
    at each tested alpha. A two-stage linear interpolation is then used only
    to render a continuous surface between those empirical cell means.
    """
    raw = load_csv("robust_residual_alpha_sweep_raw.csv")

    key_cols = ["alpha", "seed", "initial_uav_x", "initial_uav_y", "ugv_vx"]
    nominal = raw[raw.method == "Nominal Residual PPO"].copy()
    robust = raw[raw.method == "Robust Residual PPO"].copy()
    paired = nominal.merge(robust, on=key_cols, suffixes=("_nominal", "_robust"), validate="one_to_one")
    paired["steady_rmse_reduction"] = paired["steady_rmse_nominal"] - paired["steady_rmse_robust"]

    # Equal-width speed bins retain the actual randomized target-speed range.
    speed_edges = np.linspace(paired.ugv_vx.min(), paired.ugv_vx.max(), 9)
    paired["speed_bin"] = pd.cut(paired.ugv_vx, bins=speed_edges, include_lowest=True)
    cells = (
        paired.groupby(["alpha", "speed_bin"], observed=True)
        .agg(
            ugv_vx_mean=("ugv_vx", "mean"),
            steady_rmse_reduction_mean=("steady_rmse_reduction", "mean"),
            paired_episodes=("steady_rmse_reduction", "size"),
        )
        .reset_index()
    )

    # Save the empirical support used by the visualization.
    out_cells = PHASE6_RESULTS_DIR / "fig7_robustness_landscape_cells.csv"
    cells.drop(columns=["speed_bin"]).to_csv(out_cells, index=False)

    alpha_support = np.sort(cells.alpha.unique())
    speed_support = np.sort(cells.ugv_vx_mean.unique())
    matrix = (
        cells.pivot(index="alpha", columns="ugv_vx_mean", values="steady_rmse_reduction_mean")
        .loc[alpha_support, speed_support]
        .to_numpy(dtype=float)
    )

    # Continuous piecewise-linear interpolation over the measured support.
    alpha_dense = np.linspace(alpha_support.min(), alpha_support.max(), 121)
    speed_dense = np.linspace(speed_support.min(), speed_support.max(), 121)

    speed_interp = np.vstack([np.interp(speed_dense, speed_support, row) for row in matrix])
    z_dense = np.empty((len(alpha_dense), len(speed_dense)), dtype=float)
    for j in range(len(speed_dense)):
        z_dense[:, j] = np.interp(alpha_dense, alpha_support, speed_interp[:, j])

    A, V = np.meshgrid(alpha_dense, speed_dense, indexing="ij")

    fig = plt.figure(figsize=(11.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, V, z_dense, linewidth=0, antialiased=True, alpha=0.92)
    ax.scatter(
        cells.alpha.to_numpy(dtype=float),
        cells.ugv_vx_mean.to_numpy(dtype=float),
        cells.steady_rmse_reduction_mean.to_numpy(dtype=float),
        s=20,
        depthshade=True,
    )

    ax.set_xlabel("Velocity-fault severity α", labelpad=10)
    ax.set_ylabel("UGV target speed (m/s)", labelpad=10)
    ax.set_zlabel("Steady-RMSE reduction (m)\nNominal − Robust", labelpad=10)
    ax.set_title(
        "Continuous 3D robustness landscape from paired evaluations\n"
        "Positive height = lower steady-state error with Robust Residual PPO",
        pad=18,
        fontsize=13,
        weight="bold",
    )
    ax.view_init(elev=27, azim=-135)

    fig.text(
        0.5, 0.015,
        "Surface is a visualization-only linear interpolation of 40 empirical cell means "
        "(5 tested α levels × 8 target-speed bins; 100 paired seeds per α). "
        "α=1.0 was held out from robust training.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_png_svg(fig, "fig7_3d_robustness_landscape")
    plt.close(fig)

def main() -> None:
    ensure_phase6_dirs()
    fig1_method_overview()
    fig2_nominal_tradeoff()
    fig3_zero_shot_map()
    fig4_velocity_fault_sweep()
    fig5_targeted_robust_training()
    fig6_cross_robustness_tradeoffs()
    fig7_3d_robustness_landscape()
    print(f"Phase 6 core figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
