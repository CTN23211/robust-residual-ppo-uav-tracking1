import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_residual_robust_env import (
    UAVTrackingResidualRobustEnv
)


# ============================================================
# Configuration
# ============================================================

ALPHA_VALUES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]

NUM_EPISODES = 100

TEST_SEED_START = 1000

MODEL_PATH = (
    "models/residual_ppo_uav_tracking"
)

RESULTS_DIR = "results"

PLOTS_DIR = "plots"


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    method,
    alpha,
    seed,
    residual_model=None
):

    env = UAVTrackingResidualRobustEnv(
        randomize=True,
        velocity_coupling_alpha=alpha
    )

    obs, reset_info = env.reset(
        seed=seed
    )

    distances = []
    relative_speeds = []

    rel_x_values = []

    commands = []

    velocity_errors = []

    total_reward = 0.0

    terminated_flag = False

    # ========================================================
    # Run episode
    # ========================================================

    for step in range(
        env.max_steps
    ):

        # ----------------------------------------------------
        # PD+FF:
        #
        # residual is exactly zero
        # ----------------------------------------------------

        if method == "PD+FF":

            action = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Residual PPO
        # ----------------------------------------------------

        elif method == "Residual PPO":

            action, _ = (
                residual_model.predict(
                    obs,
                    deterministic=True
                )
            )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )

        distances.append(
            info["distance"]
        )

        relative_speeds.append(
            info["relative_speed"]
        )

        rel_x_values.append(
            info["rel_x"]
        )

        commands.append([
            info["command_x"],
            info["command_y"]
        ])

        velocity_errors.append(
            info[
                "velocity_estimation_error_x"
            ]
        )

        total_reward += (
            reward
        )

        if terminated:

            terminated_flag = True

        if (
            terminated
            or truncated
        ):
            break

    env.close()

    # ========================================================
    # Arrays
    # ========================================================

    distances = np.asarray(
        distances
    )

    relative_speeds = np.asarray(
        relative_speeds
    )

    rel_x_values = np.asarray(
        rel_x_values
    )

    commands = np.asarray(
        commands
    )

    velocity_errors = np.asarray(
        velocity_errors
    )

    # ========================================================
    # Tracking metrics
    # ========================================================

    rmse = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    steady_start = int(
        5.0 / 0.05
    )

    if len(distances) > steady_start:

        steady_rmse = float(
            np.sqrt(
                np.mean(
                    distances[
                        steady_start:
                    ] ** 2
                )
            )
        )

    else:

        steady_rmse = np.nan

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

    max_tracking_error = float(
        np.max(
            distances
        )
    )

    # ========================================================
    # Stable tracking
    # ========================================================

    stable_mask = (
        (distances < 0.15)
        &
        (relative_speeds < 0.10)
    )

    stable_ratio = float(
        np.mean(
            stable_mask
        )
    )

    # ========================================================
    # Capture time
    #
    # stable for 0.5 s = 10 consecutive steps
    # ========================================================

    required_steps = 10

    capture_time = np.nan

    for i in range(
        len(stable_mask)
        - required_steps
        + 1
    ):

        if np.all(
            stable_mask[
                i:
                i + required_steps
            ]
        ):

            capture_time = float(
                i * 0.05
            )

            break

    # ========================================================
    # Forward overshoot
    #
    # rel_x =
    # x_UGV - x_UAV
    #
    # rel_x < 0:
    # UAV is ahead of UGV
    #
    # Overshoot:
    #
    # max(0, -min(rel_x))
    # ========================================================

    min_rel_x = float(
        np.min(
            rel_x_values
        )
    )

    forward_overshoot = float(
        max(
            0.0,
            -min_rel_x
        )
    )

    # ========================================================
    # Time spent ahead of UGV
    # ========================================================

    ahead_ratio = float(
        np.mean(
            rel_x_values < 0.0
        )
    )

    # ========================================================
    # Command smoothness
    # ========================================================

    if len(commands) > 1:

        command_diff = np.diff(
            commands,
            axis=0
        )

        command_smoothness = float(
            np.mean(
                np.sum(
                    command_diff ** 2,
                    axis=1
                )
            )
        )

    else:

        command_smoothness = np.nan

    # ========================================================
    # Velocity-estimation-error metric
    # ========================================================

    mean_abs_velocity_error = float(
        np.mean(
            np.abs(
                velocity_errors
            )
        )
    )

    # ========================================================
    # Success
    # ========================================================

    success = (
        stable_ratio >= 0.50
        and
        final_error < 0.15
        and
        final_relative_speed < 0.10
        and
        not terminated_flag
    )

    return {

        "method":
            method,

        "alpha":
            alpha,

        "seed":
            seed,

        "initial_uav_x":
            reset_info["initial_uav_x"],

        "initial_uav_y":
            reset_info["initial_uav_y"],

        "ugv_vx":
            reset_info["ugv_vx"],

        "success":
            bool(success),

        "rmse":
            rmse,

        "steady_rmse":
            steady_rmse,

        "max_tracking_error":
            max_tracking_error,

        "final_error":
            final_error,

        "final_relative_speed":
            final_relative_speed,

        "stable_ratio":
            stable_ratio,

        "capture_time":
            capture_time,

        "forward_overshoot":
            forward_overshoot,

        "ahead_ratio":
            ahead_ratio,

        "command_smoothness":
            command_smoothness,

        "mean_abs_velocity_error":
            mean_abs_velocity_error,

        "total_reward":
            total_reward
    }


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # Load frozen Residual PPO
    # ========================================================

    residual_model = PPO.load(
        MODEL_PATH
    )

    methods = [
        "PD+FF",
        "Residual PPO"
    ]

    all_results = []

    # ========================================================
    # Run robustness sweep
    # ========================================================

    for alpha in ALPHA_VALUES:

        print()
        print("=" * 78)
        print(
            f"Velocity Compensation Fault | "
            f"alpha = {alpha:.2f}"
        )
        print("=" * 78)

        for method in methods:

            method_results = []

            for episode in range(
                NUM_EPISODES
            ):

                seed = (
                    TEST_SEED_START
                    + episode
                )

                result = evaluate_episode(
                    method=method,
                    alpha=alpha,
                    seed=seed,
                    residual_model=residual_model
                )

                all_results.append(
                    result
                )

                method_results.append(
                    result
                )

            # =================================================
            # Alpha-level summary
            # =================================================

            success_rate = np.mean([
                r["success"]
                for r in method_results
            ])

            rmse = np.asarray([
                r["rmse"]
                for r in method_results
            ])

            overshoot = np.asarray([
                r["forward_overshoot"]
                for r in method_results
            ])

            capture = np.asarray([
                r["capture_time"]
                for r in method_results
            ])

            stable_ratio = np.asarray([
                r["stable_ratio"]
                for r in method_results
            ])

            print()
            print(
                f"{method}"
            )

            print(
                f"  Success rate       : "
                f"{success_rate * 100:.2f}%"
            )

            print(
                f"  RMSE               : "
                f"{np.mean(rmse):.4f} "
                f"± {np.std(rmse):.4f} m"
            )

            print(
                f"  Forward overshoot  : "
                f"{np.mean(overshoot):.4f} "
                f"± {np.std(overshoot):.4f} m"
            )

            print(
                f"  Capture time       : "
                f"{np.nanmean(capture):.3f} "
                f"± {np.nanstd(capture):.3f} s"
            )

            print(
                f"  Stable ratio       : "
                f"{np.mean(stable_ratio) * 100:.2f}%"
            )

    # ========================================================
    # Save raw CSV
    # ========================================================

    raw_csv = os.path.join(
        RESULTS_DIR,
        "velocity_compensation_fault_raw.csv"
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        raw_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # ========================================================
    # Aggregate summary
    # ========================================================

    summary_rows = []

    for alpha in ALPHA_VALUES:

        for method in methods:

            subset = [
                r
                for r in all_results
                if (
                    r["alpha"] == alpha
                    and
                    r["method"] == method
                )
            ]

            def arr(name):

                return np.asarray([
                    r[name]
                    for r in subset
                ])

            summary_rows.append({

                "alpha":
                    alpha,

                "method":
                    method,

                "success_rate":
                    np.mean(
                        arr("success")
                    ),

                "rmse_mean":
                    np.mean(
                        arr("rmse")
                    ),

                "rmse_std":
                    np.std(
                        arr("rmse")
                    ),

                "steady_rmse_mean":
                    np.nanmean(
                        arr("steady_rmse")
                    ),

                "forward_overshoot_mean":
                    np.mean(
                        arr("forward_overshoot")
                    ),

                "forward_overshoot_std":
                    np.std(
                        arr("forward_overshoot")
                    ),

                "capture_time_mean":
                    np.nanmean(
                        arr("capture_time")
                    ),

                "stable_ratio_mean":
                    np.mean(
                        arr("stable_ratio")
                    ),

                "ahead_ratio_mean":
                    np.mean(
                        arr("ahead_ratio")
                    ),

                "command_smoothness_mean":
                    np.mean(
                        arr("command_smoothness")
                    ),

                "velocity_error_mean":
                    np.mean(
                        arr(
                            "mean_abs_velocity_error"
                        )
                    )
            })

    summary_csv = os.path.join(
        RESULTS_DIR,
        "velocity_compensation_fault_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # Plot helper
    # ========================================================

    def summary_values(
        method,
        metric
    ):

        return np.asarray([
            row[metric]
            for row in summary_rows
            if row["method"] == method
        ])

    alphas = np.asarray(
        ALPHA_VALUES
    )

    # ========================================================
    # Figure 1
    # Alpha vs Forward Overshoot
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            alphas,
            summary_values(
                method,
                "forward_overshoot_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Velocity coupling alpha"
    )

    plt.ylabel(
        "Mean forward overshoot (m)"
    )

    plt.title(
        "Velocity Compensation Fault - Forward Overshoot"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "fault_alpha_vs_forward_overshoot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 2
    # Alpha vs RMSE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            alphas,
            summary_values(
                method,
                "rmse_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Velocity coupling alpha"
    )

    plt.ylabel(
        "Mean full RMSE (m)"
    )

    plt.title(
        "Velocity Compensation Fault - RMSE"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "fault_alpha_vs_rmse.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 3
    # Alpha vs Capture Time
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            alphas,
            summary_values(
                method,
                "capture_time_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Velocity coupling alpha"
    )

    plt.ylabel(
        "Mean capture time (s)"
    )

    plt.title(
        "Velocity Compensation Fault - Capture Time"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "fault_alpha_vs_capture_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 4
    # Alpha vs Stable Ratio
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            alphas,
            100.0
            * summary_values(
                method,
                "stable_ratio_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Velocity coupling alpha"
    )

    plt.ylabel(
        "Stable tracking ratio (%)"
    )

    plt.title(
        "Velocity Compensation Fault - Stable Tracking"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "fault_alpha_vs_stable_ratio.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("=" * 78)
    print("Phase 5.1 fault sweep completed.")
    print("=" * 78)

    print(
        f"Raw results:"
        f"\n  {raw_csv}"
    )

    print(
        f"Summary:"
        f"\n  {summary_csv}"
    )


if __name__ == "__main__":
    main()