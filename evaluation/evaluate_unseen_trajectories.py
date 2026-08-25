import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_unseen_trajectory_env import (
    UAVTrackingResidualUnseenTrajectoryEnv,
    UAVTrackingDirectUnseenTrajectoryEnv
)


# ============================================================
# Configuration
# ============================================================

DT = 0.05

MANEUVER_START_TIME = 6.0

SCENARIOS = [
    "nominal",
    "speed_step",
    "lateral_sine",
    "constant_turn",
]

NUM_EPISODES = 100

TEST_SEED_START = 1000

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

RESIDUAL_MODEL_PATH = (
    "models/residual_ppo_uav_tracking"
)

DIRECT_MODEL_PATH = (
    "models/generalized_ppo_uav_tracking"
)


# ============================================================
# Safe statistics
# ============================================================

def safe_mean(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def safe_std(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.std(
            values
        )
    )


# ============================================================
# Stable-window helper
# ============================================================

def first_stable_window(
    stable_mask,
    start_index,
    required_steps
):

    for i in range(
        start_index,
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

            return i

    return None


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    method,
    scenario,
    seed,
    residual_model,
    direct_model
):

    # ========================================================
    # Environment
    # ========================================================

    if method in [
        "PD+FF",
        "Residual PPO"
    ]:

        env = (
            UAVTrackingResidualUnseenTrajectoryEnv(
                randomize=True,
                scenario=scenario,
                maneuver_start_time=
                    MANEUVER_START_TIME,
            )
        )

    elif method == "Direct PPO":

        env = (
            UAVTrackingDirectUnseenTrajectoryEnv(
                randomize=True,
                scenario=scenario,
                maneuver_start_time=
                    MANEUVER_START_TIME,
            )
        )

    else:

        raise ValueError(
            method
        )

    obs, reset_info = env.reset(
        seed=seed
    )

    # ========================================================
    # Storage
    # ========================================================

    distances = []

    relative_speeds = []

    longitudinal_errors = []

    lateral_errors = []

    commands = []

    ugv_speeds = []

    terminated_flag = False

    # ========================================================
    # Rollout
    # ========================================================

    for step in range(
        env.max_steps
    ):

        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        if method == "PD+FF":

            action = np.zeros(
                2,
                dtype=np.float32
            )

        elif method == "Residual PPO":

            action, _ = (
                residual_model.predict(
                    obs,
                    deterministic=True
                )
            )

        else:

            action, _ = (
                direct_model.predict(
                    obs,
                    deterministic=True
                )
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

        # ====================================================
        # TRUE state
        # ====================================================

        state = (
            env.simulator.get_state()
        )

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        distance = float(
            np.linalg.norm(
                rel_pos
            )
        )

        relative_speed = float(
            np.linalg.norm(
                rel_vel
            )
        )

        distances.append(
            distance
        )

        relative_speeds.append(
            relative_speed
        )

        # ====================================================
        # Along-track / cross-track coordinate system
        # ====================================================

        ugv_velocity = np.asarray(
            state["ugv_vel"],
            dtype=np.float64
        )

        ugv_speed = float(
            np.linalg.norm(
                ugv_velocity
            )
        )

        ugv_speeds.append(
            ugv_speed
        )

        if ugv_speed > 1e-8:

            heading = (
                ugv_velocity
                / ugv_speed
            )

        else:

            heading = np.array(
                [1.0, 0.0]
            )

        lateral_axis = np.array(
            [
                -heading[1],
                heading[0]
            ]
        )

        longitudinal_error = float(
            np.dot(
                rel_pos,
                heading
            )
        )

        lateral_error = float(
            np.dot(
                rel_pos,
                lateral_axis
            )
        )

        longitudinal_errors.append(
            longitudinal_error
        )

        lateral_errors.append(
            lateral_error
        )

        # ====================================================
        # Physical command
        # ====================================================

        if method == "Direct PPO":

            command = (
                np.asarray(
                    action,
                    dtype=np.float32
                )
                * env.simulator.max_uav_speed
            )

            command = np.clip(
                command,
                -env.simulator.max_uav_speed,
                env.simulator.max_uav_speed
            )

        else:

            command = np.array(
                [
                    info["command_x"],
                    info["command_y"]
                ],
                dtype=np.float32
            )

        commands.append(
            command
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

    longitudinal_errors = np.asarray(
        longitudinal_errors
    )

    lateral_errors = np.asarray(
        lateral_errors
    )

    commands = np.asarray(
        commands
    )

    ugv_speeds = np.asarray(
        ugv_speeds
    )

    times = (
        np.arange(
            1,
            len(distances) + 1
        )
        * DT
    )

    # ========================================================
    # Index masks
    # ========================================================

    post_mask = (
        times
        >= MANEUVER_START_TIME
    )

    pre_mask = (
        (times >= 5.0)
        &
        (times < MANEUVER_START_TIME)
    )

    # ========================================================
    # Full RMSE
    # ========================================================

    rmse = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    # ========================================================
    # Pre-maneuver RMSE
    # ========================================================

    if np.any(
        pre_mask
    ):

        pre_maneuver_rmse = float(
            np.sqrt(
                np.mean(
                    distances[
                        pre_mask
                    ] ** 2
                )
            )
        )

    else:

        pre_maneuver_rmse = np.nan

    # ========================================================
    # Post-maneuver tracking
    # ========================================================

    if np.any(
        post_mask
    ):

        post_distances = (
            distances[
                post_mask
            ]
        )

        post_maneuver_rmse = float(
            np.sqrt(
                np.mean(
                    post_distances ** 2
                )
            )
        )

        post_p95_error = float(
            np.percentile(
                post_distances,
                95
            )
        )

        post_max_error = float(
            np.max(
                post_distances
            )
        )

        post_lateral_rmse = float(
            np.sqrt(
                np.mean(
                    lateral_errors[
                        post_mask
                    ] ** 2
                )
            )
        )

    else:

        post_maneuver_rmse = np.nan
        post_p95_error = np.nan
        post_max_error = np.nan
        post_lateral_rmse = np.nan

    # ========================================================
    # Stable tracking
    # ========================================================

    stable_mask = (
        (distances < 0.15)
        &
        (relative_speeds < 0.10)
    )

    overall_stable_ratio = float(
        np.mean(
            stable_mask
        )
    )

    if np.any(
        post_mask
    ):

        post_stable_ratio = float(
            np.mean(
                stable_mask[
                    post_mask
                ]
            )
        )

    else:

        post_stable_ratio = np.nan

    # ========================================================
    # Reacquisition time
    #
    # After maneuver starts:
    # first continuous 0.5 s stable window.
    # ========================================================

    maneuver_index = int(
        round(
            MANEUVER_START_TIME
            / DT
        )
    )

    required_steps = int(
        0.5 / DT
    )

    reacquisition_index = (
        first_stable_window(
            stable_mask,
            maneuver_index,
            required_steps
        )
    )

    if reacquisition_index is None:

        reacquisition_time = np.nan

        reacquired = False

    else:

        reacquisition_time = float(
            reacquisition_index
            * DT
            - MANEUVER_START_TIME
        )

        reacquisition_time = max(
            0.0,
            reacquisition_time
        )

        reacquired = True

    # ========================================================
    # Along-track forward overshoot
    #
    # longitudinal_error < 0:
    # UAV is ahead of target along target heading.
    # ========================================================

    if np.any(
        post_mask
    ):

        post_longitudinal = (
            longitudinal_errors[
                post_mask
            ]
        )

        forward_overshoot = float(
            max(
                0.0,
                -np.min(
                    post_longitudinal
                )
            )
        )

    else:

        forward_overshoot = np.nan

    # ========================================================
    # Physical command smoothness
    # ========================================================

    if len(commands) > 1:

        command_diff = np.diff(
            commands,
            axis=0
        )

        command_change_norm = (
            np.linalg.norm(
                command_diff,
                axis=1
            )
        )

        command_smoothness = float(
            np.mean(
                np.sum(
                    command_diff ** 2,
                    axis=1
                )
            )
        )

        max_command_change = float(
            np.max(
                command_change_norm
            )
        )

    else:

        command_smoothness = np.nan
        max_command_change = np.nan

    # ========================================================
    # Final metrics
    # ========================================================

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

    # ========================================================
    # Dynamic-tracking success
    #
    # Post-maneuver region is the important part.
    # ========================================================

    success = (
        post_stable_ratio >= 0.50
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

        "scenario":
            scenario,

        "seed":
            seed,

        "success":
            bool(success),

        "reacquired":
            bool(reacquired),

        "rmse":
            rmse,

        "pre_maneuver_rmse":
            pre_maneuver_rmse,

        "post_maneuver_rmse":
            post_maneuver_rmse,

        "post_p95_error":
            post_p95_error,

        "post_max_error":
            post_max_error,

        "post_lateral_rmse":
            post_lateral_rmse,

        "overall_stable_ratio":
            overall_stable_ratio,

        "post_stable_ratio":
            post_stable_ratio,

        "reacquisition_time":
            reacquisition_time,

        "forward_overshoot":
            forward_overshoot,

        "command_smoothness":
            command_smoothness,

        "max_command_change":
            max_command_change,

        "final_error":
            final_error,

        "final_relative_speed":
            final_relative_speed,

        "base_ugv_speed":
            reset_info[
                "base_ugv_speed"
            ],

        "max_ugv_speed":
            float(
                np.max(
                    ugv_speeds
                )
            )
    }


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    filepath,
    rows
):

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Plot summary
# ============================================================

def plot_summary(
    summary_results
):

    methods = [
        "PD+FF",
        "Direct PPO",
        "Residual PPO"
    ]

    x = np.arange(
        len(
            SCENARIOS
        )
    )

    def metric(
        method,
        metric_name
    ):

        return np.asarray([
            row[
                metric_name
            ]
            for row
            in summary_results
            if row["method"]
            == method
        ])

    # ========================================================
    # Generic grouped line plot
    # ========================================================

    def make_plot(
        metric_name,
        ylabel,
        title,
        filename,
        percentage=False
    ):

        plt.figure(
            figsize=(10, 5)
        )

        for method in methods:

            values = metric(
                method,
                metric_name
            )

            if percentage:
                values = (
                    values
                    * 100.0
                )

            plt.plot(
                x,
                values,
                marker="o",
                label=method
            )

        plt.xticks(
            x,
            SCENARIOS
        )

        plt.xlabel(
            "Target-motion scenario"
        )

        plt.ylabel(
            ylabel
        )

        plt.title(
            title
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    make_plot(
        "success_rate",
        "Success rate (%)",
        "Unseen Target Motion - Success Rate",
        "unseen_trajectory_success_rate.png",
        percentage=True
    )

    make_plot(
        "post_maneuver_rmse_mean",
        "Post-maneuver RMSE (m)",
        "Unseen Target Motion - Post-Maneuver RMSE",
        "unseen_trajectory_post_rmse.png"
    )

    make_plot(
        "post_p95_error_mean",
        "Post-maneuver P95 error (m)",
        "Unseen Target Motion - P95 Tracking Error",
        "unseen_trajectory_p95_error.png"
    )

    make_plot(
        "post_stable_ratio_mean",
        "Post-maneuver stable ratio (%)",
        "Unseen Target Motion - Stable Tracking",
        "unseen_trajectory_stable_ratio.png",
        percentage=True
    )

    make_plot(
        "reacquisition_time_mean",
        "Reacquisition time (s)",
        "Unseen Target Motion - Reacquisition Time",
        "unseen_trajectory_reacquisition_time.png"
    )

    make_plot(
        "forward_overshoot_mean",
        "Along-track forward overshoot (m)",
        "Unseen Target Motion - Forward Overshoot",
        "unseen_trajectory_forward_overshoot.png"
    )

    make_plot(
        "post_lateral_rmse_mean",
        "Post-maneuver lateral RMSE (m)",
        "Unseen Target Motion - Lateral Tracking Error",
        "unseen_trajectory_lateral_rmse.png"
    )

    make_plot(
        "command_smoothness_mean",
        "Command smoothness cost",
        "Unseen Target Motion - Command Smoothness",
        "unseen_trajectory_command_smoothness.png"
    )


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

    residual_model = PPO.load(
        RESIDUAL_MODEL_PATH
    )

    direct_model = PPO.load(
        DIRECT_MODEL_PATH
    )

    methods = [
        "PD+FF",
        "Direct PPO",
        "Residual PPO"
    ]

    raw_results = []

    summary_results = []

    print()
    print("=" * 100)
    print(
        "PHASE 5.5 — UNSEEN TARGET TRAJECTORIES"
    )
    print("=" * 100)

    # ========================================================
    # Run sweep
    # ========================================================

    for scenario in SCENARIOS:

        print()
        print(
            f"Scenario: {scenario}"
        )

        print("-" * 100)

        for method in methods:

            episode_results = []

            for episode in range(
                NUM_EPISODES
            ):

                seed = (
                    TEST_SEED_START
                    + episode
                )

                result = evaluate_episode(
                    method=method,
                    scenario=scenario,
                    seed=seed,
                    residual_model=
                        residual_model,
                    direct_model=
                        direct_model
                )

                raw_results.append(
                    result
                )

                episode_results.append(
                    result
                )

            def values(
                name
            ):

                return np.asarray([
                    row[name]
                    for row
                    in episode_results
                ])

            summary = {

                "method":
                    method,

                "scenario":
                    scenario,

                "success_rate":
                    float(
                        np.mean(
                            values(
                                "success"
                            )
                        )
                    ),

                "reacquisition_rate":
                    float(
                        np.mean(
                            values(
                                "reacquired"
                            )
                        )
                    ),

                "rmse_mean":
                    float(
                        np.mean(
                            values(
                                "rmse"
                            )
                        )
                    ),

                "pre_maneuver_rmse_mean":
                    safe_mean(
                        values(
                            "pre_maneuver_rmse"
                        )
                    ),

                "post_maneuver_rmse_mean":
                    safe_mean(
                        values(
                            "post_maneuver_rmse"
                        )
                    ),

                "post_maneuver_rmse_std":
                    safe_std(
                        values(
                            "post_maneuver_rmse"
                        )
                    ),

                "post_p95_error_mean":
                    safe_mean(
                        values(
                            "post_p95_error"
                        )
                    ),

                "post_max_error_mean":
                    safe_mean(
                        values(
                            "post_max_error"
                        )
                    ),

                "post_lateral_rmse_mean":
                    safe_mean(
                        values(
                            "post_lateral_rmse"
                        )
                    ),

                "post_stable_ratio_mean":
                    safe_mean(
                        values(
                            "post_stable_ratio"
                        )
                    ),

                "reacquisition_time_mean":
                    safe_mean(
                        values(
                            "reacquisition_time"
                        )
                    ),

                "reacquisition_time_std":
                    safe_std(
                        values(
                            "reacquisition_time"
                        )
                    ),

                "forward_overshoot_mean":
                    safe_mean(
                        values(
                            "forward_overshoot"
                        )
                    ),

                "command_smoothness_mean":
                    safe_mean(
                        values(
                            "command_smoothness"
                        )
                    ),

                "max_command_change_mean":
                    safe_mean(
                        values(
                            "max_command_change"
                        )
                    ),

                "final_error_mean":
                    safe_mean(
                        values(
                            "final_error"
                        )
                    )
            }

            summary_results.append(
                summary
            )

            print(
                f"{method:12s} | "
                f"Success="
                f"{summary['success_rate'] * 100:6.2f}% | "
                f"Post RMSE="
                f"{summary['post_maneuver_rmse_mean']:.4f} | "
                f"P95="
                f"{summary['post_p95_error_mean']:.4f} | "
                f"Stable="
                f"{summary['post_stable_ratio_mean'] * 100:6.2f}% | "
                f"Reacq="
                f"{summary['reacquisition_time_mean']:.3f}s | "
                f"Overshoot="
                f"{summary['forward_overshoot_mean']:.4f}"
            )

    # ========================================================
    # Save
    # ========================================================

    raw_path = os.path.join(
        RESULTS_DIR,
        "unseen_trajectory_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "unseen_trajectory_summary.csv"
    )

    save_csv(
        raw_path,
        raw_results
    )

    save_csv(
        summary_path,
        summary_results
    )

    # ========================================================
    # Plots
    # ========================================================

    plot_summary(
        summary_results
    )

    print()
    print("=" * 100)

    print(
        "Phase 5.5 Unseen Target Trajectories COMPLETED."
    )

    print("=" * 100)

    print(
        f"Raw:"
        f"\n  {raw_path}"
    )

    print(
        f"Summary:"
        f"\n  {summary_path}"
    )

    print(
        f"Plots:"
        f"\n  {PLOTS_DIR}/"
    )


if __name__ == "__main__":
    main()