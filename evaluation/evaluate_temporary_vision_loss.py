import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_vision_loss_env import (
    UAVTrackingResidualVisionLossEnv,
    UAVTrackingDirectVisionLossEnv
)


# ============================================================
# Configuration
# ============================================================

DT = 0.05

LOSS_START_TIME = 3.0

LOSS_DURATIONS = [
    0.0,
    0.2,
    0.5,
    0.8,
    1.0
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

def safe_mean(values):

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
        np.mean(values)
    )


def safe_std(values):

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
        np.std(values)
    )


# ============================================================
# First continuous stable window
# ============================================================

def first_stable_window(
    stable_mask,
    start_index,
    required_steps
):

    start_index = max(
        0,
        start_index
    )

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
# Stable-to-end time with minimum suffix duration
# ============================================================

def stable_until_end_index(
    stable_mask,
    start_index,
    required_steps
):

    start_index = max(
        0,
        start_index
    )

    for i in range(
        start_index,
        len(stable_mask)
    ):

        remaining_steps = (
            len(stable_mask)
            - i
        )

        if remaining_steps < required_steps:
            continue

        if np.all(
            stable_mask[i:]
        ):
            return i

    return None


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    method,
    seed,
    loss_duration,
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
            UAVTrackingResidualVisionLossEnv(
                randomize=True,
                loss_start_time=
                    LOSS_START_TIME,
                loss_duration=
                    loss_duration
            )
        )

    elif method == "Direct PPO":

        env = (
            UAVTrackingDirectVisionLossEnv(
                randomize=True,
                loss_start_time=
                    LOSS_START_TIME,
                loss_duration=
                    loss_duration
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
    rel_x_values = []
    commands = []

    terminated_flag = False

    # ========================================================
    # Rollout
    # ========================================================

    for step in range(
        env.max_steps
    ):

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

        # TRUE state
        state = (
            env.simulator.get_state()
        )

        true_rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        true_rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        distance = float(
            np.linalg.norm(
                true_rel_pos
            )
        )

        relative_speed = float(
            np.linalg.norm(
                true_rel_vel
            )
        )

        distances.append(
            distance
        )

        relative_speeds.append(
            relative_speed
        )

        rel_x_values.append(
            float(
                true_rel_pos[0]
            )
        )

        commands.append(
            np.array(
                [
                    info["command_x"],
                    info["command_y"]
                ],
                dtype=np.float32
            )
        )

        if terminated:
            terminated_flag = True

        if terminated or truncated:
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

    # ========================================================
    # Basic tracking metrics
    # ========================================================

    rmse = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    steady_start = int(
        5.0 / DT
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
    # Loss timing
    # ========================================================

    loss_end_time = (
        LOSS_START_TIME
        + loss_duration
    )

    loss_start_index = int(
        round(
            LOSS_START_TIME
            / DT
        )
    )

    loss_end_index = int(
        round(
            loss_end_time
            / DT
        )
    )

    required_steps = int(
        0.5 / DT
    )

    # ========================================================
    # Capture time from episode start
    # ========================================================

    capture_index = first_stable_window(
        stable_mask,
        0,
        required_steps
    )

    if capture_index is None:

        capture_time = np.nan

    else:

        capture_time = float(
            capture_index
            * DT
        )

    # ========================================================
    # Recovery time
    #
    # From vision restoration until first
    # 0.5 s continuous stable window.
    # ========================================================

    recovery_index = first_stable_window(
        stable_mask,
        loss_end_index,
        required_steps
    )

    if recovery_index is None:

        recovery_time = np.nan
        recovered = False

    else:

        recovery_time = float(
            recovery_index
            * DT
            - loss_end_time
        )

        recovery_time = max(
            0.0,
            recovery_time
        )

        recovered = True

    # ========================================================
    # Retained settling after vision recovery
    #
    # Must stay stable all the way to episode end
    # AND suffix must be at least 0.5 s long.
    # ========================================================

    retained_index = stable_until_end_index(
        stable_mask,
        loss_end_index,
        required_steps
    )

    if retained_index is None:

        retained_settling_time = (
            np.nan
        )

        retained_recovery = False

    else:

        retained_settling_time = float(
            retained_index
            * DT
            - loss_end_time
        )

        retained_settling_time = max(
            0.0,
            retained_settling_time
        )

        retained_recovery = True

    # ========================================================
    # Error during loss
    # ========================================================

    if (
        loss_duration > 0.0
        and
        loss_start_index
        < len(distances)
    ):

        end_index = min(
            loss_end_index,
            len(distances)
        )

        loss_window_errors = (
            distances[
                loss_start_index:
                end_index
            ]
        )

        if len(
            loss_window_errors
        ) > 0:

            loss_window_peak_error = float(
                np.max(
                    loss_window_errors
                )
            )

        else:

            loss_window_peak_error = np.nan

    else:

        loss_window_peak_error = 0.0

    # ========================================================
    # Peak error after vision returns
    # ========================================================

    if loss_end_index < len(
        distances
    ):

        post_loss_peak_error = float(
            np.max(
                distances[
                    loss_end_index:
                ]
            )
        )

    else:

        post_loss_peak_error = np.nan

    # ========================================================
    # Forward overshoot
    # ========================================================

    forward_overshoot = float(
        max(
            0.0,
            -np.min(
                rel_x_values
            )
        )
    )

    if loss_end_index < len(
        rel_x_values
    ):

        post_loss_forward_overshoot = float(
            max(
                0.0,
                -np.min(
                    rel_x_values[
                        loss_end_index:
                    ]
                )
            )
        )

    else:

        post_loss_forward_overshoot = np.nan

    # ========================================================
    # Command smoothness
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
    # Success
    # ========================================================

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

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

        "seed":
            seed,

        "loss_start_time":
            LOSS_START_TIME,

        "loss_duration":
            loss_duration,

        "loss_end_time":
            loss_end_time,

        "success":
            bool(success),

        "recovered":
            bool(recovered),

        "retained_recovery":
            bool(
                retained_recovery
            ),

        "rmse":
            rmse,

        "steady_rmse":
            steady_rmse,

        "stable_ratio":
            stable_ratio,

        "capture_time":
            capture_time,

        "recovery_time":
            recovery_time,

        "retained_settling_time":
            retained_settling_time,

        "loss_window_peak_error":
            loss_window_peak_error,

        "post_loss_peak_error":
            post_loss_peak_error,

        "forward_overshoot":
            forward_overshoot,

        "post_loss_forward_overshoot":
            post_loss_forward_overshoot,

        "command_smoothness":
            command_smoothness,

        "max_command_change":
            max_command_change,

        "final_error":
            final_error,

        "final_relative_speed":
            final_relative_speed,

        "initial_uav_x":
            reset_info[
                "initial_uav_x"
            ],

        "initial_uav_y":
            reset_info[
                "initial_uav_y"
            ],

        "ugv_vx":
            reset_info[
                "ugv_vx"
            ]
    }


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    path,
    rows
):

    with open(
        path,
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

    durations = np.asarray(
        LOSS_DURATIONS
    )

    def metric(
        method,
        name
    ):

        return np.asarray([
            row[name]
            for row in summary_results
            if row["method"] == method
        ])

    # ========================================================
    # Generic plot
    # ========================================================

    def make_plot(
        metric_name,
        ylabel,
        title,
        filename,
        percentage=False
    ):

        plt.figure(
            figsize=(9, 5)
        )

        for method in methods:

            values = metric(
                method,
                metric_name
            )

            if percentage:
                values = 100.0 * values

            plt.plot(
                durations,
                values,
                marker="o",
                label=method
            )

        plt.xlabel(
            "Vision loss duration (s)"
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
        "Temporary Vision Loss - Success Rate",
        "vision_loss_success_rate.png",
        percentage=True
    )

    make_plot(
        "recovery_rate",
        "Recovery rate (%)",
        "Temporary Vision Loss - Recovery Rate",
        "vision_loss_recovery_rate.png",
        percentage=True
    )

    make_plot(
        "recovery_time_mean",
        "Recovery time (s)",
        "Temporary Vision Loss - Recovery Time",
        "vision_loss_recovery_time.png"
    )

    make_plot(
        "retained_settling_time_mean",
        "Retained settling time (s)",
        "Temporary Vision Loss - Stable-to-End Recovery",
        "vision_loss_retained_settling_time.png"
    )

    make_plot(
        "post_loss_peak_error_mean",
        "Post-loss peak error (m)",
        "Temporary Vision Loss - Post-Loss Peak Error",
        "vision_loss_post_loss_peak_error.png"
    )

    make_plot(
        "forward_overshoot_mean",
        "Forward overshoot (m)",
        "Temporary Vision Loss - Forward Overshoot",
        "vision_loss_forward_overshoot.png"
    )

    make_plot(
        "command_smoothness_mean",
        "Command smoothness cost",
        "Temporary Vision Loss - Command Smoothness",
        "vision_loss_command_smoothness.png"
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
    print("=" * 96)
    print(
        "PHASE 5.4 — TEMPORARY VISION LOSS"
    )
    print("=" * 96)

    for duration in LOSS_DURATIONS:

        print()
        print(
            f"Vision loss duration = "
            f"{duration:.2f} s"
        )

        print("-" * 96)

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

                    seed=seed,

                    loss_duration=
                        duration,

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

            def values(name):

                return np.asarray([
                    r[name]
                    for r
                    in episode_results
                ])

            summary = {

                "method":
                    method,

                "loss_duration":
                    duration,

                "success_rate":
                    float(
                        np.mean(
                            values(
                                "success"
                            )
                        )
                    ),

                "recovery_rate":
                    float(
                        np.mean(
                            values(
                                "recovered"
                            )
                        )
                    ),

                "retained_recovery_rate":
                    float(
                        np.mean(
                            values(
                                "retained_recovery"
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

                "steady_rmse_mean":
                    float(
                        np.mean(
                            values(
                                "steady_rmse"
                            )
                        )
                    ),

                "stable_ratio_mean":
                    float(
                        np.mean(
                            values(
                                "stable_ratio"
                            )
                        )
                    ),

                "capture_time_mean":
                    safe_mean(
                        values(
                            "capture_time"
                        )
                    ),

                "recovery_time_mean":
                    safe_mean(
                        values(
                            "recovery_time"
                        )
                    ),

                "recovery_time_std":
                    safe_std(
                        values(
                            "recovery_time"
                        )
                    ),

                "retained_settling_time_mean":
                    safe_mean(
                        values(
                            "retained_settling_time"
                        )
                    ),

                "loss_window_peak_error_mean":
                    safe_mean(
                        values(
                            "loss_window_peak_error"
                        )
                    ),

                "post_loss_peak_error_mean":
                    safe_mean(
                        values(
                            "post_loss_peak_error"
                        )
                    ),

                "forward_overshoot_mean":
                    float(
                        np.mean(
                            values(
                                "forward_overshoot"
                            )
                        )
                    ),

                "post_loss_forward_overshoot_mean":
                    safe_mean(
                        values(
                            "post_loss_forward_overshoot"
                        )
                    ),

                "command_smoothness_mean":
                    float(
                        np.mean(
                            values(
                                "command_smoothness"
                            )
                        )
                    ),

                "max_command_change_mean":
                    float(
                        np.mean(
                            values(
                                "max_command_change"
                            )
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
                f"Recovery="
                f"{summary['recovery_rate'] * 100:6.2f}% | "
                f"RMSE="
                f"{summary['rmse_mean']:.4f} | "
                f"Recovery time="
                f"{summary['recovery_time_mean']:.3f}s | "
                f"Post-loss peak="
                f"{summary['post_loss_peak_error_mean']:.4f}m | "
                f"Overshoot="
                f"{summary['forward_overshoot_mean']:.4f}m"
            )

    # ========================================================
    # Save
    # ========================================================

    raw_path = os.path.join(
        RESULTS_DIR,
        "temporary_vision_loss_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "temporary_vision_loss_summary.csv"
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
    print("=" * 96)
    print(
        "Phase 5.4 Temporary Vision Loss COMPLETED."
    )
    print("=" * 96)

    print(
        f"Raw:\n  {raw_path}"
    )

    print(
        f"Summary:\n  {summary_path}"
    )

    print(
        f"Plots:\n  {PLOTS_DIR}/"
    )


if __name__ == "__main__":
    main()