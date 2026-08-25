import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_observation_delay_env import (
    UAVTrackingResidualDelayEnv,
    UAVTrackingDirectDelayEnv
)


# ============================================================
# Configuration
# ============================================================

DT = 0.05

DELAY_STEPS_VALUES = [
    0,
    2,
    4,
    6,
    8
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
# Utility
# ============================================================

def safe_nanmean(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(
        finite_values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            finite_values
        )
    )


def safe_nanstd(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(
        finite_values
    ) == 0:

        return np.nan

    return float(
        np.std(
            finite_values
        )
    )


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    method,
    seed,
    delay_steps,
    residual_model,
    direct_model
):

    # ========================================================
    # Create environment
    # ========================================================

    if method in [
        "PD+FF",
        "Residual PPO"
    ]:

        env = UAVTrackingResidualDelayEnv(
            randomize=True,
            delay_steps=delay_steps
        )

    elif method == "Direct PPO":

        env = UAVTrackingDirectDelayEnv(
            randomize=True,
            delay_steps=delay_steps
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
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

        # ----------------------------------------------------
        # PD + FF
        # ----------------------------------------------------

        if method == "PD+FF":

            action = np.zeros(
                2,
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

        # ----------------------------------------------------
        # Generalized Direct PPO
        # ----------------------------------------------------

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
        # TRUE state for metrics
        # ====================================================

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

        # ====================================================
        # Actual PHYSICAL command
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
        distances,
        dtype=np.float32
    )

    relative_speeds = np.asarray(
        relative_speeds,
        dtype=np.float32
    )

    rel_x_values = np.asarray(
        rel_x_values,
        dtype=np.float32
    )

    commands = np.asarray(
        commands,
        dtype=np.float32
    )

    # ========================================================
    # RMSE
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

    if (
        len(distances)
        > steady_start
    ):

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
    # first 0.5 s continuous stable window
    # ========================================================

    required_steps = int(
        0.5 / DT
    )

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
                i * DT
            )

            break

    # ========================================================
    # Settling time
    #
    # earliest time after which system remains stable
    # until episode end
    # ========================================================

    settling_time = np.nan

    for i in range(
        len(stable_mask)
    ):

        if np.all(
            stable_mask[i:]
        ):

            settling_time = float(
                i * DT
            )

            break

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

    # ========================================================
    # Meaningful ahead ratio
    #
    # Only count > 5 cm ahead
    # ========================================================

    meaningful_ahead_ratio = float(
        np.mean(
            rel_x_values
            < -0.05
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
    # Final metrics / success
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

        "delay_steps":
            delay_steps,

        "delay_seconds":
            delay_steps * DT,

        "delay_ms":
            delay_steps
            * DT
            * 1000.0,

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
            ],

        "success":
            bool(success),

        "rmse":
            rmse,

        "steady_rmse":
            steady_rmse,

        "final_error":
            final_error,

        "final_relative_speed":
            final_relative_speed,

        "stable_ratio":
            stable_ratio,

        "capture_time":
            capture_time,

        "settling_time":
            settling_time,

        "forward_overshoot":
            forward_overshoot,

        "meaningful_ahead_ratio":
            meaningful_ahead_ratio,

        "command_smoothness":
            command_smoothness,

        "max_command_change":
            max_command_change
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

    delay_ms_values = np.asarray([
        step
        * DT
        * 1000.0
        for step
        in DELAY_STEPS_VALUES
    ])

    def metric(
        method,
        metric_name
    ):

        return np.asarray([
            row[metric_name]
            for row
            in summary_results
            if row["method"]
            == method
        ])

    # ========================================================
    # Plot function
    # ========================================================

    def make_plot(
        metric_name,
        ylabel,
        filename,
        title
    ):

        plt.figure(
            figsize=(9, 5)
        )

        for method in methods:

            plt.plot(
                delay_ms_values,
                metric(
                    method,
                    metric_name
                ),
                marker="o",
                label=method
            )

        plt.xlabel(
            "Observation delay (ms)"
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

    # ========================================================
    # 1. Success rate
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            delay_ms_values,
            100.0
            * metric(
                method,
                "success_rate"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Observation delay (ms)"
    )

    plt.ylabel(
        "Success rate (%)"
    )

    plt.title(
        "Observation Delay - Success Rate"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "delay_success_rate.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Other plots
    # ========================================================

    make_plot(
        "steady_rmse_mean",
        "Steady-state RMSE (m)",
        "delay_steady_rmse.png",
        "Observation Delay - Steady RMSE"
    )

    make_plot(
        "capture_time_mean",
        "Capture time (s)",
        "delay_capture_time.png",
        "Observation Delay - Capture Time"
    )

    make_plot(
        "settling_time_mean",
        "Settling time (s)",
        "delay_settling_time.png",
        "Observation Delay - Settling Time"
    )

    make_plot(
        "forward_overshoot_mean",
        "Forward overshoot (m)",
        "delay_forward_overshoot.png",
        "Observation Delay - Forward Overshoot"
    )

    make_plot(
        "command_smoothness_mean",
        "Command smoothness cost",
        "delay_command_smoothness.png",
        "Observation Delay - Command Smoothness"
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

    # ========================================================
    # Frozen models
    # ========================================================

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

    # ========================================================
    # Sweep delays
    # ========================================================

    print()
    print("=" * 92)
    print(
        "PHASE 5.3 — OBSERVATION DELAY"
    )
    print("=" * 92)

    for delay_steps in (
        DELAY_STEPS_VALUES
    ):

        delay_ms = (
            delay_steps
            * DT
            * 1000.0
        )

        print()
        print(
            f"Delay = "
            f"{delay_ms:.0f} ms "
            f"({delay_steps} steps)"
        )

        print("-" * 92)

        for method in methods:

            episode_results = []

            for episode in range(
                NUM_EPISODES
            ):

                seed = (
                    TEST_SEED_START
                    + episode
                )

                result = (
                    evaluate_episode(
                        method=method,
                        seed=seed,
                        delay_steps=
                            delay_steps,
                        residual_model=
                            residual_model,
                        direct_model=
                            direct_model
                    )
                )

                raw_results.append(
                    result
                )

                episode_results.append(
                    result
                )

            # =================================================
            # Helper
            # =================================================

            def values(
                metric_name
            ):

                return np.asarray([
                    row[metric_name]
                    for row
                    in episode_results
                ])

            success_values = (
                values(
                    "success"
                )
            )

            rmse_values = (
                values(
                    "rmse"
                )
            )

            steady_values = (
                values(
                    "steady_rmse"
                )
            )

            stable_values = (
                values(
                    "stable_ratio"
                )
            )

            capture_values = (
                values(
                    "capture_time"
                )
            )

            settling_values = (
                values(
                    "settling_time"
                )
            )

            overshoot_values = (
                values(
                    "forward_overshoot"
                )
            )

            ahead_values = (
                values(
                    "meaningful_ahead_ratio"
                )
            )

            smoothness_values = (
                values(
                    "command_smoothness"
                )
            )

            max_change_values = (
                values(
                    "max_command_change"
                )
            )

            # =================================================
            # Summary
            # =================================================

            summary = {

                "method":
                    method,

                "delay_steps":
                    delay_steps,

                "delay_ms":
                    delay_ms,

                "success_rate":
                    float(
                        np.mean(
                            success_values
                        )
                    ),

                "rmse_mean":
                    float(
                        np.mean(
                            rmse_values
                        )
                    ),

                "rmse_std":
                    float(
                        np.std(
                            rmse_values
                        )
                    ),

                "steady_rmse_mean":
                    safe_nanmean(
                        steady_values
                    ),

                "steady_rmse_std":
                    safe_nanstd(
                        steady_values
                    ),

                "stable_ratio_mean":
                    float(
                        np.mean(
                            stable_values
                        )
                    ),

                "capture_time_mean":
                    safe_nanmean(
                        capture_values
                    ),

                "capture_time_std":
                    safe_nanstd(
                        capture_values
                    ),

                "settling_time_mean":
                    safe_nanmean(
                        settling_values
                    ),

                "settling_time_std":
                    safe_nanstd(
                        settling_values
                    ),

                "forward_overshoot_mean":
                    float(
                        np.mean(
                            overshoot_values
                        )
                    ),

                "meaningful_ahead_ratio_mean":
                    float(
                        np.mean(
                            ahead_values
                        )
                    ),

                "command_smoothness_mean":
                    float(
                        np.mean(
                            smoothness_values
                        )
                    ),

                "max_command_change_mean":
                    float(
                        np.mean(
                            max_change_values
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
                f"RMSE="
                f"{summary['rmse_mean']:.4f} | "
                f"Steady="
                f"{summary['steady_rmse_mean']:.4f} | "
                f"Capture="
                f"{summary['capture_time_mean']:.3f}s | "
                f"Settle="
                f"{summary['settling_time_mean']:.3f}s | "
                f"Overshoot="
                f"{summary['forward_overshoot_mean']:.4f}"
            )

    # ========================================================
    # Save results
    # ========================================================

    raw_path = os.path.join(
        RESULTS_DIR,
        "observation_delay_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "observation_delay_summary.csv"
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

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 92)

    print(
        "Phase 5.3 Observation Delay COMPLETED."
    )

    print("=" * 92)

    print(
        f"Raw data:"
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