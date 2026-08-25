import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_observation_noise_env import (
    UAVTrackingResidualNoiseEnv,
    UAVTrackingDirectNoiseEnv
)


# ============================================================
# Configuration
# ============================================================

POSITION_NOISE_VALUES = [
    0.00,
    0.01,
    0.03,
    0.05,
    0.10
]

VELOCITY_NOISE_VALUES = [
    0.00,
    0.01,
    0.03,
    0.05,
    0.10
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
# Evaluate one episode
# ============================================================

def evaluate_episode(
    method,
    seed,
    position_noise_std,
    velocity_noise_std,
    residual_model,
    direct_model
):

    # ========================================================
    # 1. Create environment
    # ========================================================

    if method in [
        "PD+FF",
        "Residual PPO"
    ]:

        env = UAVTrackingResidualNoiseEnv(
            randomize=True,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std
        )

    elif method == "Direct PPO":

        env = UAVTrackingDirectNoiseEnv(
            randomize=True,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    obs, reset_info = env.reset(
        seed=seed
    )

    # ========================================================
    # 2. Data storage
    # ========================================================

    distances = []
    relative_speeds = []
    rel_x_values = []
    commands = []

    terminated_flag = False

    # ========================================================
    # 3. Rollout
    # ========================================================

    for step in range(
        env.max_steps
    ):

        # ----------------------------------------------------
        # PD + FF:
        # residual action = zero
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
        # Direct PPO
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
        # TRUE state for evaluation
        #
        # Metrics must use the real simulator state,
        # not noisy observation.
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
        # Physical velocity command
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
    # 4. Convert arrays
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
    # 5. Full RMSE
    # ========================================================

    rmse = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    # ========================================================
    # 6. Steady-state RMSE
    #
    # after 5 seconds
    # ========================================================

    steady_start = int(
        5.0 / env.dt
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

    # ========================================================
    # 7. Stable tracking mask
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
    # 8. Capture time
    #
    # First stable interval lasting >= 0.5 s
    # ========================================================

    required_steps = int(
        0.5 / env.dt
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
                i * env.dt
            )

            break

    # ========================================================
    # 9. Settling time
    #
    # First time after which the system remains
    # stable until the end of the episode.
    # ========================================================

    settling_time = np.nan

    for i in range(
        len(stable_mask)
    ):

        if np.all(
            stable_mask[i:]
        ):

            settling_time = float(
                i * env.dt
            )

            break

    # ========================================================
    # 10. Forward overshoot
    #
    # rel_x = x_UGV - x_UAV
    #
    # rel_x < 0 means UAV is ahead of target.
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
    # 11. Meaningful ahead ratio
    #
    # Ignore millimeter-level sign changes.
    # Only count UAV > 5 cm ahead.
    # ========================================================

    meaningful_ahead_ratio = float(
        np.mean(
            rel_x_values < -0.05
        )
    )

    # ========================================================
    # 12. Command smoothness
    #
    # Physical command smoothness:
    #
    # mean(||u_t - u_(t-1)||^2)
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

        mean_command_change = float(
            np.mean(
                command_change_norm
            )
        )

        max_command_change = float(
            np.max(
                command_change_norm
            )
        )

    else:

        command_smoothness = np.nan
        mean_command_change = np.nan
        max_command_change = np.nan

    # ========================================================
    # 13. Final metrics
    # ========================================================

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

    # ========================================================
    # 14. Success
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

    # ========================================================
    # Return
    # ========================================================

    return {

        "method":
            method,

        "seed":
            seed,

        "position_noise_std":
            position_noise_std,

        "velocity_noise_std":
            velocity_noise_std,

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

        "mean_command_change":
            mean_command_change,

        "max_command_change":
            max_command_change
    }


# ============================================================
# Run one noise sweep
# ============================================================

def run_sweep(
    noise_type,
    noise_values,
    residual_model,
    direct_model
):

    methods = [
        "PD+FF",
        "Direct PPO",
        "Residual PPO"
    ]

    raw_results = []
    summary_results = []

    print()
    print("=" * 88)

    print(
        f"OBSERVATION NOISE SWEEP: "
        f"{noise_type.upper()}"
    )

    print("=" * 88)

    for noise_std in noise_values:

        # ====================================================
        # Change only ONE noise type
        # ====================================================

        if noise_type == "position":

            sigma_p = noise_std
            sigma_v = 0.0

        elif noise_type == "velocity":

            sigma_p = 0.0
            sigma_v = noise_std

        else:

            raise ValueError(
                noise_type
            )

        print()
        print(
            f"Noise level = "
            f"{noise_std:.3f}"
        )

        print("-" * 88)

        # ====================================================
        # Evaluate three controllers
        # ====================================================

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

                    position_noise_std=
                        sigma_p,

                    velocity_noise_std=
                        sigma_v,

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

            # =================================================
            # Convert metric arrays
            # =================================================

            def values(
                metric
            ):

                return np.asarray([
                    result[metric]
                    for result in episode_results
                ])

            success_values = values(
                "success"
            )

            rmse_values = values(
                "rmse"
            )

            steady_values = values(
                "steady_rmse"
            )

            stable_values = values(
                "stable_ratio"
            )

            capture_values = values(
                "capture_time"
            )

            settling_values = values(
                "settling_time"
            )

            overshoot_values = values(
                "forward_overshoot"
            )

            ahead_values = values(
                "meaningful_ahead_ratio"
            )

            smoothness_values = values(
                "command_smoothness"
            )

            max_change_values = values(
                "max_command_change"
            )

            # =================================================
            # Aggregate
            # =================================================

            summary = {

                "noise_type":
                    noise_type,

                "noise_std":
                    noise_std,

                "method":
                    method,

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
                    float(
                        np.nanmean(
                            steady_values
                        )
                    ),

                "steady_rmse_std":
                    float(
                        np.nanstd(
                            steady_values
                        )
                    ),

                "stable_ratio_mean":
                    float(
                        np.mean(
                            stable_values
                        )
                    ),

                "capture_time_mean":
                    float(
                        np.nanmean(
                            capture_values
                        )
                    ),

                "capture_time_std":
                    float(
                        np.nanstd(
                            capture_values
                        )
                    ),

                "settling_time_mean":
                    float(
                        np.nanmean(
                            settling_values
                        )
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

            # =================================================
            # Print current result
            # =================================================

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

    return (
        raw_results,
        summary_results
    )


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
# Plot one sweep
# ============================================================

def plot_summary(
    noise_type,
    noise_values,
    summary_results
):

    methods = [
        "PD+FF",
        "Direct PPO",
        "Residual PPO"
    ]

    # ========================================================
    # Helper
    # ========================================================

    def metric(
        method,
        metric_name
    ):

        return np.asarray([
            row[metric_name]
            for row in summary_results
            if row["method"] == method
        ])

    # ========================================================
    # 1. Success Rate
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            100.0
            * metric(
                method,
                "success_rate"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Success rate (%)"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Success Rate"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_success.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 2. Steady RMSE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            metric(
                method,
                "steady_rmse_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Steady-state RMSE (m)"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Steady RMSE"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_steady_rmse.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 3. Capture Time
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            metric(
                method,
                "capture_time_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Mean capture time (s)"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Capture Time"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_capture_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 4. Settling Time
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            metric(
                method,
                "settling_time_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Mean settling time (s)"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Settling Time"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_settling_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 5. Forward Overshoot
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            metric(
                method,
                "forward_overshoot_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Mean forward overshoot (m)"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Forward Overshoot"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_forward_overshoot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 6. Command smoothness
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        plt.plot(
            noise_values,
            metric(
                method,
                "command_smoothness_mean"
            ),
            marker="o",
            label=method
        )

    plt.xlabel(
        "Noise standard deviation"
    )

    plt.ylabel(
        "Command smoothness cost"
    )

    plt.title(
        f"{noise_type.title()} Observation Noise - Command Smoothness"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{noise_type}_noise_command_smoothness.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


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
    # 1. Load frozen models
    # ========================================================

    residual_model = PPO.load(
        RESIDUAL_MODEL_PATH
    )

    direct_model = PPO.load(
        DIRECT_MODEL_PATH
    )

    # ========================================================
    # 2. Position noise
    # ========================================================

    (
        position_raw,
        position_summary
    ) = run_sweep(

        noise_type="position",

        noise_values=
            POSITION_NOISE_VALUES,

        residual_model=
            residual_model,

        direct_model=
            direct_model
    )

    # ========================================================
    # 3. Velocity noise
    # ========================================================

    (
        velocity_raw,
        velocity_summary
    ) = run_sweep(

        noise_type="velocity",

        noise_values=
            VELOCITY_NOISE_VALUES,

        residual_model=
            residual_model,

        direct_model=
            direct_model
    )

    # ========================================================
    # 4. Save all results
    # ========================================================

    raw_results = (
        position_raw
        + velocity_raw
    )

    summary_results = (
        position_summary
        + velocity_summary
    )

    raw_path = os.path.join(
        RESULTS_DIR,
        "observation_noise_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "observation_noise_summary.csv"
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
    # 5. Generate figures
    # ========================================================

    plot_summary(
        noise_type="position",
        noise_values=
            POSITION_NOISE_VALUES,
        summary_results=
            position_summary
    )

    plot_summary(
        noise_type="velocity",
        noise_values=
            VELOCITY_NOISE_VALUES,
        summary_results=
            velocity_summary
    )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 88)
    print(
        "Phase 5.2 Observation Noise Evaluation COMPLETED"
    )
    print("=" * 88)

    print()
    print(
        f"Raw data:"
        f"\n  {raw_path}"
    )

    print()
    print(
        f"Summary:"
        f"\n  {summary_path}"
    )

    print()
    print(
        f"Plots:"
        f"\n  {PLOTS_DIR}/"
    )

    print()


if __name__ == "__main__":
    main()