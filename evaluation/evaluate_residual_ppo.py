import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Configuration
# ============================================================

NUM_EPISODES = 100

TEST_SEED_START = 1000

MODEL_PATH = (
    "models/residual_ppo_uav_tracking"
)

PLOTS_DIR = "plots"

RESULTS_DIR = "results"


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    env,
    model,
    seed
):

    # ========================================================
    # 1. Reset
    # ========================================================

    obs, reset_info = env.reset(
        seed=seed
    )

    initial_uav_x = (
        reset_info["initial_uav_x"]
    )

    initial_uav_y = (
        reset_info["initial_uav_y"]
    )

    ugv_vx_initial = (
        reset_info["ugv_vx"]
    )

    # ========================================================
    # 2. Data storage
    # ========================================================

    times = []

    distances = []

    relative_speeds = []

    actions = []

    base_commands = []

    residual_commands = []

    raw_commands = []

    final_commands = []

    uav_positions = []

    ugv_positions = []

    uav_velocities = []

    ugv_velocities = []

    rewards = []

    total_reward = 0.0

    terminated_flag = False

    # ========================================================
    # 3. Run one episode
    # ========================================================

    for step in range(
        env.max_steps
    ):

        # ----------------------------------------------------
        # Deterministic policy for evaluation
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True
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

        current_time = (
            step * env.dt
        )

        # ----------------------------------------------------
        # Store time
        # ----------------------------------------------------

        times.append(
            current_time
        )

        # ----------------------------------------------------
        # Tracking states
        # ----------------------------------------------------

        distances.append(
            info["distance"]
        )

        relative_speeds.append(
            info["relative_speed"]
        )

        # ----------------------------------------------------
        # Normalized PPO action
        # ----------------------------------------------------

        actions.append(
            np.asarray(
                action,
                dtype=np.float32
            ).copy()
        )

        # ----------------------------------------------------
        # Classical base controller
        #
        # u_base = PD + velocity feedforward
        # ----------------------------------------------------

        base_commands.append([
            info["base_command_x"],
            info["base_command_y"]
        ])

        # ----------------------------------------------------
        # Physical PPO residual
        #
        # delta_u_RL
        # ----------------------------------------------------

        residual_commands.append([
            info["residual_x"],
            info["residual_y"]
        ])

        # ----------------------------------------------------
        # Raw:
        #
        # u_base + delta_u_RL
        #
        # before velocity saturation
        # ----------------------------------------------------

        raw_commands.append([
            info["raw_command_x"],
            info["raw_command_y"]
        ])

        # ----------------------------------------------------
        # Actual command sent to UAV dynamics
        #
        # after saturation
        # ----------------------------------------------------

        final_commands.append([
            info["command_x"],
            info["command_y"]
        ])

        # ----------------------------------------------------
        # Positions
        # ----------------------------------------------------

        uav_positions.append([
            info["uav_x"],
            info["uav_y"]
        ])

        ugv_positions.append([
            info["ugv_x"],
            info["ugv_y"]
        ])

        # ----------------------------------------------------
        # Velocities
        # ----------------------------------------------------

        uav_velocities.append([
            info["uav_vx"],
            info["uav_vy"]
        ])

        ugv_velocities.append([
            info["ugv_vx"],
            info["ugv_vy"]
        ])

        rewards.append(
            reward
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

    # ========================================================
    # 4. Convert to NumPy arrays
    # ========================================================

    times = np.asarray(
        times,
        dtype=np.float32
    )

    distances = np.asarray(
        distances,
        dtype=np.float32
    )

    relative_speeds = np.asarray(
        relative_speeds,
        dtype=np.float32
    )

    actions = np.asarray(
        actions,
        dtype=np.float32
    )

    base_commands = np.asarray(
        base_commands,
        dtype=np.float32
    )

    residual_commands = np.asarray(
        residual_commands,
        dtype=np.float32
    )

    raw_commands = np.asarray(
        raw_commands,
        dtype=np.float32
    )

    final_commands = np.asarray(
        final_commands,
        dtype=np.float32
    )

    uav_positions = np.asarray(
        uav_positions,
        dtype=np.float32
    )

    ugv_positions = np.asarray(
        ugv_positions,
        dtype=np.float32
    )

    uav_velocities = np.asarray(
        uav_velocities,
        dtype=np.float32
    )

    ugv_velocities = np.asarray(
        ugv_velocities,
        dtype=np.float32
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float32
    )

    # ========================================================
    # 5. Full-episode RMSE
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
    # Same definition as previous experiments:
    #
    # t >= 5 seconds
    # ========================================================

    steady_mask = (
        times >= 5.0
    )

    if np.any(
        steady_mask
    ):

        steady_rmse = float(
            np.sqrt(
                np.mean(
                    distances[
                        steady_mask
                    ] ** 2
                )
            )
        )

    else:

        steady_rmse = np.nan

    # ========================================================
    # 7. Final metrics
    # ========================================================

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
    # 8. Stable tracking ratio
    #
    # EXACTLY same thresholds as previous experiments
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
    # 9. Capture time
    #
    # Definition:
    #
    # First point where:
    #
    # distance < 0.15 m
    # relative speed < 0.10 m/s
    #
    # continuously for 0.5 seconds
    # ========================================================

    required_stable_steps = int(
        0.5 / env.dt
    )

    capture_time = np.nan

    for i in range(
        len(stable_mask)
        - required_stable_steps
        + 1
    ):

        stable_window = (
            stable_mask[
                i:
                i + required_stable_steps
            ]
        )

        if np.all(
            stable_window
        ):

            capture_time = float(
                times[i]
            )

            break

    # ========================================================
    # 10. FINAL PHYSICAL COMMAND smoothness
    #
    # IMPORTANT:
    #
    # Compare physical velocity commands,
    # not normalized PPO action.
    #
    # J_delta_u =
    #
    # mean(
    #   ||u_t - u_(t-1)||^2
    # )
    # ========================================================

    if len(
        final_commands
    ) > 1:

        command_diff = np.diff(
            final_commands,
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

        command_diff = np.empty(
            (0, 2),
            dtype=np.float32
        )

        command_change_norm = (
            np.array([])
        )

        command_smoothness = np.nan

        mean_command_change = np.nan

        max_command_change = np.nan

    # ========================================================
    # 11. Residual magnitude
    #
    # This tells us HOW MUCH PPO correction is being used.
    # ========================================================

    residual_magnitudes = (
        np.linalg.norm(
            residual_commands,
            axis=1
        )
    )

    mean_residual_magnitude = float(
        np.mean(
            residual_magnitudes
        )
    )

    max_residual_magnitude = float(
        np.max(
            residual_magnitudes
        )
    )

    rms_residual_magnitude = float(
        np.sqrt(
            np.mean(
                residual_magnitudes ** 2
            )
        )
    )

    # ========================================================
    # 12. Residual smoothness
    #
    # Separate from final command smoothness.
    # ========================================================

    if len(
        residual_commands
    ) > 1:

        residual_diff = np.diff(
            residual_commands,
            axis=0
        )

        residual_smoothness = float(
            np.mean(
                np.sum(
                    residual_diff ** 2,
                    axis=1
                )
            )
        )

    else:

        residual_smoothness = (
            np.nan
        )

    # ========================================================
    # 13. Saturation ratio
    #
    # How often final physical command reaches UAV limit
    # ========================================================

    saturation_mask = np.any(
        np.abs(
            final_commands
        )
        >= (
            env.simulator.max_uav_speed
            - 1e-6
        ),
        axis=1
    )

    saturation_ratio = float(
        np.mean(
            saturation_mask
        )
    )

    # ========================================================
    # 14. Success definition
    #
    # SAME as Generalized PPO / PD+FF
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
    # Return result
    # ========================================================

    return {

        # ----------------------------------------------------
        # Scenario
        # ----------------------------------------------------

        "seed":
            seed,

        "initial_uav_x":
            float(initial_uav_x),

        "initial_uav_y":
            float(initial_uav_y),

        "ugv_vx":
            float(ugv_vx_initial),

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------

        "episode_length":
            len(distances),

        "total_reward":
            float(total_reward),

        "success":
            bool(success),

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Command metrics
        # ----------------------------------------------------

        "command_smoothness":
            command_smoothness,

        "mean_command_change":
            mean_command_change,

        "max_command_change":
            max_command_change,

        "saturation_ratio":
            saturation_ratio,

        # ----------------------------------------------------
        # Residual metrics
        # ----------------------------------------------------

        "mean_residual":
            mean_residual_magnitude,

        "rms_residual":
            rms_residual_magnitude,

        "max_residual":
            max_residual_magnitude,

        "residual_smoothness":
            residual_smoothness,

        # ----------------------------------------------------
        # Detailed arrays for representative plots
        # ----------------------------------------------------

        "times":
            times,

        "distances":
            distances,

        "relative_speeds":
            relative_speeds,

        "actions":
            actions,

        "base_commands":
            base_commands,

        "residual_commands":
            residual_commands,

        "raw_commands":
            raw_commands,

        "final_commands":
            final_commands,

        "uav_positions":
            uav_positions,

        "ugv_positions":
            ugv_positions,

        "uav_velocities":
            uav_velocities,

        "ugv_velocities":
            ugv_velocities,

        "command_change_norm":
            command_change_norm,

        "residual_magnitudes":
            residual_magnitudes
    }


# ============================================================
# Plot representative episode
# ============================================================

def plot_representative_episode(
    result
):

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    seed = result["seed"]

    times = result["times"]

    # ========================================================
    # Figure 1
    # Trajectory
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        result["ugv_positions"][:, 0],
        result["ugv_positions"][:, 1],
        label="UGV"
    )

    plt.plot(
        result["uav_positions"][:, 0],
        result["uav_positions"][:, 1],
        label="UAV - Residual PPO"
    )

    plt.scatter(
        result["ugv_positions"][0, 0],
        result["ugv_positions"][0, 1],
        marker="o",
        label="UGV start"
    )

    plt.scatter(
        result["uav_positions"][0, 0],
        result["uav_positions"][0, 1],
        marker="x",
        s=80,
        label="UAV start"
    )

    plt.xlabel(
        "X position (m)"
    )

    plt.ylabel(
        "Y position (m)"
    )

    plt.title(
        f"Residual PPO Trajectory - Seed {seed}"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_trajectory.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 2
    # Tracking error
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["distances"]
    )

    plt.axhline(
        y=0.15,
        linestyle="--",
        label="Tracking threshold"
    )

    plt.axvline(
        x=5.0,
        linestyle="--",
        label="Steady-state start"
    )

    if np.isfinite(
        result["capture_time"]
    ):

        plt.axvline(
            x=result["capture_time"],
            linestyle=":",
            label="Capture time"
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Tracking error (m)"
    )

    plt.title(
        f"Residual PPO Tracking Error - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_tracking_error.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 3
    # Velocity response
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["uav_velocities"][:, 0],
        label="UAV vx"
    )

    plt.plot(
        times,
        result["ugv_velocities"][:, 0],
        label="UGV vx"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity (m/s)"
    )

    plt.title(
        f"Residual PPO Velocity Response - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_velocity_response.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 4
    # X command decomposition
    #
    # base + residual -> final command
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["base_commands"][:, 0],
        label="PD+FF base x"
    )

    plt.plot(
        times,
        result["residual_commands"][:, 0],
        label="RL residual x"
    )

    plt.plot(
        times,
        result["final_commands"][:, 0],
        label="Final command x"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity command (m/s)"
    )

    plt.title(
        f"Residual PPO X Command Decomposition - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_command_x_components.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 5
    # Y command decomposition
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["base_commands"][:, 1],
        label="PD+FF base y"
    )

    plt.plot(
        times,
        result["residual_commands"][:, 1],
        label="RL residual y"
    )

    plt.plot(
        times,
        result["final_commands"][:, 1],
        label="Final command y"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity command (m/s)"
    )

    plt.title(
        f"Residual PPO Y Command Decomposition - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_command_y_components.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 6
    # Final physical command
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["final_commands"][:, 0],
        label="Final command x"
    )

    plt.plot(
        times,
        result["final_commands"][:, 1],
        label="Final command y"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity command (m/s)"
    )

    plt.title(
        f"Residual PPO Final Velocity Command - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_final_commands.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 7
    # Physical command change
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times[1:],
        result["command_change_norm"]
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "||u(t) - u(t-1)|| (m/s)"
    )

    plt.title(
        f"Residual PPO Command Change - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_command_change.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 8
    # Residual magnitude
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["residual_magnitudes"]
    )

    plt.axhline(
        y=0.15,
        linestyle="--",
        label="Residual limit"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "||delta u_RL|| (m/s)"
    )

    plt.title(
        f"Residual PPO Residual Magnitude - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_residual_magnitude.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Save scalar results to CSV
# ============================================================

def save_results_csv(
    results
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    csv_path = os.path.join(
        RESULTS_DIR,
        "residual_ppo_100_episode_results.csv"
    )

    fieldnames = [
        "seed",
        "initial_uav_x",
        "initial_uav_y",
        "ugv_vx",
        "episode_length",
        "success",
        "total_reward",
        "rmse",
        "steady_rmse",
        "max_tracking_error",
        "final_error",
        "final_relative_speed",
        "stable_ratio",
        "capture_time",
        "command_smoothness",
        "mean_command_change",
        "max_command_change",
        "saturation_ratio",
        "mean_residual",
        "rms_residual",
        "max_residual",
        "residual_smoothness"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                key: result[key]
                for key in fieldnames
            }

            writer.writerow(
                row
            )

    return csv_path


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. Environment
    #
    # Same randomized distribution as:
    #
    # PD+FF
    # Generalized PPO
    # ========================================================

    env = UAVTrackingResidualEnv(
        randomize=True
    )

    # ========================================================
    # 2. Load trained model
    # ========================================================

    model = PPO.load(
        MODEL_PATH
    )

    results = []

    # ========================================================
    # 3. Run 100 frozen test scenarios
    #
    # seeds:
    #
    # 1000 ... 1099
    # ========================================================

    print()
    print("=" * 82)

    print(
        "Residual PPO - "
        "100 Randomized Episode Evaluation"
    )

    print("=" * 82)

    for episode in range(
        NUM_EPISODES
    ):

        seed = (
            TEST_SEED_START
            + episode
        )

        result = evaluate_episode(
            env=env,
            model=model,
            seed=seed
        )

        results.append(
            result
        )

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        capture_text = (
            f"{result['capture_time']:.2f}s"
            if np.isfinite(
                result["capture_time"]
            )
            else "N/A"
        )

        print(
            f"Episode {episode + 1:3d} | "
            f"seed={seed} | "
            f"UAV=("
            f"{result['initial_uav_x']:+.2f}, "
            f"{result['initial_uav_y']:+.2f}) | "
            f"UGV vx="
            f"{result['ugv_vx']:.2f} | "
            f"RMSE="
            f"{result['rmse']:.3f} | "
            f"capture="
            f"{capture_text} | "
            f"{status}"
        )

    # ========================================================
    # 4. Aggregate arrays
    # ========================================================

    success_values = np.asarray([
        r["success"]
        for r in results
    ])

    reward_values = np.asarray([
        r["total_reward"]
        for r in results
    ])

    rmse_values = np.asarray([
        r["rmse"]
        for r in results
    ])

    steady_rmse_values = np.asarray([
        r["steady_rmse"]
        for r in results
    ])

    final_error_values = np.asarray([
        r["final_error"]
        for r in results
    ])

    final_speed_values = np.asarray([
        r["final_relative_speed"]
        for r in results
    ])

    stable_ratio_values = np.asarray([
        r["stable_ratio"]
        for r in results
    ])

    capture_time_values = np.asarray([
        r["capture_time"]
        for r in results
    ])

    command_smoothness_values = np.asarray([
        r["command_smoothness"]
        for r in results
    ])

    mean_command_change_values = np.asarray([
        r["mean_command_change"]
        for r in results
    ])

    max_command_change_values = np.asarray([
        r["max_command_change"]
        for r in results
    ])

    mean_residual_values = np.asarray([
        r["mean_residual"]
        for r in results
    ])

    rms_residual_values = np.asarray([
        r["rms_residual"]
        for r in results
    ])

    max_residual_values = np.asarray([
        r["max_residual"]
        for r in results
    ])

    residual_smoothness_values = np.asarray([
        r["residual_smoothness"]
        for r in results
    ])

    saturation_ratio_values = np.asarray([
        r["saturation_ratio"]
        for r in results
    ])

    # ========================================================
    # 5. Print final summary
    # ========================================================

    print()
    print("=" * 82)

    print(
        "RESIDUAL PPO RANDOMIZED RESULTS"
    )

    print("=" * 82)

    print(
        f"Number of episodes         : "
        f"{NUM_EPISODES}"
    )

    print(
        f"Success rate               : "
        f"{np.mean(success_values) * 100:.2f}%"
    )

    print()

    print(
        f"Total reward               : "
        f"{np.mean(reward_values):.3f} "
        f"± {np.std(reward_values):.3f}"
    )

    print(
        f"Full RMSE                  : "
        f"{np.mean(rmse_values):.4f} "
        f"± {np.std(rmse_values):.4f} m"
    )

    print(
        f"Steady-state RMSE (>5s)    : "
        f"{np.mean(steady_rmse_values):.4f} "
        f"± {np.std(steady_rmse_values):.4f} m"
    )

    print(
        f"Final tracking error       : "
        f"{np.mean(final_error_values):.4f} "
        f"± {np.std(final_error_values):.4f} m"
    )

    print(
        f"Final relative speed       : "
        f"{np.mean(final_speed_values):.4f} "
        f"± {np.std(final_speed_values):.4f} m/s"
    )

    print(
        f"Stable tracking ratio      : "
        f"{np.mean(stable_ratio_values) * 100:.2f}% "
        f"± "
        f"{np.std(stable_ratio_values) * 100:.2f}%"
    )

    print(
        f"Capture time               : "
        f"{np.nanmean(capture_time_values):.3f} "
        f"± {np.nanstd(capture_time_values):.3f} s"
    )

    print()
    print(
        "--- Physical Command Metrics ---"
    )

    print(
        f"Command smoothness cost    : "
        f"{np.mean(command_smoothness_values):.6f} "
        f"± "
        f"{np.std(command_smoothness_values):.6f}"
    )

    print(
        f"Mean command change        : "
        f"{np.mean(mean_command_change_values):.6f} "
        f"± "
        f"{np.std(mean_command_change_values):.6f} m/s"
    )

    print(
        f"Max command change         : "
        f"{np.mean(max_command_change_values):.4f} "
        f"± "
        f"{np.std(max_command_change_values):.4f} m/s"
    )

    print(
        f"Command saturation ratio   : "
        f"{np.mean(saturation_ratio_values) * 100:.2f}% "
        f"± "
        f"{np.std(saturation_ratio_values) * 100:.2f}%"
    )

    print()
    print(
        "--- Residual RL Metrics ---"
    )

    print(
        f"Mean residual magnitude    : "
        f"{np.mean(mean_residual_values):.5f} "
        f"± "
        f"{np.std(mean_residual_values):.5f} m/s"
    )

    print(
        f"RMS residual magnitude     : "
        f"{np.mean(rms_residual_values):.5f} "
        f"± "
        f"{np.std(rms_residual_values):.5f} m/s"
    )

    print(
        f"Max residual magnitude     : "
        f"{np.mean(max_residual_values):.5f} "
        f"± "
        f"{np.std(max_residual_values):.5f} m/s"
    )

    print(
        f"Residual smoothness cost   : "
        f"{np.mean(residual_smoothness_values):.6f} "
        f"± "
        f"{np.std(residual_smoothness_values):.6f}"
    )

    print("=" * 82)

    # ========================================================
    # 6. Failed episodes
    # ========================================================

    failed_results = [
        result
        for result in results
        if not result["success"]
    ]

    if len(
        failed_results
    ) == 0:

        print()
        print(
            "All 100 test episodes PASSED."
        )

    else:

        print()
        print(
            f"Failed episodes: "
            f"{len(failed_results)}"
        )

        for result in failed_results:

            print(
                f"seed={result['seed']} | "
                f"RMSE={result['rmse']:.3f} | "
                f"Final error="
                f"{result['final_error']:.3f} | "
                f"Stable="
                f"{result['stable_ratio'] * 100:.1f}%"
            )

    # ========================================================
    # 7. Representative episode
    #
    # Select RMSE nearest median.
    # Avoid cherry-picking best episode.
    # ========================================================

    median_rmse = np.median(
        rmse_values
    )

    representative_index = int(
        np.argmin(
            np.abs(
                rmse_values
                - median_rmse
            )
        )
    )

    representative_result = (
        results[
            representative_index
        ]
    )

    print()
    print(
        "Representative episode:"
    )

    print(
        f"seed = "
        f"{representative_result['seed']}"
    )

    print(
        f"UAV initial position = "
        f"("
        f"{representative_result['initial_uav_x']:.3f}, "
        f"{representative_result['initial_uav_y']:.3f}"
        f")"
    )

    print(
        f"UGV vx = "
        f"{representative_result['ugv_vx']:.3f} m/s"
    )

    print(
        f"RMSE = "
        f"{representative_result['rmse']:.4f} m"
    )

    print(
        f"Capture time = "
        f"{representative_result['capture_time']:.3f} s"
    )

    print(
        f"Mean residual = "
        f"{representative_result['mean_residual']:.5f} m/s"
    )

    # ========================================================
    # 8. Save CSV
    # ========================================================

    csv_path = save_results_csv(
        results
    )

    # ========================================================
    # 9. Representative plots
    # ========================================================

    plot_representative_episode(
        representative_result
    )

    # ========================================================
    # Figure 9
    # RMSE distribution
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        rmse_values,
        bins=15
    )

    plt.xlabel(
        "Full RMSE (m)"
    )

    plt.ylabel(
        "Number of episodes"
    )

    plt.title(
        "Residual PPO - RMSE Distribution"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_rmse_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure 10
    # Initial distance vs RMSE
    # ========================================================

    initial_distances = np.asarray([
        np.sqrt(
            result["initial_uav_x"] ** 2
            +
            result["initial_uav_y"] ** 2
        )
        for result in results
    ])

    plt.figure(
        figsize=(9, 5)
    )

    plt.scatter(
        initial_distances,
        rmse_values
    )

    plt.xlabel(
        "Initial UAV-UGV distance (m)"
    )

    plt.ylabel(
        "Full RMSE (m)"
    )

    plt.title(
        "Residual PPO - Initial Distance vs RMSE"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "residual_ppo_generalization_scatter.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 82)

    print(
        f"CSV saved to:"
    )

    print(
        f"  {csv_path}"
    )

    print()

    print(
        "Plots saved to:"
    )

    print(
        f"  {PLOTS_DIR}/"
    )

    print("=" * 82)
    print()

    env.close()


if __name__ == "__main__":
    main()