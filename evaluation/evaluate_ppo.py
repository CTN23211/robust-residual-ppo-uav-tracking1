import os

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    # ==========================================================
    # 1. Create environment
    # ==========================================================

    env = UAVTrackingGymEnv()

    # ==========================================================
    # 2. Load NEW smooth PPO
    # ==========================================================

    model = PPO.load(
        "models/"
        "vanilla_ppo_smooth_uav_tracking"
    )

    # ==========================================================
    # 3. Reset environment
    # ==========================================================

    obs, info = env.reset(
        seed=42
    )

    # ==========================================================
    # 4. Data storage
    # ==========================================================

    times = []

    uav_positions = []
    ugv_positions = []

    uav_velocities = []
    ugv_velocities = []

    actions = []

    distances = []
    relative_speeds = []

    rewards = []

    total_reward = 0.0

    # ==========================================================
    # 5. Evaluation loop
    # ==========================================================

    for step in range(
        env.max_steps
    ):

        # Deterministic evaluation:
        # no stochastic exploration
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
        ) = env.step(action)

        state = (
            env.simulator.get_state()
        )

        current_time = (
            step * env.dt
        )

        # ------------------------------------------------------
        # Save data
        # ------------------------------------------------------

        times.append(
            current_time
        )

        uav_positions.append(
            state["uav_pos"].copy()
        )

        ugv_positions.append(
            state["ugv_pos"].copy()
        )

        uav_velocities.append(
            state["uav_vel"].copy()
        )

        ugv_velocities.append(
            state["ugv_vel"].copy()
        )

        actions.append(
            np.asarray(
                action,
                dtype=np.float32
            ).copy()
        )

        distances.append(
            info["distance"]
        )

        relative_speeds.append(
            info["relative_speed"]
        )

        rewards.append(
            reward
        )

        total_reward += (
            reward
        )

        if (
            terminated
            or truncated
        ):
            break

    # ==========================================================
    # 6. Convert to NumPy
    # ==========================================================

    times = np.asarray(
        times
    )

    uav_positions = np.asarray(
        uav_positions
    )

    ugv_positions = np.asarray(
        ugv_positions
    )

    uav_velocities = np.asarray(
        uav_velocities
    )

    ugv_velocities = np.asarray(
        ugv_velocities
    )

    actions = np.asarray(
        actions
    )

    distances = np.asarray(
        distances
    )

    relative_speeds = np.asarray(
        relative_speeds
    )

    rewards = np.asarray(
        rewards
    )

    # ==========================================================
    # 7. Tracking metrics
    # ==========================================================

    # ----------------------------------------------------------
    # Full episode RMSE
    # ----------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    # ----------------------------------------------------------
    # Steady-state RMSE
    #
    # Only evaluate after t >= 5 seconds
    # ----------------------------------------------------------

    steady_state_mask = (
        times >= 5.0
    )

    if np.any(
        steady_state_mask
    ):

        steady_state_rmse = (
            np.sqrt(
                np.mean(
                    distances[
                        steady_state_mask
                    ] ** 2
                )
            )
        )

    else:

        steady_state_rmse = (
            np.nan
        )

    max_error = (
        np.max(
            distances
        )
    )

    min_error = (
        np.min(
            distances
        )
    )

    final_error = (
        distances[-1]
    )

    final_relative_speed = (
        relative_speeds[-1]
    )

    peak_uav_vx = (
        np.max(
            uav_velocities[:, 0]
        )
    )

    # ==========================================================
    # 8. Stable tracking ratio
    # ==========================================================

    stable_mask = (
        (distances < 0.15)
        &
        (relative_speeds < 0.10)
    )

    stable_tracking_ratio = (
        np.mean(
            stable_mask
        )
    )

    # ==========================================================
    # 9. Action smoothness metrics
    # ==========================================================

    action_differences = np.diff(
        actions,
        axis=0
    )

    action_change_norms = (
        np.linalg.norm(
            action_differences,
            axis=1
        )
    )

    # Same basic quantity used
    # in the smoothness reward:
    #
    # mean(||a_t - a_t-1||^2)

    action_smoothness_cost = (
        np.mean(
            np.sum(
                action_differences ** 2,
                axis=1
            )
        )
    )

    mean_action_change = (
        np.mean(
            action_change_norms
        )
    )

    max_action_change = (
        np.max(
            action_change_norms
        )
    )

    # ==========================================================
    # 10. Print evaluation
    # ==========================================================

    print()
    print("=" * 65)
    print(
        "Smooth Vanilla PPO Evaluation"
    )
    print("=" * 65)

    print(
        f"Episode length          : "
        f"{len(times)} steps"
    )

    print(
        f"Total reward            : "
        f"{total_reward:.3f}"
    )

    print(
        f"Full RMSE               : "
        f"{rmse:.4f} m"
    )

    print(
        f"Steady-state RMSE (>5s) : "
        f"{steady_state_rmse:.4f} m"
    )

    print(
        f"Max tracking error      : "
        f"{max_error:.4f} m"
    )

    print(
        f"Min tracking error      : "
        f"{min_error:.4f} m"
    )

    print(
        f"Final tracking error    : "
        f"{final_error:.4f} m"
    )

    print(
        f"Final relative speed    : "
        f"{final_relative_speed:.4f} m/s"
    )

    print(
        f"Peak UAV vx             : "
        f"{peak_uav_vx:.4f} m/s"
    )

    print(
        f"Stable tracking ratio   : "
        f"{stable_tracking_ratio * 100:.2f}%"
    )

    print()
    print(
        "--- Action Smoothness ---"
    )

    print(
        f"Smoothness cost         : "
        f"{action_smoothness_cost:.6f}"
    )

    print(
        f"Mean action change      : "
        f"{mean_action_change:.6f}"
    )

    print(
        f"Max action change       : "
        f"{max_action_change:.6f}"
    )

    print("=" * 65)
    print()

    # ==========================================================
    # 11. Create plots folder
    # ==========================================================

    os.makedirs(
        "plots",
        exist_ok=True
    )

    # ==========================================================
    # Figure 1:
    # UAV / UGV trajectory
    # ==========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ugv_positions[:, 0],
        ugv_positions[:, 1],
        label="UGV"
    )

    plt.plot(
        uav_positions[:, 0],
        uav_positions[:, 1],
        label="UAV - Smooth PPO"
    )

    plt.scatter(
        ugv_positions[0, 0],
        ugv_positions[0, 1],
        marker="o",
        label="UGV start"
    )

    plt.scatter(
        uav_positions[0, 0],
        uav_positions[0, 1],
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
        "Smooth Vanilla PPO UAV Tracking"
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
        "plots/"
        "ppo_smooth_trajectory.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 2:
    # Tracking error
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        distances
    )

    plt.axhline(
        y=0.15,
        linestyle="--",
        label="Tracking threshold"
    )

    plt.axvline(
        x=5.0,
        linestyle="--",
        label="Steady-state evaluation start"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Tracking error (m)"
    )

    plt.title(
        "Smooth PPO Tracking Error"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "ppo_smooth_tracking_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 3:
    # Velocity response
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        uav_velocities[:, 0],
        label="UAV vx"
    )

    plt.plot(
        times,
        ugv_velocities[:, 0],
        label="UGV vx"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity (m/s)"
    )

    plt.title(
        "Smooth PPO Velocity Response"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "ppo_smooth_velocity_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 4:
    # PPO actions
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        actions[:, 0],
        label="PPO action x"
    )

    plt.plot(
        times,
        actions[:, 1],
        label="PPO action y"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Normalized action"
    )

    plt.title(
        "Smooth Vanilla PPO Actions"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "ppo_smooth_actions.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 5:
    # Action change magnitude
    # ==========================================================

    action_change_times = (
        times[1:]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        action_change_times,
        action_change_norms
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "||a(t) - a(t-1)||"
    )

    plt.title(
        "PPO Action Change Magnitude"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "ppo_smooth_action_change.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    env.close()


if __name__ == "__main__":
    main()