import os

import numpy as np
import matplotlib.pyplot as plt

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Evaluate one PD + FF episode
# ============================================================

def evaluate_episode(
    env,
    seed
):

    # ========================================================
    # 1. Reset with fixed test seed
    # ========================================================

    obs, info = env.reset(
        seed=seed
    )

    initial_uav_x = (
        info["initial_uav_x"]
    )

    initial_uav_y = (
        info["initial_uav_y"]
    )

    ugv_vx = (
        info["ugv_vx"]
    )

    # ========================================================
    # Data storage
    # ========================================================

    times = []

    distances = []

    relative_speeds = []

    uav_positions = []

    ugv_positions = []

    uav_velocities = []

    ugv_velocities = []

    commands = []

    base_commands = []

    total_reward = 0.0

    # ========================================================
    # 2. Run episode
    # ========================================================

    for step in range(
        env.max_steps
    ):

        # ----------------------------------------------------
        # ZERO residual:
        #
        # v_cmd = v_PD+FF + 0
        #
        # Therefore this is a pure PD + FF baseline.
        # ----------------------------------------------------

        action = np.array(
            [0.0, 0.0],
            dtype=np.float32
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

        times.append(
            current_time
        )

        distances.append(
            info["distance"]
        )

        relative_speeds.append(
            info["relative_speed"]
        )

        uav_positions.append([
            info["uav_x"],
            info["uav_y"]
        ])

        ugv_positions.append([
            info["ugv_x"],
            info["ugv_y"]
        ])

        uav_velocities.append([
            info["uav_vx"],
            info["uav_vy"]
        ])

        ugv_velocities.append([
            info["ugv_vx"],
            info["ugv_vy"]
        ])

        base_commands.append([
            info["base_command_x"],
            info["base_command_y"]
        ])

        commands.append([
            info["command_x"],
            info["command_y"]
        ])

        total_reward += (
            reward
        )

        if (
            terminated
            or truncated
        ):
            break

    # ========================================================
    # 3. NumPy conversion
    # ========================================================

    times = np.asarray(
        times
    )

    distances = np.asarray(
        distances
    )

    relative_speeds = np.asarray(
        relative_speeds
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

    base_commands = np.asarray(
        base_commands
    )

    commands = np.asarray(
        commands
    )

    # ========================================================
    # 4. Full RMSE
    # ========================================================

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    # ========================================================
    # 5. Steady-state RMSE
    #
    # t > 5 s
    # ========================================================

    steady_start = int(
        5.0 / env.dt
    )

    if (
        len(distances)
        > steady_start
    ):

        steady_rmse = np.sqrt(
            np.mean(
                distances[
                    steady_start:
                ] ** 2
            )
        )

    else:

        steady_rmse = np.nan

    # ========================================================
    # 6. Stable tracking ratio
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
    # 7. Capture time
    #
    # Definition:
    #
    # First time the controller enters the stable region:
    #
    # distance < 0.15 m
    # relative speed < 0.10 m/s
    #
    # and stays there continuously for 0.5 seconds.
    #
    # dt = 0.05 s
    # 0.5 / 0.05 = 10 steps
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

        if np.all(
            stable_mask[
                i:
                i + required_stable_steps
            ]
        ):

            capture_time = (
                times[i]
            )

            break

    # ========================================================
    # 8. Command smoothness
    #
    # IMPORTANT:
    #
    # Compare actual physical commands,
    # not normalized PPO actions.
    #
    # J_du =
    # mean(||u_t - u_(t-1)||^2)
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

        command_change_norm = (
            np.linalg.norm(
                command_diff,
                axis=1
            )
        )

        max_command_change = float(
            np.max(
                command_change_norm
            )
        )

    else:

        command_diff = np.empty(
            (0, 2)
        )

        command_change_norm = (
            np.array([])
        )

        command_smoothness = np.nan

        max_command_change = np.nan

    # ========================================================
    # 9. Final metrics
    # ========================================================

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

    # ========================================================
    # 10. Success criterion
    #
    # EXACTLY same definition as Generalized PPO
    # ========================================================

    success = (
        stable_ratio >= 0.50
        and
        final_error < 0.15
        and
        final_relative_speed < 0.10
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        "seed":
            seed,

        "initial_uav_x":
            initial_uav_x,

        "initial_uav_y":
            initial_uav_y,

        "ugv_vx":
            ugv_vx,

        "episode_length":
            len(distances),

        "reward":
            total_reward,

        "rmse":
            float(rmse),

        "steady_rmse":
            float(steady_rmse),

        "final_error":
            final_error,

        "final_relative_speed":
            final_relative_speed,

        "stable_ratio":
            stable_ratio,

        "capture_time":
            float(capture_time),

        "command_smoothness":
            command_smoothness,

        "max_command_change":
            max_command_change,

        "success":
            bool(success),

        # detailed data for plots

        "times":
            times,

        "distances":
            distances,

        "relative_speeds":
            relative_speeds,

        "uav_positions":
            uav_positions,

        "ugv_positions":
            ugv_positions,

        "uav_velocities":
            uav_velocities,

        "ugv_velocities":
            ugv_velocities,

        "base_commands":
            base_commands,

        "commands":
            commands,

        "command_change_norm":
            command_change_norm
    }


# ============================================================
# Representative episode plots
# ============================================================

def plot_representative_episode(
    result
):

    os.makedirs(
        "plots",
        exist_ok=True
    )

    seed = result["seed"]

    times = result["times"]

    # ========================================================
    # Figure 1: trajectory
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
        label="UAV - PD + FF"
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
        f"PD + FF Trajectory - Seed {seed}"
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
        "pd_randomized_trajectory.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 2: tracking error
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

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Tracking error (m)"
    )

    plt.title(
        f"PD + FF Tracking Error - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_tracking_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 3: velocity response
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
        f"PD + FF Velocity Response - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_velocity_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 4: velocity command
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        result["commands"][:, 0],
        label="Command x"
    )

    plt.plot(
        times,
        result["commands"][:, 1],
        label="Command y"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity command (m/s)"
    )

    plt.title(
        f"PD + FF Velocity Command - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_commands.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 5: command change
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
        "||u(t) - u(t-1)||"
    )

    plt.title(
        f"PD + FF Command Change - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_command_change.png",
        dpi=300,
        bbox_inches="tight"
    )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "plots",
        exist_ok=True
    )

    # ========================================================
    # 1. EXACT same randomized environment
    # ========================================================

    env = UAVTrackingResidualEnv(
        randomize=True
    )

    # ========================================================
    # 2. EXACT same 100 test seeds
    #
    # Generalized PPO:
    #
    # seeds 1000 - 1099
    #
    # PD + FF uses SAME scenarios.
    # ========================================================

    num_episodes = 100

    test_seed_start = 1000

    results = []

    print()
    print("=" * 78)

    print(
        "PD + Velocity Feedforward - "
        "100 Randomized Episode Evaluation"
    )

    print("=" * 78)

    # ========================================================
    # 3. Evaluate 100 episodes
    # ========================================================

    for episode in range(
        num_episodes
    ):

        seed = (
            test_seed_start
            + episode
        )

        result = evaluate_episode(
            env,
            seed
        )

        results.append(
            result
        )

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
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
            f"stable="
            f"{result['stable_ratio'] * 100:5.1f}% | "
            f"{status}"
        )

    # ========================================================
    # 4. Aggregate metrics
    # ========================================================

    success_values = np.asarray([
        r["success"]
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

    final_errors = np.asarray([
        r["final_error"]
        for r in results
    ])

    final_speeds = np.asarray([
        r["final_relative_speed"]
        for r in results
    ])

    stable_ratios = np.asarray([
        r["stable_ratio"]
        for r in results
    ])

    capture_times = np.asarray([
        r["capture_time"]
        for r in results
    ])

    command_smoothness_values = np.asarray([
        r["command_smoothness"]
        for r in results
    ])

    max_command_changes = np.asarray([
        r["max_command_change"]
        for r in results
    ])

    # ========================================================
    # 5. Summary
    # ========================================================

    print()
    print("=" * 78)
    print(
        "PD + FF RANDOMIZED RESULTS"
    )
    print("=" * 78)

    print(
        f"Number of episodes        : "
        f"{num_episodes}"
    )

    print(
        f"Success rate              : "
        f"{np.mean(success_values) * 100:.2f}%"
    )

    print()

    print(
        f"Full RMSE                 : "
        f"{np.mean(rmse_values):.4f} "
        f"± {np.std(rmse_values):.4f} m"
    )

    print(
        f"Steady-state RMSE (>5s)   : "
        f"{np.mean(steady_rmse_values):.4f} "
        f"± {np.std(steady_rmse_values):.4f} m"
    )

    print(
        f"Final tracking error      : "
        f"{np.mean(final_errors):.4f} "
        f"± {np.std(final_errors):.4f} m"
    )

    print(
        f"Final relative speed      : "
        f"{np.mean(final_speeds):.4f} "
        f"± {np.std(final_speeds):.4f} m/s"
    )

    print(
        f"Stable tracking ratio     : "
        f"{np.mean(stable_ratios) * 100:.2f}% "
        f"± {np.std(stable_ratios) * 100:.2f}%"
    )

    print(
        f"Capture time              : "
        f"{np.nanmean(capture_times):.3f} "
        f"± {np.nanstd(capture_times):.3f} s"
    )

    print(
        f"Command smoothness cost   : "
        f"{np.mean(command_smoothness_values):.6f} "
        f"± "
        f"{np.std(command_smoothness_values):.6f}"
    )

    print(
        f"Max command change        : "
        f"{np.mean(max_command_changes):.4f} "
        f"± "
        f"{np.std(max_command_changes):.4f} m/s"
    )

    print("=" * 78)

    # ========================================================
    # 6. Representative episode
    #
    # Select episode nearest median RMSE
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

    # ========================================================
    # 7. Representative plots
    # ========================================================

    plot_representative_episode(
        representative_result
    )

    # ========================================================
    # 8. RMSE distribution
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
        "PD + FF - RMSE Distribution"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_rmse_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # 9. Initial distance vs RMSE
    # ========================================================

    initial_distances = np.asarray([
        np.sqrt(
            r["initial_uav_x"] ** 2
            +
            r["initial_uav_y"] ** 2
        )
        for r in results
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
        "PD + FF - Initial Distance vs RMSE"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "pd_randomized_generalization_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print("=" * 78)
    print(
        "Plots saved to: plots/"
    )
    print("=" * 78)
    print()

    plt.show()

    env.close()


if __name__ == "__main__":
    main()