import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs.uav_tracking_gym_env import UAVTrackingGymEnv


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    env,
    model,
    seed,
    save_trajectory=False
):

    obs, info = env.reset(
        seed=seed
    )

    initial_uav_x = info["initial_uav_x"]
    initial_uav_y = info["initial_uav_y"]
    ugv_vx = info["ugv_vx"]

    distances = []
    relative_speeds = []
    actions = []

    uav_positions = []
    ugv_positions = []

    uav_velocities = []
    ugv_velocities = []

    times = []

    total_reward = 0.0

    # ========================================================
    # Run episode
    # ========================================================

    for step in range(
        env.max_steps
    ):

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

        state = (
            env.simulator.get_state()
        )

        time = (
            step * env.dt
        )

        times.append(
            time
        )

        distances.append(
            info["distance"]
        )

        relative_speeds.append(
            info["relative_speed"]
        )

        actions.append(
            np.asarray(
                action,
                dtype=np.float32
            ).copy()
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

        total_reward += (
            reward
        )

        if (
            terminated
            or truncated
        ):
            break

    # ========================================================
    # Convert to NumPy
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

    actions = np.asarray(
        actions
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

    # ========================================================
    # Metrics
    # ========================================================

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    # --------------------------------------------------------
    # Steady state after 5 s
    # --------------------------------------------------------

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

        steady_rmse = (
            np.nan
        )

    # --------------------------------------------------------
    # Stable tracking ratio
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Action smoothness
    # --------------------------------------------------------

    if len(actions) > 1:

        action_diff = np.diff(
            actions,
            axis=0
        )

        action_change_norm = (
            np.linalg.norm(
                action_diff,
                axis=1
            )
        )

        smoothness = float(
            np.mean(
                np.sum(
                    action_diff ** 2,
                    axis=1
                )
            )
        )

    else:

        action_diff = np.empty(
            (0, 2)
        )

        action_change_norm = (
            np.array([])
        )

        smoothness = np.nan

    final_error = float(
        distances[-1]
    )

    final_relative_speed = float(
        relative_speeds[-1]
    )

    # ========================================================
    # Success criterion
    # ========================================================

    success = (
        stable_ratio >= 0.50
        and
        final_error < 0.15
        and
        final_relative_speed < 0.10
    )

    return {

        "seed": seed,

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

        "smoothness":
            smoothness,

        "success":
            bool(success),

        # Detailed data for plots

        "times":
            times,

        "distances":
            distances,

        "relative_speeds":
            relative_speeds,

        "actions":
            actions,

        "action_change_norm":
            action_change_norm,

        "uav_positions":
            uav_positions,

        "ugv_positions":
            ugv_positions,

        "uav_velocities":
            uav_velocities,

        "ugv_velocities":
            ugv_velocities
    }


# ============================================================
# Plot representative episode
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

    uav_positions = (
        result["uav_positions"]
    )

    ugv_positions = (
        result["ugv_positions"]
    )

    uav_velocities = (
        result["uav_velocities"]
    )

    ugv_velocities = (
        result["ugv_velocities"]
    )

    distances = (
        result["distances"]
    )

    actions = (
        result["actions"]
    )

    action_change_norm = (
        result["action_change_norm"]
    )

    # ========================================================
    # Figure 1
    # Trajectory
    # ========================================================

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
        label="Generalized PPO UAV"
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
        f"Generalized PPO Trajectory - Seed {seed}"
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
        "generalized_ppo_trajectory.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 2
    # Tracking error
    # ========================================================

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
        label="Steady-state start"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Tracking error (m)"
    )

    plt.title(
        f"Generalized PPO Tracking Error - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_tracking_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 3
    # Velocity response
    # ========================================================

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
        f"Generalized PPO Velocity Response - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_velocity_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 4
    # Actions
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        actions[:, 0],
        label="Action x"
    )

    plt.plot(
        times,
        actions[:, 1],
        label="Action y"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Normalized action"
    )

    plt.title(
        f"Generalized PPO Actions - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_actions.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 5
    # Action changes
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times[1:],
        action_change_norm
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "||a(t) - a(t-1)||"
    )

    plt.title(
        f"Generalized PPO Action Change - Seed {seed}"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_action_change.png",
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
    # Randomized evaluation environment
    # ========================================================

    env = UAVTrackingGymEnv(
        randomize=True
    )

    # ========================================================
    # Load model
    # ========================================================

    model = PPO.load(
        "models/generalized_ppo_uav_tracking"
    )

    num_episodes = 100

    test_seed_start = 1000

    results = []

    print()
    print("=" * 76)
    print(
        "Generalized PPO - "
        "100 Randomized Episode Evaluation"
    )
    print("=" * 76)

    # ========================================================
    # Run 100 episodes
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
            model,
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
    # Aggregate arrays
    # ========================================================

    success_values = np.array([
        r["success"]
        for r in results
    ])

    rmse_values = np.array([
        r["rmse"]
        for r in results
    ])

    steady_rmse_values = np.array([
        r["steady_rmse"]
        for r in results
    ])

    final_errors = np.array([
        r["final_error"]
        for r in results
    ])

    final_speeds = np.array([
        r["final_relative_speed"]
        for r in results
    ])

    stable_ratios = np.array([
        r["stable_ratio"]
        for r in results
    ])

    smoothness_values = np.array([
        r["smoothness"]
        for r in results
    ])

    reward_values = np.array([
        r["reward"]
        for r in results
    ])

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 76)
    print(
        "GENERALIZED PPO RESULTS"
    )
    print("=" * 76)

    print(
        f"Number of episodes       : "
        f"{num_episodes}"
    )

    print(
        f"Success rate             : "
        f"{np.mean(success_values) * 100:.2f}%"
    )

    print(
        f"Total reward             : "
        f"{np.mean(reward_values):.3f} "
        f"± {np.std(reward_values):.3f}"
    )

    print(
        f"Full RMSE                : "
        f"{np.mean(rmse_values):.4f} "
        f"± {np.std(rmse_values):.4f} m"
    )

    print(
        f"Steady-state RMSE (>5s)  : "
        f"{np.mean(steady_rmse_values):.4f} "
        f"± {np.std(steady_rmse_values):.4f} m"
    )

    print(
        f"Final tracking error     : "
        f"{np.mean(final_errors):.4f} "
        f"± {np.std(final_errors):.4f} m"
    )

    print(
        f"Final relative speed     : "
        f"{np.mean(final_speeds):.4f} "
        f"± {np.std(final_speeds):.4f} m/s"
    )

    print(
        f"Stable tracking ratio    : "
        f"{np.mean(stable_ratios) * 100:.2f}% "
        f"± "
        f"{np.std(stable_ratios) * 100:.2f}%"
    )

    print(
        f"Action smoothness cost   : "
        f"{np.mean(smoothness_values):.6f} "
        f"± {np.std(smoothness_values):.6f}"
    )

    print("=" * 76)

    # ========================================================
    # Select representative episode
    #
    # Choose the episode whose RMSE is nearest
    # to the median RMSE.
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
    # Plot representative episode
    # ========================================================

    plot_representative_episode(
        representative_result
    )

    # ========================================================
    # Figure 6
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
        "Generalized PPO - RMSE Distribution"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_rmse_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ========================================================
    # Figure 7
    # Generalization scatter
    #
    # Initial distance vs RMSE
    # ========================================================

    initial_distances = np.array([
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
        "Generalized PPO - Initial Distance vs RMSE"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/"
        "generalized_ppo_generalization_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print("=" * 76)
    print(
        "Plots saved to: plots/"
    )
    print("=" * 76)

    plt.show()

    env.close()


if __name__ == "__main__":
    main()