from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    # ==========================================================
    # Create randomized environment
    # ==========================================================

    env = UAVTrackingGymEnv(
        randomize=True
    )

    print()
    print("=" * 72)
    print("UAV Tracking Environment Randomization Test")
    print("=" * 72)

    # ==========================================================
    # Generate 10 random episodes
    # ==========================================================

    for episode in range(10):

        obs, info = env.reset()

        uav_x = (
            info["initial_uav_x"]
        )

        uav_y = (
            info["initial_uav_y"]
        )

        ugv_vx = (
            info["ugv_vx"]
        )

        print(
            f"Episode {episode + 1:2d} | "
            f"UAV = "
            f"({uav_x:+.3f}, "
            f"{uav_y:+.3f}) | "
            f"UGV vx = "
            f"{ugv_vx:.3f} m/s"
        )

        # ======================================================
        # Verify ranges
        # ======================================================

        assert (
            -2.0
            <= uav_x
            <= -0.5
        ), (
            f"UAV x out of range: "
            f"{uav_x}"
        )

        assert (
            -1.0
            <= uav_y
            <= 1.0
        ), (
            f"UAV y out of range: "
            f"{uav_y}"
        )

        assert (
            0.20
            <= ugv_vx
            <= 0.50
        ), (
            f"UGV vx out of range: "
            f"{ugv_vx}"
        )

    # ==========================================================
    # Seed reproducibility test
    # ==========================================================

    print()
    print("-" * 72)
    print("Seed reproducibility test")
    print("-" * 72)

    obs_1, info_1 = env.reset(
        seed=42
    )

    obs_2, info_2 = env.reset(
        seed=42
    )

    print(
        "Seed 42, first reset : "
        f"UAV="
        f"("
        f"{info_1['initial_uav_x']:.6f}, "
        f"{info_1['initial_uav_y']:.6f}"
        f"), "
        f"UGV vx="
        f"{info_1['ugv_vx']:.6f}"
    )

    print(
        "Seed 42, second reset: "
        f"UAV="
        f"("
        f"{info_2['initial_uav_x']:.6f}, "
        f"{info_2['initial_uav_y']:.6f}"
        f"), "
        f"UGV vx="
        f"{info_2['ugv_vx']:.6f}"
    )

    # The two seeded resets should be identical
    assert (
        info_1["initial_uav_x"]
        == info_2["initial_uav_x"]
    )

    assert (
        info_1["initial_uav_y"]
        == info_2["initial_uav_y"]
    )

    assert (
        info_1["ugv_vx"]
        == info_2["ugv_vx"]
    )

    print()
    print(
        "Seed reproducibility PASSED."
    )

    print()
    print("=" * 72)
    print(
        "Randomization test PASSED."
    )
    print("=" * 72)
    print()

    env.close()


if __name__ == "__main__":
    main()