from stable_baselines3.common.env_checker import check_env

from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    print()
    print("=" * 60)
    print("Checking UAV Tracking Gymnasium Environments")
    print("=" * 60)

    # ==========================================================
    # 1. Check fixed environment
    # ==========================================================

    print()
    print("[1/2] Checking FIXED environment...")

    fixed_env = UAVTrackingGymEnv(
        randomize=False
    )

    check_env(
        fixed_env,
        warn=True
    )

    print(
        "Fixed environment PASSED."
    )

    fixed_env.close()

    # ==========================================================
    # 2. Check randomized environment
    # ==========================================================

    print()
    print("[2/2] Checking RANDOMIZED environment...")

    randomized_env = UAVTrackingGymEnv(
        randomize=True
    )

    check_env(
        randomized_env,
        warn=True
    )

    print(
        "Randomized environment PASSED."
    )

    randomized_env.close()

    # ==========================================================
    # Finish
    # ==========================================================

    print()
    print("=" * 60)
    print(
        "All Gymnasium environment checks PASSED."
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()