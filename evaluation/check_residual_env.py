from stable_baselines3.common.env_checker import check_env

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


def main():

    print()
    print("=" * 60)
    print("Checking Residual RL Gymnasium Environment")
    print("=" * 60)

    # ==========================================================
    # 1. Check fixed residual environment
    # ==========================================================

    print()
    print("[1/2] Checking FIXED residual environment...")

    fixed_env = UAVTrackingResidualEnv(
        randomize=False
    )

    check_env(
        fixed_env,
        warn=True
    )

    print(
        "Fixed residual environment PASSED."
    )

    fixed_env.close()

    # ==========================================================
    # 2. Check randomized residual environment
    # ==========================================================

    print()
    print("[2/2] Checking RANDOMIZED residual environment...")

    randomized_env = UAVTrackingResidualEnv(
        randomize=True
    )

    check_env(
        randomized_env,
        warn=True
    )

    print(
        "Randomized residual environment PASSED."
    )

    randomized_env.close()

    # ==========================================================
    # Finish
    # ==========================================================

    print()
    print("=" * 60)
    print(
        "Residual Gymnasium environment PASSED."
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()