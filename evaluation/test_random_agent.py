from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    env = UAVTrackingGymEnv()

    obs, info = env.reset(
        seed=42
    )

    total_reward = 0.0

    print()
    print("Random Agent Test")
    print("=" * 50)

    for step in range(500):

        action = (
            env.action_space.sample()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += reward

        if step % 20 == 0:

            print(
                f"step={step:3d} | "
                f"distance="
                f"{info['distance']:.3f} m | "
                f"relative_speed="
                f"{info['relative_speed']:.3f} m/s | "
                f"reward="
                f"{reward:.3f}"
            )

        if terminated or truncated:

            print()
            print(
                f"Episode ended at step {step}"
            )

            break

    print()
    print(
        f"Total reward: {total_reward:.3f}"
    )


if __name__ == "__main__":
    main()