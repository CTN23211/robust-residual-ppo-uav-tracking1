import numpy as np

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


def main():

    env = UAVTrackingResidualEnv(
        randomize=False
    )

    obs, info = env.reset(
        seed=42
    )

    distances = []

    print()
    print("=" * 70)
    print("Residual Environment - ZERO Residual Test")
    print("=" * 70)

    for step in range(
        env.max_steps
    ):

        # ------------------------------------------------------
        # ZERO RL correction
        #
        # This means:
        #
        # final command = PD + feedforward only
        # ------------------------------------------------------

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

        distances.append(
            info["distance"]
        )

        if (
            terminated
            or truncated
        ):
            break

    distances = np.asarray(
        distances
    )

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    final_error = (
        distances[-1]
    )

    print(
        f"Episode length      : "
        f"{len(distances)}"
    )

    print(
        f"RMSE                : "
        f"{rmse:.4f} m"
    )

    print(
        f"Final tracking error: "
        f"{final_error:.4f} m"
    )

    print()

    print(
        "Final controller components:"
    )

    print(
        f"Base command x      : "
        f"{info['base_command_x']:.4f} m/s"
    )

    print(
        f"Residual x          : "
        f"{info['residual_x']:.4f} m/s"
    )

    print(
        f"Final command x     : "
        f"{info['command_x']:.4f} m/s"
    )

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()