import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


def main():

    # ==========================================================
    # 1. Output folders
    # ==========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "logs/residual_ppo",
        exist_ok=True
    )

    # ==========================================================
    # 2. Randomized Residual RL environment
    # ==========================================================

    env = UAVTrackingResidualEnv(
        randomize=True
    )

    env = Monitor(
        env
    )

    # ==========================================================
    # 3. PPO
    #
    # Keep hyperparameters consistent with
    # Generalized Direct PPO.
    # ==========================================================

    model = PPO(

        policy="MlpPolicy",

        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.001,

        verbose=1,

        tensorboard_log=(
            "logs/residual_ppo"
        ),

        seed=42,

        device="cpu"
    )

    # ==========================================================
    # 4. Training
    # ==========================================================

    total_timesteps = 500_000

    print()
    print("=" * 72)
    print("Residual PPO Training")
    print("=" * 72)

    print(
        "Control architecture:"
    )

    print(
        "  command = PD+FF + PPO residual"
    )

    print()

    print(
        "PD controller:"
    )

    print(
        "  Kp = 0.45"
    )

    print(
        "  Kd = 0.10"
    )

    print()

    print(
        "Residual scale:"
    )

    print(
        "  ±0.15 m/s"
    )

    print()

    print(
        "Randomized environment:"
    )

    print(
        "  UAV x  : [-2.0, -0.5] m"
    )

    print(
        "  UAV y  : [-1.0, 1.0] m"
    )

    print(
        "  UGV vx : [0.20, 0.50] m/s"
    )

    print()

    print(
        f"Total timesteps: "
        f"{total_timesteps}"
    )

    print("=" * 72)
    print()

    model.learn(

        total_timesteps=(
            total_timesteps
        ),

        progress_bar=True,

        tb_log_name=(
            "Residual_PPO"
        )
    )

    # ==========================================================
    # 5. Save
    # ==========================================================

    model_path = (
        "models/"
        "residual_ppo_uav_tracking"
    )

    model.save(
        model_path
    )

    print()
    print("=" * 72)
    print(
        "Residual PPO training completed."
    )

    print(
        f"Saved to:"
    )

    print(
        f"  {model_path}.zip"
    )

    print("=" * 72)

    env.close()


if __name__ == "__main__":
    main()