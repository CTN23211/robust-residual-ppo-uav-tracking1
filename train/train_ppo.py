import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    # ==========================================================
    # 1. Create output folders
    # ==========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "logs/ppo_smooth",
        exist_ok=True
    )

    # ==========================================================
    # 2. Create Gymnasium environment
    # ==========================================================

    env = UAVTrackingGymEnv()

    # Monitor records:
    #
    # episode reward
    # episode length
    #
    # for TensorBoard / training statistics.
    env = Monitor(env)

    # ==========================================================
    # 3. Create PPO model
    # ==========================================================

    model = PPO(

        policy="MlpPolicy",

        env=env,

        # ------------------------------------------------------
        # PPO hyperparameters
        # ------------------------------------------------------

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        # ------------------------------------------------------
        # Entropy coefficient
        #
        # v1 = 0.01
        # v2 = 0.001
        #
        # Lower exploration pressure because
        # the current task is already simple.
        # ------------------------------------------------------

        ent_coef=0.001,

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        verbose=1,

        tensorboard_log=(
            "logs/ppo_smooth"
        ),

        # ------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------

        seed=42,

        # Small MLP PPO is faster on CPU
        device="cpu"
    )

    # ==========================================================
    # 4. Training settings
    # ==========================================================

    total_timesteps = 300_000

    print()
    print("=" * 65)
    print("Starting Smooth Vanilla PPO Training")
    print("=" * 65)

    print(
        f"Total timesteps : "
        f"{total_timesteps}"
    )

    print(
        "Observation dim : 10"
    )

    print(
        "Smoothness weight: 0.05"
    )

    print(
        "Entropy coef    : 0.001"
    )

    print("=" * 65)
    print()

    # ==========================================================
    # 5. Train
    # ==========================================================

    model.learn(

        total_timesteps=total_timesteps,

        progress_bar=True,

        tb_log_name="PPO_smooth"
    )

    # ==========================================================
    # 6. Save NEW model
    #
    # Do NOT overwrite PPO-v1.
    # ==========================================================

    model_path = (
        "models/"
        "vanilla_ppo_smooth_uav_tracking"
    )

    model.save(
        model_path
    )

    # ==========================================================
    # 7. Finish
    # ==========================================================

    print()
    print("=" * 65)
    print("Training completed.")
    print("=" * 65)

    print(
        f"Model saved to:\n"
        f"{model_path}.zip"
    )

    print("=" * 65)
    print()

    env.close()


if __name__ == "__main__":
    main()