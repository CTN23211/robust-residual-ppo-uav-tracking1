import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.uav_tracking_gym_env import UAVTrackingGymEnv


def main():

    # ==========================================================
    # 1. Create folders
    # ==========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "logs/ppo_generalized",
        exist_ok=True
    )

    # ==========================================================
    # 2. Create RANDOMIZED training environment
    # ==========================================================

    env = UAVTrackingGymEnv(
        randomize=True
    )

    # Monitor records episode statistics
    env = Monitor(
        env
    )

    # ==========================================================
    # 3. PPO model
    #
    # Keep hyperparameters consistent with
    # Smooth Vanilla PPO.
    #
    # The main experimental change here is:
    #
    # fixed environment
    #
    #       ↓
    #
    # randomized environment
    # ==========================================================

    model = PPO(

        policy="MlpPolicy",

        env=env,

        # ------------------------------------------------------
        # Learning rate
        # ------------------------------------------------------

        learning_rate=3e-4,

        # ------------------------------------------------------
        # PPO rollout length
        # ------------------------------------------------------

        n_steps=2048,

        # ------------------------------------------------------
        # Mini-batch size
        # ------------------------------------------------------

        batch_size=64,

        # ------------------------------------------------------
        # PPO optimization epochs
        # ------------------------------------------------------

        n_epochs=10,

        # ------------------------------------------------------
        # Discount factor
        # ------------------------------------------------------

        gamma=0.99,

        # ------------------------------------------------------
        # GAE
        # ------------------------------------------------------

        gae_lambda=0.95,

        # ------------------------------------------------------
        # PPO clipping
        # ------------------------------------------------------

        clip_range=0.2,

        # ------------------------------------------------------
        # Entropy coefficient
        #
        # Same as Smooth Vanilla PPO
        # ------------------------------------------------------

        ent_coef=0.001,

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        verbose=1,

        tensorboard_log=(
            "logs/ppo_generalized"
        ),

        # ------------------------------------------------------
        # Training seed
        # ------------------------------------------------------

        seed=42,

        # ------------------------------------------------------
        # Small MLP policy:
        # CPU is appropriate here
        # ------------------------------------------------------

        device="cpu"
    )

    # ==========================================================
    # 4. Training configuration
    # ==========================================================

    total_timesteps = 500_000

    print()
    print("=" * 72)

    print(
        "Generalized PPO Training"
    )

    print("=" * 72)

    print(
        "Environment randomization:"
    )

    print(
        "  UAV initial x : "
        "[-2.0, -0.5] m"
    )

    print(
        "  UAV initial y : "
        "[-1.0, 1.0] m"
    )

    print(
        "  UGV vx        : "
        "[0.20, 0.50] m/s"
    )

    print(
        "  UGV vy        : "
        "0.00 m/s"
    )

    print()

    print(
        "Observation dimensions : 10"
    )

    print(
        "Smoothness weight      : 0.05"
    )

    print(
        "Entropy coefficient    : 0.001"
    )

    print(
        f"Total timesteps        : "
        f"{total_timesteps}"
    )

    print("=" * 72)
    print()

    # ==========================================================
    # 5. Train
    # ==========================================================

    model.learn(

        total_timesteps=(
            total_timesteps
        ),

        progress_bar=True,

        tb_log_name=(
            "PPO_generalized"
        )
    )

    # ==========================================================
    # 6. Save
    # ==========================================================

    model_path = (
        "models/"
        "generalized_ppo_uav_tracking"
    )

    model.save(
        model_path
    )

    # ==========================================================
    # 7. Finish
    # ==========================================================

    print()
    print("=" * 72)

    print(
        "Generalized PPO training completed."
    )

    print(
        f"Model saved to:"
    )

    print(
        f"  {model_path}.zip"
    )

    print("=" * 72)
    print()

    env.close()


if __name__ == "__main__":
    main()