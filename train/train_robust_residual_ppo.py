import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback
)

from envs.uav_tracking_robust_training_env import (
    UAVTrackingRobustTrainingEnv
)


# ============================================================
# Configuration
# ============================================================

TOTAL_TIMESTEPS = 500_000

SEED = 42

ALPHA_MIN = 0.0

ALPHA_MAX = 0.75

MODEL_DIR = "models"

LOG_DIR = "logs"

CHECKPOINT_DIR = (
    "models/robust_residual_checkpoints"
)

FINAL_MODEL_PATH = (
    "models/robust_residual_ppo_alpha075"
)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. Robust domain-randomized environment
    # ========================================================

    env = UAVTrackingRobustTrainingEnv(
        randomize=True,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX
    )

    env = Monitor(
        env
    )

    # ========================================================
    # 2. PPO
    #
    # Keep hyperparameters aligned with the previously
    # trained Residual PPO.
    #
    # IMPORTANT:
    # This model is trained FROM SCRATCH.
    # ========================================================

    model = PPO(

        policy="MlpPolicy",

        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.20,

        ent_coef=0.001,

        verbose=1,

        tensorboard_log=LOG_DIR,

        seed=SEED,

        device="cpu"
    )

    # ========================================================
    # 3. Checkpoints
    # ========================================================

    checkpoint_callback = (
        CheckpointCallback(

            save_freq=100_000,

            save_path=
                CHECKPOINT_DIR,

            name_prefix=
                "robust_residual_alpha075"
        )
    )

    # ========================================================
    # 4. Train
    # ========================================================

    print()
    print("=" * 80)

    print(
        "PHASE 5B — ROBUST RESIDUAL PPO TRAINING"
    )

    print("=" * 80)

    print(
        f"Training alpha range:"
        f" [{ALPHA_MIN}, {ALPHA_MAX}]"
    )

    print(
        "Held-out severe test:"
        " alpha = 1.0"
    )

    print(
        f"Total timesteps:"
        f" {TOTAL_TIMESTEPS}"
    )

    print(
        f"Seed:"
        f" {SEED}"
    )

    print("=" * 80)
    print()

    model.learn(

        total_timesteps=
            TOTAL_TIMESTEPS,

        callback=
            checkpoint_callback,

        tb_log_name=
            "RobustResidualPPO_alpha075",

        progress_bar=True
    )

    # ========================================================
    # 5. Save
    # ========================================================

    model.save(
        FINAL_MODEL_PATH
    )

    env.close()

    print()
    print("=" * 80)

    print(
        "Robust Residual PPO training COMPLETED."
    )

    print("=" * 80)

    print(
        f"Saved model:"
        f"\n  {FINAL_MODEL_PATH}.zip"
    )


if __name__ == "__main__":
    main()