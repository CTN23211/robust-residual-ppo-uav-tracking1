import numpy as np

from envs.uav_tracking_gym_env import (
    UAVTrackingGymEnv
)

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Common noise helper
# ============================================================

def sample_noisy_measurement(
    env,
    position_noise_std,
    velocity_noise_std
):
    """
    Generate one noisy sensor observation from the TRUE state.

    position_noise_std:
        Standard deviation of relative-position measurement noise [m]

    velocity_noise_std:
        Standard deviation applied independently to
        UAV and UGV velocity measurements [m/s]

    Relative velocity is then derived consistently:

        rel_vel_obs
        =
        ugv_vel_obs - uav_vel_obs
    """

    state = env.simulator.get_state()

    true_rel_pos = (
        state["ugv_pos"]
        - state["uav_pos"]
    )

    # --------------------------------------------------------
    # Relative-position measurement noise
    # --------------------------------------------------------

    rel_pos_obs = (
        true_rel_pos
        +
        env.np_random.normal(
            loc=0.0,
            scale=position_noise_std,
            size=2
        )
    )

    # --------------------------------------------------------
    # Velocity measurement noise
    # --------------------------------------------------------

    uav_vel_obs = (
        state["uav_vel"]
        +
        env.np_random.normal(
            loc=0.0,
            scale=velocity_noise_std,
            size=2
        )
    )

    ugv_vel_obs = (
        state["ugv_vel"]
        +
        env.np_random.normal(
            loc=0.0,
            scale=velocity_noise_std,
            size=2
        )
    )

    # --------------------------------------------------------
    # Derive relative velocity consistently
    # --------------------------------------------------------

    rel_vel_obs = (
        ugv_vel_obs
        - uav_vel_obs
    )

    return {
        "rel_pos": np.asarray(
            rel_pos_obs,
            dtype=np.float32
        ),

        "rel_vel": np.asarray(
            rel_vel_obs,
            dtype=np.float32
        ),

        "uav_vel": np.asarray(
            uav_vel_obs,
            dtype=np.float32
        ),

        "ugv_vel": np.asarray(
            ugv_vel_obs,
            dtype=np.float32
        )
    }


def measurement_to_obs(
    measurement,
    prev_action
):

    obs = np.concatenate([
        measurement["rel_pos"],
        measurement["rel_vel"],
        measurement["uav_vel"],
        measurement["ugv_vel"],
        prev_action
    ])

    return obs.astype(
        np.float32
    )


# ============================================================
# Direct PPO observation-noise environment
# ============================================================

class UAVTrackingDirectNoiseEnv(
    UAVTrackingGymEnv
):
    """
    Observation-noise environment for Generalized Direct PPO.

    The simulator evolves using the TRUE state.

    The PPO receives only noisy observations.
    """

    def __init__(
        self,
        randomize=True,
        position_noise_std=0.0,
        velocity_noise_std=0.0
    ):

        super().__init__(
            randomize=randomize
        )

        self.position_noise_std = float(
            position_noise_std
        )

        self.velocity_noise_std = float(
            velocity_noise_std
        )

        self.last_measurement = None

    # ========================================================
    # Override observation generation
    # ========================================================

    def _get_obs(self):

        measurement = (
            sample_noisy_measurement(
                env=self,
                position_noise_std=
                    self.position_noise_std,
                velocity_noise_std=
                    self.velocity_noise_std
            )
        )

        self.last_measurement = (
            measurement
        )

        return measurement_to_obs(
            measurement,
            self.prev_action
        )

    def reset(
        self,
        seed=None,
        options=None
    ):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        info[
            "position_noise_std"
        ] = self.position_noise_std

        info[
            "velocity_noise_std"
        ] = self.velocity_noise_std

        return (
            obs,
            info
        )


# ============================================================
# Residual PPO / PD+FF observation-noise environment
# ============================================================

class UAVTrackingResidualNoiseEnv(
    UAVTrackingResidualEnv
):
    """
    Observation-noise environment for:

        1. PD + velocity feedforward
        2. Residual PPO

    IMPORTANT:

    PD+FF and PPO use the SAME noisy sensor snapshot.

    True simulator state is used only for:

        - dynamics
        - reward calculation
        - evaluation metrics
    """

    def __init__(
        self,
        randomize=True,
        position_noise_std=0.0,
        velocity_noise_std=0.0
    ):

        super().__init__(
            randomize=randomize
        )

        self.position_noise_std = float(
            position_noise_std
        )

        self.velocity_noise_std = float(
            velocity_noise_std
        )

        self.current_measurement = None

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        # Parent handles:
        #
        # - Gymnasium RNG
        # - randomized initial position
        # - randomized UGV velocity
        # - prev_action reset

        _, info = super().reset(
            seed=seed,
            options=options
        )

        self.current_measurement = (
            sample_noisy_measurement(
                env=self,
                position_noise_std=
                    self.position_noise_std,
                velocity_noise_std=
                    self.velocity_noise_std
            )
        )

        obs = measurement_to_obs(
            self.current_measurement,
            self.prev_action
        )

        info[
            "position_noise_std"
        ] = self.position_noise_std

        info[
            "velocity_noise_std"
        ] = self.velocity_noise_std

        return (
            obs,
            info
        )

    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action
    ):

        self.step_count += 1

        # ====================================================
        # 1. PPO residual action
        # ====================================================

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        action_change = (
            action
            - self.prev_action
        )

        # ====================================================
        # 2. Controller sees NOISY measurement
        # ====================================================

        measured_rel_pos = (
            self.current_measurement[
                "rel_pos"
            ]
        )

        measured_rel_vel = (
            self.current_measurement[
                "rel_vel"
            ]
        )

        measured_ugv_vel = (
            self.current_measurement[
                "ugv_vel"
            ]
        )

        # ====================================================
        # 3. PD + feedforward from noisy observation
        # ====================================================

        feedback_velocity = (
            self.kp
            * measured_rel_pos

            +

            self.kd
            * measured_rel_vel
        )

        base_command = (
            measured_ugv_vel
            +
            feedback_velocity
        )

        # ====================================================
        # 4. PPO residual
        # ====================================================

        residual_velocity = (
            action
            * self.residual_scale
        )

        raw_velocity_command = (
            base_command
            +
            residual_velocity
        )

        velocity_command = np.clip(
            raw_velocity_command,
            -self.simulator.max_uav_speed,
            self.simulator.max_uav_speed
        )

        # ====================================================
        # 5. Advance TRUE dynamics
        # ====================================================

        new_state = self.simulator.step(
            velocity_command
        )

        # ====================================================
        # 6. TRUE state for metrics/reward
        # ====================================================

        true_rel_pos = (
            new_state["ugv_pos"]
            - new_state["uav_pos"]
        )

        true_rel_vel = (
            new_state["ugv_vel"]
            - new_state["uav_vel"]
        )

        distance = float(
            np.linalg.norm(
                true_rel_pos
            )
        )

        relative_speed = float(
            np.linalg.norm(
                true_rel_vel
            )
        )

        # ====================================================
        # 7. Reward
        # ====================================================

        tracking_penalty = (
            -self.tracking_weight
            * distance
        )

        velocity_penalty = (
            -self.relative_velocity_weight
            * relative_speed
        )

        residual_penalty = (
            -self.residual_weight
            * float(
                np.linalg.norm(
                    action
                )
            )
        )

        smoothness_cost = float(
            np.sum(
                action_change ** 2
            )
        )

        smoothness_penalty = (
            -self.smoothness_weight
            * smoothness_cost
        )

        reward = (
            tracking_penalty
            + velocity_penalty
            + residual_penalty
            + smoothness_penalty
        )

        stable_tracking = (
            distance
            < self.distance_threshold
            and
            relative_speed
            < self.relative_speed_threshold
        )

        if stable_tracking:

            reward += (
                self.stable_bonus
            )

        # ====================================================
        # 8. Termination
        # ====================================================

        terminated = False

        if distance > 5.0:

            reward -= 20.0

            terminated = True

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # ====================================================
        # 9. Update previous action
        # ====================================================

        self.prev_action = (
            action.copy()
        )

        # ====================================================
        # 10. Generate NEXT noisy sensor measurement
        # ====================================================

        self.current_measurement = (
            sample_noisy_measurement(
                env=self,
                position_noise_std=
                    self.position_noise_std,
                velocity_noise_std=
                    self.velocity_noise_std
            )
        )

        obs = measurement_to_obs(
            self.current_measurement,
            self.prev_action
        )

        # ====================================================
        # 11. Info
        # ====================================================

        info = {

            "distance":
                distance,

            "relative_speed":
                relative_speed,

            "stable_tracking":
                stable_tracking,

            "rel_x":
                float(
                    true_rel_pos[0]
                ),

            "rel_y":
                float(
                    true_rel_pos[1]
                ),

            "base_command_x":
                float(
                    base_command[0]
                ),

            "base_command_y":
                float(
                    base_command[1]
                ),

            "residual_x":
                float(
                    residual_velocity[0]
                ),

            "residual_y":
                float(
                    residual_velocity[1]
                ),

            "command_x":
                float(
                    velocity_command[0]
                ),

            "command_y":
                float(
                    velocity_command[1]
                ),

            "uav_x":
                float(
                    new_state["uav_pos"][0]
                ),

            "uav_y":
                float(
                    new_state["uav_pos"][1]
                ),

            "uav_vx":
                float(
                    new_state["uav_vel"][0]
                ),

            "uav_vy":
                float(
                    new_state["uav_vel"][1]
                ),

            "ugv_x":
                float(
                    new_state["ugv_pos"][0]
                ),

            "ugv_y":
                float(
                    new_state["ugv_pos"][1]
                ),

            "ugv_vx":
                float(
                    new_state["ugv_vel"][0]
                ),

            "ugv_vy":
                float(
                    new_state["ugv_vel"][1]
                ),

            "position_noise_std":
                self.position_noise_std,

            "velocity_noise_std":
                self.velocity_noise_std,

            "smoothness_cost":
                smoothness_cost
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )