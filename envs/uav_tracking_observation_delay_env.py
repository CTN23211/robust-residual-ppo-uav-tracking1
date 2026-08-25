from collections import deque

import numpy as np

from envs.uav_tracking_gym_env import (
    UAVTrackingGymEnv
)

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Helper functions
# ============================================================

def get_true_measurement(env):
    """
    Get a clean measurement snapshot from the TRUE simulator state.

    This snapshot contains exactly the variables used by
    the controller / PPO observation.
    """

    state = env.simulator.get_state()

    rel_pos = (
        state["ugv_pos"]
        - state["uav_pos"]
    )

    rel_vel = (
        state["ugv_vel"]
        - state["uav_vel"]
    )

    return {
        "rel_pos": np.asarray(
            rel_pos,
            dtype=np.float32
        ).copy(),

        "rel_vel": np.asarray(
            rel_vel,
            dtype=np.float32
        ).copy(),

        "uav_vel": np.asarray(
            state["uav_vel"],
            dtype=np.float32
        ).copy(),

        "ugv_vel": np.asarray(
            state["ugv_vel"],
            dtype=np.float32
        ).copy()
    }


def copy_measurement(
    measurement
):
    """
    Deep-copy one measurement snapshot.
    """

    return {
        key: value.copy()
        for key, value
        in measurement.items()
    }


def measurement_to_obs(
    measurement,
    prev_action
):
    """
    Convert delayed measurement into the same
    10-dimensional observation used during training.
    """

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


def initialize_delay_buffer(
    initial_measurement,
    delay_steps
):
    """
    Initialize history by repeating the initial measurement.

    Example:

        delay_steps = 4

    buffer initially contains:

        s0, s0, s0, s0, s0

    This corresponds to zero-order hold before sufficient
    history becomes available.
    """

    buffer = deque(
        maxlen=delay_steps + 1
    )

    for _ in range(
        delay_steps + 1
    ):

        buffer.append(
            copy_measurement(
                initial_measurement
            )
        )

    return buffer


# ============================================================
# Direct PPO observation-delay environment
# ============================================================

class UAVTrackingDirectDelayEnv(
    UAVTrackingGymEnv
):
    """
    Observation-delay environment for Generalized Direct PPO.

    PPO receives:

        state(t - delay)

    while the simulator evolves using the TRUE current state.

    delay_steps = 0:
        nominal Generalized PPO environment

    delay_steps = 2:
        100 ms delay for dt = 0.05 s
    """

    def __init__(
        self,
        randomize=True,
        delay_steps=0
    ):

        super().__init__(
            randomize=randomize
        )

        self.delay_steps = int(
            delay_steps
        )

        if self.delay_steps < 0:

            raise ValueError(
                "delay_steps must be >= 0"
            )

        self.delay_seconds = (
            self.delay_steps
            * self.dt
        )

        self.measurement_buffer = None

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        # Parent handles random initial conditions,
        # prev_action reset and Gymnasium RNG.
        _, info = super().reset(
            seed=seed,
            options=options
        )

        initial_measurement = (
            get_true_measurement(
                self
            )
        )

        self.measurement_buffer = (
            initialize_delay_buffer(
                initial_measurement,
                self.delay_steps
            )
        )

        delayed_measurement = (
            self.measurement_buffer[0]
        )

        obs = measurement_to_obs(
            delayed_measurement,
            self.prev_action
        )

        info["delay_steps"] = (
            self.delay_steps
        )

        info["delay_seconds"] = (
            self.delay_seconds
        )

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
        # 1. PPO action
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

        # Direct PPO outputs the complete velocity command.
        velocity_command = (
            action
            * self.simulator.max_uav_speed
        )

        # ====================================================
        # 2. TRUE dynamics
        # ====================================================

        state = self.simulator.step(
            velocity_command
        )

        # ====================================================
        # 3. TRUE state for reward and evaluation
        # ====================================================

        true_rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        true_rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
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
        # 4. SAME reward as Generalized PPO
        # ====================================================

        tracking_penalty = (
            -self.tracking_weight
            * distance
        )

        velocity_penalty = (
            -self.relative_velocity_weight
            * relative_speed
        )

        action_penalty = (
            -self.action_weight
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
            + action_penalty
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
        # 5. Termination
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
        # 6. Save current action
        # ====================================================

        self.prev_action = (
            action.copy()
        )

        # ====================================================
        # 7. Push NEW true measurement into history
        # ====================================================

        new_measurement = (
            get_true_measurement(
                self
            )
        )

        self.measurement_buffer.append(
            new_measurement
        )

        # Oldest entry is exactly the delayed observation.
        delayed_measurement = (
            self.measurement_buffer[0]
        )

        obs = measurement_to_obs(
            delayed_measurement,
            self.prev_action
        )

        # ====================================================
        # 8. Info
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
                    state["uav_pos"][0]
                ),

            "uav_y":
                float(
                    state["uav_pos"][1]
                ),

            "uav_vx":
                float(
                    state["uav_vel"][0]
                ),

            "uav_vy":
                float(
                    state["uav_vel"][1]
                ),

            "ugv_x":
                float(
                    state["ugv_pos"][0]
                ),

            "ugv_y":
                float(
                    state["ugv_pos"][1]
                ),

            "ugv_vx":
                float(
                    state["ugv_vel"][0]
                ),

            "ugv_vy":
                float(
                    state["ugv_vel"][1]
                ),

            "delay_steps":
                self.delay_steps,

            "delay_seconds":
                self.delay_seconds,

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


# ============================================================
# Residual PPO / PD+FF observation-delay environment
# ============================================================

class UAVTrackingResidualDelayEnv(
    UAVTrackingResidualEnv
):
    """
    Observation-delay environment for:

        1. PD + FF
        2. Residual PPO

    Both components receive the SAME delayed measurement.

    At control time t:

        PD+FF uses measurement(t-delay)

        PPO also receives measurement(t-delay)

    TRUE state is used only for:

        dynamics
        reward
        evaluation metrics
    """

    def __init__(
        self,
        randomize=True,
        delay_steps=0
    ):

        super().__init__(
            randomize=randomize
        )

        self.delay_steps = int(
            delay_steps
        )

        if self.delay_steps < 0:

            raise ValueError(
                "delay_steps must be >= 0"
            )

        self.delay_seconds = (
            self.delay_steps
            * self.dt
        )

        self.measurement_buffer = None

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        _, info = super().reset(
            seed=seed,
            options=options
        )

        initial_measurement = (
            get_true_measurement(
                self
            )
        )

        self.measurement_buffer = (
            initialize_delay_buffer(
                initial_measurement,
                self.delay_steps
            )
        )

        delayed_measurement = (
            self.measurement_buffer[0]
        )

        obs = measurement_to_obs(
            delayed_measurement,
            self.prev_action
        )

        info["delay_steps"] = (
            self.delay_steps
        )

        info["delay_seconds"] = (
            self.delay_seconds
        )

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
        # 1. Residual PPO action
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
        # 2. Controller sees DELAYED measurement
        # ====================================================

        delayed_measurement = (
            self.measurement_buffer[0]
        )

        delayed_rel_pos = (
            delayed_measurement[
                "rel_pos"
            ]
        )

        delayed_rel_vel = (
            delayed_measurement[
                "rel_vel"
            ]
        )

        delayed_ugv_vel = (
            delayed_measurement[
                "ugv_vel"
            ]
        )

        # ====================================================
        # 3. DELAYED PD + velocity feedforward
        # ====================================================

        feedback_velocity = (
            self.kp
            * delayed_rel_pos

            +

            self.kd
            * delayed_rel_vel
        )

        base_command = (
            delayed_ugv_vel
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

        state = self.simulator.step(
            velocity_command
        )

        # ====================================================
        # 6. TRUE state for reward and evaluation
        # ====================================================

        true_rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        true_rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
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
        # 7. SAME Residual PPO reward
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
        # 9. Store current residual action
        # ====================================================

        self.prev_action = (
            action.copy()
        )

        # ====================================================
        # 10. Append current TRUE observation
        # ====================================================

        new_measurement = (
            get_true_measurement(
                self
            )
        )

        self.measurement_buffer.append(
            new_measurement
        )

        # Observation for next decision:
        # delayed state + current previous action
        next_delayed_measurement = (
            self.measurement_buffer[0]
        )

        obs = measurement_to_obs(
            next_delayed_measurement,
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

            "raw_command_x":
                float(
                    raw_velocity_command[0]
                ),

            "raw_command_y":
                float(
                    raw_velocity_command[1]
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
                    state["uav_pos"][0]
                ),

            "uav_y":
                float(
                    state["uav_pos"][1]
                ),

            "uav_vx":
                float(
                    state["uav_vel"][0]
                ),

            "uav_vy":
                float(
                    state["uav_vel"][1]
                ),

            "ugv_x":
                float(
                    state["ugv_pos"][0]
                ),

            "ugv_y":
                float(
                    state["ugv_pos"][1]
                ),

            "ugv_vx":
                float(
                    state["ugv_vel"][0]
                ),

            "ugv_vy":
                float(
                    state["ugv_vel"][1]
                ),

            "delay_steps":
                self.delay_steps,

            "delay_seconds":
                self.delay_seconds,

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