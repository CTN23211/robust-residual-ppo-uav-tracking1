import numpy as np

from envs.uav_tracking_gym_env import (
    UAVTrackingGymEnv
)

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Target measurement helpers
# ============================================================

def get_true_target_measurement(env):
    """
    Target-related information assumed to come from
    the perception / vision system.

    During vision loss these quantities are held at
    their last valid values.

    UAV self velocity is NOT stored here because it
    remains available from onboard state estimation.
    """

    state = env.simulator.get_state()

    rel_pos = (
        state["ugv_pos"]
        - state["uav_pos"]
    )

    return {
        "rel_pos": np.asarray(
            rel_pos,
            dtype=np.float32
        ).copy(),

        "ugv_vel": np.asarray(
            state["ugv_vel"],
            dtype=np.float32
        ).copy()
    }


def copy_target_measurement(
    measurement
):

    return {
        key: value.copy()
        for key, value
        in measurement.items()
    }


def build_observation(
    env,
    target_measurement,
    prev_action
):
    """
    Build the same 10-D observation used during training.

    During target loss:

        rel_pos = last valid visual relative position
        ugv_vel = last valid target velocity estimate

    while:

        uav_vel = CURRENT onboard UAV velocity

    Relative velocity is recomputed consistently:

        rel_vel = held_ugv_vel - current_uav_vel
    """

    state = env.simulator.get_state()

    current_uav_vel = (
        state["uav_vel"]
    )

    measured_ugv_vel = (
        target_measurement[
            "ugv_vel"
        ]
    )

    measured_rel_vel = (
        measured_ugv_vel
        - current_uav_vel
    )

    obs = np.concatenate([
        target_measurement[
            "rel_pos"
        ],
        measured_rel_vel,
        current_uav_vel,
        measured_ugv_vel,
        prev_action
    ])

    return obs.astype(
        np.float32
    )


# ============================================================
# Loss timing helper
# ============================================================

def vision_loss_active(
    time_seconds,
    loss_start_time,
    loss_duration
):

    if loss_duration <= 0.0:
        return False

    loss_end_time = (
        loss_start_time
        + loss_duration
    )

    return (
        time_seconds
        >= loss_start_time
        and
        time_seconds
        < loss_end_time
    )


# ============================================================
# Direct PPO Vision-Loss Environment
# ============================================================

class UAVTrackingDirectVisionLossEnv(
    UAVTrackingGymEnv
):

    def __init__(
        self,
        randomize=True,
        loss_start_time=3.0,
        loss_duration=0.0
    ):

        super().__init__(
            randomize=randomize
        )

        self.loss_start_time = float(
            loss_start_time
        )

        self.loss_duration = float(
            loss_duration
        )

        self.loss_end_time = (
            self.loss_start_time
            + self.loss_duration
        )

        self.last_valid_target = None

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

        self.last_valid_target = (
            get_true_target_measurement(
                self
            )
        )

        obs = build_observation(
            self,
            self.last_valid_target,
            self.prev_action
        )

        info["loss_start_time"] = (
            self.loss_start_time
        )

        info["loss_duration"] = (
            self.loss_duration
        )

        info["loss_end_time"] = (
            self.loss_end_time
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
        # 3. TRUE state for reward
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
        # 4. Same reward as Generalized PPO
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

        self.prev_action = (
            action.copy()
        )

        # ====================================================
        # 6. Vision system update
        # ====================================================

        next_time = (
            self.step_count
            * self.dt
        )

        loss_active = vision_loss_active(
            next_time,
            self.loss_start_time,
            self.loss_duration
        )

        # Vision available:
        # update target observation.
        #
        # Vision lost:
        # keep last valid target measurement.

        if not loss_active:

            self.last_valid_target = (
                get_true_target_measurement(
                    self
                )
            )

        obs = build_observation(
            self,
            self.last_valid_target,
            self.prev_action
        )

        # ====================================================
        # 7. Info
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

            "vision_loss_active":
                bool(
                    loss_active
                ),

            "loss_start_time":
                self.loss_start_time,

            "loss_end_time":
                self.loss_end_time
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# Residual PPO / PD+FF Vision-Loss Environment
# ============================================================

class UAVTrackingResidualVisionLossEnv(
    UAVTrackingResidualEnv
):

    def __init__(
        self,
        randomize=True,
        loss_start_time=3.0,
        loss_duration=0.0
    ):

        super().__init__(
            randomize=randomize
        )

        self.loss_start_time = float(
            loss_start_time
        )

        self.loss_duration = float(
            loss_duration
        )

        self.loss_end_time = (
            self.loss_start_time
            + self.loss_duration
        )

        self.last_valid_target = None

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

        self.last_valid_target = (
            get_true_target_measurement(
                self
            )
        )

        obs = build_observation(
            self,
            self.last_valid_target,
            self.prev_action
        )

        info["loss_start_time"] = (
            self.loss_start_time
        )

        info["loss_duration"] = (
            self.loss_duration
        )

        info["loss_end_time"] = (
            self.loss_end_time
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
        # 1. Residual action
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
        # 2. Controller uses last AVAILABLE
        #    target measurement
        # ====================================================

        state_before = (
            self.simulator.get_state()
        )

        current_uav_vel = (
            state_before[
                "uav_vel"
            ]
        )

        measured_rel_pos = (
            self.last_valid_target[
                "rel_pos"
            ]
        )

        measured_ugv_vel = (
            self.last_valid_target[
                "ugv_vel"
            ]
        )

        measured_rel_vel = (
            measured_ugv_vel
            - current_uav_vel
        )

        # ====================================================
        # 3. PD + FF
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
        # 4. Residual
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
        # 5. TRUE dynamics
        # ====================================================

        state = self.simulator.step(
            velocity_command
        )

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
        # 6. Same Residual PPO reward
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
        # 7. Termination
        # ====================================================

        terminated = False

        if distance > 5.0:

            reward -= 20.0

            terminated = True

        truncated = (
            self.step_count
            >= self.max_steps
        )

        self.prev_action = (
            action.copy()
        )

        # ====================================================
        # 8. Update vision target measurement
        # ====================================================

        next_time = (
            self.step_count
            * self.dt
        )

        loss_active = vision_loss_active(
            next_time,
            self.loss_start_time,
            self.loss_duration
        )

        if not loss_active:

            self.last_valid_target = (
                get_true_target_measurement(
                    self
                )
            )

        obs = build_observation(
            self,
            self.last_valid_target,
            self.prev_action
        )

        # ====================================================
        # 9. Info
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

            "vision_loss_active":
                bool(
                    loss_active
                ),

            "loss_start_time":
                self.loss_start_time,

            "loss_end_time":
                self.loss_end_time
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )