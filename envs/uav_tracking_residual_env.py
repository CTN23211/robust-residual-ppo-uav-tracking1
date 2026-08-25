import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.uav_tracking_env import UAVUGVSimulator


class UAVTrackingResidualEnv(gym.Env):
    """
    Residual RL environment for UAV tracking of a moving UGV.

    Control structure:

        v_cmd = v_base + delta_v_RL

    where:

        v_base
        =
        v_UGV
        + Kp * relative_position
        + Kd * relative_velocity

    PPO only learns a small residual correction:

        delta_v_RL = residual_scale * action

    Observation:
        [
            rel_x, rel_y,
            rel_vx, rel_vy,
            uav_vx, uav_vy,
            ugv_vx, ugv_vy,
            prev_action_x, prev_action_y
        ]

    Action:
        normalized residual action in [-1, 1]^2
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        randomize=True
    ):

        super().__init__()

        # ======================================================
        # 1. Environment mode
        # ======================================================

        self.randomize = randomize

        # ======================================================
        # 2. Simulation settings
        # ======================================================

        self.dt = 0.05

        self.max_episode_time = 20.0

        self.max_steps = int(
            self.max_episode_time
            / self.dt
        )

        self.step_count = 0

        # ======================================================
        # 3. Randomization ranges
        #
        # Keep EXACTLY the same as Generalized PPO.
        # ======================================================

        self.uav_x_range = (
            -2.0,
            -0.5
        )

        self.uav_y_range = (
            -1.0,
            1.0
        )

        self.ugv_vx_range = (
            0.20,
            0.50
        )

        # ======================================================
        # 4. Simulator
        # ======================================================

        self.simulator = UAVUGVSimulator(
            dt=self.dt,
            tau=0.30,
            max_uav_speed=1.0,
            max_uav_acceleration=0.8
        )

        # ======================================================
        # 5. Classical PD + feedforward controller
        # ======================================================

        self.kp = 0.45
        self.kd = 0.10

        # ======================================================
        # 6. Residual scale
        #
        # PPO action = [-1, 1]
        #
        # Physical correction:
        #
        # delta_v = action * 0.15 m/s
        # ======================================================

        self.residual_scale = 0.15

        # ======================================================
        # 7. Action space
        # ======================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ======================================================
        # 8. Observation space
        #
        # Keep 10 dimensions so Direct PPO and
        # Residual PPO have the same policy input dimension.
        # ======================================================

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32
        )

        self.prev_action = np.zeros(
            2,
            dtype=np.float32
        )

        # ======================================================
        # 9. Reward weights
        # ======================================================

        self.tracking_weight = 1.0

        self.relative_velocity_weight = 0.20

        # Penalize unnecessary residual correction
        self.residual_weight = 0.01

        # Penalize rapid residual changes
        self.smoothness_weight = 0.05

        self.stable_bonus = 1.0

        self.distance_threshold = 0.15

        self.relative_speed_threshold = 0.10

    # ==========================================================
    # Observation
    # ==========================================================

    def _get_obs(self):

        state = (
            self.simulator.get_state()
        )

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        obs = np.concatenate([
            rel_pos,
            rel_vel,
            state["uav_vel"],
            state["ugv_vel"],
            self.prev_action
        ])

        return obs.astype(
            np.float32
        )

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.step_count = 0

        self.prev_action = np.zeros(
            2,
            dtype=np.float32
        )

        # ======================================================
        # Randomized scenario
        # ======================================================

        if self.randomize:

            uav_x = self.np_random.uniform(
                self.uav_x_range[0],
                self.uav_x_range[1]
            )

            uav_y = self.np_random.uniform(
                self.uav_y_range[0],
                self.uav_y_range[1]
            )

            ugv_vx = self.np_random.uniform(
                self.ugv_vx_range[0],
                self.ugv_vx_range[1]
            )

            self.simulator.reset(

                uav_pos=np.array(
                    [
                        uav_x,
                        uav_y
                    ],
                    dtype=np.float32
                ),

                uav_vel=np.array(
                    [
                        0.0,
                        0.0
                    ],
                    dtype=np.float32
                ),

                ugv_pos=np.array(
                    [
                        0.0,
                        0.0
                    ],
                    dtype=np.float32
                ),

                ugv_vel=np.array(
                    [
                        ugv_vx,
                        0.0
                    ],
                    dtype=np.float32
                )
            )

        # ======================================================
        # Fixed scenario
        # ======================================================

        else:

            self.simulator.reset()

        state = (
            self.simulator.get_state()
        )

        obs = self._get_obs()

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        info = {

            "distance":
                float(
                    np.linalg.norm(
                        rel_pos
                    )
                ),

            "relative_speed":
                float(
                    np.linalg.norm(
                        rel_vel
                    )
                ),

            "initial_uav_x":
                float(
                    state["uav_pos"][0]
                ),

            "initial_uav_y":
                float(
                    state["uav_pos"][1]
                ),

            "ugv_vx":
                float(
                    state["ugv_vel"][0]
                )
        }

        return (
            obs,
            info
        )

    # ==========================================================
    # Step
    # ==========================================================

    def step(
        self,
        action
    ):

        self.step_count += 1

        # ======================================================
        # 1. Process RL residual action
        # ======================================================

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

        # ======================================================
        # 2. Current state BEFORE applying control
        # ======================================================

        state = (
            self.simulator.get_state()
        )

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        # ======================================================
        # 3. Classical PD + velocity feedforward
        #
        # v_base =
        # v_UGV + Kp*e + Kd*e_dot
        # ======================================================

        feedback_velocity = (
            self.kp * rel_pos
            +
            self.kd * rel_vel
        )

        base_command = (
            state["ugv_vel"]
            +
            feedback_velocity
        )

        # ======================================================
        # 4. RL residual correction
        # ======================================================

        residual_velocity = (
            action
            * self.residual_scale
        )

        # ======================================================
        # 5. Final control command
        #
        # THIS IS RESIDUAL RL
        # ======================================================

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

        # ======================================================
        # 6. Advance simulator
        # ======================================================

        new_state = self.simulator.step(
            velocity_command
        )

        # ======================================================
        # 7. New tracking state
        # ======================================================

        new_rel_pos = (
            new_state["ugv_pos"]
            - new_state["uav_pos"]
        )

        new_rel_vel = (
            new_state["ugv_vel"]
            - new_state["uav_vel"]
        )

        distance = float(
            np.linalg.norm(
                new_rel_pos
            )
        )

        relative_speed = float(
            np.linalg.norm(
                new_rel_vel
            )
        )

        # ======================================================
        # 8. Reward
        # ======================================================

        tracking_penalty = (
            -self.tracking_weight
            * distance
        )

        velocity_penalty = (
            -self.relative_velocity_weight
            * relative_speed
        )

        # PPO should only use residual when useful
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

        # ======================================================
        # 9. Stable tracking bonus
        # ======================================================

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

        # ======================================================
        # 10. Termination
        # ======================================================

        terminated = False

        if distance > 5.0:

            reward -= 20.0

            terminated = True

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # ======================================================
        # 11. Save action
        # ======================================================

        self.prev_action = (
            action.copy()
        )

        obs = self._get_obs()

        # ======================================================
        # 12. Info
        # ======================================================

        info = {

            "distance":
                distance,

            "relative_speed":
                relative_speed,

            "stable_tracking":
                stable_tracking,

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

            "smoothness_cost":
                smoothness_cost,

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
                )
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )