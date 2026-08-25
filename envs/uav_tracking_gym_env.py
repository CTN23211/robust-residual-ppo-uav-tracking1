import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.uav_tracking_env import UAVUGVSimulator


class UAVTrackingGymEnv(gym.Env):
    """
    Gymnasium environment for UAV tracking of a moving UGV.

    ------------------------------------------------------------
    Observation (10 dimensions)
    ------------------------------------------------------------

    [
        rel_x,
        rel_y,

        rel_vx,
        rel_vy,

        uav_vx,
        uav_vy,

        ugv_vx,
        ugv_vy,

        prev_action_x,
        prev_action_y
    ]

    ------------------------------------------------------------
    Action
    ------------------------------------------------------------

    PPO outputs:

        [action_x, action_y]

    each in:

        [-1, 1]

    Since maximum UAV speed is 1.0 m/s:

        velocity_command
        =
        action * max_uav_speed

    ------------------------------------------------------------
    Modes
    ------------------------------------------------------------

    randomize=False:

        Fixed original scenario.

    randomize=True:

        UAV initial position:

            x ~ U(-2.0, -0.5)
            y ~ U(-1.0,  1.0)

        UGV velocity:

            vx ~ U(0.20, 0.50) m/s
            vy = 0
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        randomize=False
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
        # 5. Action space
        # ======================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ======================================================
        # 6. Observation space
        #
        # Total = 10 dimensions
        # ======================================================

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32
        )

        # ======================================================
        # 7. Previous action
        # ======================================================

        self.prev_action = np.zeros(
            2,
            dtype=np.float32
        )

        # ======================================================
        # 8. Reward parameters
        # ======================================================

        self.tracking_weight = 1.0

        self.relative_velocity_weight = 0.20

        self.action_weight = 0.01

        self.smoothness_weight = 0.05

        self.stable_bonus = 1.0

        # ======================================================
        # 9. Stable tracking thresholds
        # ======================================================

        self.distance_threshold = 0.15

        self.relative_speed_threshold = 0.10

    # ==========================================================
    # Observation
    # ==========================================================

    def _get_obs(self):

        state = (
            self.simulator.get_state()
        )

        # Relative position:
        #
        # target - UAV

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        # Relative velocity:
        #
        # target velocity - UAV velocity

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

        # Gymnasium seed handling
        super().reset(
            seed=seed
        )

        self.step_count = 0

        # Every episode begins with
        # zero previous control action
        self.prev_action = np.zeros(
            2,
            dtype=np.float32
        )

        # ======================================================
        # RANDOMIZED scenario
        # ======================================================

        if self.randomize:

            # --------------------------------------------------
            # Random UAV position
            # --------------------------------------------------

            uav_x = self.np_random.uniform(
                self.uav_x_range[0],
                self.uav_x_range[1]
            )

            uav_y = self.np_random.uniform(
                self.uav_y_range[0],
                self.uav_y_range[1]
            )

            # --------------------------------------------------
            # Random UGV x velocity
            # --------------------------------------------------

            ugv_vx = self.np_random.uniform(
                self.ugv_vx_range[0],
                self.ugv_vx_range[1]
            )

            # --------------------------------------------------
            # Reset simulator
            # --------------------------------------------------

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
        # FIXED scenario
        # ======================================================

        else:

            self.simulator.reset()

        # ======================================================
        # Initial state
        # ======================================================

        state = (
            self.simulator.get_state()
        )

        obs = self._get_obs()

        # ======================================================
        # Initial diagnostic values
        # ======================================================

        initial_distance = float(
            np.linalg.norm(
                state["ugv_pos"]
                - state["uav_pos"]
            )
        )

        initial_relative_speed = float(
            np.linalg.norm(
                state["ugv_vel"]
                - state["uav_vel"]
            )
        )

        # ======================================================
        # Info
        # ======================================================

        info = {

            "distance":
                initial_distance,

            "relative_speed":
                initial_relative_speed,

            "initial_uav_x":
                float(
                    state["uav_pos"][0]
                ),

            "initial_uav_y":
                float(
                    state["uav_pos"][1]
                ),

            "initial_ugv_x":
                float(
                    state["ugv_pos"][0]
                ),

            "initial_ugv_y":
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

            "action_change":
                0.0,

            "smoothness_cost":
                0.0
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
        # 1. Process action
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

        # ======================================================
        # 2. Action change
        #
        # Δa(t) = a(t) - a(t-1)
        # ======================================================

        action_change = (
            action
            - self.prev_action
        )

        # ======================================================
        # 3. Convert normalized action into
        #    UAV velocity command
        # ======================================================

        velocity_command = (
            action
            * self.simulator.max_uav_speed
        )

        # ======================================================
        # 4. Advance simulator
        # ======================================================

        state = self.simulator.step(
            velocity_command
        )

        # ======================================================
        # 5. Relative state
        # ======================================================

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            state["ugv_vel"]
            - state["uav_vel"]
        )

        distance = float(
            np.linalg.norm(
                rel_pos
            )
        )

        relative_speed = float(
            np.linalg.norm(
                rel_vel
            )
        )

        # ======================================================
        # 6. Tracking penalty
        # ======================================================

        tracking_penalty = (
            -self.tracking_weight
            * distance
        )

        # ======================================================
        # 7. Relative velocity penalty
        # ======================================================

        velocity_penalty = (
            -self.relative_velocity_weight
            * relative_speed
        )

        # ======================================================
        # 8. Control magnitude penalty
        # ======================================================

        action_penalty = (
            -self.action_weight
            * float(
                np.linalg.norm(
                    action
                )
            )
        )

        # ======================================================
        # 9. Smoothness penalty
        #
        # ||a(t) - a(t-1)||²
        # ======================================================

        smoothness_cost = float(
            np.sum(
                action_change ** 2
            )
        )

        smoothness_penalty = (
            -self.smoothness_weight
            * smoothness_cost
        )

        # ======================================================
        # 10. Base reward
        # ======================================================

        reward = (
            tracking_penalty
            + velocity_penalty
            + action_penalty
            + smoothness_penalty
        )

        # ======================================================
        # 11. Stable tracking bonus
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
        # 12. Failure termination
        # ======================================================

        terminated = False

        if distance > 5.0:

            reward -= 20.0

            terminated = True

        # ======================================================
        # 13. Time limit
        # ======================================================

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # ======================================================
        # 14. Save current action
        #
        # IMPORTANT:
        # calculate smoothness BEFORE this update.
        # ======================================================

        self.prev_action = (
            action.copy()
        )

        # ======================================================
        # 15. Next observation
        # ======================================================

        obs = self._get_obs()

        # ======================================================
        # 16. Diagnostic information
        # ======================================================

        info = {

            "distance":
                distance,

            "relative_speed":
                relative_speed,

            "stable_tracking":
                stable_tracking,

            "action_change":
                float(
                    np.linalg.norm(
                        action_change
                    )
                ),

            "smoothness_cost":
                smoothness_cost,

            "tracking_penalty":
                float(
                    tracking_penalty
                ),

            "velocity_penalty":
                float(
                    velocity_penalty
                ),

            "action_penalty":
                float(
                    action_penalty
                ),

            "smoothness_penalty":
                float(
                    smoothness_penalty
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
                )
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )