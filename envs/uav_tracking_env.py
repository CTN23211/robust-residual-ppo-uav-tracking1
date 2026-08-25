import numpy as np


class UAVUGVSimulator:
    """
    Simple 2D UAV-UGV simulator.

    UAV:
        - velocity-command control
        - first-order velocity response
        - velocity saturation
        - acceleration limit

    UGV:
        - constant-velocity motion

    The simulator itself does NOT perform randomization.
    Randomization is handled by the Gymnasium environment.
    """

    def __init__(
        self,
        dt=0.05,
        tau=0.30,
        max_uav_speed=1.0,
        max_uav_acceleration=0.8
    ):

        # ======================================================
        # Simulation parameters
        # ======================================================

        self.dt = dt

        self.tau = tau

        self.max_uav_speed = (
            max_uav_speed
        )

        self.max_uav_acceleration = (
            max_uav_acceleration
        )

        # Initialize states
        self.reset()

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(
        self,
        uav_pos=None,
        uav_vel=None,
        ugv_pos=None,
        ugv_vel=None
    ):
        """
        Reset simulator states.

        If no arguments are supplied, the original fixed
        tracking scenario is used:

            UAV position = (-1.5, 0.8)
            UAV velocity = (0.0, 0.0)

            UGV position = (0.0, 0.0)
            UGV velocity = (0.35, 0.0)
        """

        # ======================================================
        # UAV initial position
        # ======================================================

        if uav_pos is None:

            self.uav_pos = np.array(
                [-1.5, 0.8],
                dtype=np.float32
            )

        else:

            self.uav_pos = np.asarray(
                uav_pos,
                dtype=np.float32
            ).copy()

        # ======================================================
        # UAV initial velocity
        # ======================================================

        if uav_vel is None:

            self.uav_vel = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

        else:

            self.uav_vel = np.asarray(
                uav_vel,
                dtype=np.float32
            ).copy()

        # ======================================================
        # UGV initial position
        # ======================================================

        if ugv_pos is None:

            self.ugv_pos = np.array(
                [0.0, 0.0],
                dtype=np.float32
            )

        else:

            self.ugv_pos = np.asarray(
                ugv_pos,
                dtype=np.float32
            ).copy()

        # ======================================================
        # UGV initial velocity
        # ======================================================

        if ugv_vel is None:

            self.ugv_vel = np.array(
                [0.35, 0.0],
                dtype=np.float32
            )

        else:

            self.ugv_vel = np.asarray(
                ugv_vel,
                dtype=np.float32
            ).copy()

        return self.get_state()

    # ==========================================================
    # Get current state
    # ==========================================================

    def get_state(self):

        return {
            "uav_pos": self.uav_pos.copy(),
            "uav_vel": self.uav_vel.copy(),

            "ugv_pos": self.ugv_pos.copy(),
            "ugv_vel": self.ugv_vel.copy()
        }

    # ==========================================================
    # Simulation step
    # ==========================================================

    def step(
        self,
        uav_velocity_command
    ):

        # ======================================================
        # 1. Convert command
        # ======================================================

        uav_velocity_command = np.asarray(
            uav_velocity_command,
            dtype=np.float32
        )

        # ======================================================
        # 2. Velocity command saturation
        # ======================================================

        uav_velocity_command = np.clip(
            uav_velocity_command,
            -self.max_uav_speed,
            self.max_uav_speed
        )

        # ======================================================
        # 3. First-order UAV velocity response
        #
        # dv =
        # dt / tau * (v_cmd - v)
        # ======================================================

        velocity_change = (
            self.dt
            / self.tau
        ) * (
            uav_velocity_command
            - self.uav_vel
        )

        # ======================================================
        # 4. Acceleration limit
        #
        # Maximum velocity change per simulation step:
        #
        # Δv_max = a_max * dt
        # ======================================================

        max_velocity_change = (
            self.max_uav_acceleration
            * self.dt
        )

        velocity_change = np.clip(
            velocity_change,
            -max_velocity_change,
            max_velocity_change
        )

        # ======================================================
        # 5. Update UAV velocity
        # ======================================================

        self.uav_vel = (
            self.uav_vel
            + velocity_change
        )

        # Safety velocity saturation
        self.uav_vel = np.clip(
            self.uav_vel,
            -self.max_uav_speed,
            self.max_uav_speed
        )

        # ======================================================
        # 6. Update UAV position
        # ======================================================

        self.uav_pos = (
            self.uav_pos
            + self.uav_vel
            * self.dt
        )

        # ======================================================
        # 7. Update UGV position
        #
        # Current version:
        # constant UGV velocity
        # ======================================================

        self.ugv_pos = (
            self.ugv_pos
            + self.ugv_vel
            * self.dt
        )

        # ======================================================
        # 8. Return new state
        # ======================================================

        return self.get_state()