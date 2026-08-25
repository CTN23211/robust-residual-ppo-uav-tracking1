import numpy as np

from envs.uav_tracking_gym_env import (
    UAVTrackingGymEnv
)

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


# ============================================================
# Trajectory configuration
# ============================================================

VALID_SCENARIOS = [
    "nominal",
    "speed_step",
    "lateral_sine",
    "constant_turn",
]


# ============================================================
# Target trajectory mixin
# ============================================================

class UnseenTargetTrajectoryMixin:

    def _init_trajectory(
        self,
        scenario,
        maneuver_start_time,
        speed_step_delta,
        sine_amplitude,
        sine_period,
        turn_rate,
    ):

        if scenario not in VALID_SCENARIOS:

            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Valid scenarios: {VALID_SCENARIOS}"
            )

        self.scenario = scenario

        self.maneuver_start_time = float(
            maneuver_start_time
        )

        self.speed_step_delta = float(
            speed_step_delta
        )

        self.sine_amplitude = float(
            sine_amplitude
        )

        self.sine_period = float(
            sine_period
        )

        self.turn_rate = float(
            turn_rate
        )

        self.base_ugv_speed = None

    # ========================================================
    # Set target velocity for the NEXT control interval
    # ========================================================

    def _update_target_velocity(
        self,
        time_seconds
    ):

        if self.base_ugv_speed is None:
            return

        base_speed = (
            self.base_ugv_speed
        )

        # ----------------------------------------------------
        # Before maneuver:
        # preserve training-like straight motion
        # ----------------------------------------------------

        if (
            time_seconds
            < self.maneuver_start_time
        ):

            new_velocity = np.array(
                [
                    base_speed,
                    0.0
                ],
                dtype=np.float32
            )

        # ----------------------------------------------------
        # NOMINAL
        # ----------------------------------------------------

        elif self.scenario == "nominal":

            new_velocity = np.array(
                [
                    base_speed,
                    0.0
                ],
                dtype=np.float32
            )

        # ----------------------------------------------------
        # SPEED STEP
        #
        # v_x jumps by +0.20 m/s
        #
        # Example:
        #
        # 0.30 -> 0.50 m/s
        #
        # Maximum limited to 0.70 m/s.
        # ----------------------------------------------------

        elif self.scenario == "speed_step":

            new_speed = min(
                base_speed
                + self.speed_step_delta,
                0.70
            )

            new_velocity = np.array(
                [
                    new_speed,
                    0.0
                ],
                dtype=np.float32
            )

        # ----------------------------------------------------
        # LATERAL SINE
        #
        # y(t) approximately follows:
        #
        # A sin(omega * tau)
        #
        # Therefore:
        #
        # vy = A * omega * cos(omega * tau)
        #
        # x velocity remains unchanged.
        # ----------------------------------------------------

        elif self.scenario == "lateral_sine":

            tau = (
                time_seconds
                - self.maneuver_start_time
            )

            omega = (
                2.0
                * np.pi
                / self.sine_period
            )

            vy = (
                self.sine_amplitude
                * omega
                * np.cos(
                    omega * tau
                )
            )

            new_velocity = np.array(
                [
                    base_speed,
                    vy
                ],
                dtype=np.float32
            )

        # ----------------------------------------------------
        # CONSTANT TURN
        #
        # Target speed remains constant,
        # heading rotates at constant angular rate.
        # ----------------------------------------------------

        elif self.scenario == "constant_turn":

            tau = (
                time_seconds
                - self.maneuver_start_time
            )

            heading = (
                self.turn_rate
                * tau
            )

            new_velocity = np.array(
                [
                    base_speed
                    * np.cos(
                        heading
                    ),

                    base_speed
                    * np.sin(
                        heading
                    )
                ],
                dtype=np.float32
            )

        else:

            raise RuntimeError(
                "Unhandled trajectory scenario."
            )

        # ----------------------------------------------------
        # Directly update TRUE target velocity.
        #
        # Simulator will integrate position normally
        # during the next dt.
        # ----------------------------------------------------

        self.simulator.ugv_vel = (
            new_velocity.copy()
        )

    # ========================================================
    # Add trajectory information
    # ========================================================

    def _add_trajectory_info(
        self,
        info
    ):

        state = (
            self.simulator.get_state()
        )

        info[
            "scenario"
        ] = self.scenario

        info[
            "maneuver_start_time"
        ] = self.maneuver_start_time

        info[
            "base_ugv_speed"
        ] = float(
            self.base_ugv_speed
        )

        info[
            "ugv_vx"
        ] = float(
            state["ugv_vel"][0]
        )

        info[
            "ugv_vy"
        ] = float(
            state["ugv_vel"][1]
        )

        info[
            "ugv_speed"
        ] = float(
            np.linalg.norm(
                state["ugv_vel"]
            )
        )

        return info


# ============================================================
# Direct PPO environment
# ============================================================

class UAVTrackingDirectUnseenTrajectoryEnv(
    UnseenTargetTrajectoryMixin,
    UAVTrackingGymEnv
):

    def __init__(
        self,
        randomize=True,
        scenario="nominal",
        maneuver_start_time=6.0,
        speed_step_delta=0.20,
        sine_amplitude=0.40,
        sine_period=8.0,
        turn_rate=0.25,
    ):

        super().__init__(
            randomize=randomize
        )

        self._init_trajectory(
            scenario=scenario,
            maneuver_start_time=
                maneuver_start_time,
            speed_step_delta=
                speed_step_delta,
            sine_amplitude=
                sine_amplitude,
            sine_period=
                sine_period,
            turn_rate=
                turn_rate,
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        state = (
            self.simulator.get_state()
        )

        self.base_ugv_speed = float(
            np.linalg.norm(
                state["ugv_vel"]
            )
        )

        # Start in normal straight-line motion.
        self._update_target_velocity(
            0.0
        )

        # Rebuild observation in case
        # velocity was normalized.
        obs = self._get_obs()

        info = (
            self._add_trajectory_info(
                info
            )
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

        # ----------------------------------------------------
        # Parent executes current control interval.
        # ----------------------------------------------------

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = super().step(
            action
        )

        # ----------------------------------------------------
        # Current simulation time after parent step.
        # ----------------------------------------------------

        current_time = (
            self.step_count
            * self.dt
        )

        # ----------------------------------------------------
        # Update target velocity for NEXT interval.
        # ----------------------------------------------------

        self._update_target_velocity(
            current_time
        )

        # ----------------------------------------------------
        # PPO must observe the updated target velocity.
        # ----------------------------------------------------

        obs = self._get_obs()

        info = (
            self._add_trajectory_info(
                info
            )
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )


# ============================================================
# Residual PPO / PD+FF environment
# ============================================================

class UAVTrackingResidualUnseenTrajectoryEnv(
    UnseenTargetTrajectoryMixin,
    UAVTrackingResidualEnv
):

    def __init__(
        self,
        randomize=True,
        scenario="nominal",
        maneuver_start_time=6.0,
        speed_step_delta=0.20,
        sine_amplitude=0.40,
        sine_period=8.0,
        turn_rate=0.25,
    ):

        super().__init__(
            randomize=randomize
        )

        self._init_trajectory(
            scenario=scenario,
            maneuver_start_time=
                maneuver_start_time,
            speed_step_delta=
                speed_step_delta,
            sine_amplitude=
                sine_amplitude,
            sine_period=
                sine_period,
            turn_rate=
                turn_rate,
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        state = (
            self.simulator.get_state()
        )

        self.base_ugv_speed = float(
            np.linalg.norm(
                state["ugv_vel"]
            )
        )

        self._update_target_velocity(
            0.0
        )

        obs = self._get_obs()

        info = (
            self._add_trajectory_info(
                info
            )
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

        (
            obs,
            reward,
            terminated,
            truncated,
            info
        ) = super().step(
            action
        )

        current_time = (
            self.step_count
            * self.dt
        )

        self._update_target_velocity(
            current_time
        )

        obs = self._get_obs()

        info = (
            self._add_trajectory_info(
                info
            )
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )