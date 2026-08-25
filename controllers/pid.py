import numpy as np


class PIDController:
    """
    2D PID/PD controller.

    Input:
        Position error:
        e = p_target - p_uav

    Output:
        Velocity correction:
        [vx_correction, vy_correction]

    Note:
        When ki = 0, this controller becomes a PD controller.
    """

    def __init__(
        self,
        kp=0.45,
        ki=0.0,
        kd=0.10,
        dt=0.05,
        integral_limit=1.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral_limit = integral_limit

        self.integral = np.zeros(
            2,
            dtype=np.float32
        )

        self.prev_error = np.zeros(
            2,
            dtype=np.float32
        )

        self.initialized = False

    def reset(self, initial_error=None):
        """
        Reset controller internal states.

        initial_error is used to avoid derivative kick
        during the first control step.
        """

        self.integral[:] = 0.0

        if initial_error is None:
            self.prev_error[:] = 0.0
            self.initialized = False

        else:
            self.prev_error = np.asarray(
                initial_error,
                dtype=np.float32
            ).copy()

            self.initialized = True

    def compute(self, error):
        """
        Calculate feedback velocity correction.

        Parameters
        ----------
        error : array-like, shape (2,)
            [target_x - uav_x,
             target_y - uav_y]

        Returns
        -------
        output : np.ndarray, shape (2,)
            Velocity feedback correction.
        """

        error = np.asarray(
            error,
            dtype=np.float32
        )

        # -----------------------------------------
        # Integral term
        # -----------------------------------------

        self.integral += (
            error * self.dt
        )

        # Anti-windup
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )

        # -----------------------------------------
        # Derivative term
        # -----------------------------------------

        if not self.initialized:

            derivative = np.zeros(
                2,
                dtype=np.float32
            )

            self.initialized = True

        else:

            derivative = (
                error - self.prev_error
            ) / self.dt

        # -----------------------------------------
        # PID / PD
        # -----------------------------------------

        proportional_term = (
            self.kp * error
        )

        integral_term = (
            self.ki * self.integral
        )

        derivative_term = (
            self.kd * derivative
        )

        output = (
            proportional_term
            + integral_term
            + derivative_term
        )

        self.prev_error = error.copy()

        return output