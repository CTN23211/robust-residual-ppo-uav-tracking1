import numpy as np

from envs.uav_tracking_residual_env import (
    UAVTrackingResidualEnv
)


class UAVTrackingResidualRobustEnv(
    UAVTrackingResidualEnv
):
    """
    Robustness-test environment for Residual RL.

    This environment introduces an erroneous UGV velocity
    estimate into the PD+FF feedforward term.

    True UGV velocity:
        v_ugv

    Estimated UGV velocity:
        v_hat_ugv
        =
        v_ugv
        + alpha * (v_uav - v_ugv)

    alpha = 0:
        perfect velocity estimate

    alpha = 1:
        estimated UGV velocity becomes UAV velocity

    IMPORTANT:
        - UGV true dynamics are NOT changed.
        - PPO observations remain true in Phase 5.1.
        - Only the feedforward velocity estimate is corrupted.
    """

    def __init__(
        self,
        randomize=True,
        velocity_coupling_alpha=0.0
    ):

        super().__init__(
            randomize=randomize
        )

        self.velocity_coupling_alpha = float(
            velocity_coupling_alpha
        )

    def step(
        self,
        action
    ):

        self.step_count += 1

        # ======================================================
        # 1. Process residual PPO action
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
        # 2. Current TRUE state
        # ======================================================

        state = (
            self.simulator.get_state()
        )

        true_ugv_velocity = (
            state["ugv_vel"].copy()
        )

        uav_velocity = (
            state["uav_vel"].copy()
        )

        # ======================================================
        # 3. True relative states
        #
        # Phase 5.1 corrupts ONLY feedforward.
        # Observation is still perfect.
        # ======================================================

        rel_pos = (
            state["ugv_pos"]
            - state["uav_pos"]
        )

        rel_vel = (
            true_ugv_velocity
            - uav_velocity
        )

        # ======================================================
        # 4. WRONG velocity estimate
        #
        # v_hat =
        # v_true + alpha * (v_uav - v_true)
        #
        # alpha = 0:
        # v_hat = v_true
        #
        # alpha = 1:
        # v_hat = v_uav
        # ======================================================

        estimated_ugv_velocity = (
            true_ugv_velocity
            +
            self.velocity_coupling_alpha
            * (
                uav_velocity
                - true_ugv_velocity
            )
        )

        # ======================================================
        # 5. Classical PD feedback
        # ======================================================

        feedback_velocity = (
            self.kp * rel_pos
            +
            self.kd * rel_vel
        )

        # ======================================================
        # 6. Faulty PD + FF base command
        #
        # IMPORTANT:
        #
        # feedforward now uses ESTIMATED target velocity.
        # ======================================================

        base_command = (
            estimated_ugv_velocity
            +
            feedback_velocity
        )

        # ======================================================
        # 7. RL residual
        # ======================================================

        residual_velocity = (
            action
            * self.residual_scale
        )

        # ======================================================
        # 8. Final command
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
        # 9. Advance TRUE simulator
        # ======================================================

        new_state = (
            self.simulator.step(
                velocity_command
            )
        )

        # ======================================================
        # 10. New true relative state
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
        # 11. Reward
        #
        # SAME reward as nominal Residual PPO.
        # ======================================================

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

        # ======================================================
        # 12. Stable tracking
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
        # 13. Termination
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
        # 14. Store action
        # ======================================================

        self.prev_action = (
            action.copy()
        )

        # Observation remains TRUE observation.
        obs = self._get_obs()

        # ======================================================
        # 15. Diagnostic information
        # ======================================================

        info = {

            "distance":
                distance,

            "relative_speed":
                relative_speed,

            "stable_tracking":
                stable_tracking,

            # ----------------------------------------------
            # Fault variables
            # ----------------------------------------------

            "velocity_coupling_alpha":
                self.velocity_coupling_alpha,

            "true_ugv_vx":
                float(
                    true_ugv_velocity[0]
                ),

            "estimated_ugv_vx":
                float(
                    estimated_ugv_velocity[0]
                ),

            "velocity_estimation_error_x":
                float(
                    estimated_ugv_velocity[0]
                    - true_ugv_velocity[0]
                ),

            # ----------------------------------------------
            # Commands
            # ----------------------------------------------

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

            # ----------------------------------------------
            # True vehicle state
            # ----------------------------------------------

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

            "rel_x":
                float(
                    new_rel_pos[0]
                ),

            "rel_y":
                float(
                    new_rel_pos[1]
                ),

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