import os

import numpy as np
import matplotlib.pyplot as plt

from envs.uav_tracking_env import UAVUGVSimulator
from controllers.pid import PIDController


def main():

    # ==========================================================
    # 1. Simulation settings
    # ==========================================================

    dt = 0.05
    simulation_time = 20.0

    total_steps = int(
        simulation_time / dt
    )

    # ==========================================================
    # 2. Simulator
    # ==========================================================

    simulator = UAVUGVSimulator(
        dt=dt,
        tau=0.30,
        max_uav_speed=1.0,
        max_uav_acceleration=0.8
    )

    # ==========================================================
    # 3. PD position-feedback controller
    #
    # ki = 0 -> PD controller
    # ==========================================================

    controller = PIDController(
        kp=0.45,
        ki=0.0,
        kd=0.10,
        dt=dt,
        integral_limit=1.0
    )

    # ==========================================================
    # 4. Reset
    # ==========================================================

    simulator.reset()

    initial_state = (
        simulator.get_state()
    )

    initial_error = (
        initial_state["ugv_pos"]
        - initial_state["uav_pos"]
    )

    # Prevent derivative kick
    controller.reset(
        initial_error=initial_error
    )

    # ==========================================================
    # 5. Data storage
    # ==========================================================

    times = []

    uav_positions = []
    ugv_positions = []

    uav_velocities = []
    ugv_velocities = []

    tracking_errors = []

    velocity_commands = []
    raw_velocity_commands = []

    feedback_velocities = []

    # ==========================================================
    # 6. Simulation loop
    # ==========================================================

    for step in range(total_steps):

        current_time = (
            step * dt
        )

        # ------------------------------------------------------
        # Current state
        # ------------------------------------------------------

        state = (
            simulator.get_state()
        )

        uav_pos = state["uav_pos"]
        ugv_pos = state["ugv_pos"]

        ugv_vel = state["ugv_vel"]

        # ------------------------------------------------------
        # Position error
        #
        # e = p_UGV - p_UAV
        # ------------------------------------------------------

        position_error = (
            ugv_pos
            - uav_pos
        )

        # ------------------------------------------------------
        # PD feedback correction
        # ------------------------------------------------------

        feedback_velocity = (
            controller.compute(
                position_error
            )
        )

        # ------------------------------------------------------
        # Velocity feedforward + feedback
        #
        # v_raw =
        #     v_UGV
        #     + v_PD
        # ------------------------------------------------------

        raw_velocity_command = (
            ugv_vel
            + feedback_velocity
        )

        # ------------------------------------------------------
        # Apply command saturation BEFORE simulator
        #
        # This guarantees that the plotted command
        # is the command actually sent to dynamics.
        # ------------------------------------------------------

        velocity_command = np.clip(
            raw_velocity_command,
            -simulator.max_uav_speed,
            simulator.max_uav_speed
        )

        # ------------------------------------------------------
        # Advance simulator
        # ------------------------------------------------------

        new_state = (
            simulator.step(
                velocity_command
            )
        )

        # ------------------------------------------------------
        # Tracking error
        # ------------------------------------------------------

        relative_position = (
            new_state["ugv_pos"]
            - new_state["uav_pos"]
        )

        tracking_error = (
            np.linalg.norm(
                relative_position
            )
        )

        # ------------------------------------------------------
        # Save data
        # ------------------------------------------------------

        times.append(
            current_time
        )

        uav_positions.append(
            new_state["uav_pos"].copy()
        )

        ugv_positions.append(
            new_state["ugv_pos"].copy()
        )

        uav_velocities.append(
            new_state["uav_vel"].copy()
        )

        ugv_velocities.append(
            new_state["ugv_vel"].copy()
        )

        tracking_errors.append(
            tracking_error
        )

        velocity_commands.append(
            velocity_command.copy()
        )

        raw_velocity_commands.append(
            raw_velocity_command.copy()
        )

        feedback_velocities.append(
            feedback_velocity.copy()
        )

    # ==========================================================
    # 7. Convert to NumPy
    # ==========================================================

    times = np.asarray(
        times
    )

    uav_positions = np.asarray(
        uav_positions
    )

    ugv_positions = np.asarray(
        ugv_positions
    )

    uav_velocities = np.asarray(
        uav_velocities
    )

    ugv_velocities = np.asarray(
        ugv_velocities
    )

    tracking_errors = np.asarray(
        tracking_errors
    )

    velocity_commands = np.asarray(
        velocity_commands
    )

    raw_velocity_commands = np.asarray(
        raw_velocity_commands
    )

    feedback_velocities = np.asarray(
        feedback_velocities
    )

    # ==========================================================
    # 8. Evaluation metrics
    # ==========================================================

    rmse = np.sqrt(
        np.mean(
            tracking_errors ** 2
        )
    )

    max_error = (
        np.max(
            tracking_errors
        )
    )

    min_error = (
        np.min(
            tracking_errors
        )
    )

    final_error = (
        tracking_errors[-1]
    )

    peak_uav_vx = (
        np.max(
            uav_velocities[:, 0]
        )
    )

    peak_command_vx = (
        np.max(
            velocity_commands[:, 0]
        )
    )

    final_uav_vx = (
        uav_velocities[-1, 0]
    )

    final_ugv_vx = (
        ugv_velocities[-1, 0]
    )

    # ==========================================================
    # 9. Print results
    # ==========================================================

    print()
    print("=" * 55)
    print("PD + Velocity Feedforward Tracking Evaluation")
    print("=" * 55)

    print(
        f"RMSE             : "
        f"{rmse:.4f} m"
    )

    print(
        f"Max error        : "
        f"{max_error:.4f} m"
    )

    print(
        f"Min error        : "
        f"{min_error:.4f} m"
    )

    print(
        f"Final error      : "
        f"{final_error:.4f} m"
    )

    print(
        f"Peak UAV vx      : "
        f"{peak_uav_vx:.4f} m/s"
    )

    print(
        f"Peak command vx  : "
        f"{peak_command_vx:.4f} m/s"
    )

    print(
        f"Final UAV vx     : "
        f"{final_uav_vx:.4f} m/s"
    )

    print(
        f"Final UGV vx     : "
        f"{final_ugv_vx:.4f} m/s"
    )

    print("=" * 55)
    print()

    # ==========================================================
    # 10. Create plots directory
    # ==========================================================

    os.makedirs(
        "plots",
        exist_ok=True
    )

    # ==========================================================
    # Figure 1:
    # 2D trajectory
    # ==========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ugv_positions[:, 0],
        ugv_positions[:, 1],
        label="UGV"
    )

    plt.plot(
        uav_positions[:, 0],
        uav_positions[:, 1],
        label="UAV"
    )

    plt.scatter(
        ugv_positions[0, 0],
        ugv_positions[0, 1],
        marker="o",
        label="UGV start"
    )

    plt.scatter(
        uav_positions[0, 0],
        uav_positions[0, 1],
        marker="x",
        s=80,
        label="UAV start"
    )

    plt.xlabel(
        "X position (m)"
    )

    plt.ylabel(
        "Y position (m)"
    )

    plt.title(
        "UAV Tracking UGV - PD + Velocity Feedforward"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/pd_ff_trajectory.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 2:
    # Tracking error
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        tracking_errors
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Tracking error (m)"
    )

    plt.title(
        "PD + Velocity Feedforward Tracking Error"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "plots/pd_ff_tracking_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 3:
    # Velocity response
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        uav_velocities[:, 0],
        label="UAV vx"
    )

    plt.plot(
        times,
        ugv_velocities[:, 0],
        label="UGV vx"
    )

    plt.plot(
        times,
        velocity_commands[:, 0],
        label="Applied command vx"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity (m/s)"
    )

    plt.title(
        "Velocity Response"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/pd_ff_velocity_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 4:
    # PD feedback correction
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        feedback_velocities[:, 0],
        label="Feedback vx"
    )

    plt.plot(
        times,
        feedback_velocities[:, 1],
        label="Feedback vy"
    )

    plt.axhline(
        y=0.0,
        linewidth=1
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "PD feedback velocity (m/s)"
    )

    plt.title(
        "PD Feedback Correction"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/pd_ff_feedback.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Figure 5:
    # Raw vs saturated velocity command
    #
    # Useful for checking whether saturation occurs.
    # ==========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        raw_velocity_commands[:, 0],
        label="Raw command vx"
    )

    plt.plot(
        times,
        velocity_commands[:, 0],
        label="Applied command vx"
    )

    plt.axhline(
        y=simulator.max_uav_speed,
        linestyle="--",
        label="Velocity limit"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Velocity command (m/s)"
    )

    plt.title(
        "Raw and Saturated Velocity Command"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/pd_ff_command_saturation.png",
        dpi=300,
        bbox_inches="tight"
    )

    # ==========================================================
    # Show figures
    # ==========================================================

    plt.show()


if __name__ == "__main__":
    main()