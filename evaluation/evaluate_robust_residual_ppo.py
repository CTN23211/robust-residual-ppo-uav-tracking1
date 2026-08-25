import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from evaluation.evaluate_velocity_compensation_fault import (
    evaluate_episode
)


# ============================================================
# Configuration
# ============================================================

ALPHA_VALUES = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]

NUM_EPISODES = 100

TEST_SEED_START = 1000

RESULTS_DIR = "results"

PLOTS_DIR = "plots"

NOMINAL_MODEL_PATH = (
    "models/residual_ppo_uav_tracking"
)

ROBUST_MODEL_PATH = (
    "models/robust_residual_ppo_alpha075"
)


# ============================================================
# Safe statistics
# ============================================================

def safe_mean(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def safe_std(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.std(
            values
        )
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    path,
    rows
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Plot
# ============================================================

def plot_metric(
    summary_results,
    metric_name,
    ylabel,
    filename,
    percentage=False
):

    methods = [
        "Nominal Residual PPO",
        "Robust Residual PPO"
    ]

    plt.figure(
        figsize=(9, 5)
    )

    for method in methods:

        rows = [
            row
            for row
            in summary_results
            if row["method"] == method
        ]

        x = [
            row["alpha"]
            for row in rows
        ]

        y = [
            row[metric_name]
            for row in rows
        ]

        if percentage:

            y = [
                value * 100.0
                for value in y
            ]

        plt.plot(
            x,
            y,
            marker="o",
            label=method
        )

    # Training limit
    plt.axvline(
        0.75,
        linestyle="--",
        label="Training alpha limit"
    )

    plt.xlabel(
        "Velocity-coupling fault severity alpha"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        "Nominal vs Robust Residual PPO"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    # ========================================================
    # Load frozen models
    # ========================================================

    nominal_model = PPO.load(
        NOMINAL_MODEL_PATH
    )

    robust_model = PPO.load(
        ROBUST_MODEL_PATH
    )

    models = {

        "Nominal Residual PPO":
            nominal_model,

        "Robust Residual PPO":
            robust_model
    }

    raw_results = []

    summary_results = []

    print()
    print("=" * 100)

    print(
        "PHASE 5B — NOMINAL vs ROBUST RESIDUAL PPO"
    )

    print("=" * 100)

    # ========================================================
    # Alpha sweep
    # ========================================================

    for alpha in ALPHA_VALUES:

        print()

        if alpha > 0.75:

            region = (
                "HELD-OUT"
            )

        else:

            region = (
                "TRAINING RANGE"
            )

        print(
            f"alpha = {alpha:.2f} "
            f"[{region}]"
        )

        print("-" * 100)

        for method_name, model in (
            models.items()
        ):

            episode_results = []

            for episode in range(
                NUM_EPISODES
            ):

                seed = (
                    TEST_SEED_START
                    + episode
                )

                result = evaluate_episode(

                    method=
                        "Residual PPO",

                    alpha=
                        alpha,

                    seed=
                        seed,

                    residual_model=
                        model
                )

                # Rename controller
                result[
                    "method"
                ] = method_name

                result[
                    "held_out"
                ] = bool(
                    alpha > 0.75
                )

                raw_results.append(
                    result
                )

                episode_results.append(
                    result
                )

            # =================================================
            # Helper
            # =================================================

            def values(
                name
            ):

                return np.asarray([
                    row[name]
                    for row
                    in episode_results
                ])

            # =================================================
            # Summary
            # =================================================

            summary = {

                "method":
                    method_name,

                "alpha":
                    alpha,

                "held_out":
                    bool(
                        alpha > 0.75
                    ),

                "success_rate":
                    float(
                        np.mean(
                            values(
                                "success"
                            )
                        )
                    ),

                "rmse_mean":
                    safe_mean(
                        values(
                            "rmse"
                        )
                    ),

                "rmse_std":
                    safe_std(
                        values(
                            "rmse"
                        )
                    ),

                "steady_rmse_mean":
                    safe_mean(
                        values(
                            "steady_rmse"
                        )
                    ),

                "capture_time_mean":
                    safe_mean(
                        values(
                            "capture_time"
                        )
                    ),

                "stable_ratio_mean":
                    safe_mean(
                        values(
                            "stable_ratio"
                        )
                    ),

                "forward_overshoot_mean":
                    safe_mean(
                        values(
                            "forward_overshoot"
                        )
                    ),

                "command_smoothness_mean":
                    safe_mean(
                        values(
                            "command_smoothness"
                        )
                    )
            }

            summary_results.append(
                summary
            )

            print(
                f"{method_name:22s} | "
                f"Success="
                f"{summary['success_rate'] * 100:6.2f}% | "
                f"RMSE="
                f"{summary['rmse_mean']:.4f} | "
                f"Steady="
                f"{summary['steady_rmse_mean']:.4f} | "
                f"Capture="
                f"{summary['capture_time_mean']:.3f}s | "
                f"Overshoot="
                f"{summary['forward_overshoot_mean']:.4f}"
            )

    # ========================================================
    # Save
    # ========================================================

    raw_path = os.path.join(
        RESULTS_DIR,
        "robust_residual_alpha_sweep_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "robust_residual_alpha_sweep_summary.csv"
    )

    save_csv(
        raw_path,
        raw_results
    )

    save_csv(
        summary_path,
        summary_results
    )

    # ========================================================
    # Plots
    # ========================================================

    plot_metric(
        summary_results,
        "success_rate",
        "Success rate (%)",
        "robust_residual_success_rate.png",
        percentage=True
    )

    plot_metric(
        summary_results,
        "forward_overshoot_mean",
        "Forward overshoot (m)",
        "robust_residual_forward_overshoot.png"
    )

    plot_metric(
        summary_results,
        "steady_rmse_mean",
        "Steady-state RMSE (m)",
        "robust_residual_steady_rmse.png"
    )

    plot_metric(
        summary_results,
        "capture_time_mean",
        "Capture time (s)",
        "robust_residual_capture_time.png"
    )

    plot_metric(
        summary_results,
        "command_smoothness_mean",
        "Command smoothness cost",
        "robust_residual_command_smoothness.png"
    )

    print()
    print("=" * 100)

    print(
        "Phase 5B evaluation COMPLETED."
    )

    print("=" * 100)

    print(
        f"Raw:"
        f"\n  {raw_path}"
    )

    print(
        f"Summary:"
        f"\n  {summary_path}"
    )


if __name__ == "__main__":
    main()