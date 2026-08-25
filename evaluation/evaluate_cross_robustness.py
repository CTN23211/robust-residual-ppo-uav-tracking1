import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# ============================================================
# Reuse VERIFIED Phase 5A evaluators
# ============================================================

from evaluation.evaluate_observation_noise import (
    evaluate_episode as evaluate_noise_episode
)

from evaluation.evaluate_observation_delay import (
    evaluate_episode as evaluate_delay_episode
)

from evaluation.evaluate_temporary_vision_loss import (
    evaluate_episode as evaluate_vision_episode
)

from evaluation.evaluate_unseen_trajectories import (
    evaluate_episode as evaluate_trajectory_episode
)


# ============================================================
# Configuration
# ============================================================

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


CONTROLLERS = [
    "Nominal Residual PPO",
    "Robust Residual PPO"
]


# ============================================================
# Cross-robustness test matrix
#
# IMPORTANT:
# These are PREVIOUSLY USED Phase 5A endpoint conditions.
# No new severity is selected after seeing robust PPO results.
# ============================================================

TEST_CONDITIONS = [

    {
        "name":
            "Nominal",

        "family":
            "noise",

        "position_noise_std":
            0.0,

        "velocity_noise_std":
            0.0
    },

    {
        "name":
            "Position noise 0.10 m",

        "family":
            "noise",

        "position_noise_std":
            0.10,

        "velocity_noise_std":
            0.0
    },

    {
        "name":
            "Velocity noise 0.10 m/s",

        "family":
            "noise",

        "position_noise_std":
            0.0,

        "velocity_noise_std":
            0.10
    },

    {
        "name":
            "Delay 400 ms",

        "family":
            "delay",

        "delay_steps":
            8
    },

    {
        "name":
            "Vision loss 1.0 s",

        "family":
            "vision",

        "loss_duration":
            1.0
    },

    {
        "name":
            "Speed step",

        "family":
            "trajectory",

        "scenario":
            "speed_step"
    },

    {
        "name":
            "Lateral sine",

        "family":
            "trajectory",

        "scenario":
            "lateral_sine"
    },

    {
        "name":
            "Constant turn",

        "family":
            "trajectory",

        "scenario":
            "constant_turn"
    }
]


# ============================================================
# Utilities
# ============================================================

def safe_mean(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.mean(values)
    )


def safe_std(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    return float(
        np.std(values)
    )


def safe_percent_change(
    new_value,
    old_value
):
    """
    Positive:
        Robust value is larger.

    Negative:
        Robust value is smaller.

    For error / overshoot / smoothness,
    negative is usually better.
    """

    if (
        not np.isfinite(old_value)
        or
        abs(old_value) < 1e-12
    ):

        return np.nan

    return float(
        100.0
        * (
            new_value
            - old_value
        )
        / old_value
    )


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
# Normalize outputs from different Phase 5A evaluators
# ============================================================

def normalize_result(
    result,
    condition,
    controller
):

    family = condition[
        "family"
    ]

    # ========================================================
    # Noise / Delay
    # ========================================================

    if family in [
        "noise",
        "delay"
    ]:

        primary_error = result[
            "steady_rmse"
        ]

        stable_ratio = result[
            "stable_ratio"
        ]

        secondary_time = result.get(
            "settling_time",
            np.nan
        )

        secondary_time_name = (
            "settling_time"
        )

        post_peak_error = np.nan

    # ========================================================
    # Temporary vision loss
    # ========================================================

    elif family == "vision":

        primary_error = result[
            "steady_rmse"
        ]

        stable_ratio = result[
            "stable_ratio"
        ]

        secondary_time = result[
            "recovery_time"
        ]

        secondary_time_name = (
            "recovery_time"
        )

        post_peak_error = result[
            "post_loss_peak_error"
        ]

    # ========================================================
    # Unseen target trajectory
    #
    # Here POST-MANEUVER performance is the relevant metric.
    # ========================================================

    elif family == "trajectory":

        primary_error = result[
            "post_maneuver_rmse"
        ]

        stable_ratio = result[
            "post_stable_ratio"
        ]

        secondary_time = result[
            "reacquisition_time"
        ]

        secondary_time_name = (
            "reacquisition_time"
        )

        post_peak_error = result[
            "post_p95_error"
        ]

    else:

        raise ValueError(
            family
        )

    return {

        "controller":
            controller,

        "condition":
            condition["name"],

        "family":
            family,

        "seed":
            result["seed"],

        "success":
            bool(
                result["success"]
            ),

        # Full-episode metric
        "full_rmse":
            float(
                result["rmse"]
            ),

        # Main condition-specific error:
        #
        # noise/delay/vision:
        #     steady RMSE
        #
        # trajectory:
        #     post-maneuver RMSE
        "primary_error":
            float(
                primary_error
            ),

        "stable_ratio":
            float(
                stable_ratio
            ),

        "forward_overshoot":
            float(
                result[
                    "forward_overshoot"
                ]
            ),

        "command_smoothness":
            float(
                result[
                    "command_smoothness"
                ]
            ),

        "secondary_time_name":
            secondary_time_name,

        "secondary_time":
            float(
                secondary_time
            )
            if np.isfinite(
                secondary_time
            )
            else np.nan,

        "post_peak_or_p95_error":
            float(
                post_peak_error
            )
            if np.isfinite(
                post_peak_error
            )
            else np.nan,

        "final_error":
            float(
                result.get(
                    "final_error",
                    np.nan
                )
            )
    }


# ============================================================
# Evaluate one condition / controller
# ============================================================

def run_one_episode(
    controller_name,
    model,
    condition,
    seed
):

    family = condition[
        "family"
    ]

    # ========================================================
    # Observation noise
    # ========================================================

    if family == "noise":

        result = evaluate_noise_episode(

            method=
                "Residual PPO",

            seed=
                seed,

            position_noise_std=
                condition[
                    "position_noise_std"
                ],

            velocity_noise_std=
                condition[
                    "velocity_noise_std"
                ],

            residual_model=
                model,

            direct_model=
                None
        )

    # ========================================================
    # Delay
    # ========================================================

    elif family == "delay":

        result = evaluate_delay_episode(

            method=
                "Residual PPO",

            seed=
                seed,

            delay_steps=
                condition[
                    "delay_steps"
                ],

            residual_model=
                model,

            direct_model=
                None
        )

    # ========================================================
    # Vision loss
    # ========================================================

    elif family == "vision":

        result = evaluate_vision_episode(

            method=
                "Residual PPO",

            seed=
                seed,

            loss_duration=
                condition[
                    "loss_duration"
                ],

            residual_model=
                model,

            direct_model=
                None
        )

    # ========================================================
    # OOD target trajectory
    # ========================================================

    elif family == "trajectory":

        result = evaluate_trajectory_episode(

            method=
                "Residual PPO",

            scenario=
                condition[
                    "scenario"
                ],

            seed=
                seed,

            residual_model=
                model,

            direct_model=
                None
        )

    else:

        raise ValueError(
            family
        )

    return normalize_result(
        result=result,
        condition=condition,
        controller=controller_name
    )


# ============================================================
# Summary
# ============================================================

def summarize_condition(
    rows,
    controller,
    condition_name
):

    subset = [

        row

        for row
        in rows

        if (
            row["controller"]
            == controller

            and

            row["condition"]
            == condition_name
        )
    ]

    def values(
        name
    ):

        return np.asarray([
            row[name]
            for row
            in subset
        ])

    return {

        "controller":
            controller,

        "condition":
            condition_name,

        "success_rate":
            float(
                np.mean(
                    values(
                        "success"
                    )
                )
            ),

        "full_rmse_mean":
            safe_mean(
                values(
                    "full_rmse"
                )
            ),

        "primary_error_mean":
            safe_mean(
                values(
                    "primary_error"
                )
            ),

        "primary_error_std":
            safe_std(
                values(
                    "primary_error"
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
            ),

        "secondary_time_mean":
            safe_mean(
                values(
                    "secondary_time"
                )
            ),

        "post_peak_or_p95_error_mean":
            safe_mean(
                values(
                    "post_peak_or_p95_error"
                )
            )
    }


# ============================================================
# Paired comparison
# ============================================================

def build_paired_comparison(
    raw_rows,
    summary_rows
):

    comparison_rows = []

    for condition in TEST_CONDITIONS:

        condition_name = (
            condition["name"]
        )

        nominal_summary = next(

            row

            for row in summary_rows

            if (
                row["controller"]
                == "Nominal Residual PPO"

                and

                row["condition"]
                == condition_name
            )
        )

        robust_summary = next(

            row

            for row in summary_rows

            if (
                row["controller"]
                == "Robust Residual PPO"

                and

                row["condition"]
                == condition_name
            )
        )

        # ====================================================
        # Paired same-seed episode outcomes
        # ====================================================

        nominal_by_seed = {

            row["seed"]:
                row

            for row in raw_rows

            if (
                row["controller"]
                == "Nominal Residual PPO"

                and

                row["condition"]
                == condition_name
            )
        }

        robust_by_seed = {

            row["seed"]:
                row

            for row in raw_rows

            if (
                row["controller"]
                == "Robust Residual PPO"

                and

                row["condition"]
                == condition_name
            )
        }

        rescued = 0

        regressed = 0

        both_success = 0

        both_fail = 0

        robust_error_better = 0

        paired_count = 0

        for seed in sorted(
            nominal_by_seed.keys()
        ):

            nominal_row = (
                nominal_by_seed[
                    seed
                ]
            )

            robust_row = (
                robust_by_seed[
                    seed
                ]
            )

            n_success = (
                nominal_row[
                    "success"
                ]
            )

            r_success = (
                robust_row[
                    "success"
                ]
            )

            if (
                not n_success
                and
                r_success
            ):

                rescued += 1

            elif (
                n_success
                and
                not r_success
            ):

                regressed += 1

            elif (
                n_success
                and
                r_success
            ):

                both_success += 1

            else:

                both_fail += 1

            if (
                robust_row[
                    "primary_error"
                ]
                <
                nominal_row[
                    "primary_error"
                ]
            ):

                robust_error_better += 1

            paired_count += 1

        comparison_rows.append({

            "condition":
                condition_name,

            # ----------------------------------------------
            # Success
            # ----------------------------------------------

            "nominal_success_rate":
                nominal_summary[
                    "success_rate"
                ],

            "robust_success_rate":
                robust_summary[
                    "success_rate"
                ],

            "success_delta_percentage_points":
                100.0
                * (
                    robust_summary[
                        "success_rate"
                    ]
                    -
                    nominal_summary[
                        "success_rate"
                    ]
                ),

            # ----------------------------------------------
            # Main tracking error
            # ----------------------------------------------

            "nominal_primary_error":
                nominal_summary[
                    "primary_error_mean"
                ],

            "robust_primary_error":
                robust_summary[
                    "primary_error_mean"
                ],

            "primary_error_change_percent":
                safe_percent_change(
                    robust_summary[
                        "primary_error_mean"
                    ],
                    nominal_summary[
                        "primary_error_mean"
                    ]
                ),

            # ----------------------------------------------
            # Stable ratio
            # ----------------------------------------------

            "stable_ratio_delta_percentage_points":
                100.0
                * (
                    robust_summary[
                        "stable_ratio_mean"
                    ]
                    -
                    nominal_summary[
                        "stable_ratio_mean"
                    ]
                ),

            # ----------------------------------------------
            # Overshoot
            # ----------------------------------------------

            "nominal_forward_overshoot":
                nominal_summary[
                    "forward_overshoot_mean"
                ],

            "robust_forward_overshoot":
                robust_summary[
                    "forward_overshoot_mean"
                ],

            "forward_overshoot_change_percent":
                safe_percent_change(
                    robust_summary[
                        "forward_overshoot_mean"
                    ],
                    nominal_summary[
                        "forward_overshoot_mean"
                    ]
                ),

            # ----------------------------------------------
            # Smoothness
            # ----------------------------------------------

            "nominal_command_smoothness":
                nominal_summary[
                    "command_smoothness_mean"
                ],

            "robust_command_smoothness":
                robust_summary[
                    "command_smoothness_mean"
                ],

            "command_smoothness_change_percent":
                safe_percent_change(
                    robust_summary[
                        "command_smoothness_mean"
                    ],
                    nominal_summary[
                        "command_smoothness_mean"
                    ]
                ),

            # ----------------------------------------------
            # Paired same-seed analysis
            # ----------------------------------------------

            "nominal_fail_robust_success":
                rescued,

            "nominal_success_robust_fail":
                regressed,

            "both_success":
                both_success,

            "both_fail":
                both_fail,

            "robust_lower_primary_error_count":
                robust_error_better,

            "paired_episode_count":
                paired_count,

            "robust_lower_primary_error_rate":
                (
                    robust_error_better
                    / paired_count
                )
        })

    return comparison_rows


# ============================================================
# Plot helper
# ============================================================

def make_grouped_plot(
    summary_rows,
    metric_name,
    ylabel,
    title,
    filename,
    percentage=False,
    log_scale=False
):

    condition_names = [
        item["name"]
        for item in TEST_CONDITIONS
    ]

    x = np.arange(
        len(condition_names)
    )

    width = 0.36

    plt.figure(
        figsize=(14, 6)
    )

    for index, controller in enumerate(
        CONTROLLERS
    ):

        values = []

        for condition_name in condition_names:

            row = next(

                row

                for row
                in summary_rows

                if (
                    row["controller"]
                    == controller

                    and

                    row["condition"]
                    == condition_name
                )
            )

            value = row[
                metric_name
            ]

            if percentage:

                value *= 100.0

            values.append(
                value
            )

        offset = (
            -width / 2
            if index == 0
            else width / 2
        )

        plt.bar(
            x + offset,
            values,
            width=width,
            label=controller
        )

    plt.xticks(
        x,
        condition_names,
        rotation=25,
        ha="right"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    if log_scale:

        plt.yscale(
            "log"
        )

    plt.grid(
        True,
        axis="y"
    )

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
    # Load frozen policies
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

    raw_rows = []

    summary_rows = []

    print()
    print("=" * 110)
    print(
        "PHASE 5B-2 — CROSS-ROBUSTNESS VALIDATION"
    )
    print("=" * 110)

    # ========================================================
    # Evaluate
    # ========================================================

    for condition in TEST_CONDITIONS:

        print()
        print(
            f"Condition: "
            f"{condition['name']}"
        )

        print("-" * 110)

        for controller_name in CONTROLLERS:

            model = models[
                controller_name
            ]

            controller_rows = []

            for episode in range(
                NUM_EPISODES
            ):

                seed = (
                    TEST_SEED_START
                    + episode
                )

                result = run_one_episode(

                    controller_name=
                        controller_name,

                    model=
                        model,

                    condition=
                        condition,

                    seed=
                        seed
                )

                raw_rows.append(
                    result
                )

                controller_rows.append(
                    result
                )

            success_rate = float(
                np.mean([
                    r["success"]
                    for r
                    in controller_rows
                ])
            )

            primary_error = safe_mean([
                r["primary_error"]
                for r
                in controller_rows
            ])

            stable_ratio = safe_mean([
                r["stable_ratio"]
                for r
                in controller_rows
            ])

            overshoot = safe_mean([
                r["forward_overshoot"]
                for r
                in controller_rows
            ])

            smoothness = safe_mean([
                r["command_smoothness"]
                for r
                in controller_rows
            ])

            print(
                f"{controller_name:22s} | "
                f"Success="
                f"{success_rate * 100:6.2f}% | "
                f"Primary error="
                f"{primary_error:.4f} | "
                f"Stable="
                f"{stable_ratio * 100:6.2f}% | "
                f"Overshoot="
                f"{overshoot:.4f} | "
                f"Smoothness="
                f"{smoothness:.6f}"
            )

    # ========================================================
    # Build summaries
    # ========================================================

    for condition in TEST_CONDITIONS:

        for controller in CONTROLLERS:

            summary_rows.append(
                summarize_condition(
                    rows=
                        raw_rows,

                    controller=
                        controller,

                    condition_name=
                        condition["name"]
                )
            )

    comparison_rows = (
        build_paired_comparison(
            raw_rows=
                raw_rows,

            summary_rows=
                summary_rows
        )
    )

    # ========================================================
    # Save
    # ========================================================

    raw_path = os.path.join(
        RESULTS_DIR,
        "cross_robustness_raw.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "cross_robustness_summary.csv"
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "cross_robustness_paired_comparison.csv"
    )

    save_csv(
        raw_path,
        raw_rows
    )

    save_csv(
        summary_path,
        summary_rows
    )

    save_csv(
        comparison_path,
        comparison_rows
    )

    # ========================================================
    # Plots
    # ========================================================

    make_grouped_plot(

        summary_rows=

            summary_rows,

        metric_name=
            "success_rate",

        ylabel=
            "Success rate (%)",

        title=
            "Cross-Robustness — Success Rate",

        filename=
            "cross_robustness_success_rate.png",

        percentage=True
    )

    make_grouped_plot(

        summary_rows=
            summary_rows,

        metric_name=
            "primary_error_mean",

        ylabel=
            "Condition-specific tracking error (m)",

        title=
            "Cross-Robustness — Tracking Error",

        filename=
            "cross_robustness_primary_error.png"
    )

    make_grouped_plot(

        summary_rows=
            summary_rows,

        metric_name=
            "forward_overshoot_mean",

        ylabel=
            "Forward overshoot (m)",

        title=
            "Cross-Robustness — Forward Overshoot",

        filename=
            "cross_robustness_forward_overshoot.png"
    )

    make_grouped_plot(

        summary_rows=
            summary_rows,

        metric_name=
            "command_smoothness_mean",

        ylabel=
            "Command smoothness cost",

        title=
            "Cross-Robustness — Physical Command Smoothness",

        filename=
            "cross_robustness_command_smoothness.png",

        log_scale=True
    )

    # ========================================================
    # Paired comparison print
    # ========================================================

    print()
    print("=" * 110)
    print(
        "PAIRED ROBUST - NOMINAL COMPARISON"
    )
    print("=" * 110)

    for row in comparison_rows:

        print()

        print(
            row["condition"]
        )

        print(
            f"  Success delta : "
            f"{row['success_delta_percentage_points']:+.2f} pp"
        )

        print(
            f"  Error change  : "
            f"{row['primary_error_change_percent']:+.2f}%"
        )

        print(
            f"  Stable delta  : "
            f"{row['stable_ratio_delta_percentage_points']:+.2f} pp"
        )

        print(
            f"  Rescue        : "
            f"{row['nominal_fail_robust_success']}"
        )

        print(
            f"  Regression    : "
            f"{row['nominal_success_robust_fail']}"
        )

        print(
            f"  Robust lower error:"
            f" "
            f"{row['robust_lower_primary_error_count']}/"
            f"{row['paired_episode_count']}"
        )

    print()
    print("=" * 110)

    print(
        "Phase 5B-2 Cross-Robustness Validation COMPLETED."
    )

    print("=" * 110)

    print(
        f"Raw:"
        f"\n  {raw_path}"
    )

    print(
        f"Summary:"
        f"\n  {summary_path}"
    )

    print(
        f"Paired comparison:"
        f"\n  {comparison_path}"
    )


if __name__ == "__main__":
    main()