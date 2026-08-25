from envs.uav_tracking_residual_robust_env import (
    UAVTrackingResidualRobustEnv
)


class UAVTrackingRobustTrainingEnv(
    UAVTrackingResidualRobustEnv
):
    """
    Phase 5B robust-training environment.

    The feedforward velocity-coupling fault severity alpha
    is randomized ONCE per episode:

        alpha ~ Uniform(alpha_min, alpha_max)

    The same alpha is held fixed during the entire episode.

    IMPORTANT:
        alpha = 1.0 is intentionally excluded from training.
        It will be used as a held-out severe fault test.
    """

    def __init__(
        self,
        randomize=True,
        alpha_min=0.0,
        alpha_max=0.75
    ):

        if alpha_min < 0.0:

            raise ValueError(
                "alpha_min must be >= 0."
            )

        if alpha_max <= alpha_min:

            raise ValueError(
                "alpha_max must be > alpha_min."
            )

        if alpha_max >= 1.0:

            raise ValueError(
                "For Phase 5B, alpha_max must remain < 1.0 "
                "so alpha=1.0 remains held-out."
            )

        self.alpha_min = float(
            alpha_min
        )

        self.alpha_max = float(
            alpha_max
        )

        # Initial value is irrelevant;
        # it will be randomized at reset.
        super().__init__(
            randomize=randomize,
            velocity_coupling_alpha=0.0
        )

    def reset(
        self,
        seed=None,
        options=None
    ):

        # ----------------------------------------------------
        # Parent reset:
        #
        # 1. seeds Gym RNG
        # 2. randomizes initial UAV state / target speed
        # 3. resets episode
        # ----------------------------------------------------

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        # ----------------------------------------------------
        # Sample ONE fault severity for this episode
        #
        # alpha in [0, 0.75]
        # ----------------------------------------------------

        self.velocity_coupling_alpha = float(
            self.np_random.uniform(
                self.alpha_min,
                self.alpha_max
            )
        )

        # ----------------------------------------------------
        # Information only.
        #
        # IMPORTANT:
        # alpha is NOT added to PPO observation.
        #
        # The policy must learn robustness without being
        # explicitly told the true fault severity.
        # ----------------------------------------------------

        info[
            "velocity_coupling_alpha"
        ] = self.velocity_coupling_alpha

        info[
            "alpha_min"
        ] = self.alpha_min

        info[
            "alpha_max"
        ] = self.alpha_max

        return (
            obs,
            info
        )