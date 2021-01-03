from typing import Dict
from typing import Union

from config_manager import base_configuration
from experiments.ach_config_template import AChConfigTemplate

import constants


class AchConfig(base_configuration.BaseConfiguration):
    """Ach RL Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(
            configuration=config,
            template=AChConfigTemplate.base_template,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Check for non-trivial associations in config.

        Raises:
            AssertionError: if any rules are broken by config.
        """
        environment = getattr(self, constants.Constants.ENVIRONMENT)
        if environment == constants.Constants.MINIGRID:
            reward_positions = getattr(self, constants.Constants.REWARD_POSITIONS)
            num_rewards = getattr(self, constants.Constants.NUM_REWARDS)
            reward_magnitudes = getattr(self, constants.Constants.REWARD_MAGNITUDES)
            assert reward_positions is None or len(reward_positions) == num_rewards, (
                "Number of reward positions must match number of rewards,"
                "or reward positions must be set to None for random placement."
            )
            assert (
                len(reward_magnitudes) == num_rewards
            ), "Number of reward magnitudes must match number of rewards,"

        # check scalars specified in logging are compatible with alg/env etc.
        scalars = getattr(self, constants.Constants.SCALARS)
        visuals = getattr(self, constants.Constants.VISUALISATIONS)
        learner = getattr(self, constants.Constants.TYPE)
        if learner == constants.Constants.DQN:
            permitted_scalars = [
                constants.Constants.TRAIN_EPISODE_REWARD,
                constants.Constants.TRAIN_EPISODE_LENGTH,
                constants.Constants.TEST_EPISODE_REWARD,
                constants.Constants.TEST_EPISODE_LENGTH,
            ]
            permitted_visuals = [
                constants.Constants.INDIVIDUAL_TRAIN_RUN,
                constants.Constants.INDIVIDUAL_TEST_RUN,
            ]
        elif learner == constants.Constants.ENSEMBLE_Q_LEARNING:
            permitted_scalars = [
                constants.Constants.TRAIN_EPISODE_REWARD,
                constants.Constants.TRAIN_EPISODE_LENGTH,
                constants.Constants.TEST_EPISODE_REWARD,
                constants.Constants.TEST_EPISODE_LENGTH,
                "_".join(
                    [
                        constants.Constants.TRAIN_EPISODE_REWARD,
                        constants.Constants.ENSEMBLE_RUNNER,
                    ]
                ),
                "_".join(
                    [
                        constants.Constants.TRAIN_EPISODE_LENGTH,
                        constants.Constants.ENSEMBLE_RUNNER,
                    ]
                ),
                constants.Constants.ENSEMBLE_EPISODE_REWARD_STD,
                constants.Constants.ENSEMBLE_EPISODE_LENGTH_STD,
                constants.Constants.MEAN_VISITATION_PENALTY,
                constants.Constants.NO_REPEAT_TEST_EPISODE_REWARD,
                constants.Constants.NO_REPEAT_TEST_EPISODE_LENGTH,
                constants.Constants.CYCLE_COUNT,
            ]
            permitted_visuals = [
                constants.Constants.INDIVIDUAL_TRAIN_RUN,
                constants.Constants.INDIVIDUAL_TEST_RUN,
                constants.Constants.VALUE_FUNCTION,
            ]
        for scalar in scalars:
            if isinstance(scalar[0], str):
                scalar_str = scalar[0]
            elif isinstance(scalar[0], list):
                scalar_str = scalar[0][0]
            assert scalar_str in permitted_scalars, (
                f"Scalar {scalar} specified in config logging, "
                f"is not compatible with learner {learner}."
            )
        for visual in visuals:
            if isinstance(visual[0], str):
                visual_str = visual[0]
            elif isinstance(visual[0], list):
                visual_str = visual[0][0]
            assert visual_str in permitted_visuals, (
                f"Visual {visual} specified in config logging, "
                f"is not compatible with learner {learner}."
            )

        # check testing procedures are compatible with alg/env etc.
        if learner == constants.Constants.DQN:
            permitted_tests = [constants.Constants.GREEDY]
        for test_type in getattr(self, constants.Constants.TESTING):
            assert (
                test_type in permitted_tests
            ), f"Test type {test_type} not compatible with learner {learner}"

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
