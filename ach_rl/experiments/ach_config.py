import itertools
from typing import Dict
from typing import List
from typing import Union

import yaml
from ach_rl import constants
from ach_rl.experiments.ach_config_template import AChConfigTemplate
from config_manager import base_configuration


class AchConfig(base_configuration.BaseConfiguration):
    """Ach RL Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        super().__init__(
            configuration=config,
            template=AChConfigTemplate.base_template,
            changes=changes,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Check for non-trivial associations in config.

        Raises:
            AssertionError: if any rules are broken by config.
        """
        environment = getattr(self, constants.ENVIRONMENT)
        if environment == constants.MINIGRID:
            reward_positions = getattr(self, constants.REWARD_POSITIONS)
            num_rewards = getattr(self, constants.NUM_REWARDS)
            reward_magnitudes = getattr(self, constants.REWARD_MAGNITUDES)
            assert reward_positions is None or len(reward_positions) == num_rewards, (
                "Number of reward positions must match number of rewards,"
                "or reward positions must be set to None for random placement."
            )
            assert (
                len(reward_magnitudes) == num_rewards
            ), "Number of reward magnitudes must match number of rewards,"

        with open(getattr(self, constants.LOG_SPECIFICATION), "r") as yaml_file:
            log_spec = yaml.load(yaml_file)

        # check scalars specified in logging are compatible with alg/env etc.
        scalars = log_spec[constants.SCALARS]
        arrays = log_spec[constants.ARRAYS]
        visuals = log_spec[constants.VISUALISATIONS]
        post_visuals = log_spec[constants.POST_VISUALISATIONS]
        learner = getattr(self, constants.TYPE)

        # if learner == constants.Q_LEARNING:
        #     permitted_scalars = [
        #         constants.TRAIN_EPISODE_REWARD,
        #         constants.TRAIN_EPISODE_LENGTH,
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #         constants.MEAN_VISITATION_PENALTY,
        #         constants.MEAN_EPSILON,
        #         constants.MEAN_LR_SCALING,
        #     ]
        #     permitted_visuals = [
        #         constants.VALUE_FUNCTION,
        #         constants.VISITATION_COUNT_HEATMAP,
        #         constants.INDIVIDUAL_TRAIN_RUN,
        #     ]
        # elif learner == constants.VANILLA_DQN:
        #     permitted_scalars = [
        #         constants.TRAIN_EPISODE_REWARD,
        #         constants.TRAIN_EPISODE_LENGTH,
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #         constants.AVERAGE_ACTION_VALUE,
        #         constants.LOSS,
        #         constants.EPSILON,
        #         constants.MEAN_VISITATION_PENALTY,
        #         constants.MEAN_EPSILON,
        #         constants.MEAN_LR_SCALING,
        #     ]
        #     permitted_visuals = [
        #         constants.INDIVIDUAL_TRAIN_RUN,
        #         constants.INDIVIDUAL_TEST_RUN,
        #     ]
        #     if environment == constants.MULTIROOM:
        #         permitted_visuals.extend(
        #             [
        #                 constants.VALUE_FUNCTION,
        #                 constants.VISITATION_COUNT_HEATMAP,
        #             ]
        #         )
        # elif learner == constants.ENSEMBLE_Q_LEARNING:
        #     permitted_scalars = [
        #         constants.TRAIN_EPISODE_REWARD,
        #         constants.TRAIN_EPISODE_LENGTH,
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #         "_".join(
        #             [
        #                 constants.TRAIN_EPISODE_REWARD,
        #                 constants.ENSEMBLE_RUNNER,
        #             ]
        #         ),
        #         "_".join(
        #             [
        #                 constants.TRAIN_EPISODE_LENGTH,
        #                 constants.ENSEMBLE_RUNNER,
        #             ]
        #         ),
        #         constants.ENSEMBLE_EPISODE_REWARD_STD,
        #         constants.ENSEMBLE_EPISODE_LENGTH_STD,
        #         constants.MEAN_VISITATION_PENALTY,
        #         f"{constants.NEXT_STATE_POLICY_ENTROPY}_{constants.MEAN}",
        #         f"{constants.CURRENT_STATE_POLICY_ENTROPY}_{constants.MEAN}",
        #         f"{constants.NEXT_STATE_MEAN_UNCERTAINTY}_{constants.MEAN}",
        #         f"{constants.NEXT_STATE_MAX_UNCERTAINTY}_{constants.MEAN}",
        #         f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_SAMPLE}",
        #         f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_MEAN}",
        #         f"{constants.TEST_EPISODE_LENGTH}_{constants.GREEDY_VOTE}",
        #         f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_SAMPLE}",
        #         f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_MEAN}",
        #         f"{constants.TEST_EPISODE_REWARD}_{constants.GREEDY_VOTE}",
        #         f"{constants.CURRENT_STATE_MAX_UNCERTAINTY}_{constants.MEAN}",
        #         f"{constants.CURRENT_STATE_MEAN_UNCERTAINTY}_{constants.MEAN}",
        #         f"{constants.CURRENT_STATE_SELECT_UNCERTAINTY}_{constants.MEAN}",
        #         constants.NO_REPEAT_TEST_EPISODE_REWARD,
        #         constants.NO_REPEAT_TEST_EPISODE_LENGTH,
        #         constants.CYCLE_COUNT,
        #         constants.MEAN_EPSILON,
        #         constants.MEAN_LR_SCALING,
        #     ]
        #     test_constants = [
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #     ]
        #     targets = [
        #         constants.GREEDY_SAMPLE,
        #         constants.GREEDY_VOTE,
        #         constants.GREEDY_MEAN,
        #     ]
        #     permitted_scalars.extend(
        #         [
        #             "_".join([test_constant, constants.NO_REP, target])
        #             for test_constant, target in itertools.product(
        #                 test_constants, targets
        #             )
        #         ]
        #     )
        #     permitted_scalars.extend(
        #         [
        #             "_".join([test_constant, target])
        #             for test_constant, target in itertools.product(
        #                 test_constants, targets
        #             )
        #         ]
        #     )

        #     permitted_visuals = [
        #         constants.INDIVIDUAL_TRAIN_RUN,
        #         constants.INDIVIDUAL_TEST_RUN,
        #         constants.VALUE_FUNCTION,
        #         constants.INDIVIDUAL_VALUE_FUNCTIONS,
        #         constants.VALUE_FUNCTION_STD,
        #     ]
        # elif learner == constants.BOOTSTRAPPED_ENSEMBLE_DQN:
        #     permitted_scalars = [
        #         constants.TRAIN_EPISODE_REWARD,
        #         constants.TRAIN_EPISODE_LENGTH,
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #         constants.LOSS,
        #         constants.EPSILON,
        #         constants.BRANCH_LOSS,
        #         constants.BRANCH_REWARD,
        #         "_".join(
        #             [
        #                 constants.TEST,
        #                 constants.ENSEMBLE_EPISODE_REWARD_STD,
        #             ]
        #         ),
        #         "_".join(
        #             [
        #                 constants.TEST,
        #                 constants.ENSEMBLE_EPISODE_LENGTH_STD,
        #             ]
        #         ),
        #         "_".join(
        #             [
        #                 constants.TEST,
        #                 constants.ENSEMBLE_EPISODE_REWARD_MEAN,
        #             ]
        #         ),
        #         "_".join(
        #             [
        #                 constants.TEST,
        #                 constants.ENSEMBLE_EPISODE_LENGTH_MEAN,
        #             ]
        #         ),
        #         constants.MEAN_VISITATION_PENALTY,
        #         constants.MEAN_INFO,
        #         constants.STD_INFO,
        #     ]
        #     test_constants = [
        #         constants.TEST_EPISODE_REWARD,
        #         constants.TEST_EPISODE_LENGTH,
        #     ]
        #     targets = [
        #         constants.GREEDY_SAMPLE,
        #         constants.GREEDY_VOTE,
        #         constants.GREEDY_MEAN,
        #         constants.GREEDY_INDIVIDUAL,
        #     ]

        #     permitted_visuals = [
        #         constants.INDIVIDUAL_TRAIN_RUN,
        #         constants.INDIVIDUAL_TEST_RUN,
        #     ]
        # else:
        #     raise ValueError(
        #         f"Permitted scalars for learner {learner} not specified..."
        #     )

        # if scalars is not None:
        #     for scalar in scalars:
        #         if isinstance(scalar[0], str):
        #             scalar_str = scalar[0]
        #         elif isinstance(scalar[0], list):
        #             scalar_str = scalar[0][0]
        #         assert scalar_str in permitted_scalars, (
        #             f"Scalar {scalar} specified in config logging, "
        #             f"is not compatible with learner {learner}."
        #         )
        # if visuals is not None:
        #     for visual in visuals:
        #         if isinstance(visual[0], str):
        #             visual_str = visual[0]
        #         elif isinstance(visual[0], list):
        #             visual_str = visual[0][0]
        #         assert visual_str in permitted_visuals, (
        #             f"Visual {visual} specified in config logging, "
        #             f"is not compatible with learner {learner}."
        #         )

        self.add_property(property_name=constants.SCALARS, property_value=scalars)
        self.add_property(property_name=constants.ARRAYS, property_value=arrays)
        self.add_property(
            property_name=constants.VISUALISATIONS, property_value=visuals
        )
        self.add_property(
            property_name=constants.POST_VISUALISATIONS, property_value=post_visuals
        )

        # check testing procedures are compatible with alg/env etc.
        if learner == constants.VANILLA_DQN:
            permitted_tests = [constants.GREEDY]
        elif learner == constants.ENSEMBLE_Q_LEARNING:
            permitted_tests = [constants.GREEDY, constants.NO_REP]

        test_types = getattr(self, constants.TESTING)
        if test_types is not None:
            for test_type in test_types:
                assert (
                    test_type in permitted_tests
                ), f"Test type {test_type} not compatible with learner {learner}"

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
