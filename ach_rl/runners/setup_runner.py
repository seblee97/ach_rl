import abc
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

from ach_rl import constants
from ach_rl.curricula import base_curriculum
from ach_rl.environments import atari
from ach_rl.environments import base_environment
from ach_rl.environments import wrapper_atari
from ach_rl.epsilon_computers import constant_epsilon_computer
from ach_rl.epsilon_computers import expected_uncertainty_epsilon_computer
from ach_rl.epsilon_computers import linear_decay_epsilon_computer
from ach_rl.epsilon_computers import unexpected_uncertainty_epsilon_computer
from ach_rl.experiments import ach_config
from ach_rl.information_computers import base_information_computer
from ach_rl.information_computers import network_information_computer
from ach_rl.information_computers import tabular_information_computer
from ach_rl.learning_rate_scalers import \
    expected_uncertainty_learning_rate_scaler
from ach_rl.learning_rate_scalers import hard_coded_learning_rate_scaler
from ach_rl.visitation_penalties import \
    adaptive_arriving_uncertainty_visitation_penalty
from ach_rl.visitation_penalties import adaptive_uncertainty_visitation_penalty
from ach_rl.visitation_penalties import base_visitation_penalty
from ach_rl.visitation_penalties import exponential_decay_visitation_penalty
from ach_rl.visitation_penalties import hard_coded_visitation_penalty
from ach_rl.visitation_penalties import linear_decay_visitation_penalty
from ach_rl.visitation_penalties import policy_entropy_penalty
from ach_rl.visitation_penalties import potential_adaptive_uncertainty_penalty
from ach_rl.visitation_penalties import potential_policy_entropy_penalty
from ach_rl.visitation_penalties import reducing_entropy_window_penalty
from ach_rl.visitation_penalties import reducing_variance_window_penalty
from ach_rl.visitation_penalties import sigmoidal_decay_visitation_penalty
from ach_rl.visitation_penalties import signed_uncertainty_window_penalty
from key_door import curriculum_env
from key_door import key_door_env
from key_door import visualisation_env
from run_modes import base_runner


class SetupRunner(base_runner.BaseRunner):
    def __init__(self, config: ach_config.AchConfig, unique_id: str) -> None:

        self._scalar_logging = self._setup_logging_frequencies(config.scalars)
        self._array_logging = self._setup_logging_frequencies(config.arrays)
        self._visualisations = self._setup_logging_frequencies(config.visualisations)
        self._post_visualisations = config.post_visualisations

        super().__init__(config=config, unique_id=unique_id)

        # logging setup
        self._array_folder_path = os.path.join(self._checkpoint_path, constants.ARRAYS)
        self._rollout_folder_path = os.path.join(
            self._checkpoint_path, constants.ROLLOUTS
        )
        self._visualisations_folder_path = os.path.join(
            self._checkpoint_path, constants.VISUALISATIONS
        )
        os.makedirs(name=self._array_folder_path, exist_ok=True)
        os.makedirs(name=self._rollout_folder_path, exist_ok=True)
        for tag in self._array_logging.keys():
            os.makedirs(name=os.path.join(self._array_folder_path, tag), exist_ok=True)
        os.makedirs(name=self._visualisations_folder_path, exist_ok=True)

        # component setup
        self._environment = self._setup_environment(config=config)
        self._information_computer = self._setup_information_computer(config=config)
        if config.visitation_penalty_type is None:
            self._visitation_penalty = None
        else:
            self._visitation_penalty = self._setup_visitation_penalty(config=config)
        self._epsilon_computer = self._setup_epsilon_computer(config=config)
        self._lr_scaler = self._setup_lr_scaler(config=config)
        self._learner = self._setup_learner(config=config)

    def _setup_logging_frequencies(
        self, logging_list: List[Tuple[Union[List, str], int]]
    ) -> Dict[str, int]:
        """Parse logging list from config into mapping from
        tag to log to frequency with which it should be logged.

        Args:
            logging_list: un-parsed list of lists consisting of attributes
                to log and how frequently they should be logged.

        Returns:
            logging_frequencies: mapping from tags to log frequencies.
        """
        logging_frequencies = {}
        if logging_list is not None:
            for i in logging_list:
                if isinstance(i[0], list):
                    for j in range(i[0][1]):
                        logging_frequencies[f"{i[0][0]}_{j}"] = i[1]
                elif isinstance(i[0], str):
                    logging_frequencies[i[0]] = i[1]
                else:
                    raise ValueError("Log list incorrectly formatted.")
        return logging_frequencies

    def _setup_environment(
        self, config: ach_config.AchConfig
    ) -> Union[base_curriculum.BaseCurriculum, base_environment.BaseEnvironment]:
        """Initialise environment specified in configuration.
        Applies curriculum wrapper where specified.
        """
        environment_args = self._get_environment_args(config=config)

        if config.environment == constants.MINIGRID:
            environment = minigrid.MiniGrid(**environment_args)
        elif config.environment == constants.ATARI:
            if config.implementation == constants.WRAPPER:
                environment = wrapper_atari.AtariEnv(**environment_args)
            elif config.implementation == constants.FUNCTIONAL:
                environment = atari.AtariEnv(**environment_args)
        elif config.environment == constants.MULTIROOM:
            environment = key_door_env.KeyDoorGridworld(**environment_args)
            environment = visualisation_env.VisualisationEnv(environment)

        if config.apply_curriculum:
            curriculum_args = self._get_curriculum_args(config=config)
            curriculum_wrapper = self.get_curriculum_wrapper(config.environment)
            environment = curriculum_wrapper(environment, **curriculum_args)

        return environment

    def _get_environment_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
        """Get arguments needed to pass to environment."""
        if config.environment == constants.MINIGRID:
            if config.reward_positions is not None:
                reward_positions = [
                    tuple(position) for position in config.reward_positions
                ]
            else:
                reward_positions = None
            if config.starting_position is not None:
                agent_starting_position = tuple(config.starting_position)
            else:
                agent_starting_position = None
            env_args = {
                constants.SIZE: tuple(config.size),
                constants.NUM_REWARDS: config.num_rewards,
                constants.REWARD_MAGNITUDES: config.reward_magnitudes,
                constants.STARTING_XY: agent_starting_position,
                constants.REWARD_XY: reward_positions,
                constants.REPEAT_REWARDS: config.repeat_rewards,
                constants.EPISODE_TIMEOUT: config.episode_timeout,
            }
        elif config.environment == constants.ATARI:
            env_args = {
                constants.ATARI_ENV_NAME: config.atari_env_name,
                constants.EPISODE_TIMEOUT: config.episode_timeout,
                constants.FRAME_STACK: config.frame_stack,
                constants.FRAME_SKIP: config.frame_skip,
            }
            if config.implementation == constants.FUNCTIONAL:
                env_args[constants.PRE_PROCESSING] = config.pre_processing
        elif config.environment == constants.MULTIROOM:
            env_args = {
                constants.MAP_ASCII_PATH: os.path.join(config.map_ascii_path),
                constants.MAP_YAML_PATH: os.path.join(config.map_yaml_path),
                constants.REPRESENTATION: config.representation,
                constants.EPISODE_TIMEOUT: config.episode_timeout,
            }
            if config.representation in [constants.PIXEL, constants.PO_PIXEL]:
                env_args[constants.SCALING] = config.scaling
            if config.representation == constants.PO_PIXEL:
                env_args[constants.FIELD_X] = config.field_of_view[0]
                env_args[constants.FIELD_Y] = config.field_of_view[1]
        return env_args

    def _get_curriculum_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
        """Get arguments needed to pass to environment curriculum wrapper."""
        if config.environment == constants.MINIGRID:
            curriculum_args = {
                constants.TRANSITION_EPISODES: config.transition_episodes,
                constants.ENVIRONMENT_CHANGES: config.environment_changes,
            }
        elif config.environment == constants.MULTIROOM:
            curriculum_args = {
                constants.TRANSITIONS: config.map_yaml_paths,
            }
        return curriculum_args

    @staticmethod
    def get_curriculum_wrapper(environment: str) -> base_curriculum.BaseCurriculum:
        """Get relevant wrapper for environment to add curriculum features."""
        if environment == constants.MINIGRID:
            wrapper = minigrid_curriculum.MinigridCurriculum
        elif environment == constants.MULTIROOM:
            wrapper = curriculum_env.CurriculumEnv
        return wrapper

    def _setup_information_computer(
        self, config: ach_config.AchConfig
    ) -> Type[base_information_computer.BaseInformationComputer]:
        if config.type in [
            constants.BOOTSTRAPPED_ENSEMBLE_DQN,
            constants.VANILLA_DQN,
            constants.INDEPENDENT_ENSEMBLE_DQN,
        ]:
            if config.shaping_implementation == constants.TRAIN_Q_NETWORK:
                use_target_network = False
            elif config.shaping_implementation == constants.TRAIN_TARGET_NETWORK:
                use_target_network = True
            information_computer = (
                network_information_computer.NetworkInformationComputer(
                    use_target_network=use_target_network,
                )
            )
        elif config.type in [
            constants.Q_LEARNING,
            constants.SARSA_LAMBDA,
            constants.ENSEMBLE_Q_LEARNING,
        ]:
            information_computer = (
                tabular_information_computer.TabularInformationComputer()
            )
        else:
            raise ValueError(
                f"Learner type {config.type} has not been grouped as tabular or network, "
                "so the correct visitation penalty class type cannot be determined."
            )
        return information_computer

    def _setup_visitation_penalty(
        self, config: ach_config.AchConfig
    ) -> base_visitation_penalty.BaseVisitationPenalty:
        """Initialise object to act as visitation penalty."""
        if config.visitation_penalty_type == constants.HARD_CODED:
            penalty_computer = hard_coded_visitation_penalty.HardCodedPenalty(
                hard_coded_penalties=config.vp_schedule
            )
        elif (
            config.visitation_penalty_type == constants.DETERMINISTIC_EXPONENTIAL_DECAY
        ):
            penalty_computer = (
                exponential_decay_visitation_penalty.ExponentialDecayPenalty(
                    A=config.A, b=config.b, c=config.c
                )
            )
        elif config.visitation_penalty_type == constants.DETERMINISTIC_LINEAR_DECAY:
            penalty_computer = linear_decay_visitation_penalty.LinearDecayPenalty(
                A=config.A, b=config.b
            )
        elif config.visitation_penalty_type == constants.DETERMINISTIC_SIGMOIDAL_DECAY:
            penalty_computer = sigmoidal_decay_visitation_penalty.SigmoidalDecayPenalty(
                A=config.A, b=config.b, c=config.c
            )
        elif config.visitation_penalty_type == constants.ADAPTIVE_UNCERTAINTY:
            penalty_computer = (
                adaptive_uncertainty_visitation_penalty.AdaptiveUncertaintyPenalty(
                    multiplicative_factor=config.multiplicative_factor,
                    action_function=config.action_function,
                )
            )
        elif config.visitation_penalty_type == constants.ADAPTIVE_ARRIVING_UNCERTAINTY:
            penalty_computer = adaptive_arriving_uncertainty_visitation_penalty.AdaptiveArrivingUncertaintyPenalty(
                multiplicative_factor=config.multiplicative_factor,
                action_function=config.action_function,
            )
        elif (
            config.visitation_penalty_type
            == constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY
        ):
            penalty_computer = potential_adaptive_uncertainty_penalty.PotentialAdaptiveUncertaintyPenalty(
                gamma=config.discount_factor,
                multiplicative_factor=config.multiplicative_factor,
                pre_action_function=config.pre_action_function,
                post_action_function=config.post_action_function,
            )
        elif config.visitation_penalty_type == constants.POLICY_ENTROPY_PENALTY:
            penalty_computer = policy_entropy_penalty.PolicyEntropyPenalty(
                multiplicative_factor=config.multiplicative_factor
            )
        elif (
            config.visitation_penalty_type
            == constants.POTENTIAL_BASED_POLICY_ENTROPY_PENALTY
        ):
            penalty_computer = (
                potential_policy_entropy_penalty.PotentialPolicyEntropyPenalty(
                    gamma=config.discount_factor,
                    multiplicative_factor=config.multiplicative_factor,
                )
            )
        elif config.visitation_penalty_type == constants.REDUCING_VARIANCE_WINDOW:
            penalty_computer = reducing_variance_window_penalty.ReducingVarianceWindowPenalty(
                expected_multiplicative_factor=config.expected_multiplicative_factor,
                unexpected_multiplicative_factor=config.unexpected_multiplicative_factor,
                action_function=config.action_function,
                moving_average_window=config.moving_average_window,
            )
        elif config.visitation_penalty_type == constants.REDUCING_ENTROPY_WINDOW:
            penalty_computer = reducing_entropy_window_penalty.ReducingEntropyWindowPenalty(
                expected_multiplicative_factor=config.expected_multiplicative_factor,
                unexpected_multiplicative_factor=config.unexpected_multiplicative_factor,
                moving_average_window=config.moving_average_window,
            )
        elif (
            config.visitation_penalty_type
            == constants.SIGNED_UNCERTAINTY_WINDOW_PENALTY
        ):
            penalty_computer = signed_uncertainty_window_penalty.SignedUncertaintyWindowPenalty(
                positive_multiplicative_factor=config.positive_multiplicative_factor,
                negative_multiplicative_factor=config.negative_multiplicative_factor,
                action_function=config.action_function,
                moving_average_window=config.moving_average_window,
            )
        else:
            raise ValueError(
                f"Visitation penalty type {config.visitation_penalty_type} not recognised"
            )

        return penalty_computer

    def _setup_epsilon_computer(self, config: ach_config.AchConfig):
        """Setup epsilon computer, for anything adaptive e.g. Doya"""
        if config.schedule == constants.EXPECTED_UNCERTAINTY:
            epsilon_computer = expected_uncertainty_epsilon_computer.ExpectedUncertaintyEpsilonComputer(
                action_function=config.epsilon_action_function,
                minimum_value=config.minimum_value,
            )
        elif config.schedule == constants.UNEXPECTED_UNCERTAINTY:
            epsilon_computer = unexpected_uncertainty_epsilon_computer.UnexpectedUncertaintyEpsilonComputer(
                action_function=config.epsilon_action_function,
                moving_average_window=config.epsilon_moving_average_window,
                minimum_value=config.minimum_value,
            )
        elif config.schedule == constants.CONSTANT:
            epsilon_computer = constant_epsilon_computer.ConstantEpsilonComputer(
                epsilon=config.value
            )
        elif config.schedule == constants.LINEAR_DECAY:
            epsilon_computer = linear_decay_epsilon_computer.LinearDecayEpsilonComputer(
                initial_value=config.initial_value,
                final_value=config.final_value,
                anneal_duration=config.anneal_duration,
                decay_timeframe=config.decay_timeframe,
            )
        return epsilon_computer

    def _setup_lr_scaler(self, config: ach_config.AchConfig):
        """Setup LR scaler"""
        if config.lr_scaler_type == constants.EXPECTED_UNCERTAINTY:
            lr_scaler = expected_uncertainty_learning_rate_scaler.ExpectedUncertaintyLearningRateScaler(
                action_function=config.lr_scaler_action_function,
                multiplicative_factor=config.lr_scaler_multiplicative_factor,
            )
        elif config.lr_scaler_type == constants.HARD_CODED:
            lr_scaler = hard_coded_learning_rate_scaler.HardCodedLearningRateScaler(
                hard_coded_lr_scaling=config.lr_scaling
            )
        return lr_scaler

    @abc.abstractmethod
    def _setup_learner(self, config: ach_config.AchConfig):
        """Instantiate learner specified in configuration."""
        pass
