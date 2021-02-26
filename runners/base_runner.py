import abc
import os
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import constants
import numpy as np
import torch
from curricula import base_curriculum
from curricula import minigrid_curriculum
from environments import atari
from environments import base_environment
from environments import minigrid
from environments import multi_room
from environments import wrapper_atari
from experiments import ach_config
from utils import data_logger
from utils import decorators
from utils import epsilon_schedules
from utils import experiment_logger
from utils import plotter
from visitation_penalties import \
    adaptive_arriving_uncertainty_visitation_penalty
from visitation_penalties import adaptive_uncertainty_visitation_penalty
from visitation_penalties import base_visitation_penalty
from visitation_penalties import hard_coded_visitation_penalty
from visitation_penalties import network_visitation_penalty
from visitation_penalties import policy_entropy_penalty
from visitation_penalties import potential_adaptive_uncertainty_penalty
from visitation_penalties import potential_policy_entropy_penalty
from visitation_penalties import tabular_visitation_penalty


class BaseRunner(abc.ABC):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig) -> None:
        self._environment = self._setup_environment(config=config)
        if config.visitation_penalty_type is None:
            self._visitation_penalty = None
        else:
            self._visitation_penalty = self._setup_visitation_penalty(config=config)
        self._epsilon_function = self._setup_epsilon_function(config=config)
        self._learner = self._setup_learner(config=config)
        self._logger = experiment_logger.get_logger(
            experiment_path=config.checkpoint_path, name=__name__
        )
        self._data_logger = self._setup_data_logger(config=config)

        self._checkpoint_path = config.checkpoint_path
        self._apply_curriculum = config.apply_curriculum
        self._print_frequency = config.print_frequency or np.inf
        self._checkpoint_frequency = config.checkpoint_frequency
        self._test_frequency = config.test_frequency
        self._test_types = config.testing
        self._num_episodes = config.num_episodes

        self._array_logging = self._setup_logging_frequencies(config.arrays)
        self._array_folder_path = os.path.join(
            self._checkpoint_path, constants.Constants.ARRAYS
        )
        os.makedirs(name=self._array_folder_path, exist_ok=True)
        for tag in self._array_logging.keys():
            os.makedirs(name=os.path.join(self._array_folder_path, tag), exist_ok=True)
        self._scalar_logging = self._setup_logging_frequencies(config.scalars)
        self._visualisations = self._setup_logging_frequencies(config.visualisations)
        self._visualisations_folder_path = os.path.join(
            self._checkpoint_path, constants.Constants.VISUALISATIONS
        )
        os.makedirs(name=self._visualisations_folder_path, exist_ok=True)
        self._post_visualisations = config.post_visualisations
        self._plotter = self._setup_plotter(config=config)

        config.save_configuration(folder_path=config.checkpoint_path)

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
                    logging_frequencies[i[0][0]] = i[1]
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

        if config.apply_curriculum:
            curriculum_args = self._get_curriculum_args(config=config)
            curriculum_wrapper = self.get_curriculum_wrapper(config.environment)
            environment = curriculum_wrapper(**environment_args, **curriculum_args)
        else:
            if config.environment == constants.Constants.MINIGRID:
                environment = minigrid.MiniGrid(**environment_args)
            elif config.environment == constants.Constants.ATARI:
                if config.implementation == constants.Constants.WRAPPER:
                    environment = wrapper_atari.AtariEnv(**environment_args)
                elif config.implementation == constants.Constants.FUNCTIONAL:
                    environment = atari.AtariEnv(**environment_args)
            elif config.environment == constants.Constants.MULTIROOM:
                environment = multi_room.MultiRoom(**environment_args)

        return environment

    def _get_environment_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
        """Get arguments needed to pass to environment."""
        if config.environment == constants.Constants.MINIGRID:
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
                constants.Constants.SIZE: tuple(config.size),
                constants.Constants.NUM_REWARDS: config.num_rewards,
                constants.Constants.REWARD_MAGNITUDES: config.reward_magnitudes,
                constants.Constants.STARTING_XY: agent_starting_position,
                constants.Constants.REWARD_XY: reward_positions,
                constants.Constants.REPEAT_REWARDS: config.repeat_rewards,
                constants.Constants.EPISODE_TIMEOUT: config.episode_timeout,
            }
        elif config.environment == constants.Constants.ATARI:
            env_args = {
                constants.Constants.ATARI_ENV_NAME: config.atari_env_name,
                constants.Constants.EPISODE_TIMEOUT: config.episode_timeout,
                constants.Constants.FRAME_STACK: config.frame_stack,
                constants.Constants.FRAME_SKIP: config.frame_skip,
            }
            if config.implementation == constants.Constants.FUNCTIONAL:
                env_args[constants.Constants.PRE_PROCESSING] = config.pre_processing
        elif config.environment == constants.Constants.MULTIROOM:
            env_args = {
                constants.Constants.ASCII_MAP_PATH: os.path.join(
                    config.run_path, config.ascii_map_path
                ),
                constants.Constants.EPISODE_TIMEOUT: config.episode_timeout,
            }
        return env_args

    def _get_curriculum_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
        """Get arguments needed to pass to environment curriculum wrapper."""
        if config.environment == constants.Constants.MINIGRID:
            curriculum_args = {
                constants.Constants.TRANSITION_EPISODES: config.transition_episodes,
                constants.Constants.ENVIRONMENT_CHANGES: config.environment_changes,
            }
        return curriculum_args

    @staticmethod
    def get_curriculum_wrapper(environment: str) -> base_curriculum.BaseCurriculum:
        """Get relevant wrapper for environment to add curriculum features."""
        if environment == constants.Constants.MINIGRID:
            return minigrid_curriculum.MinigridCurriculum

    def _setup_data_logger(
        self, config: ach_config.AchConfig
    ) -> data_logger.DataLogger:
        """Initialise logger object to record data from experiment."""
        return data_logger.DataLogger(config=config)

    def _setup_plotter(self, config: ach_config.AchConfig) -> plotter.Plotter:
        """Initialise plotter object for use in post-processing run."""
        return plotter.Plotter(
            save_folder=config.checkpoint_path,
            logfile_path=config.logfile_path,
            smoothing=config.smoothing,
        )

    def _setup_visitation_penalty(
        self, config: ach_config.AchConfig
    ) -> base_visitation_penalty.BaseVisitationPenalty:
        """Initialise object to act as visitation penalty."""
        if config.visitation_penalty_type == constants.Constants.HARD_CODED:
            penalty_computer = hard_coded_visitation_penalty.HardCodedPenalty(
                hard_coded_penalties=config.vp_schedule
            )
        elif config.visitation_penalty_type == constants.Constants.ADAPTIVE_UNCERTAINTY:
            penalty_computer = (
                adaptive_uncertainty_visitation_penalty.AdaptiveUncertaintyPenalty(
                    multiplicative_factor=config.multiplicative_factor,
                    action_function=config.action_function,
                )
            )
        elif (
            config.visitation_penalty_type
            == constants.Constants.ADAPTIVE_ARRIVING_UNCERTAINTY
        ):
            penalty_computer = adaptive_arriving_uncertainty_visitation_penalty.AdaptiveArrivingUncertaintyPenalty(
                multiplicative_factor=config.multiplicative_factor,
                action_function=config.action_function,
            )
        elif (
            config.visitation_penalty_type
            == constants.Constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY
        ):
            penalty_computer = potential_adaptive_uncertainty_penalty.PotentialAdaptiveUncertaintyPenalty(
                gamma=config.discount_factor,
                multiplicative_factor=config.multiplicative_factor,
                pre_action_function=config.pre_action_function,
                post_action_function=config.post_action_function,
            )
        elif (
            config.visitation_penalty_type == constants.Constants.POLICY_ENTROPY_PENALTY
        ):
            penalty_computer = policy_entropy_penalty.PolicyEntropyPenalty(
                multiplicative_factor=config.multiplicative_factor
            )
        elif (
            config.visitation_penalty_type
            == constants.Constants.POTENTIAL_BASED_POLICY_ENTROPY_PENALTY
        ):
            penalty_computer = (
                potential_policy_entropy_penalty.PotentialPolicyEntropyPenalty(
                    gamma=config.discount_factor,
                    multiplicative_factor=config.multiplicative_factor,
                )
            )
        else:
            raise ValueError(
                f"Visitation penalty type {config.visitation_penalty_type} not recognised"
            )

        if config.type in [
            constants.Constants.BOOTSTRAPPED_ENSEMBLE_DQN,
            constants.Constants.VANILLA_DQN,
            constants.Constants.INDEPENDENT_ENSEMBLE_DQN,
        ]:
            if config.shaping_implementation == constants.Constants.TRAIN_Q_NETWORK:
                use_target_network = False
            elif (
                config.shaping_implementation
                == constants.Constants.TRAIN_TARGET_NETWORK
            ):
                use_target_network = True
            visitation_penalty = network_visitation_penalty.NetworkVisitationPenalty(
                penalty_computer=penalty_computer,
                use_target_network=use_target_network,
            )
        elif config.type in [
            constants.Constants.Q_LEARNING,
            constants.Constants.SARSA_LAMBDA,
            constants.Constants.ENSEMBLE_Q_LEARNING,
        ]:
            visitation_penalty = tabular_visitation_penalty.TabularVisitationPenalty(
                penalty_computer=penalty_computer
            )
        else:
            raise ValueError(
                f"Learner type {config.type} has not been grouped as tabular or network, "
                "so the correct visitation penalty class type cannot be determined."
            )

        return visitation_penalty

    def _setup_epsilon_function(self, config: ach_config.AchConfig):
        """Setup epsilon function."""
        if config.schedule == constants.Constants.CONSTANT:
            epsilon_function = epsilon_schedules.ConstantEpsilon(value=config.value)
        elif config.schedule == constants.Constants.LINEAR_DECAY:
            epsilon_function = epsilon_schedules.LinearDecayEpsilon(
                initial_value=config.initial_value,
                final_value=config.final_value,
                anneal_duration=config.anneal_duration,
            )
        return epsilon_function

    @abc.abstractmethod
    def _setup_learner(self, config: ach_config.AchConfig):
        """Instantiate learner specified in configuration."""
        pass

    @abc.abstractmethod
    def _pre_episode_log(self, episode: int):
        """Define logging functionality for prior to each training episode."""
        pass

    def _array_log_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not array tag should be logged at this episode."""
        if tag in self._array_logging:
            if episode % self._array_logging[tag] == 0:
                return True
        return False

    def _scalar_log_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not scalar tag should be logged at this episode."""
        if tag in self._scalar_logging:
            if episode % self._scalar_logging[tag] == 0:
                return True
        return False

    def _visualisation_iteration(self, tag: str, episode: int) -> bool:
        """Whether or not visualistation should be produced at this episode."""
        if tag in self._visualisations:
            if episode % self._visualisations[tag] == 0:
                return True
        return False

    def _write_scalar(
        self,
        tag: str,
        episode: int,
        scalar: Union[float, int],
        df_tag: Optional[str] = None,
    ):
        """If specified, log scalar."""
        if self._scalar_log_iteration(tag=tag, episode=episode):
            df_tag = df_tag or tag
            self._data_logger.write_scalar(tag=df_tag, step=episode, scalar=scalar)

    def _write_array(
        self,
        tag: str,
        episode: int,
        array: np.ndarray,
    ) -> None:
        if self._array_log_iteration(tag=tag, episode=episode):
            file_path = os.path.join(self._array_folder_path, tag, f"{tag}_{episode}")
            np.save(file_path, array)

    def train(self) -> None:
        """Perform training (and validation) on given number of episodes."""
        train_reward: float = 0
        train_step_count: float = np.inf
        episode_duration: float = 0

        self._logger.info("Starting Training...")

        for i in range(self._num_episodes):

            episode_start_time = time.time()

            if i % self._print_frequency == 0:
                if i != 0:
                    self._logger.info(f"    Latest Episode Duration {episode_duration}")
                    self._logger.info(f"    Latest Train Reward: {train_reward}")
                    self._logger.info(f"    Latest Train Length: {train_step_count}")
                self._logger.info(f"Episode {i + 1}/{self._num_episodes}: ")

            if i % self._checkpoint_frequency == 0 and i != 0:
                self._data_logger.checkpoint()

            self._pre_episode_log(i)
            self._test_episode(episode=i)

            if self._apply_curriculum:
                if i == self._environment.next_transition_episode:
                    next(self._environment)

            train_reward, train_step_count = self._train_episode(episode=i)

            self._write_scalar(
                tag=constants.Constants.TRAIN_EPISODE_LENGTH,
                episode=i,
                scalar=train_step_count,
            )
            self._write_scalar(
                tag=constants.Constants.TRAIN_EPISODE_REWARD,
                episode=i,
                scalar=train_reward,
            )

            episode_duration = time.time() - episode_start_time

        if constants.Constants.VISITATION_COUNT_HEATMAP in self._visualisations:
            self._data_logger.plot_array_data(
                name=constants.Constants.VISITATION_COUNT_HEATMAP,
                data=self._environment.visitation_counts,
            )

        self._data_logger.checkpoint()

    @abc.abstractmethod
    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        pass

    @abc.abstractmethod
    def _run_specific_tests(self, episode: int):
        """Implement specific test runs for each runner."""
        pass

    def _test_episode(self, episode: int) -> None:
        """Perform test rollouts, once with target policy,
        and---if specified---once with non-repeat target.

        Args:
            episode: current episode number.
        """
        if episode % self._test_frequency == 0:
            self._learner.eval()

            with torch.no_grad():
                if self._test_types is not None:
                    if constants.Constants.GREEDY in self._test_types:
                        self._greedy_test_episode(episode=episode)
                    if constants.Constants.NO_REP in self._test_types:
                        self._non_repeat_test_episode(episode=episode)

                self._run_specific_tests(episode=episode)

            self._learner.train()

    def _greedy_test_episode(
        self,
        episode: int,
        action_selection_method: Optional[Callable] = None,
        action_selection_method_args: Optional[Dict] = {},
        tag_: Optional[str] = "",
    ) -> None:
        """Perform 'test' rollout with target policy

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        action_selection_method = (
            action_selection_method or self._learner.select_target_action
        )

        episode_reward = 0

        state = self._environment.reset_environment(train=False)

        while self._environment.active:
            action = action_selection_method(
                state=state, **action_selection_method_args
            )
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._data_logger.write_scalar(
            tag=constants.Constants.TEST_EPISODE_LENGTH + tag_,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._data_logger.write_scalar(
            tag=constants.Constants.TEST_EPISODE_REWARD + tag_,
            step=episode,
            scalar=episode_reward,
        )

        if episode != 0:
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_TEST_RUN + tag_, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TEST_RUN + tag_}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def _non_repeat_test_episode(
        self,
        episode: int,
        action_selection_method: Optional[Callable] = None,
        action_selection_method_args: Optional[Dict] = {},
        tag_: Optional[str] = "",
    ) -> None:
        """Perform 'test' rollout with target policy, except actions are never
        repeated in same state. Instead next best action is chosen.
        This is to break loops in test rollouts.

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        action_selection_method = (
            action_selection_method or self._learner.select_target_action
        )

        episode_reward = 0

        states_visited = {}

        state = self._environment.reset_environment(train=False)
        states_visited[state] = []

        while self._environment.active:
            action = action_selection_method(
                state=state,
                excluded_actions=states_visited.get(state, []),
                **action_selection_method_args,
            )
            if state not in states_visited:
                states_visited[state] = []
            states_visited[state].append(action)
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._data_logger.write_scalar_df(
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_LENGTH,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._data_logger.write_scalar_df(
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_REWARD,
            step=episode,
            scalar=episode_reward,
        )

        if episode != 0:
            if self._visualisation_iteration(
                constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN, episode
            ):
                self._data_logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def post_process(self) -> None:
        """Solidify any data and make plots."""
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
        self._post_visualisation()

    @abc.abstractmethod
    def _post_visualisation(self):
        """Create visualisations at end of training."""
        pass
