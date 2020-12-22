import abc
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

import constants
from curricula import base_curriculum
from curricula import minigrid_curriculum
from environments import atari
from environments import multi_room
from environments import base_environment
from environments import minigrid
from experiments import ach_config
from utils import cycle_counter
from utils import epsilon_schedules
from utils import logger
from utils import plotter
from visitation_penalties import base_visistation_penalty
from visitation_penalties import hard_coded_visitation_penalty
from visitation_penalties import adaptive_uncertainty_visitation_penalty


class BaseRunner(abc.ABC):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig) -> None:
        self._environment = self._setup_environment(config=config)
        self._visitation_penalty = self._setup_visitation_penalty(config=config)
        self._epsilon_function = self._setup_epsilon_function(config=config)
        self._learner = self._setup_learner(config=config)
        self._logger = self._setup_logger(config=config)
        self._plotter = self._setup_plotter(config=config)

        self._apply_curriculum = config.apply_curriculum
        self._print_frequency = config.print_frequency or np.inf
        self._checkpoint_frequency = config.checkpoint_frequency
        self._test_frequency = config.test_frequency
        self._train_log_frequency = config.train_log_frequency
        self._full_test_log_frequency = config.full_test_log_frequency
        self._test_types = config.testing
        self._scalar_logging = config.columns or []
        self._array_logging = config.arrays or []
        self._plot_logging = config.plots or []
        self._num_episodes = config.num_episodes

        config.save_configuration(folder_path=config.checkpoint_path)

    def _setup_environment(
        self, config: ach_config.AchConfig
    ) -> Union[base_curriculum.BaseCurriculum, base_environment.BaseEnvironment]:
        """Initialise environment specified in configuration."""
        environment_args = self._get_environment_args(config=config)

        if config.apply_curriculum:
            curriculum_args = self._get_curriculum_args(config=config)
            curriculum_wrapper = self.get_curriculum_wrapper(config.environment)
            environment = curriculum_wrapper(**environment_args, **curriculum_args)
        else:
            if config.environment == constants.Constants.MINIGRID:
                environment = minigrid.MiniGrid(**environment_args)
            elif config.environment == constants.Constants.ATARI:
                environment = atari.AtariEnv(**environment_args)
            elif config.environment == constants.Constants.MULTIROOM:
                environment = multi_room.MultiRoom(**environment_args)
        return environment

    def _get_environment_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
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
                constants.Constants.PRE_PROCESSING: config.pre_processing,
                constants.Constants.FRAME_STACK: config.frame_stack,
                constants.Constants.FRAME_SKIP: config.frame_skip,
            }
        elif config.environment == constants.Constants.MULTIROOM:
            env_args = {
                constants.Constants.ASCII_MAP_PATH: config.ascii_map_path,
                constants.Constants.EPISODE_TIMEOUT: config.episode_timeout,
            }
        return env_args

    def _get_curriculum_args(self, config: ach_config.AchConfig) -> Dict[str, Any]:
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

    def _setup_logger(self, config: ach_config.AchConfig) -> logger.Logger:
        """Initialise logger object to record data from experiment."""
        return logger.Logger(config=config)

    def _setup_plotter(self, config: ach_config.AchConfig) -> plotter.Plotter:
        """Initialise plotter object for use in post-processing run."""
        return plotter.Plotter(
            save_folder=config.checkpoint_path,
            logfile_path=config.logfile_path,
            plot_tags=config.plot_tags,
            smoothing=config.smoothing,
        )

    def _setup_visitation_penalty(
        self, config: ach_config.AchConfig
    ) -> base_visistation_penalty.BaseVisitationPenalty:
        """Initialise object to act as visitation penalty."""
        if config.visitation_penalty_type == constants.Constants.HARD_CODED:
            visitation_penalty = hard_coded_visitation_penalty.HardCodedPenalty(
                config=config
            )
        elif config.visitation_penalty_type == constants.Constants.ADAPTIVE_UNCERTAINTY:
            visitation_penalty = (
                adaptive_uncertainty_visitation_penalty.AdaptiveUncertaintyPenalty(
                    config=config
                )
            )
        else:
            raise ValueError(
                f"Visitation penalty type {config.visitation_penalty} not recognised"
            )
        return visitation_penalty

    def _setup_epsilon_function(self, config: ach_config.AchConfig):
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
        pass

    def train(self) -> None:
        """Perform training (and validation) on given number of episodes."""
        train_reward: float
        train_step_count: float

        print("Starting Training...")

        for i in range(self._num_episodes):

            if i % self._print_frequency == 0:
                print(f"Episode {i}/{self._num_episodes}: ")
                if i != 0:
                    print(f"    Latest Train Reward: {train_reward}")
                    print(f"    Latest Train Length: {train_step_count}")

            if i % self._checkpoint_frequency == 0:
                self._logger.checkpoint_df()

            if i % self._train_log_frequency == 0:
                self._pre_episode_log(i)

            if i % self._test_frequency == 0:
                self._test_episode(episode=i)

            if self._apply_curriculum:
                if i == self._environment.next_transition_episode:
                    next(self._environment)

            train_reward, train_step_count = self._train_episode(episode=i)

            self._logger.write_scalar_df(
                tag=constants.Constants.TRAIN_EPISODE_LENGTH,
                step=i,
                scalar=train_step_count,
            )
            self._logger.write_scalar_df(
                tag=constants.Constants.TRAIN_EPISODE_REWARD,
                step=i,
                scalar=train_reward,
            )

        if constants.Constants.VISITATION_COUNT_HEATMAP in self._plot_logging:
            self._logger.plot_array_data(
                name=constants.Constants.VISITATION_COUNT_HEATMAP,
                data=self._environment.visitation_counts,
            )

        self._logger.checkpoint_df()

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

    def _test_episode(self, episode: int) -> None:
        """Perform test rollouts, once with target policy,
        and---if specified---once with non-repeat target.

        Args:
            episode: current episode number.
        """

        self._learner.eval()

        with torch.no_grad():
            if constants.Constants.GREEDY in self._test_types:
                self._greedy_test_episode(episode=episode)
            if constants.Constants.NO_REP in self._test_types:
                self._non_repeat_test_episode(episode=episode)

        self._learner.train()

    def _greedy_test_episode(self, episode: int) -> None:
        """Perform 'test' rollout with target policy

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        state = self._environment.reset_environment(train=False)

        while self._environment.active:
            action = self._learner.select_target_action(state)
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._logger.write_scalar_df(
            tag=constants.Constants.TEST_EPISODE_LENGTH,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._logger.write_scalar_df(
            tag=constants.Constants.TEST_EPISODE_REWARD,
            step=episode,
            scalar=episode_reward,
        )

        if episode % self._full_test_log_frequency == 0 and episode != 0:
            if constants.Constants.INDIVIDUAL_TEST_RUN in self._array_logging:
                self._logger.write_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )
            if constants.Constants.INDIVIDUAL_TEST_RUN in self._plot_logging:
                self._logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def _non_repeat_test_episode(self, episode: int) -> None:
        """Perform 'test' rollout with target policy, except actions are never
        repeated in same state. Instead next best action is chosen.
        This is to break loops in test rollouts.

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        states_visited = {}

        state = self._environment.reset_environment(train=False)
        states_visited[state] = []

        while self._environment.active:
            action = self._learner.non_repeat_greedy_action(
                state, excluded_actions=states_visited.get(state, [])
            )
            if state not in states_visited:
                states_visited[state] = []
            states_visited[state].append(action)
            reward, state = self._environment.step(action)
            episode_reward += reward

        self._logger.write_scalar_df(
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_LENGTH,
            step=episode,
            scalar=self._environment.episode_step_count,
        )
        self._logger.write_scalar_df(
            tag=constants.Constants.NO_REPEAT_TEST_EPISODE_REWARD,
            step=episode,
            scalar=episode_reward,
        )

        if episode % self._full_test_log_frequency == 0:
            if constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN in self._array_logging:
                self._logger.write_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )
            if constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN in self._plot_logging:
                self._logger.plot_array_data(
                    name=f"{constants.Constants.INDIVIDUAL_NO_REP_TEST_RUN}_{episode}",
                    data=self._environment.plot_episode_history(),
                )

    def post_process(self) -> None:
        """Solidify any data and make plots."""
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
        if constants.Constants.VALUE_FUNCTION in self._plot_logging:
            self._plotter.plot_value_function(
                grid_size=self._grid_size,
                state_action_values=self._learner.state_action_values,
            )
