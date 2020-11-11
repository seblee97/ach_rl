import abc
from typing import List
from typing import Tuple

import constants
from environments import base_environment
from environments import minigrid
from experiments import ach_config
from utils import logger
from utils import plotter
from utils import cycle_counter


class BaseRunner(abc.ABC):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig) -> None:
        self._environment = self._setup_environment(config=config)
        self._learner = self._setup_learner(config=config)
        self._logger = self._setup_logger(config=config)
        self._plotter = self._setup_plotter(config=config)

        self._checkpoint_frequency = config.checkpoint_frequency
        self._test_frequency = config.test_frequency
        self._train_log_frequency = config.train_log_frequency
        self._full_test_log_frequency = config.full_test_log_frequency
        self._scalar_logging = config.columns or []
        self._array_logging = config.arrays or []
        self._plot_logging = config.plots or []
        self._num_episodes = config.num_episodes
        self._grid_size = tuple(config.size)

        config.save_configuration(folder_path=config.checkpoint_path)

    def _setup_environment(
        self, config: ach_config.AchConfig
    ) -> base_environment.BaseEnvironment:
        """Initialise environment specified in configuration."""
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
            environment = minigrid.MiniGrid(
                size=tuple(config.size),
                num_rewards=config.num_rewards,
                reward_magnitudes=config.reward_magnitudes,
                starting_xy=agent_starting_position,
                reward_xy=reward_positions,
                repeat_rewards=config.repeat_rewards,
                episode_timeout=config.episode_timeout,
            )

        return environment

    def _setup_logger(self, config: ach_config.AchConfig) -> logger.Logger:
        """Initialise logger object to record data from experiment."""
        return logger.Logger(config=config)

    def _setup_plotter(self, config: ach_config.AchConfig) -> plotter.Plotter:
        """Initialise plotter object for use in post-processing run."""
        return plotter.Plotter(
            save_folder=config.checkpoint_path,
            logfile_path=config.logfile_path,
            plot_tags=config.plot_tags,
        )

    @abc.abstractmethod
    def _setup_learner(self, config: ach_config.AchConfig):
        """Instantiate learner specified in configuration."""
        pass

    def train(self) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Perform training (and validation) on given number of episodes."""
        train_episode_rewards = []
        train_episode_lengths = []

        test_episode_rewards = []
        test_episode_lengths = []

        for i in range(self._num_episodes):

            if i % self._checkpoint_frequency == 0:
                self._logger.checkpoint_df()

            if i % self._test_frequency == 0:
                self._test_episode(episode=i)

            train_reward, train_step_count = self._train_episode()

            if i % self._train_log_frequency == 0:
                if constants.Constants.INDIVIDUAL_TRAIN_RUN in self._plot_logging:
                    self._logger.plot_array_data(
                        name=f"{constants.Constants.INDIVIDUAL_TRAIN_RUN}_{i}",
                        data=self._environment.plot_episode_history(),
                    )
                if constants.Constants.VALUE_FUNCTION in self._plot_logging:
                    self._plotter.plot_value_function(
                        grid_size=self._grid_size,
                        state_action_values=self._learner.state_action_values,
                        extra_tag=f"{i}_",
                    )

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

            if constants.Constants.CYCLE_COUNT in self._scalar_logging:
                num_cycles = cycle_counter.evaluate_loops_on_value_function(
                    size=self._grid_size,
                    state_action_values=self._learner.state_action_values,
                )
                self._logger.write_scalar_df(
                    tag=constants.Constants.CYCLE_COUNT,
                    step=i,
                    scalar=num_cycles,
                )

        if constants.Constants.VISITATION_COUNT_HEATMAP in self._plot_logging:
            self._logger.plot_array_data(
                name=constants.Constants.VISITATION_COUNT_HEATMAP,
                data=self._environment.visitation_counts,
            )

        self._logger.checkpoint_df()

        return (
            train_episode_rewards,
            train_episode_lengths,
            test_episode_rewards,
            test_episode_lengths,
        )

    @abc.abstractmethod
    def _train_episode(self):
        """Perform single training loop."""
        pass

    def _test_episode(self, episode: int) -> Tuple[float, int]:
        """Perform test rollouts, once with target policy,
        and---if specified---once with non-repeat target.

        Args:
            episode: current episode number.
        """
        self._greedy_test_episode(episode=episode)
        self._non_repeat_test_episode(episode=episode)

    def _greedy_test_episode(self, episode: int) -> None:
        """Perform 'test' rollout with target policy

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        self._environment.reset_environment(train=False)
        state = self._environment.agent_position

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

        if episode % self._full_test_log_frequency == 0:
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

        self._environment.reset_environment(train=False)
        state = self._environment.agent_position
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
        self._plotter.plot_value_function(
            grid_size=self._grid_size,
            state_action_values=self._learner.state_action_values,
        )
