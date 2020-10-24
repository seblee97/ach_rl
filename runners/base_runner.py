import abc
from typing import List, Tuple

import constants
from environments import base_environment, minigrid
from experiments import ach_config
from utils import logger


class BaseRunner(abc.ABC):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig) -> None:
        self._environment = self._setup_environment(config=config)
        self._learner = self._setup_learner(config=config)
        self._logger = self._setup_logger(config=config)

        self._checkpoint_frequency = config.checkpoint_frequency
        self._test_frequency = config.test_frequency
        self._num_episodes = config.num_episodes

        config.save_configuration(folder_path=config.checkpoint_path)

    def _setup_environment(
        self, config: ach_config.AchConfig
    ) -> base_environment.BaseEnvironment:
        """Initialise environment specified in configuration."""
        if config.environment == constants.Constants.MINIGRID:
            if config.reward_position is not None:
                reward_position = tuple(config.reward_position)
            else:
                reward_position = None
            if config.starting_position is not None:
                agent_starting_position = tuple(config.starting_position)
            else:
                agent_starting_position = None
            environment = minigrid.MiniGrid(
                size=tuple(config.size),
                starting_xy=agent_starting_position,
                reward_xy=reward_position,
                episode_timeout=config.episode_timeout,
            )

        return environment

    def _setup_logger(self, config: ach_config.AchConfig) -> logger.Logger:
        """Initialise logger object to record data from experiment."""
        return logger.Logger(config=config)

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

            train_reward, train_step_count = self._train_episode()

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

            if i % self._test_frequency == 0:
                test_reward, test_step_count = self._test_episode()

                self._logger.write_scalar_df(
                    tag=constants.Constants.TEST_EPISODE_LENGTH,
                    step=i,
                    scalar=test_step_count,
                )
                self._logger.write_scalar_df(
                    tag=constants.Constants.TEST_EPISODE_REWARD,
                    step=i,
                    scalar=test_reward,
                )

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

    def _test_episode(self) -> Tuple[float, int]:
        """Perform 'test' rollout with target policy

        Returns:
            episode_reward: scalar reward accumulated over episode.
            num_steps: number of steps taken for episode.
        """
        episode_reward = 0

        self._environment.reset_environment()
        state = self._environment.agent_position

        while self._environment.active:
            action = self._learner.select_target_action(state)
            reward, state = self._environment.step(action)
            episode_reward += reward

        return episode_reward, self._environment.episode_step_count
