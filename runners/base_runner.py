import abc
import ach_config

from environments import base_environment
from environments import minigrid
import constants


class BaseRunner(abc.ABC):
    """Base class for runners (orchestrating training, testing, logging etc.)."""

    def __init__(self, config: ach_config.AchConfig) -> None:
        self._environment = self._setup_environment(config=config)
        self._learner = self._setup_learner(config=config)

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

    @abc.abstractmethod
    def _setup_learner(self, config: ach_config.AchConfig):
        """Instantiate learner specified in configuration."""
        pass

    @abc.abstractmethod
    def train(self, num_episodes: int):
        """Perform training loop for given number of episodes."""
        pass
