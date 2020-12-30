from typing import Dict
from typing import Union

from config_manager import base_configuration
from config_manager import config_template

import constants


class AchConfig(base_configuration.BaseConfiguration):
    """Ach RL Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(
        self, config: Union[str, Dict], template: config_template.Template
    ) -> None:
        super().__init__(configuration=config, template=template)

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

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
