from typing import Dict, Union

from config_manager import base_configuration
from config_manager import config_template


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
        assert (
            self.reward_positions is None
            or len(self.reward_positions) == self.num_rewards
        ), (
            "Number of reward positions must match number of rewards,"
            "or reward positions must be set to None for random placement."
        )
        assert (
            len(self.reward_magnitudes) == self.num_rewards
        ), "Number of reward magnitudes must match number of rewards,"

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
