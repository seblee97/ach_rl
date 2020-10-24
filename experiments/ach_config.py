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
        pass
