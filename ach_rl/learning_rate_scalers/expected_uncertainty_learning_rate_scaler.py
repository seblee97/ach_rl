from typing import Any
from typing import Dict
from typing import Optional

from ach_rl import constants
from ach_rl.learning_rate_scalers import base_learning_rate_scaler


class ExpectedUncertaintyLearningRateScaler(
    base_learning_rate_scaler.BaseLearningRateScaler
):
    """LR tuned to 'expected' uncertainty over an ensemble."""

    def __init__(self, action_function: str, multiplicative_factor: float):
        self._action_function = action_function
        self._multiplicative_factor = multiplicative_factor

    def __call__(
        self,
        episode: int,
        lr_scaling_info: Dict[str, Any],
        state_label: Optional[str] = None,
    ):
        if self._action_function == constants.MAX:
            lr_scaling = lr_scaling_info[
                f"{state_label}_{constants.STATE_MAX_UNCERTAINTY}"
            ]
        elif self._action_function == constants.MEAN:
            lr_scaling = lr_scaling_info[
                f"{state_label}_{constants.STATE_MEAN_UNCERTAINTY}"
            ]

        return self._multiplicative_factor * lr_scaling
