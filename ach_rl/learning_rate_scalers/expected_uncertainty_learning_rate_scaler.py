from typing import Any
from typing import Dict

from ach_rl import constants
from ach_rl.learning_rate_scalers import base_learning_rate_scaler


class ExpectedUncertaintyLearningRateScaler(
    base_learning_rate_scaler.BaseLearningRateScaler
):
    """LR tuned to 'expected' uncertainty over an ensemble."""

    def __init__(self, action_function: str):
        self._action_function = action_function

    def __call__(self, episode: int, lr_scaling_info: Dict[str, Any]):
        if self._action_function == constants.MAX:
            lr_scaling = lr_scaling_info[constants.CURRENT_STATE_MAX_UNCERTAINTY]
        elif self._action_function == constants.MEAN:
            lr_scaling = lr_scaling_info[constants.CURRENT_STATE_MEAN_UNCERTAINTY]

        return lr_scaling
