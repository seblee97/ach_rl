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

    def _compute_lr_scaling(self, episode: int, lr_info: Dict[str, Any]):
        if self._action_function == constants.Constants.MAX:
            lr_scaling = lr_info[constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY]
        elif self._action_function == constants.Constants.MEAN:
            lr_scaling = lr_info[constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY]

        return lr_scaling
