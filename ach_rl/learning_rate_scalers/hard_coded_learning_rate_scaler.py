from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from ach_rl.learning_rate_scalers import base_learning_rate_scaler


class HardCodedLearningRateScaler(base_learning_rate_scaler.BaseLearningRateScaler):
    """LR scaling that is constant."""

    def __init__(self, hard_coded_lr_scaling: Union[float, int]):
        self._hard_coded_lr_scaling = hard_coded_lr_scaling

    def __call__(
        self,
        episode: int,
        lr_scaling_info: Dict[str, Any],
        state_label: Optional[str] = None,
    ):
        return self._hard_coded_lr_scaling
