from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from ach_rl import constants
from ach_rl.epsilon_computers import base_epsilon_computer


class PercentileLinearUncertaintySqueezeEpsilonComputer(
    base_epsilon_computer.BaseEpsilonComputer
):
    """Epsilon tuned to uncertainty over an ensemble."""

    def __init__(
        self,
        action_function: str,
        minimum_value: Union[int, float],
        max_value_percentile: int = 100,
    ):
        self._action_function = action_function
        self._minimum_value = minimum_value
        self._max_value_percentile = max_value_percentile

        self._maximum_uncertainty: float = 0
        self._uncertainty_history = []

    def __call__(
        self,
        epsilon_info: Dict[str, Any],
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        if self._action_function == constants.MAX:
            uncertainty_metric = epsilon_info[constants.CURRENT_STATE_MAX_UNCERTAINTY]
        elif self._action_function == constants.MEAN:
            uncertainty_metric = epsilon_info[constants.CURRENT_STATE_MEAN_UNCERTAINTY]

        assert (
            len(uncertainty_metric.shape) == 1 and uncertainty_metric.shape[0] == 1
        ), (
            "uncertainty_metric for epsilon calculation must be scalar."
            f"Dimensions here are {uncertainty_metric.shape}."
        )

        uncertainty = uncertainty_metric.item()
        self._uncertainty_history.append(uncertainty)
        self._maximum_uncertainty = np.percentile(
            a=self._uncertainty_history, q=self._max_value_percentile
        )

        computed_epsilon = self._minimum_value + (
            uncertainty / self._maximum_uncertainty
        ) * (1 - self._minimum_value)

        return min([computed_epsilon, 1])
