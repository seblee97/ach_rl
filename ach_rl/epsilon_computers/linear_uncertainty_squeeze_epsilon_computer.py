from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from ach_rl import constants
from ach_rl.epsilon_computers import base_epsilon_computer


class LinearUncertaintySqueezeEpsilonComputer(
    base_epsilon_computer.BaseEpsilonComputer
):
    """Epsilon tuned to uncertainty over an ensemble."""

    def __init__(
        self,
        action_function: str,
        minimum_value: Union[int, float],
    ):
        self._action_function = action_function
        self._minimum_value = minimum_value

        self._maximum_uncertainty: float = 0

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
        # update max uncertainty if needed
        if uncertainty > self._maximum_uncertainty:
            self._maximum_uncertainty = uncertainty

        computed_epsilon = self._minimum_value + (
            uncertainty / self._maximum_uncertainty
        ) * (1 - self._minimum_value)

        return computed_epsilon
