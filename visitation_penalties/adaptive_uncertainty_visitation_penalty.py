from typing import Any
from typing import Dict
from typing import Union

import constants
from visitation_penalties import base_visistation_penalty


class AdaptiveUncertaintyPenalty(base_visistation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to uncertainty over an ensemble."""

    def __init__(self, multiplicative_factor: Union[float, int], action_function: str):

        self._multiplicative_factor = multiplicative_factor
        self._action_function = action_function

        super().__init__()

    def _compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        if self._action_function == constants.Constants.MAX:
            return (
                self._multiplicative_factor
                * penalty_info[constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY]
            )
        elif self._action_function == constants.Constants.MEAN:
            return (
                self._multiplicative_factor
                * penalty_info[constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY]
            )
        elif self._action_function == constants.Constants.SELECT:
            return (
                self._multiplicative_factor
                * penalty_info[constants.Constants.CURRENT_STATE_SELECT_UNCERTAINTY]
            )
