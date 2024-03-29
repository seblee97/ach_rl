from typing import Any
from typing import Dict
from typing import Union

from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class AdaptiveArrivingUncertaintyPenalty(base_visitation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to uncertainty of NEXT state over an ensemble."""

    def __init__(self, multiplicative_factor: Union[float, int], action_function: str):

        self._multiplicative_factor = multiplicative_factor
        self._action_function = action_function

        super().__init__()

    def __call__(self, episode: int, penalty_info: Dict[str, Any]):
        if self._action_function == constants.MAX:
            return (
                self._multiplicative_factor
                * penalty_info[constants.NEXT_STATE_MAX_UNCERTAINTY]
            )
        elif self._action_function == constants.MEAN:
            return (
                self._multiplicative_factor
                * penalty_info[constants.NEXT_STATE_MEAN_UNCERTAINTY]
            )
