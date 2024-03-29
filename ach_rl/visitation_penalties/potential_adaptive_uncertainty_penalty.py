from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class PotentialAdaptiveUncertaintyPenalty(
    base_visitation_penalty.BaseVisitationPenalty
):
    """Visitation penalty tuned to uncertainty,
    in the potential-based reward shaping formulation of Ng, Harada, Russell (1999)."""

    def __init__(
        self,
        gamma: float,
        multiplicative_factor: Union[int, float],
        pre_action_function: str,
        post_action_function: str,
    ):
        self._state_action_values: List[Dict[Tuple[int], float]]

        self._gamma = gamma
        self._multiplicative_factor = multiplicative_factor
        self._pre_action_function = pre_action_function
        self._post_action_function = post_action_function

    def __call__(self, episode: int, penalty_info: Dict[str, Any]):
        if self._pre_action_function == constants.MAX:
            pre_uncertainty = penalty_info[constants.CURRENT_STATE_MAX_UNCERTAINTY]
        elif self._pre_action_function == constants.MEAN:
            pre_uncertainty = penalty_info[constants.CURRENT_STATE_MEAN_UNCERTAINTY]
        elif self._pre_action_function == constants.SELECT:
            pre_uncertainty = penalty_info[constants.CURRENT_STATE_SELECT_UNCERTAINTY]
        if self._post_action_function == constants.MAX:
            post_uncertainty = penalty_info[constants.NEXT_STATE_MAX_UNCERTAINTY]
        elif self._post_action_function == constants.MEAN:
            post_uncertainty = penalty_info[constants.NEXT_STATE_MEAN_UNCERTAINTY]

        penalty = self._multiplicative_factor * (
            self._gamma * post_uncertainty - pre_uncertainty
        )
        return penalty
