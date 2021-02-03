from typing import Any
from typing import Dict
from typing import Union

import constants
from visitation_penalties import base_visistation_penalty


class PolicyEntropyPenalty(base_visistation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to policy entropy over an ensemble."""

    def __init__(self, multiplicative_factor: Union[float, int]):

        self._multiplicative_factor = multiplicative_factor

        super().__init__()

    def _compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        return (
            self._multiplicative_factor
            * penalty_info[constants.Constants.CURRENT_STATE_POLICY_ENTROPY]
        )
