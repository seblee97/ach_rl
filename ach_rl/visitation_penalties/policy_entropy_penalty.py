from typing import Any
from typing import Dict
from typing import Union

from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class PolicyEntropyPenalty(base_visitation_penalty.BaseVisitationPenalty):
    """Visitation penalty tuned to policy entropy over an ensemble."""

    def __init__(self, multiplicative_factor: Union[float, int]):

        self._multiplicative_factor = multiplicative_factor

    def __call__(self, episode: int, penalty_info: Dict[str, Any]):
        return (
            self._multiplicative_factor
            * penalty_info[constants.CURRENT_STATE_POLICY_ENTROPY]
        )
