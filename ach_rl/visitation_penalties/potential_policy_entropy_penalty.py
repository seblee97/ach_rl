from typing import Any
from typing import Dict
from typing import Union

from ach_rl import constants
from ach_rl.visitation_penalties import base_visitation_penalty


class PotentialPolicyEntropyPenalty(base_visitation_penalty.BaseVisitationPenalty):
    """Visitation penalty based on potential-based form of policy entropy over an ensemble."""

    def __init__(self, gamma: float, multiplicative_factor: Union[float, int]):
        self._gamma = gamma
        self._multiplicative_factor = multiplicative_factor

    def __call__(self, episode: int, penalty_info: Dict[str, Any]):
        current_state_policy_entropy = penalty_info[
            constants.CURRENT_STATE_POLICY_ENTROPY
        ]
        next_state_policy_entropy = penalty_info[constants.NEXT_STATE_POLICY_ENTROPY]
        return self._multiplicative_factor * (
            self._gamma * next_state_policy_entropy - current_state_policy_entropy
        )
