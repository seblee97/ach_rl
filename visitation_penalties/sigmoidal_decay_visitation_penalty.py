from typing import Any
from typing import Dict
from typing import Union

import constants
import numpy as np
from visitation_penalties import base_visitation_penalty


class SigmoidalDecayPenalty(base_visitation_penalty.BaseVisitationPenaltyComputer):
    """Hard-coded (or deterministic) sigmoidal penalty."""

    def __init__(self, A: Union[int, float], b: Union[int, float], c: Union[int, float]):
        self._A = A
        self._b = b
        self._c = c

    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]) -> float:
    	# Logistic function: f(x) = A / (1 + e^(-b(x-c)))
        current_penalty = self._A  / (1 + np.exp(-self._b * (episode - self._c))) 

        reference_measure = penalty_info[
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY
        ]
        if isinstance(reference_measure, float):
            return current_penalty
        elif isinstance(reference_measure, np.ndarray):
            batch_dimension = reference_measure.shape[0]
            return current_penalty * np.ones(batch_dimension)
