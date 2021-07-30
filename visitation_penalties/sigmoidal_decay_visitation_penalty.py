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
        return self._A  / (1 + np.exp(-self._b * (episode - self._c))) 
