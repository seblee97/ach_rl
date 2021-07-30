from typing import Any
from typing import Dict
from typing import Union

import constants
import numpy as np
from visitation_penalties import base_visitation_penalty


class ExponentialDecayPenalty(base_visitation_penalty.BaseVisitationPenaltyComputer):
    """Hard-coded (or deterministic) exponential penalty."""

    def __init__(self, A: Union[int, float], b: Union[int, float], c: Union[int, float]):
        # default 0
        self._A = A
        self._b = b
        self._c = c

    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]) -> float:
        return self._A * self._b ** (self._c * episode)
