from typing import Any
from typing import Dict
from typing import Union

import constants
import numpy as np
from visitation_penalties import base_visitation_penalty


class LinearDecayPenalty(base_visitation_penalty.BaseVisitationPenaltyComputer):
    """Hard-coded (or deterministic) linear function penalty."""

    def __init__(self, A: Union[int, float], b: Union[int, float]):
        self._A = A
        self._b = b

    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]) -> float:
        return self._A * episode + self._b
