from typing import List
from typing import Tuple

import numpy as np

from learners import base_learner


class EnsembleLearner(base_learner.BaseLearner):
    def __init__(self, learner_ensemble: List[base_learner.BaseLearner]):
        self._learner_ensemble = learner_ensemble

    @property
    def ensemble(self) -> List:
        return self._learner_ensemble

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass
