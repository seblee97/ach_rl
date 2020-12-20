from typing import List
from typing import Tuple

import numpy as np

from learners import base_learner


class EnsembleLearner(base_learner.BaseLearner):
    def __init__(self, learner_ensemble: List[base_learner.BaseLearner]):
        self._learner_ensemble = learner_ensemble

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def select_target_action(self, state: Tuple[int, int]):
        # get original (np array) state action values for each learner.

        import pdb

        pdb.set_trace()
        all_state_action_values = [
            learner._state_action_values for learner in self._learner_ensemble
        ]
        average_state_action_values = np.mean(all_state_action_values)
        raise NotImplementedError
