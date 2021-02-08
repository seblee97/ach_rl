import abc
from typing import Any
from typing import List

import numpy as np
from learners import base_learner


class EnsembleLearner(base_learner.BaseLearner):
    """Learner consisting of ensemble."""
    def __init__(self, learner_ensemble: List[base_learner.BaseLearner]):
        """Class constructor.

        Args:
            learner_ensemble: list of learners forming ensemble.
        """
        self._learner_ensemble = learner_ensemble

    @property
    def state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        averaged_values = {}

        states = all_state_action_values[0].keys()
        for state in states:
            state_values = [
                values[state] for values in all_state_action_values
            ]
            mean_state_values = np.mean(state_values, axis=0)
            averaged_values[state] = mean_state_values
        return averaged_values

    @property
    def individual_learner_state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]
        return all_state_action_values

    @property
    def state_action_values_std(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        values_std = {}

        states = all_state_action_values[0].keys()
        for state in states:
            state_values = [
                values[state] for values in all_state_action_values
            ]
            state_values_std = np.std(state_values, axis=0)
            values_std[state] = state_values_std
        return values_std

    @property
    def ensemble(self) -> List:
        return self._learner_ensemble

    @ensemble.setter
    def ensemble(self, learner_ensemble: List[base_learner.BaseLearner]):
        self._learner_ensemble = learner_ensemble

    def select_target_action(self, state: Any) -> None:
        pass

    def eval(self) -> None:
        for learner in self._learner_ensemble:
            learner.eval()

    def train(self) -> None:
        for learner in self._learner_ensemble:
            learner.train()
