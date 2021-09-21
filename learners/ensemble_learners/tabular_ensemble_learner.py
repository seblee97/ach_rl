import pickle
from typing import Any
from typing import List
from typing import Optional

import numpy as np
from learners import base_learner
from utils import custom_functions


class TabularEnsembleLearner(base_learner.BaseLearner):
    """Learner consisting of ensemble."""

    def __init__(
        self,
        learner_ensemble_path: Optional[str] = None,
        learner_ensemble: Optional[List[base_learner.BaseLearner]] = None,
    ):
        """Class constructor.

        Args:
            learner_ensemble_path: path to saved pretrained model
            learner_ensemble: list of learners forming ensemble.
        """
        assert (
            sum([learner_ensemble_path is not None, learner_ensemble is not None]) == 1
        ), "either a learner ensemble or a path to a saved learner ensemble must be provided."

        if learner_ensemble_path is not None:
            self._learner_ensemble = self._load_model(model_path=learner_ensemble_path)
        else:
            self._learner_ensemble = learner_ensemble

    def _load_model(self, model_path: str):
        with open(model_path, "rb") as file:
            learner_ensemble = pickle.load(file)
        return learner_ensemble

    def checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "wb") as file:
            pickle.dump(self._learner_ensemble, file)

    @property
    def state_visitation_counts(self):
        all_state_visitation_counts = [
            learner._state_visitation_counts for learner in self._learner_ensemble
        ]

        averaged_counts = {}

        states = all_state_visitation_counts[0].keys()

        for state in states:
            state_counts = [counts[state] for counts in all_state_visitation_counts]
            mean_state_counts = np.mean(state_counts, axis=0)
            averaged_counts[state] = mean_state_counts

        return averaged_counts

    @property
    def state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        averaged_values = {}

        states = all_state_action_values[0].keys()
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
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
            state_values = [values[state] for values in all_state_action_values]
            state_values_std = np.std(state_values, axis=0)
            values_std[state] = state_values_std
        return values_std

    @property
    def policy_entropy(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        policy_entropy = {}

        states = all_state_action_values[0].keys()

        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            max_action_indices = np.argmax(state_values, axis=1)
            state_policy_entropy = custom_functions.policy_entropy(
                state_max_action_indices=max_action_indices,
                num_actions=len(state_values[0]),
            )
            policy_entropy[state] = state_policy_entropy
        return policy_entropy

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
