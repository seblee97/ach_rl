from typing import List
from typing import Tuple

import numpy as np
import copy

from learners.ensemble_learners import ensemble_learner


class SampleGreedyEnsemble(ensemble_learner.EnsembleLearner):
    """Ensemble learner where action in a given state
    is selected based on sample of maximum action-value
    for that state over each learner in ensemble.
    """

    @property
    def state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        possible_actions = self._learner_ensemble[0].action_space
        states = all_state_action_values[0].keys()

        weighted_values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            max_indices = np.argmax(state_values, axis=1)
            max_index_counts = np.bincount(
                max_indices, minlength=len(possible_actions)
            ) / len(max_indices)

            weighted_values[state] = np.average(
                state_values, axis=0, weights=max_index_counts
            )

        return weighted_values

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action for policy based on sample of maximums in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]
        max_action_values = [
            np.argmax(state_action_values[state])
            for state_action_values in all_state_action_values
        ]
        action_probabilities = np.bincount(max_action_values) / len(max_action_values)

        action = np.random.choice(
            range(len(action_probabilities)), p=action_probabilities
        )
        return action

    def non_repeat_greedy_action(
        self, state: Tuple[int, int], excluded_actions: List[int]
    ) -> int:
        """Find action with highest value in given state not included
        in set of excluded actions.

        Args:
            state: state for which to find action with (modified) highest value.
            excluded_actions: set of actions to exclude from consideration.

        Returns:
            action: action with (modified) highest value in state given.
        """
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        max_action_values = []
        for state_action_values in all_state_action_values:
            values_copy = copy.deepcopy(state_action_values[state])
            for action in excluded_actions:
                values_copy[action] = -np.inf
            max_action_values.append(np.argmax(values_copy))

        action_probabilities = np.bincount(max_action_values) / len(max_action_values)

        action = np.random.choice(
            range(len(action_probabilities)), p=action_probabilities
        )

        return action
