from typing import List
from typing import Tuple

import copy
import numpy as np

from learners.ensemble_learners import ensemble_learner


class MajorityVoteEnsemble(ensemble_learner.EnsembleLearner):
    """Ensemble learner where action in a given state
    is selected based on most common argmax of state-action
    values in ensemble.
    """

    @property
    def weighted_state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        possible_actions = self._learner_ensemble[0].action_space
        states = all_state_action_values[0].keys()

        values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            max_indices = np.argmax(state_values, axis=1)
            max_index_counts = np.bincount(
                max_indices, minlength=len(possible_actions)
            ) / len(max_indices)

            values[state] = np.mean(state_values, axis=0)[np.argmax(max_index_counts)]

        return values

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action for policy based on most common max action value
        over learners in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        possible_actions = self._learner_ensemble[0].action_space

        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner.state_action_values[state] for learner in self._learner_ensemble
        ]
        max_action_values = [
            np.argmax(state_action_values)
            for state_action_values in all_state_action_values
        ]
        max_action_value_counts = np.bincount(
            max_action_values, minlength=len(possible_actions)
        )
        action = np.argmax(max_action_value_counts)

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
        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner.state_action_values[state] for learner in self._learner_ensemble
        ]
        max_action_values = [
            np.argmax(state_action_values)
            for state_action_values in all_state_action_values
        ]

        max_action_values = []
        for state_action_values in all_state_action_values:
            values_copy = copy.deepcopy(state_action_values[state])
            for action in excluded_actions:
                values_copy[action] = -np.inf
            max_action_values.append(np.argmax(values_copy))

        max_action_value_counts = np.bincount(max_action_values)
        action = np.argmax(max_action_value_counts)

        return action
