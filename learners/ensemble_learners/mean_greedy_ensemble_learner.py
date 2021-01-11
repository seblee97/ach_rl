from typing import List
from typing import Tuple

import copy
import numpy as np

from learners.ensemble_learners import ensemble_learner


class MeanGreedyEnsemble(ensemble_learner.EnsembleLearner):
    @property
    def state_action_values(self):
        all_state_action_values = [
            learner.state_action_values for learner in self._learner_ensemble
        ]

        possible_actions = self._learner_ensemble[0].action_space
        states = all_state_action_values[0].keys()

        values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            mean_state_values = np.mean(state_values, axis=0)
            values[state] = np.max(mean_state_values)

        return values

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action for policy based on max from mean over learners in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner.state_action_values[state] for learner in self._learner_ensemble
        ]
        average_state_action_values = np.mean(all_state_action_values, axis=0)

        action = np.argmax(average_state_action_values)

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
        average_state_action_values = np.mean(all_state_action_values, axis=0)
        average_state_action_values_copy = copy.deepcopy(average_state_action_values)
        for action in excluded_actions:
            average_state_action_values_copy[action] = -np.inf

        action = np.argmax(average_state_action_values_copy)

        return action
