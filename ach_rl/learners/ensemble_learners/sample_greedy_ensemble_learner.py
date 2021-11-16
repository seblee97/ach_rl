import copy
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from ach_rl.learners import base_learner


class SampleGreedyEnsemble:
    """Ensemble learner where action in a given state
    is selected based on sample of maximum action-value
    for that state over each learner in ensemble.
    """

    @staticmethod
    def weighted_state_action_values(learners: List[base_learner.BaseLearner]):
        all_state_action_values = [learner.state_action_values for learner in learners]

        possible_actions = learners[0].action_space
        states = all_state_action_values[0].keys()

        weighted_values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            max_indices = np.argmax(state_values, axis=1)
            max_index_counts = np.bincount(
                max_indices, minlength=len(possible_actions)
            ) / len(max_indices)

            weighted_state_values = max_index_counts * state_values

            weighted_values[state] = np.mean(weighted_state_values, axis=0)

        return weighted_values

    @staticmethod
    def select_target_action(
        learners: List[base_learner.BaseLearner],
        state: Tuple[int, int],
        excluded_actions: Optional[List[int]] = [],
    ) -> int:
        """Select action for policy based on sample of maximums in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        possible_actions = learners[0].action_space
        all_state_action_values = [learner.state_action_values for learner in learners]

        max_action_values = []
        for state_action_values in all_state_action_values:
            if excluded_actions:
                for action in excluded_actions:
                    state_action_values[state][action] = -np.inf
            max_action_values.append(np.argmax(state_action_values))

        action_probabilities = np.bincount(
            max_action_values, minlength=len(possible_actions)
        ) / len(max_action_values)

        action = np.random.choice(possible_actions, p=action_probabilities)

        return action
