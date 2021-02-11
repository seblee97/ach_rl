from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from learners import base_learner


class MajorityVoteEnsemble:
    """Ensemble learner where action in a given state
    is selected based on most common argmax of state-action
    values in ensemble.
    """

    @staticmethod
    def weighted_state_action_values(learners: List[base_learner.BaseLearner]):
        all_state_action_values = [
            learner.state_action_values for learner in learners
        ]

        possible_actions = learners[0].action_space
        states = all_state_action_values[0].keys()

        values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            max_indices = np.argmax(state_values, axis=1)
            max_index_counts = np.bincount(
                max_indices, minlength=len(possible_actions)) / len(max_indices)

            values[state] = np.mean(state_values,
                                    axis=0)[np.argmax(max_index_counts)]

        return values

    @staticmethod
    def select_target_action(
        learners: List[base_learner.BaseLearner],
        state: Tuple[int, int],
        excluded_actions: Optional[List[int]] = [],
    ) -> int:
        """Select action for policy based on most common max action value
        over learners in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        possible_actions = learners[0].action_space

        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner.state_action_values[state] for learner in learners
        ]

        max_action_values = []
        for state_action_values in all_state_action_values:
            if excluded_actions:
                for action in excluded_actions:
                    state_action_values[action] = -np.inf
            max_action_values.append(np.argmax(state_action_values))

        max_action_value_counts = np.bincount(max_action_values,
                                              minlength=len(possible_actions))
        action = np.argmax(max_action_value_counts)

        return action
