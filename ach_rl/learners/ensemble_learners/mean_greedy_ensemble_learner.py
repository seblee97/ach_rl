import copy
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from ach_rl.learners import base_learner


class MeanGreedyEnsemble:
    """Ensemble learner where action in a given state
    is selected based on argmax over mean of action-values
    across learners in ensemble.
    """

    @staticmethod
    def weighted_state_action_values(learners: List[base_learner.BaseLearner]):
        all_state_action_values = [learner.state_action_values for learner in learners]

        states = all_state_action_values[0].keys()

        values = {}
        for state in states:
            state_values = [values[state] for values in all_state_action_values]
            mean_state_values = np.mean(state_values, axis=0)
            values[state] = np.max(mean_state_values)

        return values

    @staticmethod
    def select_target_action(
        learners: List[base_learner.BaseLearner],
        state: Tuple[int, int],
        excluded_actions: Optional[List[int]] = [],
    ) -> int:
        """Select action for policy based on max from mean over learners in ensemble.

        Args:
            state: state for which to select optimal action.
        """
        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner.state_action_values[state] for learner in learners
        ]
        average_state_action_values = np.mean(all_state_action_values, axis=0)

        if excluded_actions:
            for action in excluded_actions:
                average_state_action_values[action] = -np.inf

        action = np.argmax(average_state_action_values)

        return action
