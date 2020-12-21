from typing import Tuple

import numpy as np

from learners.ensemble_learners import ensemble_learner


class SampleGreedyEnsemble(ensemble_learner.EnsembleLearner):
    def select_target_action(self, state: Tuple[int, int]) -> int:
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
