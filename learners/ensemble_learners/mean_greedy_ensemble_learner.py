from learners.ensemble_learners import ensemble_learner

from typing import Tuple


class MeanGreedyEnsemble(ensemble_learner.EnsembleLearner):
    def select_target_action(self, state: Tuple[int, int]):
        # get original (np array) state action values for each learner.
        all_state_action_values = [
            learner._state_action_values for learner in self._learner_ensemble
        ]
        average_state_action_values = np.mean(all_state_action_values)
        raise NotImplementedError
