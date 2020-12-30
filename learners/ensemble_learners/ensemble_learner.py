from typing import List

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
    def ensemble(self) -> List:
        return self._learner_ensemble

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass
