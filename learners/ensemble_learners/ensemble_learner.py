import abc
from typing import Any
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

    @abc.abstractmethod
    def select_target_action(self, state: Any) -> None:
        pass

    def eval(self) -> None:
        for learner in self._learner_ensemble:
            learner.eval()

    def train(self) -> None:
        for learner in self._learner_ensemble:
            learner.train()
