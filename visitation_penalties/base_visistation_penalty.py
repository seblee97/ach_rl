import abc
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

import constants


class BaseVisitationPenalty(abc.ABC):
    """Base class for visitation penalties.

    Generally calls to this 'penalty' should follow
    the potential-based reward structure of
    Ng, Harada, Russell (1999).
    """

    # very small value to avoid log (0) in entropy calculation
    EPSILON = 1e-8

    def __init__(self, *args, **kwargs):
        self._state_action_values: List[Dict[Tuple[int], List[float]]]

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def __call__(self, episode, state, action, next_state, *args, **kwargs):
        """compute series of uncertainties over states culminating in penalty."""
        current_state_values = [s[state] for s in self._state_action_values]
        next_state_values = [s[next_state] for s in self._state_action_values]

        num_actions = len(current_state_values[0])

        current_state_max_action_values = [np.max(s) for s in current_state_values]
        current_state_max_action_indices = [np.argmax(s) for s in current_state_values]
        current_state_max_uncertainty = np.std(current_state_max_action_values)
        current_state_mean_uncertainty = np.mean(np.std(current_state_values, axis=0))

        current_state_select_uncertainty = np.std(
            [s[action] for s in current_state_values]
        )
        current_state_max_action_index_probabilities = np.bincount(
            current_state_max_action_indices, minlength=num_actions
        ) / len(current_state_max_action_indices)
        current_state_policy_entropy = -np.sum(
            [
                (probability + self.EPSILON) * np.log(probability + self.EPSILON)
                for probability in current_state_max_action_index_probabilities
            ]
        )

        next_state_max_action_values = [np.max(s) for s in next_state_values]
        next_state_max_action_indices = [np.argmax(s) for s in next_state_values]
        next_state_max_uncertainty = np.std(next_state_max_action_values)
        next_state_mean_uncertainty = np.mean(np.std(next_state_values, axis=0))
        next_state_max_action_index_probabilities = np.bincount(
            next_state_max_action_indices, minlength=num_actions
        ) / len(current_state_max_action_indices)
        next_state_policy_entropy = -np.sum(
            [
                (probability + self.EPSILON) * np.log(probability + self.EPSILON)
                for probability in next_state_max_action_index_probabilities
            ]
        )

        penalty_info = {
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY: current_state_max_uncertainty,
            constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY: current_state_mean_uncertainty,
            constants.Constants.CURRENT_STATE_SELECT_UNCERTAINTY: current_state_select_uncertainty,
            constants.Constants.CURRENT_STATE_POLICY_ENTROPY: current_state_policy_entropy,
            constants.Constants.NEXT_STATE_MAX_UNCERTAINTY: next_state_max_uncertainty,
            constants.Constants.NEXT_STATE_MEAN_UNCERTAINTY: next_state_mean_uncertainty,
            constants.Constants.NEXT_STATE_POLICY_ENTROPY: next_state_policy_entropy,
        }

        penalty = self._compute_penalty(episode=episode, penalty_info=penalty_info)

        return penalty, penalty_info

    @abc.abstractmethod
    def _compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        pass
