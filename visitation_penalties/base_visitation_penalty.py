import abc
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import constants
import numpy as np


class BaseVisitationPenaltyComputer(abc.ABC):
    """Base class for visitation penalty computations. 
    Child classes of this class are argument to overall visitation penalty classes."""

    @abc.abstractmethod
    def compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        pass


class BaseVisitationPenalty(abc.ABC):
    """Base class for visitation penalties.

    Generally calls to this 'penalty' should follow
    the potential-based reward structure of
    Ng, Harada, Russell (1999).
    """

    # very small value to avoid log (0) in entropy calculation
    EPSILON = 1e-8

    def __init__(self, penalty_computer: BaseVisitationPenaltyComputer):
        self._penalty_computer = penalty_computer
        self._state_action_values: List[Dict[Tuple[int], List[float]]]

    @abc.abstractmethod
    def _compute_state_values(self, state):
        pass

    def _get_penalty_info(
        self,
        state,
        action,
        next_state,
    ):
        """compute series of uncertainties over states culminating in penalty."""
        current_state_values = self._compute_state_values(state=state)
        next_state_values = self._compute_state_values(state=next_state)

        num_actions = len(current_state_values[0])

        current_state_max_action_values = np.amax(current_state_values, axis=1)
        current_state_max_action_indices = np.argmax(current_state_values,
                                                     axis=1)

        current_state_max_uncertainty = np.std(current_state_max_action_values)
        current_state_mean_uncertainty = np.mean(
            np.std(current_state_values, axis=0))

        current_state_select_uncertainty = np.std(current_state_values[:,
                                                                       action])
        current_state_max_action_index_probabilities = np.bincount(
            current_state_max_action_indices,
            minlength=num_actions) / len(current_state_max_action_indices)
        current_state_policy_entropy = -np.sum(
            (current_state_max_action_index_probabilities + self.EPSILON) *
            np.log(current_state_max_action_index_probabilities + self.EPSILON))

        next_state_max_action_values = np.max(next_state_values, axis=1)
        next_state_max_action_indices = np.argmax(next_state_values, axis=1)

        next_state_max_uncertainty = np.std(next_state_max_action_values)
        next_state_mean_uncertainty = np.mean(np.std(next_state_values, axis=0))
        next_state_max_action_index_probabilities = np.bincount(
            next_state_max_action_indices,
            minlength=num_actions) / len(current_state_max_action_indices)
        next_state_policy_entropy = -np.sum(
            (next_state_max_action_index_probabilities + self.EPSILON) *
            np.log(next_state_max_action_index_probabilities + self.EPSILON))

        return {
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY:
                current_state_max_uncertainty,
            constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY:
                current_state_mean_uncertainty,
            constants.Constants.CURRENT_STATE_SELECT_UNCERTAINTY:
                current_state_select_uncertainty,
            constants.Constants.CURRENT_STATE_POLICY_ENTROPY:
                current_state_policy_entropy,
            constants.Constants.NEXT_STATE_MAX_UNCERTAINTY:
                next_state_max_uncertainty,
            constants.Constants.NEXT_STATE_MEAN_UNCERTAINTY:
                next_state_mean_uncertainty,
            constants.Constants.NEXT_STATE_POLICY_ENTROPY:
                next_state_policy_entropy,
        }

    def __call__(self, episode, state, action, next_state, *args, **kwargs):
        penalty_info = self._get_penalty_info(state=state,
                                              action=action,
                                              next_state=next_state)

        penalty = self._penalty_computer.compute_penalty(
            episode=episode, penalty_info=penalty_info)

        return penalty, penalty_info