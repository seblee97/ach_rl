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
        current_state_values = self._compute_state_values(
            state=state
        )  # dims: E x B x A or E x A
        next_state_values = self._compute_state_values(
            state=next_state
        )  # dims: E x B x A or E x A

        num_actions = current_state_values.shape[-1]

        def _state_quantities(state_values):
            state_max_action_values = np.amax(state_values, axis=-1)  # E x B
            state_max_action_indices = np.argmax(state_values, axis=-1)  # E x B

            state_max_uncertainty = np.std(state_max_action_values, axis=0)
            state_mean_uncertainty = np.mean(np.std(state_values, axis=0), axis=-1)

            state_max_action_index_probabilities = [
                np.bincount(indices, minlength=num_actions) / len(indices)
                for indices in state_max_action_indices.T
            ]
            state_policy_entropy = np.array(
                [
                    -np.sum(
                        (probabilities + self.EPSILON)
                        * np.log(probabilities + self.EPSILON)
                    )
                    for probabilities in state_max_action_index_probabilities
                ]
            )
            return state_max_uncertainty, state_mean_uncertainty, state_policy_entropy

        (
            current_state_max_uncertainty,
            current_state_mean_uncertainty,
            current_state_policy_entropy,
        ) = _state_quantities(current_state_values)

        (
            next_state_max_uncertainty,
            next_state_mean_uncertainty,
            next_state_policy_entropy,
        ) = _state_quantities(next_state_values)

        current_state_select_uncertainty = np.std(
            current_state_values[:, :, action], axis=0
        )

        return {
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY: current_state_max_uncertainty,
            constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY: current_state_mean_uncertainty,
            constants.Constants.CURRENT_STATE_SELECT_UNCERTAINTY: current_state_select_uncertainty,
            constants.Constants.CURRENT_STATE_POLICY_ENTROPY: current_state_policy_entropy,
            constants.Constants.NEXT_STATE_MAX_UNCERTAINTY: next_state_max_uncertainty,
            constants.Constants.NEXT_STATE_MEAN_UNCERTAINTY: next_state_mean_uncertainty,
            constants.Constants.NEXT_STATE_POLICY_ENTROPY: next_state_policy_entropy,
        }

    def __call__(self, episode, state, action, next_state, *args, **kwargs):
        penalty_info = self._get_penalty_info(
            state=state, action=action, next_state=next_state
        )

        penalty = self._penalty_computer.compute_penalty(
            episode=episode, penalty_info=penalty_info
        )

        return penalty, penalty_info
