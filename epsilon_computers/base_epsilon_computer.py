import abc
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import constants
import numpy as np
from utils import custom_functions


class BaseEpsilonComputer(abc.ABC):
    """Base class for adaptive epsilon computations."""

    @abc.abstractmethod
    def _compute_epsilon(self, episode: int, epsilon_info: Dict[str, Any]):
        pass

    @property
    def state_action_values(self):
        return self._state_action_values

    @state_action_values.setter
    def state_action_values(self, state_action_values: List[Dict[Tuple[int], float]]):
        self._state_action_values = state_action_values

    def _compute_state_values(self, state):
        return np.array([s[state] for s in self._state_action_values])

    def _get_epsilon_info(self, state):
        """compute series of uncertainties over states."""
        current_state_values = self._compute_state_values(
            state=state
        )  # dims: E x B x A or E x A

        num_actions = current_state_values.shape[-1]

        def _state_quantities(state_values):
            state_max_action_values = np.amax(state_values, axis=-1)  # E x B
            state_max_action_indices = np.argmax(state_values, axis=-1)  # E x B

            state_max_uncertainty = np.std(state_max_action_values, axis=0)
            state_mean_uncertainty = np.mean(np.std(state_values, axis=0), axis=-1)

            if len(state_max_action_indices.shape) == 1:
                state_policy_entropy = custom_functions.policy_entropy(
                    state_max_action_indices, num_actions=num_actions
                )
            elif len(state_max_action_indices.shape) == 2:
                state_policy_entropy = np.array(
                    [
                        custom_functions.policy_entropy(
                            indices, num_actions=num_actions
                        )
                        for indices in state_max_action_indices.T
                    ]
                )

            return state_max_uncertainty, state_mean_uncertainty, state_policy_entropy

        (
            current_state_max_uncertainty,
            current_state_mean_uncertainty,
            current_state_policy_entropy,
        ) = _state_quantities(current_state_values)

        return {
            constants.Constants.STATE: state,
            constants.Constants.CURRENT_STATE_MAX_UNCERTAINTY: current_state_max_uncertainty,
            constants.Constants.CURRENT_STATE_MEAN_UNCERTAINTY: current_state_mean_uncertainty,
            constants.Constants.CURRENT_STATE_POLICY_ENTROPY: current_state_policy_entropy,
            constants.Constants.NORMALISED_POLICY_ENTROPY: current_state_policy_entropy / np.log(num_actions)
        }

    def __call__(self, episode, state, *args, **kwargs):
        epsilon_info = self._get_epsilon_info(state=state)

        epsilon = self._compute_epsilon(
            episode=episode, epsilon_info=epsilon_info
        )

        return epsilon, epsilon_info
