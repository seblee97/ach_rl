import abc

import numpy as np
from ach_rl import constants
from ach_rl.utils import custom_functions


class BaseInformationComputer(abc.ABC):
    """Object to computer various metric of agent and environment at any given time.

    The outputs of this computer can be used downstream for e.g. visitation penalties.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def _compute_state_values(self, state):
        pass

    def _state_quantities(self, state_values):

        num_actions = state_values.shape[-1]

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
                    custom_functions.policy_entropy(indices, num_actions=num_actions)
                    for indices in state_max_action_indices.T
                ]
            )

        return state_max_uncertainty, state_mean_uncertainty, state_policy_entropy

    def compute_state_information(self, state, state_label: str):

        state_values = self._compute_state_values(
            state=state
        )  # dims: E x B x A or E x A

        (
            state_max_uncertainty,
            state_mean_uncertainty,
            state_policy_entropy,
        ) = self._state_quantities(state_values)

        return {
            f"{state_label}_{constants.STATE_MAX_UNCERTAINTY}": state_max_uncertainty,
            f"{state_label}_{constants.STATE_MEAN_UNCERTAINTY}": state_mean_uncertainty,
            f"{state_label}_{constants.STATE_POLICY_ENTROPY}": state_policy_entropy,
        }

    def compute_state_select_information(self, state, action):
        state_values = self._compute_state_values(
            state=state
        )  # dims: E x B x A or E x A
        if len(state_values.shape) == 2:
            state_select = state_values[:, action]
        elif len(state_values.shape) == 3:
            state_select = state_values[:, :, action]
        state_select_uncertainty = np.std(state_select, axis=0)
        return {constants.CURRENT_STATE_SELECT_UNCERTAINTY: state_select_uncertainty}
