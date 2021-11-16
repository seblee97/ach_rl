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

    def __call__(
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

        (
            next_state_max_uncertainty,
            next_state_mean_uncertainty,
            next_state_policy_entropy,
        ) = _state_quantities(next_state_values)

        if len(current_state_values.shape) == 2:
            current_state_select = current_state_values[:, action]
        elif len(current_state_values.shape) == 3:
            current_state_select = current_state_values[:, :, action]
        current_state_select_uncertainty = np.std(current_state_select, axis=0)

        return {
            constants.CURRENT_STATE_MAX_UNCERTAINTY: current_state_max_uncertainty,
            constants.CURRENT_STATE_MEAN_UNCERTAINTY: current_state_mean_uncertainty,
            constants.CURRENT_STATE_SELECT_UNCERTAINTY: current_state_select_uncertainty,
            constants.CURRENT_STATE_POLICY_ENTROPY: current_state_policy_entropy,
            constants.NEXT_STATE_MAX_UNCERTAINTY: next_state_max_uncertainty,
            constants.NEXT_STATE_MEAN_UNCERTAINTY: next_state_mean_uncertainty,
            constants.NEXT_STATE_POLICY_ENTROPY: next_state_policy_entropy,
        }
