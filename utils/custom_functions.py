import constants
import numpy as np


def policy_entropy(state_max_action_indices: np.ndarray, num_actions: int):
    state_max_action_index_probabilities = np.bincount(
        state_max_action_indices, minlength=num_actions
    ) / len(state_max_action_indices)
    entropy = -np.sum(
        (state_max_action_index_probabilities + constants.Constants.LOG_EPSILON)
        * np.log(state_max_action_index_probabilities + constants.Constants.LOG_EPSILON)
    )
    return entropy
