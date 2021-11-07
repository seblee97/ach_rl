import numpy as np
from ach_rl import constants


def policy_entropy(state_max_action_indices: np.ndarray, num_actions: int):
    state_max_action_index_probabilities = np.bincount(
        state_max_action_indices, minlength=num_actions
    ) / len(state_max_action_indices)
    entropy = -np.sum(
        (state_max_action_index_probabilities + constants.LOG_EPSILON)
        * np.log(state_max_action_index_probabilities + constants.LOG_EPSILON)
    )
    return entropy


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # rgb channel last
    grayscale = np.dot(rgb[..., :3], [[0.299], [0.587], [0.114]])
    return grayscale
