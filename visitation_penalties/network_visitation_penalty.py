import abc
from typing import Any
from typing import Dict
from typing import List

import constants
import numpy as np
import torch
import torch.nn as nn


class NetworkVisitationPenalty:

    EPSILON = 1e-8

    def __init__(self):
        self._q_network: nn.Module

    @property
    def q_network(self):
        return self._q_network

    @q_network.setter
    def q_network(self, q_network: nn.Module):
        self._q_network = q_network

    def _get_penalty_info(
        self,
        state,
        action,
        next_state,
    ):
        """compute series of uncertainties over states culminating in penalty."""
        with torch.no_grad():
            current_state_values = np.array([
                self._q_network(state, branch=i).numpy().flatten()
                for i in range(len(self._q_network._branched_layers))
            ])
            next_state_values = np.array([
                self._q_network(next_state, branch=i).numpy().flatten()
                for i in range(len(self._q_network._branched_layers))
            ])

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

        penalty = self._compute_penalty(episode=episode,
                                        penalty_info=penalty_info)

        return penalty, penalty_info

    @abc.abstractmethod
    def _compute_penalty(self, episode: int, penalty_info: Dict[str, Any]):
        pass
