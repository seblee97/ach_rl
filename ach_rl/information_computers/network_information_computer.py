import torch
import torch.nn as nn
from ach_rl.information_computers import base_information_computer


class NetworkInformationComputer(base_information_computer.BaseInformationComputer):
    def __init__(
        self,
        use_target_network: bool = False,
    ):
        self._q_network: nn.Module
        self._target_q_network: nn.Module
        self._use_target_network = use_target_network
        super().__init__()

    @property
    def q_network(self):
        return self._q_network

    @q_network.setter
    def q_network(self, q_network: nn.Module):
        self._q_network = q_network

    @property
    def target_q_network(self):
        return self._target_q_network

    @target_q_network.setter
    def target_q_network(self, target_q_network: nn.Module):
        self._target_q_network = target_q_network

    def _compute_state_values(self, state):
        with torch.no_grad():
            if self._use_target_network:
                state_values = self._target_q_network.forward_all_heads(state)
            else:
                state_values = self._q_network.forward_all_heads(state)
        # output of forward_all_heads method has dimensions [NUM_LEARNERS x BATCH_SIZE x NUM_ACTIONS]
        # for penalty compuation we have batch size of 1 and want to squash this dimension
        return state_values.cpu().numpy()
