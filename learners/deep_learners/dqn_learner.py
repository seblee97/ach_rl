import copy
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import constants
from learners import base_learner
from learners.deep_learners.components import q_network
from utils import epsilon_schedules


class DQNLearner(base_learner.BaseLearner):
    """Simple DQN agent: Mnih, et al. 2015"""

    def __init__(
        self,
        action_space: List[int],
        state_dimensions: Tuple[int, int, int],
        layer_specifications: List[Dict[str, Any]],
        optimiser_type: str,
        network_initialisation: str,
        learning_rate: float,
        gamma: float,
        epsilon: epsilon_schedules.EpsilonSchedule,
        target_network_update_period: int,
        device: torch.device,
        momentum: float = 0,
        eps: float = 1e-8,
    ):
        """Class constructor.

        Args:
            action_space: possible action list.
            state_dimensions: dimensions of state space.
            layer_specifications: network architecture for q-networks.
            optimiser_type: name of optimiser e.g. adam.
            network_initialisation: name of network initialisation e.g. xavier_uniform.
            learning_rate: learning rate.
            gamma: discount factor.
            epsilon: exploration percentage in epsilon-greedy.
            target_network_update_period: number of steps between
                re-callibrating q-network.
            device: cpu or gpu.
            momentum: momentum of optimiser (e.g. for adam).
            eps: eps of optimiser (e.g. for adam).
        """
        self._state_dimensions = state_dimensions
        self._action_space = action_space
        self._layer_specifications = layer_specifications
        self._optimiser_type = optimiser_type
        self._network_initialisation = network_initialisation
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._eps = eps
        self._gamma = gamma
        self._epsilon = epsilon
        self._target_network_update_period = target_network_update_period
        self._device = device

        self._q_network = self._initialise_q_network().to(self._device)
        self._target_q_network = self._initialise_q_network().to(self._device)

        self._loss_module = nn.MSELoss()

        self._optimiser = self._setup_optimiser()

        self._num_training_steps = 0

    @property
    def q_network(self) -> nn.Module:
        return self._q_network

    def _initialise_q_network(self):
        """Setup a q-network."""
        return q_network.QNetwork(
            state_dim=self._state_dimensions,
            num_actions=len(self._action_space),
            layer_specifications=self._layer_specifications,
            initialisation=self._network_initialisation,
        )

    def _setup_optimiser(self):
        """Setup optimiser.

        Supports adam and rms_prop."""
        if self._optimiser_type == constants.Constants.ADAM:
            optimiser = torch.optim.Adam(
                self._q_network.parameters(), lr=self._learning_rate
            )
        elif self._optimiser_type == constants.Constants.RMS_PROP:
            optimiser = torch.optim.RMSprop(
                self._q_network.parameters(),
                lr=self._learning_rate,
                momentum=self._momentum,
                eps=self._eps,
            )
        return optimiser

    def train(self) -> None:
        """Set to train."""
        self._q_network.train()
        self._target_q_network.train()

    def eval(self) -> None:
        """Set to evaluation."""
        self._q_network.eval()
        self._target_q_network.eval()

    def _update_target_network(self):
        """Re-set q-network to target network."""
        q_network_state_dict = copy.deepcopy(self._q_network.state_dict())
        self._target_q_network.load_state_dict(q_network_state_dict)

    def select_behaviour_action(self, state: np.ndarray):
        """Action to select for behaviour i.e. for training."""
        # cast state to tensor
        state = torch.from_numpy(state).to(torch.float).to(self._device)

        if random.random() < self._epsilon.value:
            action = random.choice(self._action_space)
        else:
            state_action_values = self._q_network(state)
            action = torch.argmax(state_action_values)
        return action

    def select_target_action(self, state: np.ndarray):
        """Action to select for target, i.e. final policy."""
        # cast state to tensor
        state = torch.from_numpy(state).to(torch.float).to(self._device)

        state_action_values = self._q_network(state)
        action = torch.argmax(state_action_values).item()
        return action

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        active: torch.Tensor,
        visitation_penalty: float,
    ) -> None:
        """Training step."""
        estimate = torch.max(self._q_network(next_state), axis=1).values

        target = reward + active * self._gamma * torch.max(
            self._target_q_network(next_state)
        )

        self._optimiser.zero_grad()
        loss = self._loss_module(target, estimate)
        loss.backward()
        self._optimiser.step()

        if self._num_training_steps % self._target_network_update_period == 0:
            self._update_target_network()

        self._num_training_steps += 1

        # step epsilon
        next(self._epsilon)
