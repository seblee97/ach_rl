import copy
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from ach_rl import constants
from ach_rl.learners import base_learner
from ach_rl.learners.deep_learners.components import q_network
from ach_rl.utils import epsilon_schedules


class DQNLearner(base_learner.BaseLearner):
    """Simple DQN agent: Mnih, et al. 2015"""

    def __init__(
        self,
        action_space: List[int],
        state_dimensions: Tuple[int, int, int],
        layer_specifications: List[Dict[str, Any]],
        optimiser_type: str,
        learning_rate: float,
        gamma: float,
        epsilon: epsilon_schedules.EpsilonSchedule,
        target_network_update_period: int,
        device: torch.device,
        gradient_clipping: Union[Tuple, None],
        momentum: float = 0,
        eps: float = 1e-8,
    ):
        """Class constructor.

        Args:
            action_space: possible action list.
            state_dimensions: dimensions of state space.
            layer_specifications: network architecture for q-networks.
            optimiser_type: name of optimiser e.g. adam.
            network_weight_initialisation: name of network initialisation e.g. xavier_uniform.
            network_bias_initialisation: type of initialisation for bias e.g. zeros.
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
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._eps = eps
        self._gamma = gamma
        self._epsilon = epsilon
        self._target_network_update_period = target_network_update_period
        self._device = device
        self._gradient_clipping = gradient_clipping

        self._q_network = self._initialise_q_network().to(self._device)
        self._target_q_network = self._initialise_q_network().to(self._device)
        self._update_target_network()

        self._loss_module = nn.MSELoss()

        self._optimiser = self._setup_optimiser()

        self._num_training_steps = 0

    @property
    def q_network(self) -> nn.Module:
        return self._q_network

    @property
    def target_q_network(self) -> nn.Module:
        return self._target_q_network

    def _initialise_q_network(self):
        """Setup a q-network."""
        return q_network.QNetwork(
            state_dim=self._state_dimensions,
            num_actions=len(self._action_space),
            layer_specifications=self._layer_specifications,
        )

    def _setup_optimiser(self):
        """Setup optimiser.

        Supports adam and rms_prop."""
        if self._optimiser_type == constants.ADAM:
            optimiser = torch.optim.Adam(
                self._q_network.parameters(), lr=self._learning_rate
            )
        elif self._optimiser_type == constants.RMS_PROP:
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

    def eval(self) -> None:
        """Set to evaluation."""
        self._q_network.eval()

    def _update_target_network(self):
        """Re-set q-network to target network."""
        self._target_q_network.load_state_dict(self._q_network.state_dict())
        self._target_q_network.eval()

    def select_behaviour_action(
        self,
        state: np.ndarray,
    ):
        """Action to select for behaviour i.e. for training."""
        # cast state to tensor
        state = torch.from_numpy(state).to(torch.float).to(self._device)

        if random.random() < self._epsilon.value:
            action = random.choice(self._action_space)
        else:
            self._q_network.eval()
            with torch.no_grad():
                state_action_values = self._q_network(state)
                action = torch.argmax(state_action_values).item()
            self._q_network.train()
        return action

    def select_target_action(
        self,
        state: np.ndarray,
    ):
        """Action to select for target, i.e. final policy."""
        with torch.no_grad():
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
    ) -> Tuple[float, float]:
        """Training step."""

        state_value_estimates = self._q_network(state)

        # add batch dimension to action for indexing
        action_index = action.unsqueeze(-1).to(torch.int64)
        estimate = torch.gather(state_value_estimates, 1, action_index).squeeze()

        max_target = torch.max(
            self._target_q_network(
                next_state,
            ),
            axis=1,
        ).values.detach()

        target = reward + visitation_penalty + active * self._gamma * max_target

        loss = self._loss_module(estimate, target)
        self._optimiser.zero_grad()
        loss.backward()

        # clip gradients
        if self._gradient_clipping is not None:
            for param in self._q_network.parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(
                        self._gradient_clipping[0], self._gradient_clipping[1]
                    )

        self._optimiser.step()

        if self._num_training_steps % self._target_network_update_period == 0:
            self._update_target_network()

        self._num_training_steps += 1

        # step epsilon
        next(self._epsilon)

        return loss.item(), self._epsilon.value
