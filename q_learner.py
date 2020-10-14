import random
from typing import List, Dict, Optional, Tuple

import constants


class TabularQLearner:
    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        alpha: float,
        gamma: float,
        policy: str,
        epsilon: Optional[float] = None,
    ):
        self._action_space = action_space
        self._state_space = state_space
        self._state_action_values = self._initialise_values(action_space, state_space)
        self._alpha = alpha
        self._gamma = gamma
        self._policy = policy
        self._epsilon = epsilon

    def _initialise_values(
        self, action_space: List[int], state_space: List[Tuple[int, int]]
    ) -> Dict[int, float]:
        """Initialise values for each state, action pair in state-action space.

        Args:
            action_space: list of permissible actions.
            state_space: list of states in environment.

        Returns:
            initial_values: dictionary containing state-action id / value mapping.
        """
        return {
            state: {action: random.random() for action in action_space}
            for state in state_space
        }

    def _select_epsilon_greedy_action(self, state: Tuple[int, int]) -> int:
        """Perform epsilon-greedy action selection.

        Args:
            state: current state.

        Returns:
            action: selected action from epsilon-greedy policy.
        """
        if random.random() < self._epsilon:
            action = random.choice(self._action_space)
        else:
            action = self._arg_max_state_action_value(state=state)
        return action

    def _arg_max_state_action_value(self, state: Tuple[int, int]) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.

        Returns:
            action: action with highest value in state given.
        """
        action_values = self._state_action_values[state]
        return max(action_values, key=lambda key: action_values[key])

    def _max_state_action_value(self, state: Tuple[int, int]) -> float:
        """Find highest value in given state.

        Args:
            state: state for which to find highest value.

        Returns:
            value: corresponding highest value.
        """
        return max(self._state_action_values[state].values())

    def select_action(self, state: Tuple[int, int]) -> int:
        """Select action according to specified policy.

        Args:
            state: current state.

        Returns:
            action: selected action from specified policy.
        """
        if self._policy == constants.Constants.EPSILON_GREEDY:
            action = self._select_epsilon_greedy_action(state)
        return action

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
    ) -> None:
        """Update state-action values.

        Make q-learning update:
        Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [
                            r_{t+1}
                            + \gamma * max_a(Q(s_{t+1}, a))
                            - Q(s_t, at_t)
                            ]

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
        """
        initial_state_action_value = self._state_action_values[state][action]
        updated_state_action_value = initial_state_action_value + self._alpha * (
            reward
            + self._gamma * self._max_state_action_value(state=new_state)
            - initial_state_action_value
        )
        self._state_action_values[state][action] = updated_state_action_value
