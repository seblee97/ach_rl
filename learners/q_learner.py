import random
from typing import List, Dict, Optional, Tuple

import constants

from learners import base_learner


class TabularQLearner(base_learner.BaseLearner):
    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: str,
        visitation_penalty: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        self._action_space = action_space
        self._state_space = state_space
        self._state_action_values = self._initialise_values(
            action_space, state_space, initialisation_strategy=initialisation_strategy
        )
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._visitation_penalty = visitation_penalty
        self._epsilon = epsilon

    def _initialise_values(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        initialisation_strategy: str,
    ) -> Dict[int, float]:
        """Initialise values for each state, action pair in state-action space.

        Args:
            action_space: list of permissible actions.
            state_space: list of states in environment.

        Returns:
            initial_values: dictionary containing state-action id / value mapping.
        """
        if initialisation_strategy == constants.Constants.RANDOM:
            return {
                state: {action: random.random() for action in action_space}
                for state in state_space
            }
        elif initialisation_strategy == constants.Constants.ZEROS:
            return {
                state: {action: 0.0 for action in action_space} for state in state_space
            }
        elif initialisation_strategy == constants.Constants.ONES:
            return {
                state: {action: 1.0 for action in action_space} for state in state_space
            }

    def _max_state_action_value(self, state: Tuple[int, int]) -> float:
        """Find highest value in given state.

        Args:
            state: state for which to find highest value.

        Returns:
            value: corresponding highest value.
        """
        return max(self._state_action_values[state].values())

    def _greedy_action(self, state: Tuple[int, int]) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.

        Returns:
            action: action with highest value in state given.
        """
        action_values = self._state_action_values[state]
        return max(action_values, key=lambda key: action_values[key])

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action according to target policy, i.e. policy being learned.
        For Q-learning this corresponds to the greedy policy, so action with
        highest value in given state is selected.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        action = self._greedy_action(state=state)
        return action

    def select_behaviour_action(self, state: Tuple[int, int]) -> Tuple[int, float]:
        """Select action with behaviour policy, i.e. policy collecting trajectory data
        and generating behaviour. For Q-learning this corresponds to an
        epsilon-greedy policy. This chooses the greedy policy with probability
        (1 - epsilon) and a random action with probability epsilon.

        Args:
            state: current state.

        Returns:
            action: selected action from specified policy.
        """
        if random.random() < self._epsilon:
            action = random.choice(self._action_space)
        else:
            action = self._greedy_action(state=state)
        return action

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        bootstrapped_return: float,
    ):
        """Update state-action values.

        Make q-learning update (Equation 7.9 in Sutton & Barto):
        Q(s, a) <- Q(s, a) + alpha * rho * [G - Q(s, a)]
        where G is a n-step bootstrapped return.

        Args:
            state: state to be updated.
            action: action to be updated.
            importance_weight: weighting to account for potential difference
            between behaviour and target policies.
            bootstrapped_return: reward in next n-steps plus bootstrapped Q value.
        """
        initial_state_action_value = self._state_action_values[state][action]
        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate * (bootstrapped_return - initial_state_action_value)
        )
        self._state_action_values[state][action] = updated_state_action_value

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
                            + gamma * max_a(Q(s_{t+1}, a))
                            - Q(s_t, at_t)
                            ]

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
        """
        initial_state_action_value = self._state_action_values[state][action]
        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate
            * (
                reward
                + self._gamma * self._max_state_action_value(state=new_state)
                - initial_state_action_value
            )
        )
        self._state_action_values[state][action] = (
            updated_state_action_value - self._visitation_penalty
        )
