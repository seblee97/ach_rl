import abc
from typing import List, Optional, Tuple

import constants
import numpy as np
import random


class TabularLearner(abc.ABC):
    """Base class for learners in tabular settings."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: str,
        behaviour: str,
        target: str,
        visitation_penalty: Optional[float] = None,
    ):
        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )

        self._behaviour = behaviour
        self._target = target
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._visitation_penalty = visitation_penalty

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        if initialisation_strategy == constants.Constants.RANDOM:
            return np.random.rand(len(self._state_space), len(self._action_space))
        elif initialisation_strategy == constants.Constants.ZEROS:
            return np.zeros((len(self._state_space), len(self._action_space)))
        elif initialisation_strategy == constants.Constants.ONES:
            return np.ones((len(self._state_space), len(self._action_space)))

    def _max_state_action_value(self, state: Tuple[int, int]) -> float:
        """Find highest value in given state.

        Args:
            state: state for which to find highest value.

        Returns:
            value: corresponding highest value.
        """
        state_id = self._state_id_mapping[state]
        return np.amax(self._state_action_values[state_id])

    def _greedy_action(self, state: Tuple[int, int]) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.

        Returns:
            action: action with highest value in state given.
        """
        state_id = self._state_id_mapping[state]
        return np.argmax(self._state_action_values[state_id])

    def non_repeat_greedy_action(
        self, state: Tuple[int, int], excluded_actions: List[int]
    ) -> int:
        """Find action with highest value in given state not included set of excluded actions.

        Args:
            state: state for which to find action with (modified) highest value.
            excluded_actions: set of actions to exclude from consideration.

        Returns:
            action: action with (modified) highest value in state given.
        """
        state_id = self._state_id_mapping[state]
        actions_available = [
            action if i not in excluded_actions else -np.inf
            for (i, action) in enumerate(self._state_action_values[state_id])
        ]
        return np.argmax(actions_available)

    def _epsilon_greedy_action(self, state: Tuple[int, int], epsilon: float) -> int:
        """Choose greedy policy with probability
        (1 - epsilon) and a random action with probability epsilon.

        Args:
            state: state for which to find action.
            epsilon: parameter controlling randomness.

        Returns:
            action: action chosen according to epsilon greedy.
        """
        if random.random() < self._epsilon:
            action = random.choice(self._action_space)
        else:
            action = self._greedy_action(state=state)
        return action

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action according to target policy, i.e. policy being learned.
        Here, action with highest value in given state is selected.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        if self._target == constants.Constants.GREEDY:
            action = self._greedy_action(state=state)
        elif self._target == constants.Constants.EPSILON_GREEDY:
            action = self._epsilon_greedy_action(state=state, epsilon=self._epsilon)
        return action

    def select_behaviour_action(self, state: Tuple[int, int]) -> Tuple[int, float]:
        """Select action with behaviour policy, i.e. policy collecting trajectory data
        and generating behaviour. Sarsa lambda is on-policy so this is the same as the
        target policy, namely the greedy action.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        if self._behaviour == constants.Constants.GREEDY:
            action = self._greedy_action(state=state)
        elif self._behaviour == constants.Constants.EPSILON_GREEDY:
            action = self._epsilon_greedy_action(state=state, epsilon=self._epsilon)
        return action

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Update relevant data for learner."""
        pass
