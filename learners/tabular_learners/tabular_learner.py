import abc
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import constants
import numpy as np
from learners import base_learner
from utils import epsilon_schedules


class TabularLearner(base_learner.BaseLearner):
    """Base class for learners in tabular settings."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        epsilon: epsilon_schedules.EpsilonSchedule,
        initialisation_strategy: str,
        behaviour: str,
        target: str,
    ):
        """Class constructor.

        Args:
            action_space: list of actions available.
            state_space: list of states.
            learning_rate: learning_rage.
            gamma: discount factor.
            initialisation_strategy: name of network initialisation strategy.
            behaviour: name of behaviour type e.g. epsilon_greedy.
            target: name of target type e.g. greedy.
            epsilon: exploration parameter.
        """
        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )

        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._behaviour = behaviour
        self._target = target
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon

    def train(self):
        pass

    def eval(self):
        pass

    @property
    def action_space(self) -> List[int]:
        return self._action_space

    @property
    def state_id_mapping(self) -> Dict:
        return self._state_id_mapping

    @property
    def id_state_mapping(self) -> Dict:
        return self._id_state_mapping

    @property
    def state_visitation_counts(self) -> Dict[Tuple[int, int], int]:
        return self._state_visitation_counts

    @property
    def state_action_values(self) -> Dict[Tuple[int, int], np.ndarray]:
        return {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        if isinstance(initialisation_strategy, (int, float)):
            return initialisation_strategy * np.ones(
                (len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy == constants.Constants.RANDOM:
            return np.random.normal(loc=0, scale=0.1, size=(len(self._state_space), len(self._action_space)))
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
        if random.random() < epsilon:
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
            action = self._epsilon_greedy_action(
                state=state, epsilon=self._epsilon.value
            )
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
            action = self._epsilon_greedy_action(
                state=state, epsilon=self._epsilon.value
            )
        return action

    @abc.abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Update relevant data for learner."""
        pass
