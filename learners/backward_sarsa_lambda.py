from typing import Dict, List, Optional, Tuple

import constants
import numpy as np

from learners import base_learner


class BackwardSARSALambda(base_learner.BaseLearner):
    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: str,
        trace_lambda: float,
        visitation_penalty: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        self._action_space = action_space
        self._state_space = state_space
        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )
        self._state_action_eligibility_traces = self._initialise_values(
            initialisation_strategy=constants.Constants.ZEROS
        )
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._visitation_penalty = visitation_penalty
        self._epsilon = epsilon
        self._trace_lambda = trace_lambda

    def _initialise_values(self, initialisation_strategy: str) -> Dict[int, float]:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: dictionary containing state-action id / value mapping.
        """
        if initialisation_strategy == constants.Constants.RANDOM:
            return {
                state: np.random.rand(len(self._action_space))
                for state in self._state_space
            }
        elif initialisation_strategy == constants.Constants.ZEROS:
            return {
                state: np.zeros(len(self._action_space)) for state in self._state_space
            }
        elif initialisation_strategy == constants.Constants.ONES:
            return {
                state: np.ones(len(self._action_space)) for state in self._state_space
            }

    def _max_state_action_value(self, state: Tuple[int, int]) -> float:
        """Find highest value in given state.

        Args:
            state: state for which to find highest value.

        Returns:
            value: corresponding highest value.
        """
        return np.amax(self._state_action_values[state])

    def _greedy_action(self, state: Tuple[int, int]) -> int:
        """Find action with highest value in given state.

        Args:
            state: state for which to find action with highest value.

        Returns:
            action: action with highest value in state given.
        """
        return np.argmax(self._state_action_values[state])

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
        action = self._greedy_action(state=state)
        return action
        # if random.random() < self._epsilon:
        #     action = random.choice(self._action_space)
        # else:
        #     action = self._greedy_action(state=state)
        # return action

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        next_action: int,
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
            next_action: action taken by agent one step later.
        """
        current_state_action_value = self._state_action_values[state][action]
        current_next_state_action_value = self._state_action_values[new_state][
            next_action
        ]

        # bootstrapped error in one-step return
        delta = (
            reward
            + self._gamma * current_next_state_action_value
            - current_state_action_value
        )

        # increment trace associated with visited state, action pair
        self._state_action_eligibility_traces[state][action] += 1

        # decay eligibility traces and update other state-action values
        for state in self._state_space:
            change = (
                self._learning_rate
                * delta
                * self._state_action_eligibility_traces[state]
                - self._visitation_penalty
            )
            self._state_action_values[state] += change
            self._state_action_eligibility_traces[state] *= (
                self._gamma * self._trace_lambda
            )
