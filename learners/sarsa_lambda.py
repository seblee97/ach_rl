from typing import List, Optional, Tuple

import constants

from learners import tabular_learner


class TabularSARSALambda(tabular_learner.TabularLearner):
    """Backward SARSA(lambda)."""

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
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            initialisation_strategy=initialisation_strategy,
            visitation_penalty=visitation_penalty,
        )
        self._state_action_eligibility_traces = self._initialise_values(
            initialisation_strategy=constants.Constants.ZEROS
        )
        self._trace_lambda = trace_lambda

    def select_target_action(self, state: Tuple[int, int]) -> int:
        """Select action according to target policy, i.e. policy being learned.
        Here, action with highest value in given state is selected.

        Args:
            state: current state.

        Returns:
            action: greedy action.
        """
        action = self._greedy_action(state=state)
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
        action = self._greedy_action(state=state)
        return action

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
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
            next_state: next state.
            next_action: action taken by agent one step later.
        """
        state_id = self._state_id_mapping[state]
        next_state_id = self._state_id_mapping[next_state]

        current_state_action_value = self._state_action_values[state_id][action]
        current_next_state_action_value = self._state_action_values[next_state_id][
            next_action
        ]

        # bootstrapped error in one-step return
        delta = (
            reward
            + self._gamma * current_next_state_action_value
            - current_state_action_value
        )

        # increment trace associated with visited state, action pair
        self._state_action_eligibility_traces[state_id][action] += 1

        # update other state-action values
        self._state_action_values += (
            self._learning_rate * delta * self._state_action_eligibility_traces
        )

        # decay eligibility traces
        self._state_action_eligibility_traces *= self._gamma * self._trace_lambda

        # apply visitation penalty
        self._state_action_values[state_id][action] -= self._visitation_penalty
