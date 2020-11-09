import random
from typing import List, Optional, Tuple

from learners import tabular_learner


class TabularQLearner(tabular_learner.TabularLearner):
    """Q-learning (Watkins)."""

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
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            initialisation_strategy=initialisation_strategy,
            visitation_penalty=visitation_penalty,
        )
        self._epsilon = epsilon

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
        action = self._epsilon_greedy_action(state=state, epsilon=self._epsilon)
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
                            + gamma * max_a(Q(s_{t+1}, a))
                            - Q(s_t, at_t)
                            ]

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
        """
        state_id = self._state_id_mapping[state]
        initial_state_action_value = self._state_action_values[state_id][action]

        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate
            * (
                reward
                + self._gamma * self._max_state_action_value(state=new_state)
                - initial_state_action_value
            )
        )
        self._state_action_values[state_id][action] = (
            updated_state_action_value - self._visitation_penalty
        )
