from typing import List
from typing import Tuple
from typing import Dict

from learners.tabular_learners import tabular_learner
from utils import epsilon_schedules


class TabularQLearner(tabular_learner.TabularLearner):
    """Q-learning (Watkins)."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: Dict,
        behaviour: str,
        target: str,
        epsilon: epsilon_schedules.EpsilonSchedule,
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
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            initialisation_strategy=initialisation_strategy,
            behaviour=behaviour,
            target=target,
        )
        self._epsilon = epsilon

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
        visitation_penalty: float,
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
            active: whether episode is still ongoing.
            visitation_penalty: penalty to apply to state-action pair for visit.
        """
        state_id = self._state_id_mapping[state]
        initial_state_action_value = self._state_action_values[state_id][action]

        if active:
            discount = self._gamma
        else:
            discount = 0

        self._state_visitation_counts[state] += 1

        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate
            * (
                reward
                + visitation_penalty
                + discount * self._max_state_action_value(state=new_state)
                - initial_state_action_value
            )
        )
        self._state_action_values[state_id][action] = updated_state_action_value

        # step epsilon
        next(self._epsilon)
