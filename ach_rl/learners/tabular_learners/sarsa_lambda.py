from typing import List
from typing import Tuple

from ach_rl import constants
from ach_rl.learners.tabular_learners import tabular_learner
from ach_rl.utils import epsilon_schedules


class TabularSARSALambda(tabular_learner.TabularLearner):
    """Backward SARSA(lambda)."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: float,
        gamma: float,
        initialisation_strategy: str,
        behaviour: str,
        target: str,
        trace_lambda: float,
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
            trace_lambda: trace parameter, how far to extend trace.
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
        self._state_action_eligibility_traces = self._initialise_values(
            initialisation_strategy=constants.ZEROS
        )
        self._trace_lambda = trace_lambda

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        next_action: int,
        active: bool,
        visitation_penalty: float,
    ) -> None:
        """Update state-action values via sarsa-lambda.

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            next_state: next state.
            next_action: action taken by agent one step later.
            active: whether episode is still ongoing.
            visitation_penalty: penalty to apply to state-action pair for visit.
        """
        state_id = self._state_id_mapping[state]
        next_state_id = self._state_id_mapping[next_state]

        current_state_action_value = self._state_action_values[state_id][action]
        current_next_state_action_value = self._state_action_values[next_state_id][
            next_action
        ]

        if active:
            discount = self._gamma
        else:
            discount = 0

        # bootstrapped error in one-step return
        delta = (
            reward
            + discount * current_next_state_action_value
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
        self._state_action_values[state_id][action] -= visitation_penalty

        # step epsilon
        next(self._epsilon)
