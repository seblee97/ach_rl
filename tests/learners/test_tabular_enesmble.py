import itertools
import pdb

import numpy as np
import pytest
from learners.ensemble_learners import tabular_ensemble_learner
from learners.tabular_learners import tabular_learner

DUMMY_ACTION_SPACE = [0, 1]
DUMMY_STATE_SPACE = list(itertools.product(range(3), range(2)))
EPSILON = 1e-8


TEST_STATE_ACTION_VALUES_1 = np.array(
    [
        np.array([3, 4]),
        np.array([4, 5]),
        np.array([4, 3]),
        np.array([1, 2]),
        np.array([3, 2]),
        np.array([0, 1]),
    ]
)
TEST_STATE_ACTION_VALUES_2 = np.array(
    [
        np.array([1, 3]),
        np.array([-5, -7]),
        np.array([17, 19]),
        np.array([7, 5]),
        np.array([3, 5]),
        np.array([-1, 1]),
    ]
)
TEST_STATE_ACTION_VALUES_3 = np.array(
    [
        np.array([-2, 0]),
        np.array([2, 0]),
        np.array([-8, -10]),
        np.array([7, 9]),
        np.array([1, -1]),
        np.array([8, 10]),
    ]
)

MAX_1 = [1, 1, 0, 1, 0, 1]
MAX_2 = [1, 0, 1, 0, 1, 1]
MAX_3 = [1, 0, 0, 1, 0, 1]

MAX_PROBABILITIES_STATE_1 = np.array([0, 1]) + EPSILON
MAX_PROBABILITIES_STATE_2 = np.array([2 / 3, 1 / 3]) + EPSILON
MAX_PROBABILITIES_STATE_3 = np.array([2 / 3, 1 / 3]) + EPSILON
MAX_PROBABILITIES_STATE_4 = np.array([1 / 3, 2 / 3]) + EPSILON
MAX_PROBABILITIES_STATE_5 = np.array([1 / 3, 2 / 3]) + EPSILON
MAX_PROBABILITIES_STATE_6 = np.array([0, 1]) + EPSILON

ENTROPY_1 = -np.sum(MAX_PROBABILITIES_STATE_1 * np.log(MAX_PROBABILITIES_STATE_1))
ENTROPY_2 = -np.sum(MAX_PROBABILITIES_STATE_2 * np.log(MAX_PROBABILITIES_STATE_2))
ENTROPY_3 = -np.sum(MAX_PROBABILITIES_STATE_3 * np.log(MAX_PROBABILITIES_STATE_3))
ENTROPY_4 = -np.sum(MAX_PROBABILITIES_STATE_4 * np.log(MAX_PROBABILITIES_STATE_4))
ENTROPY_5 = -np.sum(MAX_PROBABILITIES_STATE_5 * np.log(MAX_PROBABILITIES_STATE_5))
ENTROPY_6 = -np.sum(MAX_PROBABILITIES_STATE_6 * np.log(MAX_PROBABILITIES_STATE_6))


class DummyTabularLearner(tabular_learner.TabularLearner):
    def step(self):
        pass


def _setup_single_learner(state_action_values):
    learner = DummyTabularLearner(
        action_space=DUMMY_ACTION_SPACE,
        state_space=DUMMY_STATE_SPACE,
        learning_rate=0.01,
        gamma=0.9,
        epsilon=0.1,
        initialisation_strategy="random",
        behaviour="greedy",
        target="greedy",
    )
    # manually set state action values for controlled test cases.
    learner._state_action_values = state_action_values
    return learner


def _setup():
    dummy_learners = [
        _setup_single_learner(state_action_values)
        for state_action_values in [
            TEST_STATE_ACTION_VALUES_1,
            TEST_STATE_ACTION_VALUES_2,
            TEST_STATE_ACTION_VALUES_3,
        ]
    ]

    dummy_ensemble_learner = tabular_ensemble_learner.TabularEnsembleLearner(
        learner_ensemble=dummy_learners
    )
    return dummy_ensemble_learner


ENTROPIES = [ENTROPY_1, ENTROPY_2, ENTROPY_3, ENTROPY_4, ENTROPY_5, ENTROPY_6]
POLICY_ENTROPY = {
    state: entropy for state, entropy in zip(DUMMY_STATE_SPACE, ENTROPIES)
}


def test_policy_entropy():
    ensemble_learner = _setup()
    for key, value in ensemble_learner.policy_entropy.items():
        print(value, POLICY_ENTROPY[key])
        assert value == pytest.approx(POLICY_ENTROPY[key])
