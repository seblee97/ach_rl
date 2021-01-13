import itertools
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pytest

from visitation_penalties import potential_adaptive_uncertainty_penalty


def _setup():
    _x, _y = np.arange(5), np.arange(5)
    _states = list(itertools.product(_x, _y))
    _actions = np.arange(4)
    _state_action_state_space = list(itertools.product(_states, _actions, _states))

    _learner_1 = {tuple(s): i * _actions for i, s in enumerate(_states)}
    _learner_2 = {tuple(s): 2 * i * _actions for i, s in enumerate(_states)}
    _learner_3 = {tuple(s): 1.0 * _actions for i, s in enumerate(_states)}

    _state_action_values = [_learner_1, _learner_2, _learner_3]

    _expected_penalties_0 = {
        sas_tuple: 2
        * (
            0.99
            * np.std([learner[sas_tuple[2]][-1] for learner in _state_action_values])
            - np.std([learner[sas_tuple[0]][-1] for learner in _state_action_values])
        )
        for sas_tuple in _state_action_state_space
    }
    _expected_penalties_1 = {
        sas_tuple: 2
        * (
            0.99
            * np.std(
                [
                    learner[sas_tuple[2]][sas_tuple[1]]
                    for learner in _state_action_values
                ]
            )
            - np.std(
                [
                    learner[sas_tuple[0]][sas_tuple[1]]
                    for learner in _state_action_values
                ]
            )
        )
        for sas_tuple in _state_action_state_space
    }
    _expected_penalties_2 = {sas_tuple: 0.0 for sas_tuple in _state_action_state_space}
    _expected_penalties_3 = {sas_tuple: 0.0 for sas_tuple in _state_action_state_space}
    _expected_penalties_4 = {
        sas_tuple: -2
        * (
            0.99
            * np.std([learner[sas_tuple[2]][-1] for learner in _state_action_values])
            - np.std([learner[sas_tuple[0]][-1] for learner in _state_action_values])
        )
        for sas_tuple in _state_action_state_space
    }
    _expected_penalties_5 = {
        sas_tuple: -2
        * (
            0.99
            * np.std(
                [
                    learner[sas_tuple[2]][sas_tuple[1]]
                    for learner in _state_action_values
                ]
            )
            - np.std(
                [
                    learner[sas_tuple[0]][sas_tuple[1]]
                    for learner in _state_action_values
                ]
            )
        )
        for sas_tuple in _state_action_state_space
    }

    test_params = [
        (2.0, True, _state_action_values, _expected_penalties_0),
        (2.0, False, _state_action_values, _expected_penalties_1),
        (0.0, True, _state_action_values, _expected_penalties_2),
        (0.0, False, _state_action_values, _expected_penalties_3),
        (-2.0, True, _state_action_values, _expected_penalties_4),
        (-2.0, False, _state_action_values, _expected_penalties_5),
    ]

    return _state_action_state_space, test_params


class TestPotentialAdaptiveUncertaintyPenalty:

    STATE_ACTION_SPACE, test_params = _setup()

    GAMMA = 0.99

    string_arg_signature = (
        "multiplicative_factor, max_over_actions, "
        "state_action_values, expected_penalties"
    )

    @pytest.mark.parametrize(string_arg_signature, test_params)
    def test_binary_class_labels(
        self,
        multiplicative_factor: Union[float, int],
        max_over_actions: bool,
        state_action_values: List[Dict[Tuple[int], List[float]]],
        expected_penalties,
    ):
        penalty_class = (
            potential_adaptive_uncertainty_penalty.PotentialAdaptiveUncertaintyPenalty(
                gamma=self.GAMMA,
                multiplicative_factor=multiplicative_factor,
                max_over_actions=max_over_actions,
            )
        )
        penalty_class.state_action_values = state_action_values
        all_penalties = {
            sas_tuple: penalty_class(
                state=sas_tuple[0], action=sas_tuple[1], next_state=sas_tuple[2]
            )
            for sas_tuple in self.STATE_ACTION_SPACE
        }
        assert all_penalties == expected_penalties
