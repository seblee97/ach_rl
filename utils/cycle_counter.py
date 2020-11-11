from typing import Dict
from typing import List
from typing import Tuple

import numpy as np


def loop_on_state(
    size: Tuple[int, int],
    state_action_values: Dict[Tuple[int, int], np.ndarray],
    state: Tuple[int, int],
):
    state_history = [state]
    env = Tabular(size=size)
    current_state = state
    while True:
        action = np.argmax(state_action_values[current_state])
        current_state = env.step(current_state, action)
        if current_state in state_history:
            if current_state == state:
                return True, len(state_history)
            else:
                return False
        else:
            state_history.append(current_state)


def evaluate_loops_on_value_function(
    size: Tuple[int, int], state_action_values: Dict[Tuple[int, int], np.ndarray]
):
    loops = {}
    for state in state_action_values.keys():
        state_loop = loop_on_state(
            size=size, state_action_values=state_action_values, state=state
        )
        loops[state] = state_loop
    return loops


class Tabular:
    def __init__(self, size: Tuple[int, int]):
        self._size = size

    def step(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        state = list(state)
        if action == 0:
            new_state = self._move_agent_left(state)
        elif action == 1:
            new_state = self._move_agent_up(state)
        elif action == 2:
            new_state = self._move_agent_right(state)
        elif action == 3:
            new_state = self._move_agent_down(state)
        return tuple(new_state)

    def _move_agent_left(self, state) -> List[int]:
        """Move agent one position to the left. If already at left-most point, no-op."""
        if state[0] == 0:
            pass
        else:
            state[0] -= 1
        return state

    def _move_agent_up(self, state) -> List[int]:
        """Move agent one position upwards. If already at upper-most point, no-op."""
        if state[1] == self._size[1] - 1:
            pass
        else:
            state[1] += 1
        return state

    def _move_agent_right(self, state) -> List[int]:
        """Move agent one position upwards. If already at right-most point, no-op."""
        if state[0] == self._size[0] - 1:
            pass
        else:
            state[0] += 1
        return state

    def _move_agent_down(self, state) -> List[int]:
        """Move agent one position downwards. If already at bottom-most point, no-op."""
        if state[1] == 0:
            pass
        else:
            state[1] -= 1
        return state
