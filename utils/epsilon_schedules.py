from typing import Union

import abc


class EpsilonSchedule(abc.ABC):
    def __init__(self, value: Union[int, float]):
        self._value = value

    @abc.abstractmethod
    def __next__(self):
        pass

    @property
    def value(self):
        return self._value


class ConstantEpsilon(EpsilonSchedule):
    def __init__(self, value: Union[int, float]):
        super().__init__(value=value)

    def __next__(self):
        pass


class LinearDecayEpsilon(EpsilonSchedule):
    def __init__(
        self,
        initial_value: Union[int, float],
        final_value: Union[int, float],
        anneal_duration: int,
    ):

        self._step_size = (final_value - initial_value) / anneal_duration

        super().__init__(value=initial_value)

    def __next__(self):
        self._value -= self._step_size
