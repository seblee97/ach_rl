from collections import namedtuple

from ach_rl import constants

Transition = namedtuple(
    typename=constants.TRANSITION,
    field_names=[
        constants.STATE_ENCODING,
        constants.ACTION,
        constants.REWARD,
        constants.NEXT_STATE_ENCODING,
        constants.ACTIVE,
    ],
)

MaskedTransition = namedtuple(
    typename=constants.TRANSITION,
    field_names=[
        constants.STATE_ENCODING,
        constants.ACTION,
        constants.REWARD,
        constants.NEXT_STATE_ENCODING,
        constants.ACTIVE,
        constants.MASK,
    ],
)

MaskedPenaltyTransition = namedtuple(
    typename=constants.TRANSITION,
    field_names=[
        constants.STATE_ENCODING,
        constants.ACTION,
        constants.REWARD,
        constants.NEXT_STATE_ENCODING,
        constants.ACTIVE,
        constants.MASK,
        constants.PENALTY,
    ],
)

PenaltyTransition = namedtuple(
    typename=constants.TRANSITION,
    field_names=[
        constants.STATE_ENCODING,
        constants.ACTION,
        constants.REWARD,
        constants.NEXT_STATE_ENCODING,
        constants.ACTIVE,
        constants.PENALTY,
    ],
)
