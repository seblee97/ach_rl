from collections import namedtuple

import constants

Transition = namedtuple(
    typename=constants.Constants.TRANSITION,
    field_names=[
        constants.Constants.STATE_ENCODING,
        constants.Constants.ACTION,
        constants.Constants.REWARD,
        constants.Constants.NEXT_STATE_ENCODING,
        constants.Constants.ACTIVE,
    ],
)

MaskedTransition = namedtuple(
    typename=constants.Constants.TRANSITION,
    field_names=[
        constants.Constants.STATE_ENCODING,
        constants.Constants.ACTION,
        constants.Constants.REWARD,
        constants.Constants.NEXT_STATE_ENCODING,
        constants.Constants.ACTIVE,
        constants.Constants.MASK,
    ],
)

MaskedPenaltyTransition = namedtuple(
    typename=constants.Constants.TRANSITION,
    field_names=[
        constants.Constants.STATE_ENCODING,
        constants.Constants.ACTION,
        constants.Constants.REWARD,
        constants.Constants.NEXT_STATE_ENCODING,
        constants.Constants.ACTIVE,
        constants.Constants.MASK,
        constants.Constants.PENALTY,
    ],
)

PenaltyTransition = namedtuple(
    typename=constants.Constants.TRANSITION,
    field_names=[
        constants.Constants.STATE_ENCODING,
        constants.Constants.ACTION,
        constants.Constants.REWARD,
        constants.Constants.NEXT_STATE_ENCODING,
        constants.Constants.ACTIVE,
        constants.Constants.PENALTY,
    ],
)
