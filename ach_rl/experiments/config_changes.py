"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

# CONFIG_CHANGES = {
#     f"vp_{i}": [{
#         "learner": {
#             "hard_coded": {
#                 "vp_schedule": [[0, float(i)]]
#             }
#         }
#     }] for i in range(2)
# }

import itertools

hard_coded_penalty_changes = {
    f"hard_coded_penatly_{i}": [
        {
            "learner": {
                "visitation_penalty": {
                    "type": "hard_coded",
                    "hard_coded": {"vp_schedule": [[0, i]]},
                }
            }
        }
    ]
    for i in [-0.01, -0.1, -1]
}

policy_entropy_penalty_changes = {
    f"policy_entropy_penatly_{i}": [
        {
            "learner": {
                "visitation_penalty": {
                    "type": "policy_entropy_penalty",
                    "policy_entropy_penalty": {"multiplicative_factor": i},
                }
            }
        }
    ]
    for i in [-0.01, -0.1, -1]
}

uncertainty_penalty_changes = {
    f"uncertainty_penatly_{i}": [
        {
            "learner": {
                "visitation_penalty": {
                    "type": "adaptive_uncertainty",
                    "adaptive_uncertainty": {"multiplicative_factor": i},
                }
            }
        },
    ]
    for i in [-0.01, -0.1, -1]
}

baseline_penalty_changes = {
    "baseline_penatly": [
        {
            "learner": {
                "visitation_penalty": {
                    "type": "hard_coded",
                    "hard_coded": {"vp_schedule": [[0, 0]]},
                }
            }
        },
    ],
}

hard_coded_lr_scaler_changes = {
    f"hard_coded_lr_scaler_{i}": [
        {
            "learner": {
                "lr_scaler": {"type": "hard_coded", "hard_coded": {"lr_scaling": i}}
            }
        }
    ]
    for i in [0.01, 0.1, 1, 10]
}

expected_uncertainty_max_lr_scaler_changes = {
    f"expected_uncertainty_max_lr_scaler_{i}": [
        {
            "learner": {
                "lr_scaler": {
                    "type": "expected_uncertainty",
                    "expected_uncertainty": {
                        "action_function": "max",
                        "multiplicative_factor": i,
                    },
                }
            }
        }
    ]
    for i in [0.001, 0.01, 0.1, 1, 10]
}

constant_epsilon_changes = {
    f"constant_epsilon_{i}": [
        {"learner": {"epsilon": {"schedule": "constant", "constant": {"value": i}}}}
    ]
    for i in [0.01, 0.1]
}

unexpected_uncertainty_epsilon_changes = {
    f"unexpected_uncertainty_epsilon_{i}": [
        {
            "learner": {
                "epsilon": {
                    "schedule": "unexpected_uncertainty",
                    "unexpected_uncertainty": {
                        "action_function": "max",
                        "moving_average_window": i,
                        "minimum_value": j,
                    },
                }
            }
        }
    ]
    for i, j in itertools.product([5, 10, 50], [0.01, 0.1, 1])
}

expected_uncertainty_epsilon_changes = {
    f"expected_uncertainty_epsilon_{i}": [
        {
            "learner": {
                "epsilon": {
                    "schedule": "expected_uncertainty",
                    "expected_uncertainty": {
                        "action_function": "max",
                        "minimum_value": i,
                    },
                }
            }
        }
    ]
    for i in [0.01, 0.1, 1]
}

vp_types = {
    **baseline_penalty_changes,
    **hard_coded_penalty_changes,
    **policy_entropy_penalty_changes,
    **uncertainty_penalty_changes,
}

lr_scaler_types = {
    **hard_coded_lr_scaler_changes,
    **expected_uncertainty_max_lr_scaler_changes,
}

epsilon_types = {
    **constant_epsilon_changes,
    **unexpected_uncertainty_epsilon_changes,
    **expected_uncertainty_epsilon_changes,
}

CONFIG_CHANGES = {
    "_".join(key_combo): vp_types[key_combo[0]]
    + lr_scaler_types[key_combo[1]]
    + epsilon_types[key_combo[2]]
    for key_combo in itertools.product(
        vp_types.keys(), lr_scaler_types.keys(), epsilon_types.keys()
    )
}
