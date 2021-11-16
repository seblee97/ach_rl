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

hard_coded_changes = {
    f"hard_coded_{i}": [
        {"learner": {"hard_coded": {"vp_schedule": [[0, i]]}}},
        {"learner": {"visitation_penalty_type": "hard_coded"}},
    ]
    for i in [-0.1, 0.01, 0.01, 0.1]
}

policy_entropy_changes = {
    f"policy_entropy_{i}": [
        {"learner": {"policy_entropy_penalty": {"multiplicative_factor": i}}},
        {"learner": {"visitation_penalty_type": "policy_entropy_penalty"}},
    ]
    for i in [-0.1, 0.01, 0.01, 0.1]
}

uncertainty_changes = {
    f"uncertainty_{i}": [
        {"learner": {"adaptive_uncertainty": {"multiplicative_factor": i}}},
        {"learner": {"visitation_penalty_type": "adaptive_uncertainty"}},
    ]
    for i in [-0.1, 0.01, 0.01, 0.1]
}

baseline_changes = {
    "baseline": [
        {"learner": {"hard_coded": {"vp_schedule": [[0, 0]]}}},
        {"learner": {"visitation_penalty_type": "hard_coded"}},
    ],
}

CONFIG_CHANGES = {
    **baseline_changes,
    **hard_coded_changes,
    **policy_entropy_changes,
    **uncertainty_changes,
}
