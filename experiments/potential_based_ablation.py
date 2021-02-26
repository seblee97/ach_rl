_baseline = {
    "baseline": [
        {"learner": {"visitation_penalty_type": "hard_coded"}},
        {"hard_coded": {"vp_schedule": [[0, 0]]}},
    ],
}

_hard_coded = {
    f"hard_coded_{i}": [
        {
            "learner": {
                "visitation_penalty_type": "hard_coded",
                "hard_coded": {"vp_schedule": [[0, i]]},
            }
        },
    ]
    for i in [-0.1, -0.01, -1]
}

_policy_entropy = {
    f"policy_entropy_{i}": [
        {
            "learner": {
                "visitation_penalty_type": "policy_entropy_penalty",
                "policy_entropy_penalty": {"multiplicative_factor": i},
            }
        },
    ]
    for i in [-0.1, -0.01, -1]
}

_potential_policy_entropy = {
    f"potential_policy_entropy_{i}": [
        {
            "learner": {
                "visitation_penalty_type": "potential_based_policy_entropy_penalty",
                "potential_based_policy_entropy_penalty": {"multiplicative_factor": i},
            }
        },
    ]
    for i in [-0.1, -0.01, -1]
}

_adaptive_uncertainty = {
    f"uncertainty_{i}": [
        {
            "learner": {
                "visitation_penalty_type": "adaptive_uncertainty",
                "adaptive_uncertainty": {"multiplicative_factor": i},
            }
        },
    ]
    for i in [-0.1, -0.01, -1]
}

_potential_based_adaptive_uncertainty = {
    f"potential_uncertainty_{i}": [
        {
            "learner": {
                "visitation_penalty_type": "potential_based_adaptive_uncertainty",
                "potential_based_adaptive_uncertainty": {"multiplicative_factor": i},
            }
        },
    ]
    for i in [-0.1, -0.01, -1]
}

CONFIG_CHANGES = {
    **_baseline,
    **_hard_coded,
    **_policy_entropy,
    **_potential_policy_entropy,
    **_adaptive_uncertainty,
    **_potential_based_adaptive_uncertainty,
}
