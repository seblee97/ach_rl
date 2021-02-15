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

CONFIG_CHANGES = {
    "baseline": [],
    "hard_coded": [{
        "learner": {
            "hard_coded": {
                "vp_schedule": [[0, -0.025]]
            }
        }
    }, {
        "learner": {
            "visitation_penalty_type": "hard_coded"
        }
    }],
    "policy_entropy": [{
        "learner": {
            "policy_entropy_penalty": {
                "multiplicative_factor": -0.01
            }
        }
    }, {
        "learner": {
            "visitation_penalty_type": "policy_entropy_penalty"
        }
    }]
}
