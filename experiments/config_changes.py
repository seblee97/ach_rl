"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""


class ConfigChange:

    config_changes = {
        "vp_0": [("vp_schedule", [[0, 0.0]])],
        "vp_0.001": [("vp_schedule", [[0, 0.001]])],
        "vp_0.002": [("vp_schedule", [[0, 0.002]])],
        "vp_0.003": [("vp_schedule", [[0, 0.003]])],
        "optimistic_initialisation": [
            ("vp_schedule", [[0, 0]]),
            ("initialisation", 1.5),
        ],
    }
