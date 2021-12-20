"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import itertools

timeout_changes = {
    f"timeout_{i}": [{"multiroom": {"episode_timeout": i}}] for i in [100, 400, 1600]
}

num_learner_changes = {
    f"num_learners_{i}": [
        {"learner": {"bootstrapped_ensemble_dqn": {"num_learners": i}}}
    ]
    for i in [4, 16, 32]
}

shared_layer_changes = {
    "shared_layer_1": [
        {"learner": {"bootstrapped_ensemble_dqn": {"shared_layers": [0, 1, 2, 3]}}},
    ],
    "shared_layer_2": [
        {"learner": {"bootstrapped_ensemble_dqn": {"shared_layers": [0, 1, 2]}}},
    ],
}

mask_probability_changes = {
    f"mask_probability_{i}": [
        {"learner": {"bootstrapped_ensemble_dqn": {"mask_probability": i}}}
    ]
    for i in [0.2, 0.5, 0.8, 1]
}

buffer_size_changes = {
    f"buffer_size_{i}": [{"learner": {"dqn": {"replay_buffer_size": i}}}]
    for i in [1000, 10000, 100000]
}

target_net_update_changes = {
    f"update_period_{i}": [{"learner": {"dqn": {"target_network_update_period": i}}}]
    for i in [500, 1000, 10000, 50000]
}

learning_rate_changes = {
    f"lr_{i}": [{"learner": {"learning_rate": i}}] for i in [0.1, 0.01, 0.001, 0.0001]
}

anneal_duration_changes = {
    f"anneal_{i}": [{"learner": {"epsilon": {"linear_decy": {"anneal_duration": i}}}}]
    for i in [10000, 100000, 1000000, 5000000]
}

key_combos = itertools.product(
    timeout_changes.keys(),
    num_learner_changes.keys(),
    shared_layer_changes.keys(),
    mask_probability_changes.keys(),
    buffer_size_changes.keys(),
    target_net_update_changes.keys(),
    learning_rate_changes.keys(),
    anneal_duration_changes.keys(),
)

CONFIG_CHANGES = {
    "_".join(key_combo): timeout_changes[key_combo[0]]
    + num_learner_changes[key_combo[1]]
    + shared_layer_changes[key_combo[2]]
    + mask_probability_changes[key_combo[3]]
    + buffer_size_changes[key_combo[4]]
    + target_net_update_changes[key_combo[5]]
    + learning_rate_changes[key_combo[6]]
    + anneal_duration_changes[key_combo[7]]
    for key_combo in key_combos
}
