from ach_rl import constants
from config_manager import config_field
from config_manager import config_template


class AChConfigTemplate:

    _minigrid_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.SIZE,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.LIVING_REWARD,
                types=[float, int, type(None)],
                requirements=[lambda x: x <= 0],
            ),
            config_field.Field(
                name=constants.NO_OP_PENALTY,
                types=[float, int, type(None)],
                requirements=[lambda x: x is None or x >= 0],
            ),
            config_field.Field(
                name=constants.STARTING_POSITION,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or (len(x) == 2 and all(y >= 0 for y in x))
                ],
            ),
            config_field.Field(
                name=constants.NUM_REWARDS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.REWARD_POSITIONS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, list) for y in x),
                    lambda x: x is None or all(len(y) == 2 for y in x),
                    lambda x: x is None or all(all(z >= 0 for z in y) for y in x),
                ],
            ),
            config_field.Field(
                name=constants.REWARD_MAGNITUDES,
                types=[list],
                requirements=[lambda x: all(isinstance(r, float) for r in x)],
            ),
            config_field.Field(
                name=constants.REPEAT_REWARDS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.EPISODE_TIMEOUT,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.PLOT_ORIGIN,
                types=[str],
                requirements=[lambda x: x in [constants.UPPER, constants.LOWER]],
            ),
        ],
        dependent_variables=[constants.ENVIRONMENT],
        dependent_variables_required_values=[[constants.MINIGRID]],
        level=[constants.MINIGRID],
    )

    _minigrid_curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TRANSITION_EPISODES,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) for y in x)],
            ),
            config_field.Field(
                name=constants.ENVIRONMENT_CHANGES,
                types=[list],
            ),
        ],
        level=[constants.MINIGRID_CURRICULUM],
        dependent_variables=[
            constants.ENVIRONMENT,
            constants.APPLY_CURRICULUM,
        ],
        dependent_variables_required_values=[[constants.MINIGRID], [True]],
    )

    _multiroom_curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TRANSITION_EPISODES,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) for y in x)],
            ),
            config_field.Field(
                name=constants.ENVIRONMENT_CHANGES,
                types=[list],
            ),
        ],
        level=[constants.MULTIIROOM_CURRICULUM],
        dependent_variables=[
            constants.ENVIRONMENT,
            constants.APPLY_CURRICULUM,
        ],
        dependent_variables_required_values=[[constants.MULTIROOM], [True]],
    )

    _multiroom_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.MAP_ASCII_PATH, types=[str]),
            config_field.Field(name=constants.MAP_YAML_PATH, types=[str]),
            config_field.Field(
                name=constants.EPISODE_TIMEOUT,
                types=[type(None), int],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.PLOT_ORIGIN,
                types=[str],
                requirements=[lambda x: x in [constants.UPPER, constants.LOWER]],
            ),
            config_field.Field(
                name=constants.REWARD_SPECIFICATIONS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, dict) for y in x)],
            ),
            config_field.Field(
                name=constants.REPRESENTATION,
                types=[str],
                requirements=[
                    lambda x: x in [constants.PIXEL, constants.AGENT_POSITION]
                ],
            ),
            config_field.Field(
                name=constants.ENCODED_STATE_DIMENSIONS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.FRAME_STACK,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.MULTIROOM],
        dependent_variables=[
            constants.ENVIRONMENT,
        ],
        dependent_variables_required_values=[[constants.MULTIROOM]],
    )

    _atari_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ATARI_ENV_NAME,
                types=[str],
                requirements=[lambda x: x in constants.ATARI_ENVS],
            ),
            config_field.Field(
                name=constants.IMPLEMENTATION,
                types=[str],
                requirements=[lambda x: x in [constants.FUNCTIONAL, constants.WRAPPER]],
            ),
            config_field.Field(
                name=constants.PRE_PROCESSING,
                types=[list],
                requirements=[lambda x: all(isinstance(y, dict) for y in x)],
            ),
            config_field.Field(
                name=constants.FRAME_STACK,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.FRAME_SKIP,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.EPISODE_TIMEOUT,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.ENCODED_STATE_DIMENSIONS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.PLOT_ORIGIN,
                types=[str],
                requirements=[lambda x: x in [constants.UPPER, constants.LOWER]],
            ),
        ],
        dependent_variables=[constants.ENVIRONMENT],
        dependent_variables_required_values=[[constants.ATARI]],
        level=[constants.ATARI],
    )

    _hard_coded_vp_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.VP_SCHEDULE,
                types=[list],
                requirements=[
                    lambda x: all(isinstance(y, list) and len(y) == 2 for y in x)
                ],
            )
        ],
        level=[constants.LEARNER, constants.HARD_CODED],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.HARD_CODED]],
    )

    _exponential_decay_vp_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.A, types=[float, int]),
            config_field.Field(name=constants.b, types=[float, int]),
            config_field.Field(name=constants.c, types=[float, int]),
        ],
        level=[
            constants.LEARNER,
            constants.DETERMINISTIC_EXPONENTIAL_DECAY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[
            [constants.DETERMINISTIC_EXPONENTIAL_DECAY]
        ],
    )

    _linear_decay_vp_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.A, types=[float, int]),
            config_field.Field(name=constants.b, types=[float, int]),
        ],
        level=[
            constants.LEARNER,
            constants.DETERMINISTIC_LINEAR_DECAY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.DETERMINISTIC_LINEAR_DECAY]],
    )

    _sigmoidal_decay_vp_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.A, types=[float, int]),
            config_field.Field(name=constants.b, types=[float, int]),
            config_field.Field(name=constants.c, types=[float, int]),
        ],
        level=[
            constants.LEARNER,
            constants.DETERMINISTIC_SIGMOIDAL_DECAY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.DETERMINISTIC_SIGMOIDAL_DECAY]],
    )

    _adaptive_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                        constants.SELECT,
                    ]
                ],
            ),
        ],
        level=[constants.LEARNER, constants.ADAPTIVE_UNCERTAINTY],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.ADAPTIVE_UNCERTAINTY]],
    )

    _adaptive_arriving_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                    ]
                ],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.ADAPTIVE_ARRIVING_UNCERTAINTY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.ADAPTIVE_ARRIVING_UNCERTAINTY]],
    )

    _potential_based_adaptive_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.PRE_ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                        constants.SELECT,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.POST_ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                    ]
                ],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[
            [constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY]
        ],
    )

    _policy_entropy_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.POLICY_ENTROPY_PENALTY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.POLICY_ENTROPY_PENALTY]],
    )

    _potential_based_policy_entropy_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.POTENTIAL_BASED_POLICY_ENTROPY_PENALTY,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[
            [constants.POTENTIAL_BASED_POLICY_ENTROPY_PENALTY]
        ],
    )

    _reducing_variance_window_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.EXPECTED_MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.UNEXPECTED_MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.MOVING_AVERAGE_WINDOW,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.REDUCING_VARIANCE_WINDOW,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.REDUCING_VARIANCE_WINDOW]],
    )

    _reducing_entropy_window_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.EXPECTED_MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.UNEXPECTED_MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.MOVING_AVERAGE_WINDOW,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.REDUCING_ENTROPY_WINDOW,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.REDUCING_ENTROPY_WINDOW]],
    )

    _signed_uncertainty_window_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.POSITIVE_MULITPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.NEGATIVE_MULITPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MEAN,
                        constants.MAX,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.MOVING_AVERAGE_WINDOW,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.SIGNED_UNCERTAINTY_WINDOW,
        ],
        dependent_variables=[constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[
            [constants.SIGNED_UNCERTAINTY_WINDOW_PENALTY]
        ],
    )

    _hard_coded_lr_scaler_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.LR_SCALING, types=[float, int]),
        ],
        level=[constants.LEARNER, constants.LR_SCALER, constants.HARD_CODED],
        dependent_variables=[constants.LR_SCALER_TYPE],
        dependent_variables_required_values=[
            [constants.HARD_CODED],
        ],
    )

    _expected_uncertainty_lr_scaler_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[lambda x: x in [constants.MAX, constants.MEAN]],
                key=constants.LR_SCALER_ACTION_FUNCTION,
            ),
        ],
        level=[
            constants.LEARNER,
            constants.LR_SCALER,
            constants.EXPECTED_UNCERTAINTY,
        ],
        dependent_variables=[constants.LR_SCALER_TYPE],
        dependent_variables_required_values=[
            [constants.EXPECTED_UNCERTAINTY],
        ],
    )

    _lr_scaler_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                types=[str, type(None)],
                requirements=[
                    lambda x: x is None
                    or x in [constants.EXPECTED_UNCERTAINTY, constants.HARD_CODED]
                ],
                key=constants.LR_SCALER_TYPE,
            ),
        ],
        level=[constants.LEARNER, constants.LR_SCALER],
        nested_templates=[
            _hard_coded_lr_scaler_template,
            _expected_uncertainty_lr_scaler_template,
        ],
    )

    _constant_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            )
        ],
        level=[
            constants.LEARNER,
            constants.EPSILON,
            constants.CONSTANT,
        ],
        dependent_variables=[constants.SCHEDULE],
        dependent_variables_required_values=[[constants.CONSTANT]],
    )

    _linear_decay_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INITIAL_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.FINAL_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.ANNEAL_DURATION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.EPSILON,
            constants.LINEAR_DECAY,
        ],
        dependent_variables=[constants.SCHEDULE],
        dependent_variables_required_values=[
            [constants.LINEAR_DECAY],
        ],
    )

    _expected_uncertainty_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[lambda x: x in [constants.MAX, constants.MEAN]],
                key=constants.EPSILON_ACTION_FUNCTION,
            ),
            config_field.Field(
                name=constants.MINIMUM_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.EPSILON,
            constants.EXPECTED_UNCERTAINTY,
        ],
        dependent_variables=[constants.SCHEDULE],
        dependent_variables_required_values=[
            [constants.EXPECTED_UNCERTAINTY],
        ],
    )

    _unexpected_uncertainty_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ACTION_FUNCTION,
                types=[str],
                requirements=[lambda x: x in [constants.MAX, constants.MEAN]],
                key=constants.EPSILON_ACTION_FUNCTION,
            ),
            config_field.Field(
                name=constants.MOVING_AVERAGE_WINDOW,
                types=[int],
                requirements=[lambda x: x > 0],
                key=constants.EPSILON_MOVING_AVERAGE_WINDOW,
            ),
            config_field.Field(
                name=constants.MINIMUM_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
        ],
        level=[
            constants.LEARNER,
            constants.EPSILON,
            constants.UNEXPECTED_UNCERTAINTY,
        ],
        dependent_variables=[constants.SCHEDULE],
        dependent_variables_required_values=[
            [constants.UNEXPECTED_UNCERTAINTY],
        ],
    )

    _epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.SCHEDULE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.CONSTANT,
                        constants.LINEAR_DECAY,
                        constants.UNEXPECTED_UNCERTAINTY,
                    ]
                ],
            ),
        ],
        level=[constants.LEARNER, constants.EPSILON],
        nested_templates=[
            _constant_epsilon_template,
            _linear_decay_epsilon_template,
            _expected_uncertainty_epsilon_template,
            _unexpected_uncertainty_epsilon_template,
        ],
    )

    _random_uniform_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.LOWER_BOUND, types=[float, int]),
            config_field.Field(name=constants.UPPER_BOUND, types=[float, int]),
        ],
        level=[constants.LEARNER, constants.RANDOM_UNIFORM],
        dependent_variables=[constants.INITIALISATION],
        dependent_variables_required_values=[[constants.RANDOM_UNIFORM]],
    )

    _random_normal_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.MEAN, types=[float, int]),
            config_field.Field(name=constants.VARIANCE, types=[float, int]),
        ],
        level=[constants.LEARNER, constants.RANDOM_NORMAL],
        dependent_variables=[constants.INITIALISATION],
        dependent_variables_required_values=[[constants.RANDOM_NORMAL]],
    )

    _learner_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.SARSA_LAMBDA,
                        constants.Q_LEARNING,
                        constants.VANILLA_DQN,
                        constants.ENSEMBLE_Q_LEARNING,
                        constants.BOOTSTRAPPED_ENSEMBLE_DQN,
                        constants.INDEPENDENT_ENSEMBLE_DQN,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.PRETRAINED_MODEL_PATH,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.GRADIENT_MOMENTUM,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.SQUARED_GRADIENT_MOMENTUM,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.MIN_SQUARED_GRADIENT,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.DISCOUNT_FACTOR,
                types=[float, int],
                requirements=[lambda x: x <= 1 and x >= 0],
            ),
            config_field.Field(
                name=constants.INITIALISATION,
                types=[str, float, int],
                requirements=[
                    lambda x: x
                    in [
                        constants.RANDOM_UNIFORM,
                        constants.RANDOM_NORMAL,
                        constants.ZEROS,
                        constants.ONES,
                    ]
                    or isinstance(x, (int, float))
                ],
            ),
            config_field.Field(
                name=constants.SPLIT_VALUE_FUNCTION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.VISITATION_PENALTY_TYPE,
                types=[str, type(None)],
                requirements=[
                    lambda x: x is None
                    or x
                    in [
                        constants.HARD_CODED,
                        constants.DETERMINISTIC_EXPONENTIAL_DECAY,
                        constants.DETERMINISTIC_LINEAR_DECAY,
                        constants.DETERMINISTIC_SIGMOIDAL_DECAY,
                        constants.ADAPTIVE_UNCERTAINTY,
                        constants.ADAPTIVE_ARRIVING_UNCERTAINTY,
                        constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY,
                        constants.POLICY_ENTROPY_PENALTY,
                        constants.POTENTIAL_BASED_POLICY_ENTROPY_PENALTY,
                        constants.REDUCING_VARIANCE_WINDOW,
                        constants.REDUCING_ENTROPY_WINDOW,
                        constants.SIGNED_UNCERTAINTY_WINDOW_PENALTY,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.SHAPING_IMPLEMENTATION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.ACT,
                        constants.TRAIN_Q_NETWORK,
                        constants.TRAIN_TARGET_NETWORK,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.INFORMATION_COMPUTER_UPDATE_PERIOD,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.LEARNER],
        nested_templates=[
            _epsilon_template,
            _lr_scaler_template,
            _random_uniform_template,
            _random_normal_template,
            _hard_coded_vp_template,
            _exponential_decay_vp_template,
            _linear_decay_vp_template,
            _sigmoidal_decay_vp_template,
            _adaptive_uncertainty_template,
            _adaptive_arriving_uncertainty_template,
            _potential_based_adaptive_uncertainty_template,
            _policy_entropy_penalty_template,
            _potential_based_policy_entropy_penalty_template,
            _reducing_variance_window_penalty_template,
            _reducing_entropy_window_penalty_template,
            _signed_uncertainty_window_penalty_template,
        ],
    )

    _sarsa_lambda_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TRACE_LAMBDA,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[
            [constants.SARSA_LAMBDA],
        ],
        level=[constants.SARSA_LAMBDA],
    )

    _q_learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[[constants.Q_LEARNING]],
        level=[constants.Q_LEARNING],
    )

    _ensemble_q_learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_LEARNERS,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.COPY_LEARNER_INITIALISATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGETS,
                types=[list],
                requirements=[
                    lambda x: x is None
                    or all(
                        y
                        in [
                            constants.GREEDY_SAMPLE,
                            constants.GREEDY_MEAN,
                            constants.GREEDY_VOTE,
                        ]
                        for y in x
                    )
                ],
            ),
            config_field.Field(name=constants.PARALLELISE_ENSEMBLE, types=[bool]),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[[constants.ENSEMBLE_Q_LEARNING]],
        level=[constants.ENSEMBLE_Q_LEARNING],
    )

    _vanilla_dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGETS,
                types=[list, type(None)],
                requirements=[lambda x: x is None or all(y in [] for y in x)],
            ),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[
            [constants.VANILLA_DQN],
        ],
        level=[constants.VANILLA_DQN],
    )

    _independent_ensemble_dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_LEARNERS,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.COPY_LEARNER_INITIALISATION,
                types=[bool],
            ),
            config_field.Field(name=constants.SHARE_REPLAY_BUFFER, types=[bool]),
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGETS,
                types=[list, type(None)],
                requirements=[lambda x: x is None or all(y in [] for y in x)],
            ),
            config_field.Field(name=constants.PARALLELISE_ENSEMBLE, types=[bool]),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[
            [constants.INDEPENDENT_ENSEMBLE_DQN],
        ],
        level=[constants.INDEPENDENT_ENSEMBLE_DQN],
    )

    _bootstrapped_ensemble_dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_LEARNERS,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.COPY_LEARNER_INITIALISATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SHARED_LAYERS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None
                    or (isinstance(x, list) and all(isinstance(y, int) for y in x))
                ],
            ),
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [constants.GREEDY, constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.TARGETS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None
                    or all(
                        y
                        in [
                            constants.GREEDY_INDIVIDUAL,
                            constants.GREEDY_SAMPLE,
                            constants.GREEDY_MEAN,
                            constants.GREEDY_VOTE,
                        ]
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.MASK_PROBABILITY,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[
            [constants.BOOTSTRAPPED_ENSEMBLE_DQN],
        ],
        level=[constants.BOOTSTRAPPED_ENSEMBLE_DQN],
    )

    _dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NUM_REPLAY_FILL_TRAJECTORIES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.REPLAY_BUFFER_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TARGET_NETWORK_UPDATE_PERIOD,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(name=constants.NORMALISE_STATE, types=[bool]),
            config_field.Field(
                name=constants.GRADIENT_CLIPPING,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None
                    or (
                        len(x) == 2
                        and all((isinstance(y, float) or isinstance(y, int)) for y in x)
                    )
                ],
            ),
            config_field.Field(
                name=constants.OPTIMISER,
                types=[str],
                requirements=[lambda x: x in [constants.ADAM, constants.RMS_PROP]],
            ),
            config_field.Field(
                name=constants.LAYER_SPECIFICATIONS,
                types=[list],
            ),
        ],
        dependent_variables=[constants.TYPE],
        dependent_variables_required_values=[
            [
                constants.VANILLA_DQN,
                constants.BOOTSTRAPPED_ENSEMBLE_DQN,
                constants.INDEPENDENT_ENSEMBLE_DQN,
            ]
        ],
        level=[constants.DQN],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_EPISODES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TESTING,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x)
                ],
            ),
        ],
        level=[constants.TRAINING],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.PRINT_FREQUENCY,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.MODEL_CHECKPOINT_FREQUENCY,
                types=[type(None), int],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.ANIMATION_LIBRARY,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MATPLOTLIB_ANIMATION,
                        constants.IMAGEIO,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.ANIMATION_FILE_FORMAT,
                types=[str],
                requirements=[lambda x: x in [constants.MP4, constants.GIF]],
            ),
            config_field.Field(
                name=constants.ARRAYS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None
                    or all(isinstance(y, list) and isinstance(y[1], int) for y in x),
                    lambda x: x is None
                    or all(
                        (
                            isinstance(z, str)
                            or (
                                isinstance(z, list)
                                and isinstance(z[0], str)
                                and isinstance(z[1], int)
                            )
                            for z in y[0]
                        )
                        for y in x
                    ),
                ],
            ),
            config_field.Field(
                name=constants.SCALARS,
                types=[list],
                requirements=[
                    lambda x: all(
                        isinstance(y, list) and isinstance(y[1], int) for y in x
                    ),
                    lambda x: all(
                        (
                            isinstance(z, str)
                            or (
                                isinstance(z, list)
                                and isinstance(z[0], str)
                                and isinstance(z[1], int)
                            )
                            for z in y[0]
                        )
                        for y in x
                    ),
                ],
            ),
            config_field.Field(
                name=constants.VISUALISATIONS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None
                    or all(isinstance(y, list) and isinstance(y[1], int) for y in x),
                    lambda x: x is None
                    or all(
                        (
                            isinstance(z, str)
                            or (
                                isinstance(z, list)
                                and isinstance(z[0], str)
                                and isinstance(z[1], int)
                            )
                            for z in y[0]
                        )
                        for y in x
                    ),
                ],
            ),
            config_field.Field(
                name=constants.POST_VISUALISATIONS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x),
                ],
            ),
        ],
        level=[constants.LOGGING],
    )

    _plotting_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.XLABEL, types=[str]),
            config_field.Field(name=constants.SMOOTHING, types=[int]),
        ],
        level=[constants.PLOTTING],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(name=constants.USE_GPU, types=[bool]),
            config_field.Field(
                name=constants.GPU_ID,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.ENVIRONMENT,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.MINIGRID,
                        constants.ATARI,
                        constants.MULTIROOM,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.APPLY_CURRICULUM,
                types=[bool],
            ),
        ],
        nested_templates=[
            _minigrid_template,
            _minigrid_curriculum_template,
            _multiroom_curriculum_template,
            _multiroom_template,
            _atari_template,
            _learner_template,
            _sarsa_lambda_template,
            _q_learning_template,
            _vanilla_dqn_template,
            _ensemble_q_learning_template,
            _bootstrapped_ensemble_dqn_template,
            _independent_ensemble_dqn_template,
            _dqn_template,
            _training_template,
            _logging_template,
            _plotting_template,
        ],
    )
