import constants
from config_manager import config_field
from config_manager import config_template


class AChConfigTemplate:

    _minigrid_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.SIZE,
                types=[list],
                requirements=[
                    lambda x: all(isinstance(y, int) and y > 0 for y in x)
                ],
            ),
            config_field.Field(
                name=constants.Constants.LIVING_REWARD,
                types=[float, int, type(None)],
                requirements=[lambda x: x <= 0],
            ),
            config_field.Field(
                name=constants.Constants.NO_OP_PENALTY,
                types=[float, int, type(None)],
                requirements=[lambda x: x is None or x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.STARTING_POSITION,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or
                    (len(x) == 2 and all(y >= 0 for y in x))
                ],
            ),
            config_field.Field(
                name=constants.Constants.NUM_REWARDS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.REWARD_POSITIONS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, list) for y in x),
                    lambda x: x is None or all(len(y) == 2 for y in x),
                    lambda x: x is None or all(
                        all(z >= 0 for z in y) for y in x),
                ],
            ),
            config_field.Field(
                name=constants.Constants.REWARD_MAGNITUDES,
                types=[list],
                requirements=[lambda x: all(isinstance(r, float) for r in x)],
            ),
            config_field.Field(
                name=constants.Constants.REPEAT_REWARDS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.EPISODE_TIMEOUT,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.Constants.PLOT_ORIGIN,
                types=[str],
                requirements=[
                    lambda x: x in
                    [constants.Constants.UPPER, constants.Constants.LOWER]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.ENVIRONMENT],
        dependent_variables_required_values=[[constants.Constants.MINIGRID]],
        level=[constants.Constants.MINIGRID],
    )

    _minigrid_curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TRANSITION_EPISODES,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) for y in x)],
            ),
            config_field.Field(
                name=constants.Constants.ENVIRONMENT_CHANGES,
                types=[list],
            ),
        ],
        level=[constants.Constants.MINIGRID_CURRICULUM],
        dependent_variables=[
            constants.Constants.ENVIRONMENT,
            constants.Constants.APPLY_CURRICULUM,
        ],
        dependent_variables_required_values=[[constants.Constants.MINIGRID],
                                             [True]],
    )

    _multiroom_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.Constants.ASCII_MAP_PATH,
                               types=[str]),
            config_field.Field(
                name=constants.Constants.EPISODE_TIMEOUT,
                types=[type(None), int],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.Constants.PLOT_ORIGIN,
                types=[str],
                requirements=[
                    lambda x: x in
                    [constants.Constants.UPPER, constants.Constants.LOWER]
                ],
            ),
        ],
        level=[constants.Constants.MULTIROOM],
        dependent_variables=[
            constants.Constants.ENVIRONMENT,
        ],
        dependent_variables_required_values=[[constants.Constants.MULTIROOM]],
    )

    _atari_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.ATARI_ENV_NAME,
                types=[str],
                requirements=[lambda x: x in constants.Constants.ATARI_ENVS],
            ),
            config_field.Field(
                name=constants.Constants.IMPLEMENTATION,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.FUNCTIONAL, constants.Constants.
                        WRAPPER
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.PRE_PROCESSING,
                types=[list],
                requirements=[lambda x: all(isinstance(y, dict) for y in x)],
            ),
            config_field.Field(
                name=constants.Constants.FRAME_STACK,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.FRAME_SKIP,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.EPISODE_TIMEOUT,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.Constants.ENCODED_STATE_DIMENSIONS,
                types=[list],
                requirements=[
                    lambda x: all(isinstance(y, int) and y > 0 for y in x)
                ],
            ),
            config_field.Field(
                name=constants.Constants.PLOT_ORIGIN,
                types=[str],
                requirements=[
                    lambda x: x in
                    [constants.Constants.UPPER, constants.Constants.LOWER]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.ENVIRONMENT],
        dependent_variables_required_values=[[constants.Constants.ATARI]],
        level=[constants.Constants.ATARI],
    )

    _hard_coded_vp_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.VP_SCHEDULE,
                types=[list],
                requirements=[
                    lambda x: all(
                        isinstance(y, list) and len(y) == 2 for y in x)
                ],
            )
        ],
        level=[constants.Constants.LEARNER, constants.Constants.HARD_CODED],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.Constants.HARD_CODED]],
    )

    _adaptive_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.Constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MEAN,
                        constants.Constants.MAX,
                        constants.Constants.SELECT,
                    ]
                ],
            ),
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.ADAPTIVE_UNCERTAINTY
        ],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[
            constants.Constants.ADAPTIVE_UNCERTAINTY
        ]],
    )

    _adaptive_arriving_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.Constants.ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MEAN,
                        constants.Constants.MAX,
                    ]
                ],
            ),
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.ADAPTIVE_ARRIVING_UNCERTAINTY,
        ],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[
            constants.Constants.ADAPTIVE_ARRIVING_UNCERTAINTY
        ]],
    )

    _potential_based_adaptive_uncertainty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.Constants.PRE_ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MEAN,
                        constants.Constants.MAX,
                        constants.Constants.SELECT,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.POST_ACTION_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MEAN,
                        constants.Constants.MAX,
                    ]
                ],
            ),
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY,
        ],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[
            constants.Constants.POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY
        ]],
    )

    _policy_entropy_penalty_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.MULTIPLICATIVE_FACTOR,
                types=[float, int],
            ),
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.POLICY_ENTROPY_PENALTY,
        ],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[
            constants.Constants.POLICY_ENTROPY_PENALTY
        ]],
    )

    _constant_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            )
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.EPSILON,
            constants.Constants.CONSTANT,
        ],
        dependent_variables=[constants.Constants.SCHEDULE],
        dependent_variables_required_values=[[constants.Constants.CONSTANT]],
    )

    _linear_decay_epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.INITIAL_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.Constants.FINAL_VALUE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.Constants.ANNEAL_DURATION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[
            constants.Constants.LEARNER,
            constants.Constants.EPSILON,
            constants.Constants.LINEAR_DECAY,
        ],
        dependent_variables=[constants.Constants.SCHEDULE],
        dependent_variables_required_values=[
            [constants.Constants.LINEAR_DECAY],
        ],
    )

    _epsilon_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.SCHEDULE,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.CONSTANT, constants.Constants.
                        LINEAR_DECAY
                    ]
                ],
            ),
        ],
        level=[constants.Constants.LEARNER, constants.Constants.EPSILON],
        nested_templates=[
            _constant_epsilon_template, _linear_decay_epsilon_template
        ],
    )

    _learner_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TYPE,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.SARSA_LAMBDA, constants.Constants.
                        Q_LEARNING, constants.Constants.DQN, constants.Constants
                        .ENSEMBLE_Q_LEARNING, constants.Constants.ENSEMBLE_DQN
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.LEARNING_RATE,
                types=[float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.GRADIENT_MOMENTUM,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.SQUARED_GRADIENT_MOMENTUM,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.MIN_SQUARED_GRADIENT,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.DISCOUNT_FACTOR,
                types=[float, int],
                requirements=[lambda x: x <= 1 and x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.INITIALISATION,
                types=[str, float, int],
                requirements=[
                    lambda x: x in [
                        constants.Constants.RANDOM,
                        constants.Constants.ZEROS,
                        constants.Constants.ONES,
                    ] or isinstance(x, (int, float))
                ],
            ),
            config_field.Field(
                name=constants.Constants.VISITATION_PENALTY_TYPE,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.HARD_CODED,
                        constants.Constants.ADAPTIVE_UNCERTAINTY,
                        constants.Constants.ADAPTIVE_ARRIVING_UNCERTAINTY,
                        constants.Constants.
                        POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY,
                        constants.Constants.POLICY_ENTROPY_PENALTY,
                    ]
                ],
            ),
        ],
        level=[constants.Constants.LEARNER],
        nested_templates=[
            _epsilon_template,
            _hard_coded_vp_template,
            _adaptive_uncertainty_template,
            _adaptive_arriving_uncertainty_template,
            _potential_based_adaptive_uncertainty_template,
            _policy_entropy_penalty_template,
        ],
    )

    _sarsa_lambda_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TRACE_LAMBDA,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.Constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[
            [constants.Constants.SARSA_LAMBDA],
        ],
        level=[constants.Constants.SARSA_LAMBDA],
    )

    _q_learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[[constants.Constants.Q_LEARNING]],
        level=[constants.Constants.Q_LEARNING],
    )

    _ensemble_q_learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.NUM_LEARNERS,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.Constants.COPY_LEARNER_INITIALISATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGETS,
                types=[list],
                requirements=[
                    lambda x: x is None or all(y in [
                        constants.Constants.GREEDY_SAMPLE,
                        constants.Constants.GREEDY_MEAN,
                        constants.Constants.GREEDY_VOTE,
                    ] for y in x)
                ],
            ),
            config_field.Field(name=constants.Constants.PARALLELISE_ENSEMBLE,
                               types=[bool]),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[[
            constants.Constants.ENSEMBLE_Q_LEARNING
        ]],
        level=[constants.Constants.ENSEMBLE_Q_LEARNING],
    )

    _ensemble_dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.NUM_LEARNERS,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.Constants.COPY_LEARNER_INITIALISATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.GREEDY, constants.Constants.
                        EPSILON_GREEDY
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGETS,
                types=[list, type(None)],
                requirements=[lambda x: x is None or all(y in [] for y in x)],
            ),
            config_field.Field(name=constants.Constants.PARALLELISE_ENSEMBLE,
                               types=[bool]),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[
            [constants.Constants.ENSEMBLE_DQN],
        ],
        level=[constants.Constants.ENSEMBLE_DQN],
    )

    _dqn_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.NUM_REPLAY_FILL_TRAJECTORIES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.REPLAY_BUFFER_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TARGET_NETWORK_UPDATE_PERIOD,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(name=constants.Constants.NORMALISE_STATE,
                               types=[bool]),
            config_field.Field(
                name=constants.Constants.GRADIENT_CLIPPING,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or (len(x) == 2 and all(
                        (isinstance(y, float) or isinstance(y, int))
                        for y in x))
                ],
            ),
            config_field.Field(
                name=constants.Constants.OPTIMISER,
                types=[str],
                requirements=[
                    lambda x: x in
                    [constants.Constants.ADAM, constants.Constants.RMS_PROP]
                ],
            ),
            config_field.Field(
                name=constants.Constants.LAYER_SPECIFICATIONS,
                types=[list],
            ),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[[
            constants.Constants.DQN, constants.Constants.ENSEMBLE_DQN
        ]],
        level=[constants.Constants.DQN],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.NUM_EPISODES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TESTING,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x)
                ],
            ),
        ],
        level=[constants.Constants.TRAINING],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.PRINT_FREQUENCY,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.Constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.ANIMATION_LIBRARY,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MATPLOTLIB_ANIMATION,
                        constants.Constants.IMAGEIO,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.ANIMATION_FILE_FORMAT,
                types=[str],
                requirements=[
                    lambda x: x in
                    [constants.Constants.MP4, constants.Constants.GIF]
                ],
            ),
            config_field.Field(
                name=constants.Constants.ARRAYS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x)
                ],
            ),
            config_field.Field(
                name=constants.Constants.SCALARS,
                types=[list],
                requirements=[
                    lambda x: all(
                        isinstance(y, list) and isinstance(y[1], int)
                        for y in x),
                    lambda x: all((isinstance(z, str) or
                                   (isinstance(z, list) and isinstance(
                                       z[0], str) and isinstance(z[1], int))
                                   for z in y[0])
                                  for y in x),
                ],
            ),
            config_field.Field(
                name=constants.Constants.VISUALISATIONS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(
                        isinstance(y, list) and isinstance(y[1], int)
                        for y in x),
                    lambda x: x is None or all(
                        (isinstance(z, str) or
                         (isinstance(z, list) and isinstance(z[0], str) and
                          isinstance(z[1], int)) for z in y[0]) for y in x),
                ],
            ),
        ],
        level=[constants.Constants.LOGGING],
    )

    _post_processing_template = config_template.Template(
        fields=[
            # config_field.Field(
            #     name=constants.Constants.PLOT_TAGS,
            #     types=[list],
            #     requirements=[lambda x: all(isinstance(y, str) for y in x)],
            # ),
            config_field.Field(
                name=constants.Constants.SMOOTHING,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.Constants.POST_PROCESSING],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(name=constants.Constants.USE_GPU, types=[bool]),
            config_field.Field(
                name=constants.Constants.GPU_ID,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.ENVIRONMENT,
                types=[str],
                requirements=[
                    lambda x: x in [
                        constants.Constants.MINIGRID,
                        constants.Constants.ATARI,
                        constants.Constants.MULTIROOM,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.APPLY_CURRICULUM,
                types=[bool],
            ),
        ],
        nested_templates=[
            _minigrid_template,
            _minigrid_curriculum_template,
            _multiroom_template,
            _atari_template,
            _learner_template,
            _sarsa_lambda_template,
            _q_learning_template,
            _ensemble_q_learning_template,
            _ensemble_dqn_template,
            _dqn_template,
            _training_template,
            _logging_template,
            _post_processing_template,
        ],
    )
