from config_manager import config_field
from config_manager import config_template

import constants


class ConfigTemplate:

    _minigrid_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.SIZE,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
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
                    lambda x: x is None or (len(x) == 2 and all(y >= 0 for y in x))
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
                    lambda x: x is None or all(all(z >= 0 for z in y) for y in x),
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
        dependent_variables_required_values=[[constants.Constants.MINIGRID], [True]],
    )

    _hard_coded_vp_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.SCHEDULE,
                types=[list],
                requirements=[
                    lambda x: all(isinstance(y, list) and len(y) == 2 for y in x)
                ],
            )
        ],
        level=[constants.Constants.LEARNER, constants.Constants.HARD_CODED],
        dependent_variables=[constants.Constants.VISITATION_PENALTY_TYPE],
        dependent_variables_required_values=[[constants.Constants.HARD_CODED]],
    )

    _learner_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.Constants.SARSA_LAMBDA,
                        constants.Constants.Q_LEARNING,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.Constants.LEARNING_RATE,
                types=[float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.DISCOUNT_FACTOR,
                types=[float, int],
                requirements=[lambda x: x <= 1 and x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.EPSILON,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.Constants.INITIALISATION,
                types=[str, float, int],
                requirements=[
                    lambda x: x
                    in [
                        constants.Constants.RANDOM,
                        constants.Constants.ZEROS,
                        constants.Constants.ONES,
                    ]
                    or isinstance(x, (int, float))
                ],
            ),
            config_field.Field(
                name=constants.Constants.VISITATION_PENALTY_TYPE,
                types=[str],
                requirements=[lambda x: x in [constants.Constants.HARD_CODED]],
            ),
        ],
        level=[constants.Constants.LEARNER],
        nested_templates=[_hard_coded_vp_template],
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
                    lambda x: x
                    in [constants.Constants.GREEDY, constants.Constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.GREEDY, constants.Constants.EPSILON_GREEDY]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[constants.Constants.SARSA_LAMBDA],
        level=[constants.Constants.SARSA_LAMBDA],
    )

    _q_learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.BEHAVIOUR,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.GREEDY, constants.Constants.EPSILON_GREEDY]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TARGET,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.GREEDY, constants.Constants.EPSILON_GREEDY]
                ],
            ),
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[constants.Constants.Q_LEARNING],
        level=[constants.Constants.Q_LEARNING],
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
                name=constants.Constants.TRAIN_LOG_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.FULL_TEST_LOG_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.Constants.TRAINING],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.COLUMNS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, str) for y in x)],
            ),
            config_field.Field(
                name=constants.Constants.ARRAYS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x)
                ],
            ),
            config_field.Field(
                name=constants.Constants.PLOTS,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or all(isinstance(y, str) for y in x)
                ],
            ),
        ],
        level=[constants.Constants.LOGGING],
    )

    _post_processing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.PLOT_TAGS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, str) for y in x)],
            )
        ],
        level=[constants.Constants.POST_PROCESSING],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.Constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.ENVIRONMENT,
                types=[str],
                requirements=[lambda x: x in [constants.Constants.MINIGRID]],
            ),
            config_field.Field(
                name=constants.Constants.APPLY_CURRICULUM,
                types=[bool],
            ),
        ],
        nested_templates=[
            _minigrid_template,
            _minigrid_curriculum_template,
            _learner_template,
            _sarsa_lambda_template,
            _q_learning_template,
            _training_template,
            _logging_template,
            _post_processing_template,
        ],
    )
