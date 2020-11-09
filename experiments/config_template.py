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
                name=constants.Constants.VISITATION_PENALTY,
                types=[float, int, type(None)],
                requirements=[lambda x: x is None or x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.INITIALISATION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.Constants.RANDOM,
                        constants.Constants.ZEROS,
                        constants.Constants.ONES,
                    ]
                ],
            ),
        ],
        level=[constants.Constants.LEARNER],
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
        ],
        dependent_variables=[constants.Constants.TYPE],
        dependent_variables_required_values=[constants.Constants.SARSA_LAMBDA],
        level=[constants.Constants.SARSA_LAMBDA],
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
        ],
        nested_templates=[
            _minigrid_template,
            _learner_template,
            _sarsa_lambda_template,
            _training_template,
            _logging_template,
        ],
    )