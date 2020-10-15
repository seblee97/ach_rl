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
                name=constants.Constants.REWARD_POSITION,
                types=[list, type(None)],
                requirements=[
                    lambda x: x is None or (len(x) == 2 and all(y >= 0 for y in x))
                ],
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

    _epislon_greedy_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.EPSILON,
                types=[float],
                requirements=[lambda x: x <= 1 and x >= 0],
            )
        ],
        dependent_variables=[constants.Constants.POLICY],
        dependent_variables_required_values=[[constants.Constants.EPSILON_GREEDY]],
        level=[constants.Constants.Q_LEARNER, constants.Constants.EPSILON_GREEDY],
    )

    _q_learner_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.ALPHA,
                types=[float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.GAMMA,
                types=[float],
                requirements=[lambda x: x <= 1 and x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.POLICY,
                types=[str],
                requirements=[lambda x: x in [constants.Constants.EPSILON_GREEDY]],
            ),
        ],
        dependent_variables=[constants.Constants.LEARNER],
        dependent_variables_required_values=[[constants.Constants.Q_LEARNER]],
        nested_templates=[_epislon_greedy_template],
        level=[constants.Constants.Q_LEARNER],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.NUM_EPISODES,
                types=[int],
                requirements=[lambda x: x > 0],
            )
        ],
        level=[constants.Constants.TRAINING],
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
                name=constants.Constants.LEARNER,
                types=[str],
                requirements=[lambda x: x in [constants.Constants.Q_LEARNER]],
            ),
        ],
        nested_templates=[_minigrid_template, _q_learner_template, _training_template],
    )
