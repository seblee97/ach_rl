class Constants:

    EPSILON_GREEDY = "epsilon_greedy"

    # config
    SIZE = "size"
    LIVING_REWARD = "living_reward"
    NO_OP_PENALTY = "no_op_penalty"
    STARTING_POSITION = "starting_position"
    REWARD_POSITIONS = "reward_positions"
    ENVIRONMENT = "environment"
    MINIGRID = "minigrid"
    EPISODE_TIMEOUT = "episode_timeout"
    EPSILON = "epsilon"
    POLICY = "policy"
    EPSILON_GREEDY = "epsilon_greedy"
    Q_LEARNER = "q_learner"
    SARSA_LAMBDA = "sarsa_lambda"
    TRACE_LAMBDA = "trace_lambda"
    LEARNING_RATE = "learning_rate"
    DISCOUNT_FACTOR = "discount_factor"
    LEARNER = "learner"
    LEARNERS = "learners"
    TRAINING = "training"
    NUM_EPISODES = "num_episodes"
    EXPERIMENT_NAME = "experiment_name"
    SEED = "seed"
    LOGGING = "logging"
    SCALARS = "scalars"
    VISUALISATIONS = "visualisations"
    ARRAYS = "arrays"
    PLOTS = "plots"
    TRAIN_EPISODE_LENGTH = "train_episode_length"
    TRAIN_EPISODE_REWARD = "train_episode_reward"
    TEST_EPISODE_LENGTH = "test_episode_length"
    TEST_EPISODE_REWARD = "test_episode_reward"
    TEST_FREQUENCY = "test_frequency"
    LOGFILE_PATH = "logfile_path"
    CHECKPOINT_FREQUENCY = "checkpoint_frequency"
    EPISODE = "episode"
    INDIVIDUAL_TEST_RUN = "individual_test_run"
    INDIVIDUAL_TRAIN_RUN = "individual_train_run"
    INDIVIDUAL_NO_REP_TEST_RUN = "individual_no_rep_test_run"
    INDIVIDUAL_NO_REP_TRAIN_RUN = "individual_no_rep_train_run"
    FULL_TEST_LOG_FREQUENCY = "full_test_log_frequency"
    VISITATION_COUNT_HEATMAP = "visitation_count_heatmap"
    TRAIN_LOG_FREQUENCY = "train_log_frequency"
    GREEDY = "greedy"
    BEHAVIOUR = "behaviour"
    TARGET = "target"
    TARGETS = "targets"
    NUM_REWARDS = "num_rewards"
    REWARD_MAGNITUDES = "reward_magnitudes"
    REPEAT_REWARDS = "repeat_rewards"
    PLOT_EPISODE_REWARDS = "plot_episode_rewards"
    PLOT_EPISODE_LENGTHS = "plot_episode_lengths"
    RESULTS = "results"
    CHECKPOINT_PATH = "checkpoint_path"
    EXPERIMENT_TIMESTAMP = "experiment_timestamp"
    VISITATION_PENALTY_TYPE = "visitation_penalty_type"
    VISITATION_PENALTY = "visitation_penalty"
    INITIALISATION = "initialisation"
    RANDOM = "random"
    ZEROS = "zeros"
    ONES = "ones"
    TYPE = "type"
    SARSA = "sarsa"
    EXPECTED_SARSA = "expected_sarsa"
    Q_LEARNING = "q_learning"
    POST_PROCESSING = "post_processing"
    PLOT_TAGS = "plot_tags"
    PLOT_PDF = "plot.pdf"
    NO_REPEAT_TEST_EPISODE_LENGTH = "no_repeat_test_episode_length"
    NO_REPEAT_TEST_EPISODE_REWARD = "no_repeat_test_episode_reward"
    MAX_VALUES_PDF = "max_values.pdf"
    QUIVER_VALUES_PDF = "quiver_values.pdf"
    QUIVER_MAX_VALUES_PDF = "quiver_max_values.pdf"
    VALUE_FUNCTION = "value_function"
    CYCLE_COUNT = "cycle_count"
    PARALLEL = "parallel"
    SERIAL = "serial"
    STARTING_XY = "starting_xy"
    REWARD_XY = "reward_xy"
    TRANSITION_EPISODES = "transition_episodes"
    ENVIRONMENT_CHANGES = "environment_changes"
    APPLY_CURRICULUM = "apply_curriculum"
    CHANGE_STARTING_POSITION = "change_starting_position"
    MINIGRID_CURRICULUM = "minigrid_curriculum"
    HARD_CODED = "hard_coded"
    VP_SCHEDULE = "vp_schedule"
    SINGLE = "single"
    PRINT_FREQUENCY = "print_frequency"
    CONV = "conv"
    FC = "fc"
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"
    PADDING = "padding"
    IN_FEATURES = "in_features"
    OUT_FEATURES = "out_features"
    RELU = "relu"
    IDENTITY = "identity"
    LAYER_TYPE = "layer_type"
    LAYER_DIMENSIONS = "layer_dimensions"
    NONLINEARITY = "nonlinearity"
    TRANSITION = "transition"
    STATE_ENCODING = "state_encoding"
    ACTION = "action"
    REWARD = "reward"
    NEXT_STATE_ENCODING = "next_state_encoding"
    ATARI = "atari"
    ATARI_ENV_NAME = "atari_env_name"
    DQN = "dqn"
    REPLAY_BUFFER_SIZE = "replay_buffer_size"
    FRAME_STACK = "frame_stack"
    FRAME_SKIP = "frame_skip"
    PRE_PROCESSING = "pre_processing"
    DOWN_SAMPLE = "down_sample"
    GRAY_SCALE = "gray_scale"
    LAYER_SPECIFICATIONS = "layer_specifications"
    ENCODED_STATE_DIMENSIONS = "encoded_state_dimensions"
    WIDTH = "width"
    HEIGHT = "height"
    MAX_OVER = "max_over"
    NUM_FRAMES = "num_frames"
    NUM_FILTERS = "num_filters"
    FLATTEN = "flatten"
    TESTING = "testing"
    NO_REP = "no_rep"
    TARGET_NETWORK_UPDATE_PERIOD = "target_network_update_period"
    OPTIMISER = "optimiser"
    ADAM = "adam"
    RMS_PROP = "rms_prop"
    GRADIENT_MOMENTUM = "gradient_momentum"
    SQUARED_GRADIENT_MOMENTUM = "squared_gradient_momentum"
    MIN_SQUARED_GRADIENT = "min_squared_gradient"
    ACTIVE = "active"
    BATCH_SIZE = "batch_size"
    NUM_REPLAY_FILL_TRAJECTORIES = "num_replay_fill_trajectories"
    VALUE = "value"
    SCHEDULE = "schedule"
    LINEAR_DECAY = "linear_decay"
    ANNEAL_DURATION = "anneal_duration"
    INITIAL_VALUE = "initial_value"
    FINAL_VALUE = "final_value"
    CONSTANT = "constant"
    USE_GPU = "use_gpu"
    GPU_ID = "gpu_id"
    USING_GPU = "using_gpu"
    EXPERIMENT_DEVICE = "experiment_device"
    AVERAGE_ACTION_VALUE = "average_action_value"
    ANIMATION_FILE_FORMAT = "animation_file_format"
    NORMAL = "normal"
    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    NETWORK_INITIALISATION = "network_initialisation"
    WEIGHT_INITIALISATION = "weight_initialisation"
    BIAS_INITIALISATION = "bias_initialisation"
    NETWORK_WEIGHT_INITIALISATION = "network_weight_initialisation"
    NETWORK_BIAS_INITIALISATION = "network_bias_initialisation"
    MULTIROOM = "multiroom"
    ASCII_MAP_PATH = "ascii_map_path"
    WALL_CHARACTER = "#"
    OPEN_CHARACTER = " "
    REWARD_CHARACTER = "R"
    KEY_CHARACTER = "K"
    DOOR_CHARACTER = "D"
    START_CHARACTER = "S"
    PLOT_ORIGIN = "plot_origin"
    UPPER = "upper"
    LOWER = "lower"
    ENSEMBLE_Q_LEARNING = "ensemble_q_learning"
    NUM_LEARNERS = "num_learners"
    GREEDY_SAMPLE = "greedy_sample"
    GREEDY_MEAN = "greedy_mean"
    GREEDY_VOTE = "greedy_vote"
    PARALLELISE_ENSEMBLE = "parallelise_ensemble"
    ADAPTIVE_UNCERTAINTY = "adaptive_uncertainty"
    POTENTIAL_BASED_ADAPTIVE_UNCERTAINTY = "potential_based_adaptive_uncertainty"
    ENSEMBLE_RUNNER = "ensemble_runner"
    ENSEMBLE_EPISODE_REWARD_STD = "ensemble_episode_reward_std"
    ENSEMBLE_EPISODE_LENGTH_STD = "ensemble_episode_length_std"
    MEAN_VISITATION_PENALTY = "mean_visitation_penalty"
    SMOOTHING = "smoothing"
    MULTIPLICATIVE_FACTOR = "multiplicative_factor"
    LOSS = "loss"
    GRADIENT_CLIPPING = "gradient_clipping"
    MAX_OVER_ACTIONS = "max_over_actions"
    NORMALISE_STATE = "normalise_state"
    IMPLEMENTATION = "implementation"
    FUNCTIONAL = "functional"
    WRAPPER = "wrapper"
    ANIMATION_LIBRARY = "animation_library"
    MATPLOTLIB_ANIMATION = "matplotlib_animation"
    IMAGEIO = "imageio"
    MP4 = "mp4"
    GIF = "gif"
    CONFIG_CHANGES = "config_changes"
    MAX = "max"
    SELECT = "select"
    MEAN = "mean"
    ACTION_FUNCTION = "action_function"
    PRE_ACTION_FUNCTION = "pre_action_function"
    POST_ACTION_FUNCTION = "post_action_function"
    MEAN_PENALTY_INFO = "mean_penalty_info"
    UNCERTAINTY = "uncertainty"
    CURRENT_STATE_UNCERTAINTY = "current_state_uncertainty"
    CURRENT_STATE_MAX_UNCERTAINTY = "current_state_max_uncertainty"
    CURRENT_STATE_MEAN_UNCERTAINTY = "current_state_mean_uncertainty"
    CURRENT_STATE_SELECT_UNCERTAINTY = "current_state_select_uncertainty"
    NEXT_STATE_UNCERTAINTY = "next_state_uncertainty"
    NEXT_STATE_MEAN_UNCERTAINTY = "next_state_mean_uncertainty"
    NEXT_STATE_MAX_UNCERTAINTY = "next_state_max_uncertainty"
    ADAPTIVE_ARRIVING_UNCERTAINTY = "adaptive_arriving_uncertainty"
    INDIVIDUAL_VALUE_FUNCTIONS = "individual_value_functions"
    VALUE_FUNCTION_STD = "value_function_std"
    VALUE_FUNCTION_STD_PDF = "value_function_std.pdf"
    NEXT_STATE_POLICY_ENTROPY = "next_state_policy_entropy"
    CURRENT_STATE_POLICY_ENTROPY = "current_state_policy_entropy"
    RUN_PATH = "run_path"
    STD = "std"
    COPY_LEARNER_INITIALISATION = "copy_learner_initialisation"
    POLICY_ENTROPY_PENALTY = "policy_entropy_penalty"
    LOG_FILE_NAME = "experiment.log"
    LOG_FORMAT = "%(asctime)s  %(name)8s  %(levelname)5s  %(message)s"
    ENSEMBLE_DQN = "ensemble_dqn"
    CLUSTER = "cluster"
    ERROR_FILE_NAME = "error.txt"
    OUTPUT_FILE_NAME = "output.txt"
    INDIVIDUAL = "individual"
    STD_PENALTY_INFO = "std_penalty_info"
    CLUSTER_ARRAY = "cluster_array"
    CONFIG_CHANGES_SYM_PATH = "config_changes_symbolic_dir"
    ERROR_FILES_SYM_PATH = "error_files_symbolic_dir"
    OUTPUT_FILES_SYM_PATH = "output_files_symbolic_dir"
    CHECKPOINTS_SYM_PATH = "checkpoints_symbolic_dir"
    CONFIG_CHANGES_JSON = "config_changes.json"
    ACH = "ach"
    JOB_SCRIPT = "job_script"
    VALUE_FUNCTION_ANIMATION = "value_function_animation"
    INDIVIDUAL_VALUE_FUNCTION_ANIMATION = "individual_value_function_animation"
    VALUE_FUNCTION_STD_ANIMATION = "value_function_std_animation"
    POST_VISUALISATIONS = "post_visualisations"
    SHARED_LAYERS = "shared_layers"
    SHARE_REPLAY_BUFFER = "share_replay_buffer"
    MASK = "mask"
    BOOTSTRAPPED_ENSEMBLE_DQN = "bootstrapped_ensemble_dqn"
    INDEPENDENT_ENSEMBLE_DQN = "independent_ensemble_dqn"
    MASK_PROBABILITY = "mask_probability"
    POLICY_ENTROPY = "policy_entropy"
    SHAPING_IMPLEMENTATION = "shaping_implementation"
    ACT = "act"
    TRAIN_Q_NETWORK = "train_q_network"
    TRAIN_TARGET_NETWORK = "train_target_network"
    PENALTY = "penalty"
    LOG_EPSILON = 1e-8

    ATARI_ENVS = ["PongNoFrameskip-v4"]

    GRAPH_LAYOUTS = {
        1: (1, 1),
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
        10: (4, 3),
        11: (4, 3),
        12: (4, 3),
    }
