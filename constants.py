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
    TRAINING = "training"
    NUM_EPISODES = "num_episodes"
    EXPERIMENT_NAME = "experiment_name"
    SEED = "seed"
    LOGGING = "logging"
    COLUMNS = "columns"
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

    ATARI_ENVS = ["Pong-v0"]