class K:
    EPISODES = 3000
    LEARNING_RATE = 0.25e-3
    MEMORY_SIZE = 100_000
    EPSILON_DECAY = 0.99995
    EPSILON_MIN = 0.05
    EPSILON_INIT = 1
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.99
    STATE_SIZE = 8
    ACTION_SIZE = 4
    PRINT_EVERY = 100
    MODEL_EXPORT_NAME = 'results/model_success.h5'

    def __init__(self, env):
        # You can override if you want to pass in a different environment
        # but we'll hard code the lunar landar's because that's what we're using
        self.STATE_SIZE = env.observation_space.shape[0]
        self.ACTION_SIZE = env.action_space.n
